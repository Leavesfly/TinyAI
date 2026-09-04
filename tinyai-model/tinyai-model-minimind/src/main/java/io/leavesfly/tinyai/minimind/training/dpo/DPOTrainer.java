package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.minimind.training.dataset.DPODataset;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * DPO (Direct Preference Optimization) 训练器
 * 
 * DPO直接从偏好数据优化策略模型,无需显式奖励模型
 * 核心思想:通过最大化preferred响应相对于rejected响应的隐式奖励差异
 * 
 * 训练流程:
 * 1. 加载SFT模型作为初始模型
 * 2. 创建参考模型(冻结的模型副本)
 * 3. 对每个偏好对(prompt, chosen, rejected):
 *    - 前向传播计算策略模型和参考模型的log概率
 *    - 计算DPO损失
 *    - 反向传播更新策略模型
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPOTrainer extends BaseTrainer {
    
    private final MiniMindModel referenceModel;   // 参考模型(冻结)
    private final MiniMindConfig config;
    private final DPODataset dataset;
    private final DPOConfig dpoConfig;
    private final DPOLoss dpoLoss;
    private final Adam optimizer;
    
    // 训练配置
    private float learningRate;
    private List<Float> accuracyHistory;  // 记录chosen>rejected的比例
    
    /**
     * 构造函数
     * 
     * @param policyModel 策略模型(将被训练)
     * @param dataset DPO数据集
     * @param dpoConfig DPO配置
     */
    public DPOTrainer(MiniMindModel policyModel, DPODataset dataset, DPOConfig dpoConfig) {
        super(policyModel);
        this.config = policyModel.getConfig();
        this.dataset = dataset;
        this.dpoConfig = dpoConfig;
        
        // 验证配置
        dpoConfig.validate();
        
        // 创建参考模型（参数深拷贝 + eval + requireGrad=false，一步到位）
        this.referenceModel = policyModel.createFrozenCopy("reference_model");
        
        // 创建DPO损失函数
        this.dpoLoss = new DPOLoss(dpoConfig.getBeta(), dpoConfig.getLabelSmoothing(),
                                   dpoConfig.isUseLengthNormalization());
        
        // 默认训练配置
        this.maxEpochs = 3;
        this.learningRate = 5e-6f;  // DPO通常使用非常小的学习率
        this.maxGradNorm = 1.0f;
        this.logInterval = 10;
        this.saveInterval = 500;
        this.checkpointDir = "./checkpoints/minimind/dpo";

        // 创建优化器
        this.optimizer = new Adam(policyModel, learningRate, 0.9f, 0.999f, 1e-8f);
        
        this.accuracyHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public DPOTrainer configure(int maxEpochs, float learningRate, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.maxGradNorm = maxGradNorm;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 开始训练
     * <p>
     * 复用 {@link BaseTrainer#train()} 的训练循环（含结束后自动复位 eval 模式），
     * 仅在完成后补打统计信息。
     */
    @Override
    public void train() {
        super.train();
        printTrainingStats();
    }
        
    /**
     * 训练一个epoch
     */
    @Override
    protected void trainOneEpoch() {
        prepareDataset();
        model.setTraining(true);
            
        long epochStartTime = System.currentTimeMillis();
        double epochLoss = 0.0;
        double epochAccuracy = 0.0;
        int batchCount = 0;
        int skippedCount = 0;
            
        while (hasNextBatch()) {
            Object batch = getNextBatch();
            float[] metrics = trainStepImpl(batch);
            float stepLoss = metrics[0];
            float stepAccuracy = metrics[1];
                
            currentStep++;
                
            // 跳过 NaN/Inf 损失，避免污染统计
            if (Float.isNaN(stepLoss) || Float.isInfinite(stepLoss)) {
                skippedCount++;
                System.err.printf("警告: Step %d 损失异常(%s)，已跳过累加与历史记录%n",
                    currentStep, Float.isNaN(stepLoss) ? "NaN" : "Inf");
            } else {
                epochLoss += stepLoss;
                epochAccuracy += stepAccuracy;
                batchCount++;
                lossHistory.add(stepLoss);
                accuracyHistory.add(stepAccuracy);
            }
                
            // 打印日志
            if (currentStep % logInterval == 0) {
                printTrainingLog();
            }
                
            // 保存检查点
            if (currentStep % saveInterval == 0) {
                saveCheckpoint();
            }
        }
            
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        double avgEpochAcc = batchCount > 0 ? epochAccuracy / batchCount : 0.0;
            
        if (skippedCount > 0) {
            System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 平均准确率: %.2f%% | 跳过批次: %d | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, avgEpochAcc * 100, skippedCount, epochEndTime - epochStartTime);
        } else {
            System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 平均准确率: %.2f%% | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, avgEpochAcc * 100, epochEndTime - epochStartTime);
        }
            
        onEpochEnd();
        resetDataset();
    }
        
    /**
     * 训练一步（私有实现，返回loss和accuracy）
     * 
     * 计算图设计要点:
     * 1. 参考模型使用独立的 NdArray 副本，避免与策略模型共享输入导致数据污染
     * 2. 策略模型的 chosen 和 rejected 拼接为一次前向传播，避免两次 predict 导致
     *    模型内部中间状态被覆盖
     * 3. 拆回 chosen / rejected 时用 sliceRange 而非 split：split 是多输出函数，
     *    两个分片汇入同一损失时，先被访问分片的梯度会被后一个分片的
     *    multi-output backward 重复计入（其 .grad 未被清理），造成 chosen 分支梯度翻倍
     * 4. clearGrads 在前向传播之前调用，确保梯度从干净状态开始累积
     * 
     * @return [loss, accuracy]
     */
    private float[] trainStepImpl(Object batch) {
        DPODataset.Batch dpoBatch = (DPODataset.Batch) batch;
            
        // 获取数据（chosen / rejected 各自拥有独立掩码）
        NdArray chosenInput = dpoBatch.getChosenInput();
        NdArray chosenLabels = dpoBatch.getChosenLabels();
        NdArray rejectedInput = dpoBatch.getRejectedInput();
        NdArray rejectedLabels = dpoBatch.getRejectedLabels();
        NdArray chosenMask = dpoBatch.getChosenMask();
        NdArray rejectedMask = dpoBatch.getRejectedMask();
            
        int batchSize = chosenInput.getShape().getDimension(0);
            
        // ========== 1. 参考模型前向传播 (不需要梯度) ==========
        // 使用 NdArray 的深拷贝作为参考模型输入，避免与策略模型共享底层数据
        referenceModel.setTraining(false);
        Variable refChosenLogits = referenceModel.predict(new Variable(chosenInput.copy())).detach();
        Variable refRejectedLogits = referenceModel.predict(new Variable(rejectedInput.copy())).detach();
            
        float[] refChosenLogProb = dpoLoss.computeLogProbsDetached(
            refChosenLogits, chosenLabels, chosenMask);
        float[] refRejectedLogProb = dpoLoss.computeLogProbsDetached(
            refRejectedLogits, rejectedLabels, rejectedMask);
            
        // 释放参考模型的计算图，节省内存
        refChosenLogits.unChainBackward();
        refRejectedLogits.unChainBackward();
            
        // ========== 2. 策略模型前向传播 (保持计算图) ==========
        model.setTraining(true);
        model.clearGrads();  // 在前向传播前清零梯度，确保干净状态
            
        // 沿 batch 维度拼接: [chosen_batch; rejected_batch] -> [2*batch, seq_len]
        Variable combinedInput = Variable.cat(
            new Variable[]{new Variable(chosenInput), new Variable(rejectedInput)}, 0);
            
        // 单次前向传播，计算图完整连贯（同时取出 MoE 负载均衡损失）
        MiniMindBlock.MoEOutput output = forwardWithAux(combinedInput);
        Variable combinedLogits = output.getOutput();
            
        // 沿 batch 维度拆回 chosen 和 rejected
        Variable policyChosenLogits = combinedLogits.sliceRange(0, 0, batchSize);
        Variable policyRejectedLogits = combinedLogits.sliceRange(0, batchSize, 2 * batchSize);
            
        // ========== 3. 计算策略模型的 log 概率 (在计算图中) ==========
        Variable chosenLabelsVar = new Variable(chosenLabels);
        Variable rejectedLabelsVar = new Variable(rejectedLabels);
        Variable chosenMaskVar = new Variable(chosenMask);
        Variable rejectedMaskVar = new Variable(rejectedMask);
        chosenLabelsVar.setRequireGrad(false);
        rejectedLabelsVar.setRequireGrad(false);
        chosenMaskVar.setRequireGrad(false);
        rejectedMaskVar.setRequireGrad(false);
            
        Variable policyChosenLogProbs = dpoLoss.computeLogProbs(
            policyChosenLogits, chosenLabelsVar, chosenMaskVar);
        Variable policyRejectedLogProbs = dpoLoss.computeLogProbs(
            policyRejectedLogits, rejectedLabelsVar, rejectedMaskVar);
            
        // ========== 4. 计算 DPO 损失 ==========
        Variable loss = dpoLoss.loss(policyChosenLogProbs, policyRejectedLogProbs,
                                     refChosenLogProb, refRejectedLogProb);
        loss = withMoeAuxLoss(loss, output);
            
        float lossValue = loss.getValue().getNumber().floatValue();
            
        // ========== 5. 计算准确率（逐样本比较，再求 batch 平均）==========
        float[] chosenValues = policyChosenLogProbs.getValue().getArray();
        float[] rejectedValues = policyRejectedLogProbs.getValue().getArray();
        int correctCount = 0;
        for (int i = 0; i < batchSize; i++) {
            if (chosenValues[i] > rejectedValues[i]) {
                correctCount++;
            }
        }
        float accuracy = batchSize > 0 ? (float) correctCount / batchSize : 0.0f;
            
        // 损失异常时不反向，并释放计算图
        if (Float.isNaN(lossValue) || Float.isInfinite(lossValue)) {
            System.err.println("警告: DPO 损失异常 (" + lossValue + "), 跳过此batch");
            loss.unChainBackward();
            model.clearGrads();
            return new float[]{Float.NaN, accuracy};
        }
            
        // ========== 6. 反向传播 ==========
        loss.backward();
            
        // ========== 7. 梯度裁剪 ==========
        clipGradients();
            
        // ========== 8. 参数更新 ==========
        optimizer.update();
        model.clearGrads();
            
        // ========== 9. 释放计算图 ==========
        loss.unChainBackward();
            
        return new float[]{lossValue, accuracy};
    }
        
    /**
     * 打印训练统计
     */
    private void printTrainingStats() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("训练统计");
        System.out.println("=".repeat(70));
        
        if (!lossHistory.isEmpty()) {
            double avgLoss = lossHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            double finalLoss = lossHistory.get(lossHistory.size() - 1);
            System.out.printf("平均损失: %.4f%n", avgLoss);
            System.out.printf("最终损失: %.4f%n", finalLoss);
        }
        
        if (!accuracyHistory.isEmpty()) {
            double avgAcc = accuracyHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            double finalAcc = accuracyHistory.get(accuracyHistory.size() - 1);
            System.out.printf("平均准确率: %.2f%%%n", avgAcc * 100);
            System.out.printf("最终准确率: %.2f%%%n", finalAcc * 100);
        }
        
        System.out.println("总训练步数: " + currentStep);
        System.out.println("=".repeat(70));
    }
    
    // ==================== 实现 BaseTrainer 的抽象方法 ====================
    
    @Override
    protected float trainStep(Object batch) {
        return trainStepImpl((DPODataset.Batch) batch)[0];
    }
    
    @Override
    protected String getTrainerName() {
        return "DPO";
    }
    
    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(70));
        System.out.println("开始DPO训练");
        System.out.println("=".repeat(70));
        System.out.println("策略模型: " + config.getModelSize());
        System.out.println("DPO配置: " + dpoConfig);
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("学习率: " + learningRate);
        System.out.println("Beta (KL惩罚): " + dpoConfig.getBeta());
        System.out.println("=".repeat(70));
    }
    
    @Override
    protected void prepareDataset() {
        dataset.prepare(true);
    }
    
    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNext();
    }
    
    @Override
    protected Object getNextBatch() {
        return dataset.nextBatch();
    }
    
    @Override
    protected void resetDataset() {
        dataset.reset();
    }
    
    @Override
    protected String getCheckpointPrefix() {
        return "dpo";
    }
    
    // Getters
    
    public List<Float> getAccuracyHistory() {
        return new ArrayList<>(accuracyHistory);
    }
}