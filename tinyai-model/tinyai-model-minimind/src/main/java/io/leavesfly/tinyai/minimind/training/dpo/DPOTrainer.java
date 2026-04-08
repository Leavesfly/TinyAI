package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.minimind.training.dataset.DPODataset;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
        
        // 创建参考模型(冻结)
        this.referenceModel = createReferenceModel(policyModel);
        freezeModel(referenceModel);
        
        // 创建DPO损失函数
        this.dpoLoss = new DPOLoss(dpoConfig.getBeta(), dpoConfig.getLabelSmoothing());
        
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
     * 创建参考模型
     */
    private MiniMindModel createReferenceModel(MiniMindModel sourceModel) {
        // 创建新模型实例
        MiniMindModel refModel = new MiniMindModel("reference_model", config);
        
        // 复制参数
        copyModelParameters(sourceModel, refModel);
        
        return refModel;
    }
    
    /**
     * 复制模型参数（深拷贝）
     *
     * 使用Parameter原生的deepCopy()方法，将源模型的参数数据深拷贝到目标模型中，
     * 确保两个模型的参数内存完全独立。
     */
    private void copyModelParameters(MiniMindModel source, MiniMindModel target) {
        Map<String, Parameter> sourceParams = source.getAllParams();
        Map<String, Parameter> targetParams = target.getAllParams();

        for (Map.Entry<String, Parameter> entry : sourceParams.entrySet()) {
            String paramName = entry.getKey();
            if (targetParams.containsKey(paramName)) {
                Parameter sourceParam = entry.getValue();
                Parameter deepCopied = sourceParam.deepCopy();
                targetParams.get(paramName).setData(deepCopied.data());
            }
        }
    }
    
    /**
     * 冻结模型参数
     */
    private void freezeModel(MiniMindModel model) {
        model.setTraining(false);
        // 参考模型不需要梯度
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
     * 设置检查点保存
     */
    public DPOTrainer setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    /**
     * 开始训练
     */
    @Override
    public void train() {
        printTrainingInfo();
        
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        System.out.println("\nDPO训练完成!");
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
        float epochLoss = 0.0f;
        float epochAccuracy = 0.0f;
        int batchCount = 0;
        
        while (hasNextBatch()) {
            Object batch = getNextBatch();
            float[] metrics = trainStepImpl(batch);
            float stepLoss = metrics[0];
            float stepAccuracy = metrics[1];
            
            epochLoss += stepLoss;
            epochAccuracy += stepAccuracy;
            batchCount++;
            currentStep++;
            
            lossHistory.add(stepLoss);
            accuracyHistory.add(stepAccuracy);
            
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
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 平均准确率: %.2f%% | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, avgEpochAcc * 100, epochEndTime - epochStartTime);
        
        resetDataset();
    }
    
    /**
     * 训练一步（私有实现，返回loss和accuracy）
     * 
     * @return [loss, accuracy]
     */
    private float[] trainStepImpl(Object batch) {
        DPODataset.Batch dpoBatch = (DPODataset.Batch) batch;
        
        // 获取数据
        NdArray chosenInput = dpoBatch.getChosenInput();
        NdArray chosenLabels = dpoBatch.getChosenLabels();
        NdArray rejectedInput = dpoBatch.getRejectedInput();
        NdArray rejectedLabels = dpoBatch.getRejectedLabels();
        NdArray promptMask = dpoBatch.getPromptMask();
        
        Variable chosenInputVar = new Variable(chosenInput);
        Variable rejectedInputVar = new Variable(rejectedInput);
        Variable chosenLabelsVar = new Variable(chosenLabels);
        Variable rejectedLabelsVar = new Variable(rejectedLabels);
        Variable maskVar = new Variable(promptMask);
        
        // 1. 参考模型前向传播 (不需要梯度，使用 detached 计算)
        // 修复：在参考模型前向传播前设置training=false，确保不计算梯度
        referenceModel.setTraining(false);
        Variable refChosenLogits = referenceModel.predict(new Variable(chosenInput));
        Variable refRejectedLogits = referenceModel.predict(new Variable(rejectedInput));
        
        // 确保参考模型的输出不参与梯度计算
        refChosenLogits = refChosenLogits.detach();
        refRejectedLogits = refRejectedLogits.detach();
        
        float refChosenLogProb = dpoLoss.computeLogProbsDetached(refChosenLogits, chosenLabelsVar, maskVar);
        float refRejectedLogProb = dpoLoss.computeLogProbsDetached(refRejectedLogits, rejectedLabelsVar, maskVar);
        
        // 2. 策略模型前向传播 (保持计算图，支持反向传播)
        model.setTraining(true);
        Variable policyChosenLogits = model.predict(chosenInputVar);
        Variable policyRejectedLogits = model.predict(rejectedInputVar);
        
        // 3. 计算策略模型的 log 概率 (在计算图中)
        Variable policyChosenLogProbs = dpoLoss.computeLogProbs(policyChosenLogits, chosenLabelsVar, maskVar);
        Variable policyRejectedLogProbs = dpoLoss.computeLogProbs(policyRejectedLogits, rejectedLabelsVar, maskVar);
        
        // 4. 计算 DPO 损失 (参考模型的 log prob 作为常量传入)
        Variable loss = dpoLoss.loss(policyChosenLogProbs, policyRejectedLogProbs,
                                     refChosenLogProb, refRejectedLogProb);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 5. 计算准确率 (chosen 的 log 概率是否 > rejected)
        float chosenProb = policyChosenLogProbs.getValue().getNumber().floatValue();
        float rejectedProb = policyRejectedLogProbs.getValue().getNumber().floatValue();
        float accuracy = chosenProb > rejectedProb ? 1.0f : 0.0f;
        
        // 6. 反向传播
        model.clearGrads();
        loss.backward();
        
        // 7. 梯度裁剪 (继承自 BaseTrainer)
        clipGradients();
        
        // 8. 参数更新
        optimizer.update();
        
        // 9. 断开计算图，释放内存
        loss.unChainBackward();
        
        return new float[]{lossValue, accuracy};
    }
    
    /**
     * 保存检查点
     */
    @Override
    protected void saveCheckpoint() {
        String filename = String.format("%s_checkpoint_epoch%d_step%d.model", 
            getCheckpointPrefix(), currentEpoch, currentStep);
        String filepath = java.nio.file.Paths.get(checkpointDir, filename).toString();
        
        try {
            model.save(new java.io.File(filepath));
            System.out.println(getTrainerName() + "检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
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