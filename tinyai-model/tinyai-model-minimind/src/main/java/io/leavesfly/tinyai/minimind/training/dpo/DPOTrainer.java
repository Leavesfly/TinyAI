package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.DPODataset;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

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
public class DPOTrainer {
    
    private final MiniMindModel policyModel;      // 策略模型(被训练)
    private final MiniMindModel referenceModel;   // 参考模型(冻结)
    private final MiniMindConfig config;
    private final DPODataset dataset;
    private final DPOConfig dpoConfig;
    private final DPOLoss dpoLoss;
    private final Adam optimizer;
    
    // 训练配置
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private int logInterval;
    private int saveInterval;
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int currentStep;
    private List<Float> lossHistory;
    private List<Float> accuracyHistory;  // 记录chosen>rejected的比例
    
    /**
     * 构造函数
     * 
     * @param policyModel 策略模型(将被训练)
     * @param dataset DPO数据集
     * @param dpoConfig DPO配置
     */
    public DPOTrainer(MiniMindModel policyModel, DPODataset dataset, DPOConfig dpoConfig) {
        this.policyModel = policyModel;
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
        this.checkpointDir = "./dpo_checkpoints";
        
        // 创建优化器
        this.optimizer = new Adam(policyModel, learningRate, 0.9f, 0.999f, 1e-8f);
        
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.lossHistory = new ArrayList<>();
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
     * 复制模型参数
     */
    private void copyModelParameters(MiniMindModel source, MiniMindModel target) {
        Map<String, Parameter> sourceParams = source.getAllParams();
        Map<String, Parameter> targetParams = target.getAllParams();
        
        for (String name : sourceParams.keySet()) {
            if (targetParams.containsKey(name)) {
                Parameter sourceParam = sourceParams.get(name);
                Parameter targetParam = targetParams.get(name);
                
                // 复制数据
                NdArray sourceData = sourceParam.getValue();
                NdArray targetData = targetParam.getValue();
                
                // 简化:直接赋值(实际应该深拷贝)
                float[] srcBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) sourceData).buffer;
                float[] tgtBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) targetData).buffer;
                System.arraycopy(srcBuffer, 0, tgtBuffer, 0, srcBuffer.length);
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
    public void train() {
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
    private void trainOneEpoch() {
        dataset.prepare(true);  // 打乱数据
        
        long epochStartTime = System.currentTimeMillis();
        float epochLoss = 0.0f;
        float epochAccuracy = 0.0f;
        int batchCount = 0;
        
        while (dataset.hasNext()) {
            DPODataset.Batch batch = dataset.nextBatch();
            
            float[] metrics = trainStep(batch);
            float stepLoss = metrics[0];
            float stepAccuracy = metrics[1];
            
            epochLoss += stepLoss;
            epochAccuracy += stepAccuracy;
            batchCount++;
            currentStep++;
            
            lossHistory.add(stepLoss);
            accuracyHistory.add(stepAccuracy);
            
            if (currentStep % logInterval == 0) {
                double avgLoss = lossHistory.stream()
                    .skip(Math.max(0, lossHistory.size() - logInterval))
                    .mapToDouble(Float::doubleValue)
                    .average()
                    .orElse(0.0);
                
                double avgAcc = accuracyHistory.stream()
                    .skip(Math.max(0, accuracyHistory.size() - logInterval))
                    .mapToDouble(Float::doubleValue)
                    .average()
                    .orElse(0.0);
                
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | Acc: %.2f%%%n",
                    currentEpoch + 1, maxEpochs, currentStep, avgLoss, avgAcc * 100);
            }
            
            if (currentStep % saveInterval == 0) {
                saveCheckpoint();
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        double avgEpochAcc = batchCount > 0 ? epochAccuracy / batchCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 平均准确率: %.2f%% | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, avgEpochAcc * 100, epochEndTime - epochStartTime);
        
        dataset.reset();
    }
    
    /**
     * 训练一步
     * 
     * @return [loss, accuracy]
     */
    private float[] trainStep(DPODataset.Batch batch) {
        // 获取数据
        NdArray chosenInput = batch.getChosenInput();
        NdArray chosenLabels = batch.getChosenLabels();
        NdArray rejectedInput = batch.getRejectedInput();
        NdArray rejectedLabels = batch.getRejectedLabels();
        NdArray promptMask = batch.getPromptMask();
        
        Variable chosenInputVar = new Variable(chosenInput);
        Variable rejectedInputVar = new Variable(rejectedInput);
        Variable chosenLabelsVar = new Variable(chosenLabels);
        Variable rejectedLabelsVar = new Variable(rejectedLabels);
        Variable maskVar = new Variable(promptMask);
        
        // 1. 参考模型前向传播 (不需要梯度，使用 detached 计算)
        referenceModel.setTraining(false);
        Variable refChosenLogits = referenceModel.predict(new Variable(chosenInput));
        Variable refRejectedLogits = referenceModel.predict(new Variable(rejectedInput));
        float refChosenLogProb = dpoLoss.computeLogProbsDetached(refChosenLogits, chosenLabelsVar, maskVar);
        float refRejectedLogProb = dpoLoss.computeLogProbsDetached(refRejectedLogits, rejectedLabelsVar, maskVar);
        
        // 2. 策略模型前向传播 (保持计算图，支持反向传播)
        policyModel.setTraining(true);
        Variable policyChosenLogits = policyModel.predict(chosenInputVar);
        Variable policyRejectedLogits = policyModel.predict(rejectedInputVar);
        
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
        policyModel.clearGrads();
        loss.backward();
        
        // 7. 梯度裁剪
        clipGradients();
        
        // 8. 参数更新
        optimizer.update();
        
        // 9. 断开计算图，释放内存
        loss.unChainBackward();
        
        return new float[]{lossValue, accuracy};
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        if (maxGradNorm <= 0) {
            return;
        }
        
        float totalNorm = 0.0f;
        Map<String, Parameter> params = policyModel.getAllParams();
        
        for (Parameter param : params.values()) {
            if (param.getGrad() != null) {
                float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) param.getGrad()).buffer;
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        totalNorm = (float) Math.sqrt(totalNorm);
        
        if (totalNorm > maxGradNorm) {
            float scale = maxGradNorm / (totalNorm + 1e-6f);
            for (Parameter param : params.values()) {
                if (param.getGrad() != null) {
                    float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) param.getGrad()).buffer;
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= scale;
                    }
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint() {
        System.out.println("保存检查点: step_" + currentStep);
        // 实际实现应该保存模型参数
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
    
    // Getters
    
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    public List<Float> getAccuracyHistory() {
        return new ArrayList<>(accuracyHistory);
    }
}
