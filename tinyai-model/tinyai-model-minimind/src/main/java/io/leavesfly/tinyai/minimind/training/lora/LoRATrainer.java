package io.leavesfly.tinyai.minimind.training.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;

/**
 * LoRA训练器
 * 
 * 使用LoRA进行参数高效微调
 * 只训练LoRA参数,冻结原始模型参数
 * 
 * @author leavesfly
 * @since 2024
 */
public class LoRATrainer extends BaseTrainer {
    
    /**
     * 标签忽略位标记（对齐 SoftmaxCE 的 ignore_index 约定）
     */
    private static final int IGNORE_INDEX = -100;
    
    private final SFTDataset dataset;
    private final LoRAConfig loraConfig;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    private float learningRate;
    private float currentLearningRate;
    
    /**
     * 构造函数
     */
    public LoRATrainer(MiniMindModel model, SFTDataset dataset, LoRAConfig loraConfig) {
        super(model);
        this.dataset = dataset;
        this.loraConfig = loraConfig;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认配置(LoRA通常使用更高的学习率)
        this.maxEpochs = 3;
        this.learningRate = 1e-4f;
        this.currentLearningRate = learningRate;
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.saveInterval = 500;
        this.checkpointDir = "./checkpoints/minimind/lora";
        
        // 创建优化器(Adam，本框架未实现权重衰减)
        // 实际只有 LoRA 参数会拿到梯度，其余参数在 update 前被清零，Adam 对 null 梯度会跳过
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
        
        // 冻结非LoRA参数
        freezeNonLoRAParams();
    }
    
    /**
     * 是否存在 LoRA 参数（决定 trainStep 是否需要在 update 前清零非 LoRA 梯度）
     */
    private boolean hasLoRAParams;
    
    /**
     * 冻结非LoRA参数
     * <p>
     * 说明：{@code Parameter.clearGrad()} 只能清零"当前累积的梯度"，
     * 但在 backward 时这些参数仍会被重新填充梯度并被 Adam 更新。
     * 因此这里仅做统计和标记：
     *   - 若存在 LoRA 参数：设置 {@link #hasLoRAParams} = true，
     *     实际的"冻结"由 {@link #zeroOutNonLoRAGrads()} 在每步 update 前执行；
     *   - 若不存在 LoRA 参数：退化为全参数微调，hasLoRAParams = false。
     */
    private void freezeNonLoRAParams() {
        int frozenCount = 0;
        int loraCount = 0;
        int totalParams = 0;
        
        // 先统计LoRA参数数量
        for (var entry : model.getAllParams().entrySet()) {
            String paramName = entry.getKey();
            totalParams++;
            if (paramName.toLowerCase().contains("lora")) {
                loraCount++;
            }
        }
        
        // 如果没有LoRA参数,退化为全参数微调
        if (loraCount == 0) {
            this.hasLoRAParams = false;
            System.out.println("⚠️ 未检测到LoRA参数,退化为全参数微调模式");
            System.out.println("可训练参数: " + totalParams);
            return;
        }
        
        this.hasLoRAParams = true;
        // 统计冻结参数数量（实际清零发生在 trainStep 的 update 之前）
        for (var entry : model.getAllParams().entrySet()) {
            String paramName = entry.getKey();
            if (!paramName.toLowerCase().contains("lora")) {
                frozenCount++;
            }
        }
        
        System.out.println("冻结参数: " + frozenCount + "（在每步 update 前清零梯度实现冻结）");
        System.out.println("LoRA可训练参数: " + loraCount);
    }
    
    /**
     * 将所有非 LoRA 参数的梯度清零，实现真正的参数冻结。
     * <p>
     * 必须在 backward 完成、optimizer.update 执行之前调用。
     * 只有这样 Adam 才不会对非 LoRA 参数做参数更新（梯度为 0，动量项也会衰减）。
     */
    private void zeroOutNonLoRAGrads() {
        if (!hasLoRAParams) {
            return; // 全参数微调模式，不冻结任何参数
        }
        for (var entry : model.getAllParams().entrySet()) {
            String paramName = entry.getKey();
            Parameter param = entry.getValue();
            if (!paramName.toLowerCase().contains("lora")) {
                // clearGrad 会把梯度置为 null；Adam.update 对 null 梯度会跳过更新
                param.clearGrad();
            }
        }
    }
    
    /**
     * 配置训练参数
     */
    public LoRATrainer configure(int maxEpochs, float learningRate, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.currentLearningRate = learningRate;
        this.maxGradNorm = maxGradNorm;
        
        optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 训练一步
     * <p>
     * 关键顺序：backward → 清零非 LoRA 梯度 → 梯度裁剪 → update。
     * 裁剪必须放在清零之后：否则裁剪系数是按"全参数梯度范数"算的（基座梯度通常
     * 比 LoRA 大好几个量级），会把实际参与更新的 LoRA 梯度无谓地缩小。
     */
    @Override
    protected float trainStep(Object batch) {
        // 更新学习率（余弦退火 with 10% floor，与其他训练器保持一致）
        updateLearningRate();
        
        SFTDataset.Batch sftBatch = (SFTDataset.Batch) batch;
        NdArray inputArray = sftBatch.getInput();
        // answer-only：prompt / padding 位置置 -100（仅按 pad token 屏蔽会漏掉 prompt 部分）
        NdArray labelArray = applyIgnoreIndex(
            sftBatch.getLabels(), sftBatch.getLossMask(), IGNORE_INDEX);
        
        Variable input = new Variable(inputArray);
        Variable labels = new Variable(labelArray);
        labels.setRequireGrad(false);
        
        // 前向传播（同时取出 MoE 负载均衡损失；LoRA 不支持 MoE 时为 0 常量）
        MiniMindBlock.MoEOutput output = forwardWithAux(input);
        Variable logits = output.getOutput();
        
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int totalTokens = logitsShape[0] * logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
        Variable labelsReshaped = labels.reshape(Shape.of(totalTokens, 1));
        
        Variable ceLoss = lossFunction.loss(labelsReshaped, logitsReshaped);
        Variable loss = withMoeAuxLoss(ceLoss, output);
        float lossValue = loss.getValue().getNumber().floatValue();
        
        if (Float.isNaN(lossValue) || Float.isInfinite(lossValue)) {
            System.err.println("警告: 损失值异常 (" + lossValue + "), 跳过此batch");
            loss.unChainBackward();
            model.clearGrads();
            return Float.NaN;
        }
        
        // 反向传播
        loss.backward();
        
        // 先将非 LoRA 参数的梯度清零，实现真正的参数冻结；
        // 否则 backward 会给所有参数填充梯度，Adam 将更新所有参数，相当于全参数微调
        zeroOutNonLoRAGrads();
        
        // 再做梯度裁剪：此时范数只统计 LoRA 梯度，裁剪系数才是对的
        clipGradients();
        
        // 更新参数（仅 LoRA 参数会被实际更新，其他参数梯度为 null）
        optimizer.update();
        
        // 清除梯度 + 断开计算图
        model.clearGrads();
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 更新学习率（余弦退火 with 10% floor）
     */
    private void updateLearningRate() {
        int totalSteps = maxEpochs * dataset.getBatchCount();
        currentLearningRate = computeScheduledLearningRate(
            learningRate, currentStep, totalSteps, 0);
        optimizer.setLearningRate(currentLearningRate);
    }
    
    @Override
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
            .skip(Math.max(0, lossHistory.size() - logInterval))
            .mapToDouble(Float::doubleValue)
            .average()
            .orElse(0.0);
        
        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | LR: %.6f%n",
            currentEpoch + 1, maxEpochs, currentStep, avgLoss, currentLearningRate);
    }
    
    /**
     * 获取可训练参数统计
     */
    public void printTrainableParams() {
        int totalParams = 0;
        int loraParams = 0;
        
        for (var entry : model.getAllParams().entrySet()) {
            String paramName = entry.getKey();
            Parameter param = entry.getValue();
            int paramCount = param.getValue().getShape().size();
            
            totalParams += paramCount;
            if (paramName.toLowerCase().contains("lora")) {
                loraParams += paramCount;
            }
        }
        
        // 如果没有LoRA参数,所有参数都是可训练的(全参数微调模式)
        int trainableParams = loraParams > 0 ? loraParams : totalParams;
        float percentage = (float) trainableParams / totalParams * 100;
        
        System.out.println("=".repeat(60));
        System.out.println("参数统计:");
        System.out.println("  总参数: " + totalParams);
        System.out.println("  可训练参数: " + trainableParams + (loraParams == 0 ? " (全参数微调)" : " (LoRA)"));
        System.out.println("  训练参数占比: " + String.format("%.2f%%", percentage));
        System.out.println("=".repeat(60));
    }
    
    // ==================== 实现 BaseTrainer 的抽象方法 ====================
    
    @Override
    protected String getTrainerName() {
        return "LoRA";
    }
    
    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(60));
        System.out.println("开始LoRA微调");
        System.out.println("=".repeat(60));
        System.out.println("LoRA配置: " + loraConfig);
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("学习率: " + learningRate);
        System.out.println("=".repeat(60));
    }
    
    @Override
    protected void prepareDataset() {
        dataset.prepare(true);
    }
    
    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNextBatch();
    }
    
    @Override
    protected Object getNextBatch() {
        return dataset.getNextBatch();
    }
    
    @Override
    protected void resetDataset() {
        dataset.reset();
    }
    
    @Override
    protected String getCheckpointPrefix() {
        return "lora";
    }
}