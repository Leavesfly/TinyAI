package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.base.training.DeepSeekBasePretrain;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Block;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * DeepSeek-V3预训练器（V2版本 - 基于共享基类）
 * 
 * 继承 DeepSeekBasePretrain，实现因果语言建模预训练。
 * V3 特点：
 * - 使用 Adam 优化器
 * - MoE 负载均衡损失
 * - 任务感知训练
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3PretrainV2 extends DeepSeekBasePretrain<DeepSeekV3Model> {
    
    private final DeepSeekV3Dataset dataset;
    private float moeLoadBalanceWeight;
    
    /**
     * 构造函数
     */
    public DeepSeekV3PretrainV2(DeepSeekV3Model model, DeepSeekV3Dataset dataset) {
        super(model);
        this.dataset = dataset;
        this.checkpointDir = "./checkpoints/deepseek_v3_pretrain";
        this.moeLoadBalanceWeight = (float) model.getConfig().getLoadBalanceLossWeight();
    }
    
    /**
     * 创建 Adam 优化器（V3 特有）
     */
    @Override
    protected Optimizer createOptimizer() {
        return new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
    }
    
    /**
     * 计算损失（语言模型损失 + MoE 负载均衡损失）
     */
    @Override
    protected Variable computeLoss(NdArray inputIds, NdArray targetIds) {
        // 前向传播（带详细信息以获取 MoE 损失）
        Variable inputVar = new Variable(inputIds);
        TaskType taskType = getTaskType(inputIds);  // 获取任务类型
        
        DeepSeekV3Block.DetailedForwardResult result = 
            model.getV3Block().forwardWithDetails(inputVar, taskType);
        
        Variable logits = result.logits;
        double moeLoss = result.avgMoELoss;
        
        // SoftmaxCE 要求 2D 输入
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));
        
        // 语言模型损失
        Variable lmLoss = lossFunction.loss(targetVar, logits2D);
        
        // 总损失 = 语言模型损失 + MoE 负载均衡损失
        double moeScaledLoss = moeLoss * moeLoadBalanceWeight;
        Variable totalLoss = lmLoss.add(new Variable(NdArray.of(moeScaledLoss)));
        
        return totalLoss;
    }
    
    /**
     * 获取任务类型（简化版，可根据实际需求扩展）
     */
    private TaskType getTaskType(NdArray inputIds) {
        // 简化实现：默认返回 GENERAL
        // 实际应用中可根据数据集元数据或输入特征判断
        return TaskType.GENERAL;
    }
    
    // ========== 数据集访问实现 ==========
    
    @Override
    protected void prepareDataset(boolean shuffle) {
        dataset.prepare(shuffle);
    }
    
    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNext();
    }
    
    @Override
    protected Batch nextBatch() {
        DeepSeekV3Dataset.Batch batch = dataset.nextBatch();
        return new Batch(batch.getInputIds(), batch.getTargetIds());
    }
    
    @Override
    protected int getStepsPerEpoch() {
        return 1000;  // 简化实现
    }
}
