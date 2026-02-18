package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.base.training.DeepSeekBasePretrain;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1Dataset;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * DeepSeek-R1预训练器（V2版本 - 基于共享基类）
 * 
 * 继承 DeepSeekBasePretrain，实现因果语言建模预训练。
 * R1 特点：
 * - 使用 SGD 优化器（减少内存占用）
 * - 纯 MoE 架构，推理能力通过 RL 训练自然涌现
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1PretrainV2 extends DeepSeekBasePretrain<DeepSeekR1Model> {
    
    private final DeepSeekR1Dataset dataset;
    
    /**
     * 构造函数
     */
    public DeepSeekR1PretrainV2(DeepSeekR1Model model, DeepSeekR1Dataset dataset) {
        super(model);
        this.dataset = dataset;
        this.checkpointDir = "./checkpoints/deepseek_r1_pretrain";
    }
    
    /**
     * 创建 SGD 优化器（R1 特有）
     */
    @Override
    protected Optimizer createOptimizer() {
        return new SGD(model, initialLearningRate);
    }
    
    /**
     * 计算损失（标准语言模型损失）
     */
    @Override
    protected Variable computeLoss(NdArray inputIds, NdArray targetIds) {
        // 前向传播
        Variable inputVar = new Variable(inputIds);
        DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
        Variable logits = result.logits;
        
        // SoftmaxCE 要求 2D 输入：[batch_size * seq_len, vocab_size]
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        
        // 目标也要展平成 1D：[batch_size * seq_len, 1]
        Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));
        
        // 计算损失
        Variable loss = lossFunction.loss(targetVar, logits2D);
        
        return loss;
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
        DeepSeekR1Dataset.Batch batch = dataset.nextBatch();
        return new Batch(batch.getInputIds(), batch.getTargetIds());
    }
    
    @Override
    protected int getStepsPerEpoch() {
        // 简化实现：返回1000作为默认步数估计
        // 实际步数在训练时动态计算
        return 1000;
    }
}
