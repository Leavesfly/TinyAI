package io.leavesfly.tinyai.ml.parallel;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.Map;
import java.util.concurrent.Callable;

/**
 * 并行批次处理器 - 在单独线程中处理一个batch的训练
 * 
 * 每个处理器负责：
 * 1. 执行前向传播
 * 2. 计算损失
 * 3. 执行反向传播
 * 4. 收集梯度
 * 5. 提交梯度到聚合器
 */
public class ParallelBatchProcessor implements Callable<ParallelBatchProcessor.BatchProcessResult> {
    
    private final Batch batch;
    private final Model model;
    private final Loss loss;
    private final GradientAggregator gradientAggregator;
    private final int threadId;
    
    /**
     * 构造并行批次处理器
     * @param batch 要处理的数据批次
     * @param model 深拷贝的模型实例（每个线程独立）
     * @param loss 损失函数
     * @param gradientAggregator 梯度聚合器
     * @param threadId 线程ID，用于调试
     */
    public ParallelBatchProcessor(Batch batch, Model model, Loss loss,
                                  GradientAggregator gradientAggregator, int threadId) {
        this.batch = batch;
        this.model = model;
        this.loss = loss;
        this.gradientAggregator = gradientAggregator;
        this.threadId = threadId;
    }
    
    @Override
    public BatchProcessResult call() throws Exception {
        try {
            // 1. 准备输入数据
            Variable variableX = batch.toVariableX().setName("x_" + threadId).setRequireGrad(false);
            Variable variableY = batch.toVariableY().setName("y_" + threadId).setRequireGrad(false);
            
            // 2. 前向传播
            Variable predictY = model.forward(variableX);
            
            // 3. 计算损失
            Variable lossVariable = loss.loss(variableY, predictY);
            lossVariable.setName("loss_" + threadId);
            
            float lossValue = lossVariable.getValue().getNumber().floatValue();
            
            // 4. 清空梯度并执行反向传播
            model.clearGrads();
            lossVariable.backward();
            
            // 5. 获取梯度并提交到聚合器
            Map<String, Parameter> gradients = model.getAllParams();
            gradientAggregator.submitGradients(gradients);
            
            // 6. 清理计算图
            lossVariable.unChainBackward();
            
            return new BatchProcessResult(threadId, lossValue, batch.getSize(), true, null);
            
        } catch (Exception e) {
            // 如果处理失败，仍然要提交空梯度以免阻塞其他线程
            try {
                Map<String, Parameter> emptyGradients = model.getAllParams();
                // 清空梯度
                for (Parameter param : emptyGradients.values()) {
                    param.clearGrad();
                }
                gradientAggregator.submitGradients(emptyGradients);
            } catch (Exception submitEx) {
                // 忽略提交异常
            }
            
            return new BatchProcessResult(threadId, 0.0f, batch.getSize(), false, e);
        }
    }
    
    /**
     * 批次处理结果
     */
    public static class BatchProcessResult {
        private final int threadId;
        private final float lossValue;
        private final int batchSize;
        private final boolean success;
        private final Exception exception;
        
        public BatchProcessResult(int threadId, float lossValue, int batchSize, 
                                boolean success, Exception exception) {
            this.threadId = threadId;
            this.lossValue = lossValue;
            this.batchSize = batchSize;
            this.success = success;
            this.exception = exception;
        }
        
        public int getThreadId() { return threadId; }
        public float getLossValue() { return lossValue; }
        public int getBatchSize() { return batchSize; }
        public boolean isSuccess() { return success; }
        public Exception getException() { return exception; }
        
        @Override
        public String toString() {
            return String.format("BatchResult[thread=%d, loss=%.4f, size=%d, success=%b]", 
                               threadId, lossValue, batchSize, success);
        }
    }
}