package io.leavesfly.tinyai.ml.callback;

/**
 * 训练回调接口，用于在训练过程的不同阶段执行自定义逻辑，提升扩展性。
 *
 * @author TinyAI
 * @version 1.0
 */
public interface TrainingCallback {
    
    /**
     * 训练开始时的回调
     */
    default void onTrainingStart() {
        // 默认空实现
    }
    
    /**
     * 训练结束时的回调
     * 
     * @param epoch 最终轮次
     * @param finalLoss 最终损失
     */
    default void onTrainingEnd(int epoch, float finalLoss) {
        // 默认空实现
    }
    
    /**
     * 每个epoch开始时的回调
     * 
     * @param epoch 当前轮次
     */
    default void onEpochStart(int epoch) {
        // 默认空实现
    }
    
    /**
     * 每个epoch结束时的回调
     * 
     * @param epoch 当前轮次
     * @param loss 训练损失
     * @param accuracy 训练准确率（如果可用）
     */
    default void onEpochEnd(int epoch, float loss, Float accuracy) {
        // 默认空实现
    }
    
    /**
     * 每个批次结束时的回调
     * 
     * @param epoch 当前轮次
     * @param batchIndex 批次索引
     * @param batchLoss 批次损失
     */
    default void onBatchEnd(int epoch, int batchIndex, float batchLoss) {
        // 默认空实现
    }
    
    /**
     * 验证开始时的回调
     * 
     * @param epoch 当前轮次
     */
    default void onValidationStart(int epoch) {
        // 默认空实现
    }
    
    /**
     * 验证结束时的回调
     * 
     * @param epoch 当前轮次
     * @param valLoss 验证损失
     * @param valAccuracy 验证准确率（如果可用）
     */
    default void onValidationEnd(int epoch, float valLoss, Float valAccuracy) {
        // 默认空实现
    }
    
    /**
     * 检查是否应该停止训练，返回 true 时训练器会提前停止。
     *
     * @return true 如果应该停止训练
     */
    default boolean shouldStop() {
        return false;
    }
}

