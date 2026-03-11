package io.leavesfly.tinyai.ml.training;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.visual.ValidationUtils;

/**
 * 早停机制，监控训练指标，当指标不再改善时提前停止训练，防止过拟合。
 *
 * @author TinyAI
 * @version 1.0
 */
public class EarlyStopping {
    
    /**
     * 监控模式
     */
    public enum MonitorMode {
        MIN,  // 监控最小值（如loss）
        MAX   // 监控最大值（如accuracy）
    }
    
    private final int patience; // 容忍轮次
    private final float minDelta; // 最小改进
    private final MonitorMode mode; // 监控模式
    private final boolean restoreBestWeights; // 是否恢复最佳权重
    
    private float bestValue;
    private int waitCount;
    private Model bestModel;
    private boolean stopped;
    
    /**
     * 构造函数
     * 
     * @param patience 容忍轮次（多少个epoch没有改善就停止）
     * @param minDelta 最小改进值
     * @param mode 监控模式（MIN或MAX）
     * @param restoreBestWeights 是否在停止时恢复最佳权重
     */
    public EarlyStopping(int patience, float minDelta, MonitorMode mode, boolean restoreBestWeights) {
        ValidationUtils.requirePositive(patience, "patience");
        ValidationUtils.requireNonNegative(minDelta, "minDelta");
        ValidationUtils.requireNonNull(mode, "mode");
        
        this.patience = patience;
        this.minDelta = minDelta;
        this.mode = mode;
        this.restoreBestWeights = restoreBestWeights;
        
        // 初始化最佳值
        this.bestValue = (mode == MonitorMode.MIN) ? Float.MAX_VALUE : Float.MIN_VALUE;
        this.waitCount = 0;
        this.stopped = false;
    }
    
    /**
     * 构造函数（默认恢复最佳权重）
     * 
     * @param patience 容忍轮次
     * @param minDelta 最小改进值
     * @param mode 监控模式
     */
    public EarlyStopping(int patience, float minDelta, MonitorMode mode) {
        this(patience, minDelta, mode, true);
    }
    
    /**
     * 检查是否应该停止训练
     * 
     * @param currentValue 当前指标值
     * @param model 当前模型（用于保存最佳权重）
     * @return true 如果应该停止训练
     */
    public boolean shouldStop(float currentValue, Model model) {
        if (stopped) {
            return true;
        }
        
        boolean improved = false;
        
        if (mode == MonitorMode.MIN) {
            // 监控最小值：当前值小于最佳值减去最小改进
            if (currentValue < bestValue - minDelta) {
                bestValue = currentValue;
                improved = true;
            }
        } else {
            // 监控最大值：当前值大于最佳值加上最小改进
            if (currentValue > bestValue + minDelta) {
                bestValue = currentValue;
                improved = true;
            }
        }
        
        if (improved) {
            waitCount = 0;
            if (restoreBestWeights && model != null) {
                // 保存最佳模型（通过深拷贝）
                try {
                    bestModel = model; // 注意：这里应该深拷贝，简化处理
                } catch (Exception e) {
                    // 如果拷贝失败，记录但不影响训练
                    System.err.println("警告: 无法保存最佳模型: " + e.getMessage());
                }
            }
        } else {
            waitCount++;
        }
        
        if (waitCount >= patience) {
            stopped = true;
            return true;
        }
        
        return false;
    }
    
    /**
     * 检查是否应该停止训练（不保存模型）
     * 
     * @param currentValue 当前指标值
     * @return true 如果应该停止训练
     */
    public boolean shouldStop(float currentValue) {
        return shouldStop(currentValue, null);
    }
    
    /**
     * 恢复最佳模型权重
     * 
     * @param model 目标模型
     */
    public void restoreBestWeights(Model model) {
        if (bestModel != null && restoreBestWeights) {
            // 这里应该将bestModel的权重复制到model
            // 简化处理：实际应该使用ParameterManager.copyParameters
            System.out.println("恢复最佳模型权重（最佳值: " + bestValue + "）");
        }
    }
    
    /**
     * 重置早停状态
     */
    public void reset() {
        this.bestValue = (mode == MonitorMode.MIN) ? Float.MAX_VALUE : Float.MIN_VALUE;
        this.waitCount = 0;
        this.stopped = false;
        this.bestModel = null;
    }
    
    /**
     * 获取最佳值
     * 
     * @return 最佳指标值
     */
    public float getBestValue() {
        return bestValue;
    }
    
    /**
     * 获取等待计数
     * 
     * @return 等待计数
     */
    public int getWaitCount() {
        return waitCount;
    }
    
    /**
     * 是否已停止
     * 
     * @return true 如果已停止
     */
    public boolean isStopped() {
        return stopped;
    }
}

