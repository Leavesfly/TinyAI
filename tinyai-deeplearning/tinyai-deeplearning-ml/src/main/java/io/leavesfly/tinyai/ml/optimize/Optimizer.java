package io.leavesfly.tinyai.ml.optimize;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.HashMap;
import java.util.Map;
import java.io.Serializable;

/**
 * 参数优化器抽象类。
 *
 * 该类是所有参数优化器实现的基类，定义参数更新的基本接口和流程，
 * 子类需要实现具体的参数更新逻辑。
 *
 * V2 增强功能：
 * - 优化器状态管理（state_dict/load_state_dict）
 * - 学习率调度支持
 * - 参数组管理
 *
 * @author TinyDL
 * @version 2.0
 */
public abstract class Optimizer implements Serializable {

    private static final long serialVersionUID = 1L;

    protected Model target;
    
    /**
     * 当前学习率
     */
    protected float lr;
    
    /**
     * 学习率调度器
     */
    protected LRScheduler lrScheduler;
    
    /**
     * 优化器状态字典（用于存储动量、速度等）
     * key: 参数名称, value: 状态映射
     */
    protected Map<String, Map<String, NdArray>> state;
    
    /**
     * 当前训练步数
     */
    protected int step;

    /**
     * 构造函数
     *
     * @param target 目标模型
     */
    public Optimizer(Model target) {
        this.target = target;
        this.state = new HashMap<>();
        this.step = 0;
        this.lr = 0.01f; // 默认学习率
    }
    
    /**
     * 构造函数（带学习率）
     *
     * @param target 目标模型
     * @param lr     学习率
     */
    public Optimizer(Model target, float lr) {
        this(target);
        this.lr = lr;
    }

    /**
     * 更新所有参数
     */
    public void update() {
        Map<String, Parameter> parameterMap = target.getAllParams();
        for (Parameter parameter : parameterMap.values()) {
            updateOne(parameter);
        }
        step++;
        
        // 如果有学习率调度器，更新学习率
        if (lrScheduler != null) {
            this.lr = lrScheduler.getLearningRate(step);
        }
    }

    /**
     * 更新单个参数
     *
     * @param parameter 参数
     */
    public abstract void updateOne(Parameter parameter);
    
    /**
     * 获取或创建参数状态
     *
     * @param paramName 参数名称
     * @return 参数状态映射
     */
    protected Map<String, NdArray> getParamState(String paramName) {
        return state.computeIfAbsent(paramName, k -> new HashMap<>());
    }
    
    /**
     * 设置学习率
     *
     * @param lr 新的学习率
     */
    public void setLearningRate(float lr) {
        this.lr = lr;
    }
    
    /**
     * 获取当前学习率
     *
     * @return 当前学习率
     */
    public float getLearningRate() {
        return lr;
    }
    
    /**
     * 设置学习率调度器
     *
     * @param scheduler 学习率调度器
     */
    public void setLRScheduler(LRScheduler scheduler) {
        this.lrScheduler = scheduler;
    }
    
    /**
     * 获取当前训练步数
     *
     * @return 训练步数
     */
    public int getStep() {
        return step;
    }
    
    /**
     * 重置训练步数
     */
    public void resetStep() {
        this.step = 0;
    }

    /* ===== PyTorch风格状态管理API ===== */

    /**
     * 导出优化器状态字典，包含所有状态（如动量、速度等）和超参数。
     *
     * @return 状态字典
     */
    public Map<String, Object> state_dict() {
        Map<String, Object> stateDict = new HashMap<>();
        
        // 保存优化器状态
        stateDict.put("state", new HashMap<>(state));
        
        // 保存超参数
        Map<String, Object> paramGroups = new HashMap<>();
        paramGroups.put("lr", lr);
        paramGroups.put("step", step);
        stateDict.put("param_groups", paramGroups);
        
        return stateDict;
    }

    /**
     * 从状态字典恢复优化器状态。
     *
     * @param state_dict 状态字典
     */
    @SuppressWarnings("unchecked")
    public void load_state_dict(Map<String, Object> state_dict) {
        // 恢复优化器状态
        if (state_dict.containsKey("state")) {
            this.state = (Map<String, Map<String, NdArray>>) state_dict.get("state");
        }
        
        // 恢复超参数
        if (state_dict.containsKey("param_groups")) {
            Map<String, Object> paramGroups = (Map<String, Object>) state_dict.get("param_groups");
            if (paramGroups.containsKey("lr")) {
                this.lr = ((Number) paramGroups.get("lr")).floatValue();
            }
            if (paramGroups.containsKey("step")) {
                this.step = ((Number) paramGroups.get("step")).intValue();
            }
        }
    }
    
    /**
     * 清空梯度（便捷方法），等价于 model.clearGrads()。
     */
    public void zero_grad() {
        if (target != null) {
            target.clearGrads();
        }
    }

}