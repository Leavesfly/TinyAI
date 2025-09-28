package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;

/**
 * 强化学习策略抽象基类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * Policy类定义了强化学习中策略的标准接口。
 * 策略负责根据状态选择动作，可以是确定性策略或随机策略。
 * 支持不同类型的策略实现，如ε-贪婪策略、Softmax策略、高斯策略等。
 */
public abstract class Policy {
    
    /**
     * 策略名称
     */
    protected String name;
    
    /**
     * 状态空间维度
     */
    protected int stateDim;
    
    /**
     * 动作空间维度
     */
    protected int actionDim;
    
    /**
     * 构造函数
     * 
     * @param name 策略名称
     * @param stateDim 状态空间维度
     * @param actionDim 动作空间维度
     */
    public Policy(String name, int stateDim, int actionDim) {
        this.name = name;
        this.stateDim = stateDim;
        this.actionDim = actionDim;
    }
    
    /**
     * 根据状态选择动作
     * 
     * @param state 当前状态
     * @return 选择的动作
     */
    public abstract Variable selectAction(Variable state);
    
    /**
     * 计算动作概率分布
     * 
     * @param state 当前状态
     * @return 动作概率分布
     */
    public abstract Variable getActionProbabilities(Variable state);
    
    /**
     * 计算特定状态-动作对的概率
     * 
     * @param state 状态
     * @param action 动作
     * @return 动作概率
     */
    public abstract float getActionProbability(Variable state, Variable action);
    
    /**
     * 计算策略的对数概率（用于策略梯度）
     * 
     * @param state 状态
     * @param action 动作
     * @return 对数概率
     */
    public abstract Variable getLogProbability(Variable state, Variable action);
    
    /**
     * 获取策略名称
     * 
     * @return 策略名称
     */
    public String getName() {
        return name;
    }
    
    /**
     * 获取状态空间维度
     * 
     * @return 状态空间维度
     */
    public int getStateDim() {
        return stateDim;
    }
    
    /**
     * 获取动作空间维度
     * 
     * @return 动作空间维度
     */
    public int getActionDim() {
        return actionDim;
    }
}