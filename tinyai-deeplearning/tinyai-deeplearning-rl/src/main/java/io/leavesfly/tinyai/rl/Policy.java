package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;

/**
 * 强化学习策略接口
 * 
 * @author leavesfly
 * @version 0.02
 * 
 * 【设计说明】
 * Policy定义了强化学习中策略的标准行为契约。
 * 策略负责根据状态选择动作，可以是确定性策略或随机策略。
 * 
 * 将Policy设计为接口而非抽象类，使其可以被灵活组合到不同的Agent中:
 * - ValueBasedAgent通过组合EpsilonGreedyPolicy实现探索
 * - PolicyBasedAgent直接通过神经网络输出策略分布
 * - BanditAgent使用各自特有的选择策略
 * 
 * 支持的策略类型:
 * - ε-贪婪策略(EpsilonGreedyPolicy): 概率探索
 * - Softmax策略: 温度参数控制
 * - 高斯策略: 连续动作空间
 */
public interface Policy {
    
    /**
     * 根据状态选择动作
     * 
     * @param state 当前状态
     * @return 选择的动作
     */
    Variable selectAction(Variable state);
    
    /**
     * 计算动作概率分布
     * 
     * @param state 当前状态
     * @return 动作概率分布
     */
    Variable getActionProbabilities(Variable state);
    
    /**
     * 计算特定状态-动作对的概率
     * 
     * @param state 状态
     * @param action 动作
     * @return 动作概率
     */
    float getActionProbability(Variable state, Variable action);
    
    /**
     * 计算策略的对数概率（用于策略梯度）
     * 
     * @param state 状态
     * @param action 动作
     * @return 对数概率
     */
    Variable getLogProbability(Variable state, Variable action);
    
    /**
     * 获取策略名称
     * 
     * @return 策略名称
     */
    String getName();
    
    /**
     * 获取状态空间维度
     * 
     * @return 状态空间维度
     */
    int getStateDim();
    
    /**
     * 获取动作空间维度
     * 
     * @return 动作空间维度
     */
    int getActionDim();
}