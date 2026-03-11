package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;

/**
 * 动作空间接口
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【设计说明】
 * ActionSpace抽象了强化学习中的动作空间概念,区分离散和连续两种类型:
 * - 离散动作空间(DiscreteActionSpace): 有限个动作选择,如上下左右
 * - 连续动作空间(ContinuousActionSpace): 连续值动作,如力的大小、角度
 * 
 * 这一区分对算法选择至关重要:
 * - 离散空间适合Value-Based方法(DQN等),可以枚举所有动作的Q值
 * - 连续空间适合Policy-Based方法(PPO/DDPG等),直接输出动作值
 * 
 * 【与Environment的关系】
 * Environment通过getActionSpace()暴露其动作空间类型,
 * Agent可以据此选择合适的策略和网络结构。
 */
public interface ActionSpace {
    
    /**
     * 获取动作空间维度
     * - 离散空间: 动作的数量
     * - 连续空间: 动作向量的维度
     * 
     * @return 动作空间维度
     */
    int getDimension();
    
    /**
     * 随机采样一个动作
     * 
     * @return 随机动作
     */
    Variable sample();
    
    /**
     * 检查动作是否在合法范围内
     * 
     * @param action 要检查的动作
     * @return 是否合法
     */
    boolean contains(Variable action);
    
    /**
     * 是否为离散动作空间
     * 
     * @return true表示离散,false表示连续
     */
    boolean isDiscrete();
}
