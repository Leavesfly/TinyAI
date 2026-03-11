package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;

/**
 * 强化学习智能体接口
 * 
 * @author leavesfly
 * @version 0.02
 * 
 * 【强化学习核心概念 - Agent】
 * Agent(智能体)是强化学习系统的核心决策者,负责:
 * 1. 感知环境状态(State Perception): 接收环境反馈的状态信息
 * 2. 选择执行动作(Action Selection): 根据策略或价值函数选择动作
 * 3. 从经验中学习(Learning): 通过奖励信号优化决策策略
 * 4. 平衡探索与利用(Exploration vs Exploitation): ε-贪婪、Softmax等策略
 * 
 * 【MDP框架中的Agent】
 * 在马尔可夫决策过程(MDP)中,Agent与Environment形成交互循环:
 * {@code
 * t=0: s0 ----[Agent选择a0]----> Environment
 *             <----[返回r1,s1]---- Environment
 * t=1: s1 ----[Agent选择a1]----> Environment
 *             <----[返回r2,s2]---- Environment
 * }
 * 
 * 【设计说明 - 接口 vs 抽象类】
 * Agent被设计为接口而非抽象类,因为不同类型的Agent差异很大:
 * - 基于神经网络的Agent(DQN/PPO)需要model、optimizer等字段
 * - 多臂老虎机Agent(Bandit)不需要神经网络,只需统计信息
 * - 使用接口可以让各分支自由选择继承体系,避免被迫继承无用字段
 * 
 * 通用的神经网络Agent实现请参见 {@link io.leavesfly.tinyai.rl.AbstractAgent}
 * 
 * 【算法分类体系】
 * TinyAI强化学习模块按算法类型组织Agent层次:
 * {@code
 * Agent (接口)
 *  ├── AbstractAgent (基于神经网络的通用基类)
 *  │    ├── ValueBasedAgent (基于值函数)
 *  │    │    ├── DQNAgent: 深度Q网络
 *  │    │    └── DoubleDQNAgent: 双Q网络(解决过估计)
 *  │    │
 *  │    └── PolicyBasedAgent (基于策略)
 *  │         ├── REINFORCEAgent: 策略梯度
 *  │         └── PPOAgent: 近端策略优化
 *  │
 *  └── BanditAgent (多臂老虎机,直接实现Agent接口)
 *       ├── EpsilonGreedyBanditAgent
 *       ├── UCBBanditAgent
 *       └── ThompsonSamplingBanditAgent
 * }
 */
public interface Agent {
    
    /**
     * 根据当前状态选择动作
     * 
     * @param state 当前状态
     * @return 选择的动作
     */
    Variable selectAction(Variable state);
    
    /**
     * 从经验中学习更新模型
     * 
     * @param experience 经验数据(s, a, r, s', done)
     */
    void learn(Experience experience);
    
    /**
     * 批量学习更新模型
     * 
     * @param experiences 经验批次
     */
    void learnBatch(Experience[] experiences);
    
    /**
     * 存储经验（用于经验回放）
     * 
     * @param experience 要存储的经验
     */
    void storeExperience(Experience experience);
    
    /**
     * 设置训练模式
     * 
     * @param training 是否为训练模式
     */
    void setTraining(boolean training);
    
    /**
     * 检查是否处于训练模式
     * 
     * @return 是否为训练模式
     */
    boolean isTraining();
    
    /**
     * 获取智能体名称
     * 
     * @return 智能体名称
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
    
    /**
     * 获取训练步数
     * 
     * @return 训练步数
     */
    int getTrainingStep();
    
    /**
     * 重置智能体状态
     */
    void reset();
    
    /**
     * 保存模型参数
     * 
     * @param filepath 保存路径
     */
    void saveModel(String filepath);
    
    /**
     * 加载模型参数
     * 
     * @param filepath 加载路径
     */
    void loadModel(String filepath);
    
    /**
     * 设置评估模式（禁用探索，使用确定性策略）
     */
    default void eval() {
        setTraining(false);
    }
    
    /**
     * 设置训练模式（启用探索）
     */
    default void train() {
        setTraining(true);
    }
    
    /**
     * 校验Agent与Environment的维度兼容性
     * 
     * Agent的stateDim必须等于Environment的stateDim（神经网络输入维度 = 状态向量维度），
     * Agent的actionDim必须等于Environment的actionDim（神经网络输出维度 = 动作空间维度）。
     * 维度不匹配会导致运行时矩阵运算错误，此方法提供提前检测能力。
     * 
     * @param environment 要校验的环境
     * @throws IllegalArgumentException 如果维度不匹配
     */
    default void validateCompatibility(Environment environment) {
        if (getStateDim() != environment.getStateDim()) {
            throw new IllegalArgumentException(String.format(
                "Agent '%s' 的 stateDim(%d) 与 Environment 的 stateDim(%d) 不匹配。" +
                "Agent 的神经网络输入维度必须等于环境的状态向量维度。",
                getName(), getStateDim(), environment.getStateDim()));
        }
        if (getActionDim() != environment.getActionDim()) {
            throw new IllegalArgumentException(String.format(
                "Agent '%s' 的 actionDim(%d) 与 Environment 的 actionDim(%d) 不匹配。" +
                "Agent 的神经网络输出维度必须等于环境的动作空间维度。",
                getName(), getActionDim(), environment.getActionDim()));
        }
    }
}