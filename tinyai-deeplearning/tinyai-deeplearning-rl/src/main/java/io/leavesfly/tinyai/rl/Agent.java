package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * 强化学习智能体抽象基类
 * 
 * @author leavesfly
 * @version 0.01
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
 * ```
 * t=0: s0 ----[Agent选择a0]----> Environment
 *             <----[返回r1,s1]---- Environment
 * t=1: s1 ----[Agent选择a1]----> Environment
 *             <----[返回r2,s2]---- Environment
 * ...
 * ```
 * 
 * 【设计模式应用】
 * 1. 模板方法模式: learn()定义学习流程,子类实现具体算法
 *    - ValueBasedAgent: 经验回放 + TD学习
 *    - PolicyBasedAgent: 回合采样 + 蒙特卡罗回报
 * 
 * 2. 策略模式: selectAction()支持不同的动作选择策略
 *    - ε-贪婪策略: 概率探索
 *    - Softmax策略: 温度参数控制
 *    - 确定性策略: DDPG等
 * 
 * 【算法分类体系】
 * TinyAI强化学习模块按算法类型组织Agent层次:
 * 
 * ```
 * Agent (抽象基类)
 *  ├── ValueBasedAgent (基于值函数)
 *  │    ├── DQNAgent: 深度Q网络
 *  │    ├── DoubleDQNAgent: 双Q网络(解决过估计)
 *  │    └── DuelingDQNAgent: 对偶网络(分离V和A)
 *  │
 *  ├── PolicyBasedAgent (基于策略)
 *  │    ├── REINFORCEAgent: 策略梯度
 *  │    ├── A2CAgent: Actor-Critic
 *  │    └── PPOAgent: 近端策略优化
 *  │
 *  └── BanditAgent (多臂老虎机)
 *       ├── EpsilonGreedyBanditAgent
 *       ├── UCBBanditAgent
 *       └── ThompsonSamplingBanditAgent
 * ```
 * 
 * 【教学价值】
 * 通过这个分类体系,学习者可以:
 * - 理解Value-Based vs Policy-Based的本质区别
 * - 看到不同算法共享的通用模式
 * - 快速定位算法差异点(如DQN vs DoubleDQN)
 * - 轻松扩展新算法(只需继承并实现2-3个方法)
 * 
 * 【关键方法说明】
 * - selectAction(): 根据当前状态选择动作
 * - learn(): 从单个经验中学习
 * - learnBatch(): 从批量经验中学习(提高效率)
 * - storeExperience(): 存储经验到缓冲区
 */
public abstract class Agent {
    
    /**
     * 智能体名称
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
     * 主要的神经网络模型
     */
    protected Model model;
    
    /**
     * 学习率
     */
    protected float learningRate;
    
    /**
     * 探索率（epsilon-greedy策略中的epsilon）
     */
    protected float epsilon;
    
    /**
     * 折扣因子
     */
    protected float gamma;
    
    /**
     * 训练步数计数器
     */
    protected int trainingStep;
    
    /**
     * 是否处于训练模式
     */
    protected boolean training;
    
    /**
     * 构造函数
     * 
     * @param name 智能体名称
     * @param stateDim 状态空间维度
     * @param actionDim 动作空间维度
     * @param learningRate 学习率
     * @param epsilon 初始探索率
     * @param gamma 折扣因子
     */
    public Agent(String name, int stateDim, int actionDim, float learningRate, float epsilon, float gamma) {
        this.name = name;
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.gamma = gamma;
        this.trainingStep = 0;
        this.training = true;
    }
    
    /**
     * 根据当前状态选择动作
     * 
     * @param state 当前状态
     * @return 选择的动作
     */
    public abstract Variable selectAction(Variable state);
    
    /**
     * 从经验中学习更新模型
     * 
     * 【学习机制差异】
     * 不同类型的Agent有不同的学习方式:
     * 
     * 1. ValueBasedAgent (如DQN):
     *    - 经验回放: 存入ReplayBuffer,随机采样打破相关性
     *    - TD学习: L = (r + γ*max Q_target(s',a') - Q(s,a))^2
     *    - 批量更新: 每次从buffer采样batch_size个经验
     * 
     * 2. PolicyBasedAgent (如REINFORCE):
     *    - 回合采样: 收集完整回合的(s,a,r)序列
     *    - 蒙特卡罗回报: G_t = Σ γ^k * r_{t+k}
     *    - 回合结束后更新: ∇θ J = Σ ∇log π(a|s) * G_t
     * 
     * 3. BanditAgent (如UCB):
     *    - 增量更新: Q_new = Q_old + α * (r - Q_old)
     *    - 无状态依赖: 直接更新动作价值估计
     * 
     * 【为什么需要不同的学习方式】
     * - ValueBased需要稳定的训练目标 → 经验回放+目标网络
     * - PolicyBased需要完整轨迹的回报 → 回合采样+蒙特卡罗
     * - Bandit无时序依赖 → 增量式更新即可
     * 
     * @param experience 经验数据(s, a, r, s', done)
     */
    public abstract void learn(Experience experience);
    
    /**
     * 批量学习更新模型
     * 
     * @param experiences 经验批次
     */
    public abstract void learnBatch(Experience[] experiences);
    
    /**
     * 存储经验（用于经验回放）
     * 
     * @param experience 要存储的经验
     */
    public abstract void storeExperience(Experience experience);
    
    /**
     * 获取模型的所有参数
     * 
     * @return 参数映射
     */
    public Map<String, Parameter> getAllParams() {
        if (model != null) {
            return model.getAllParams();
        }
        return new HashMap<>();
    }
    
    /**
     * 清空梯度
     */
    public void clearGrads() {
        if (model != null) {
            model.clearGrads();
        }
    }
    
    /**
     * 设置训练模式
     * 
     * @param training 是否为训练模式
     */
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    /**
     * 设置评估模式（禁用探索，使用确定性策略）
     * 等同于 setTraining(false)
     */
    public void eval() {
        this.training = false;
    }
    
    /**
     * 设置训练模式（启用探索）
     * 等同于 setTraining(true)
     */
    public void train() {
        this.training = true;
    }
    
    /**
     * 检查是否处于训练模式
     * 
     * @return 是否为训练模式
     */
    public boolean isTraining() {
        return training;
    }
    
    /**
     * 获取当前探索率
     * 
     * @return 当前探索率
     */
    public float getEpsilon() {
        return epsilon;
    }
    
    /**
     * 设置探索率
     * 
     * @param epsilon 新的探索率
     */
    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }
    
    /**
     * 衰减探索率
     * 
     * @param decayRate 衰减率
     */
    public void decayEpsilon(float decayRate) {
        this.epsilon = Math.max(0.01f, this.epsilon * decayRate);
    }
    
    /**
     * 获取训练步数
     * 
     * @return 训练步数
     */
    public int getTrainingStep() {
        return trainingStep;
    }
    
    /**
     * 增加训练步数
     */
    protected void incrementTrainingStep() {
        this.trainingStep++;
    }
    
    /**
     * 获取智能体名称
     * 
     * @return 智能体名称
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
    
    /**
     * 获取学习率
     * 
     * @return 学习率
     */
    public float getLearningRate() {
        return learningRate;
    }
    
    /**
     * 设置学习率
     * 
     * @param learningRate 新的学习率
     */
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * 获取折扣因子
     * 
     * @return 折扣因子
     */
    public float getGamma() {
        return gamma;
    }
    
    /**
     * 重置智能体状态
     */
    public void reset() {
        if (model != null) {
            model.resetState();
        }
    }
    
    /**
     * 保存模型参数
     * 
     * @param filepath 保存路径
     */
    public abstract void saveModel(String filepath);
    
    /**
     * 加载模型参数
     * 
     * @param filepath 加载路径
     */
    public abstract void loadModel(String filepath);
}