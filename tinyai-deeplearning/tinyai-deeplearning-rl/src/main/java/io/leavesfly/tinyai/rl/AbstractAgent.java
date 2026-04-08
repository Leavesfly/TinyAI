package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * 基于神经网络的强化学习智能体抽象基类
 * 
 * @author leavesfly
 * @version 0.02
 * 
 * 【设计说明】
 * AbstractAgent为使用神经网络的RL算法提供通用基础设施:
 * - 神经网络模型(model)管理
 * - 学习率、折扣因子等超参数
 * - 探索率(epsilon)及其衰减机制
 * - 训练/评估模式切换
 * - 训练步数统计
 * 
 * 不需要神经网络的算法(如多臂老虎机)应直接实现{@link Agent}接口,
 * 而非继承此基类,以避免携带无用字段。
 * 
 * 【继承体系】
 * {@code
 * AbstractAgent (本类)
 *  ├── ValueBasedAgent (基于值函数: DQN, DoubleDQN等)
 *  └── PolicyBasedAgent (基于策略: REINFORCE, PPO等)
 * }
 */
public abstract class AbstractAgent implements Agent {
    
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
    public AbstractAgent(String name, int stateDim, int actionDim, 
                         float learningRate, float epsilon, float gamma) {
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
    
    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    @Override
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
    
    @Override
    public int getTrainingStep() {
        return trainingStep;
    }
    
    /**
     * 增加训练步数
     */
    protected void incrementTrainingStep() {
        this.trainingStep++;
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public int getStateDim() {
        return stateDim;
    }
    
    @Override
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
    
    @Override
    public void reset() {
        if (model != null) {
            model.resetState();
        }
    }
}
