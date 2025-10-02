package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * 强化学习智能体抽象基类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * Agent类定义了强化学习智能体的标准接口，包括动作选择、学习更新等功能。
 * 智能体负责与环境交互，根据状态选择动作，并从经验中学习改进策略。
 * 支持不同的强化学习算法实现，如Q-Learning、Policy Gradient等。
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
     * @param experience 经验数据
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