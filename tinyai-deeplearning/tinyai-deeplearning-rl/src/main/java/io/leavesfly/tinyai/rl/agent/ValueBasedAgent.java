package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.ReplayBuffer;
import io.leavesfly.tinyai.rl.policy.EpsilonGreedyPolicy;
import io.leavesfly.tinyai.rl.util.ModelUtil;
import io.leavesfly.tinyai.rl.util.QValueComputer;
import io.leavesfly.tinyai.rl.util.TrainingStatistics;

/**
 * 基于值函数的强化学习智能体抽象基类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【算法分类 - Value-Based RL】
 * 基于值函数的强化学习算法学习状态-动作价值函数 Q(s,a),
 * 然后通过选择Q值最大的动作来制定策略。这类算法包括:
 * - DQN (Deep Q-Network)
 * - DoubleDQN (Double DQN)
 * - DuelingDQN (Dueling Network Architecture)
 * - PrioritizedDQN (Prioritized Experience Replay)
 * 
 * 【核心组件】
 * 1. 经验回放缓冲区(Replay Buffer): 存储和采样历史经验,打破数据相关性
 * 2. 目标网络(Target Network): 稳定训练目标,避免自举导致的不稳定
 * 3. ε-贪婪策略(Epsilon-Greedy): 平衡探索与利用
 * 4. TD学习(Temporal Difference): 使用时序差分更新Q值
 * 
 * 【统一训练流程 - 模板方法模式】
 * learn() 方法定义了Value-Based算法的标准训练流程:
 * 1. 存储经验到缓冲区
 * 2. 检查是否有足够经验进行学习
 * 3. 采样批次经验
 * 4. 计算目标Q值 (子类实现 computeTargetQValues)
 * 5. 计算当前Q值 (子类实现 computeCurrentQValues)
 * 6. 计算TD损失并反向传播
 * 7. 更新目标网络(定期)
 * 
 * 【算法差异点 - 子类实现】
 * 不同Value-Based算法的主要差异在于目标Q值的计算方式:
 * - DQN: y = r + γ * max_a' Q_target(s', a')
 * - DoubleDQN: y = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
 * - DuelingDQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
 * 
 * 【教学价值】
 * 通过这个基类,学习者可以清晰理解:
 * - Value-Based算法的通用框架
 * - 经验回放机制的作用
 * - 目标网络的必要性
 * - 不同算法仅在目标Q值计算上有差异
 */
public abstract class ValueBasedAgent extends Agent {
    
    // ========== 核心组件 ==========
    
    /**
     * 经验回放缓冲区
     * 【作用】存储历史经验,随机采样打破数据相关性
     */
    protected final ReplayBuffer replayBuffer;
    
    /**
     * 目标网络
     * 【作用】提供稳定的训练目标,定期从在线网络同步权重
     */
    protected Model targetModel;
    
    /**
     * 优化器
     * 【作用】更新网络参数,常用Adam或RMSprop
     */
    protected Optimizer optimizer;
    
    /**
     * 损失函数
     * 【作用】计算TD误差,通常使用MSE或Huber Loss
     */
    protected final Loss lossFunction;
    
    /**
     * ε-贪婪策略
     * 【作用】平衡探索与利用
     */
    protected EpsilonGreedyPolicy policy;
    
    // ========== 训练超参数 ==========
    
    /**
     * 批次大小
     */
    protected final int batchSize;
    
    /**
     * 目标网络更新频率
     */
    protected final int targetUpdateFreq;
    
    // ========== 训练统计 ==========
    
    /**
     * 训练统计信息
     */
    protected final TrainingStatistics stats;
    
    /**
     * 构造函数
     * 
     * @param name 智能体名称
     * @param stateDim 状态空间维度
     * @param actionDim 动作空间维度
     * @param learningRate 学习率
     * @param epsilon 初始探索率
     * @param gamma 折扣因子
     * @param batchSize 批次大小
     * @param bufferSize 经验回放缓冲区大小
     * @param targetUpdateFreq 目标网络更新频率
     */
    public ValueBasedAgent(String name, int stateDim, int actionDim,
                          float learningRate, float epsilon, float gamma,
                          int batchSize, int bufferSize, int targetUpdateFreq) {
        super(name, stateDim, actionDim, learningRate, epsilon, gamma);
        
        this.batchSize = batchSize;
        this.targetUpdateFreq = targetUpdateFreq;
        this.replayBuffer = new ReplayBuffer(bufferSize);
        this.lossFunction = new MeanSquaredLoss();
        this.stats = new TrainingStatistics();
        
        // targetModel, optimizer, policy由子类在super()后初始化
    }
    
    /**
     * 选择动作
     * 
     * 【策略】
     * - 训练模式: 使用ε-贪婪策略(探索)
     * - 评估模式: 使用贪婪策略(利用)
     */
    @Override
    public Variable selectAction(Variable state) {
        if (training) {
            return policy.selectAction(state);
        } else {
            // 评估模式: 总是选择Q值最大的动作
            Variable qValues = model.forward(state);
            return QValueComputer.selectGreedyAction(qValues, actionDim);
        }
    }
    
    /**
     * 存储经验到缓冲区
     * 
     * 【经验回放】
     * 将(s,a,r,s',done)五元组存入缓冲区,
     * 后续通过随机采样进行学习,打破数据相关性。
     */
    @Override
    public void storeExperience(Experience experience) {
        replayBuffer.push(experience);
    }
    
    /**
     * 学习更新 - 模板方法
     * 
     * 【Value-Based算法统一流程】
     * 这是所有Value-Based算法的标准学习流程,体现了:
     * - 经验回放机制
     * - 批量学习
     * - TD学习更新
     * - 目标网络同步
     * 
     * 子类只需实现 computeTargetQValues 和 computeCurrentQValues
     */
    @Override
    public void learn(Experience experience) {
        // 1. 存储经验
        storeExperience(experience);
        
        // 2. 检查是否有足够经验进行学习
        if (shouldLearn()) {
            // 3. 采样批次经验
            Experience[] batch = sampleBatch();
            
            // 4. 批量学习
            learnBatch(batch);
        }
    }
    
    /**
     * 批量学习 - 核心训练逻辑
     * 
     * 【TD学习流程】
     * 1. 计算目标Q值: y = r + γ * Q_target(s', a')
     * 2. 计算当前Q值: Q(s, a)
     * 3. 计算TD误差: loss = (y - Q(s,a))^2
     * 4. 反向传播更新网络
     * 5. 定期同步目标网络
     */
    @Override
    public void learnBatch(Experience[] experiences) {
        if (experiences.length == 0) return;
        
        // 1. 计算目标Q值 (子类实现算法特定逻辑)
        Variable targetQValues = computeTargetQValues(experiences);
        
        // 2. 计算当前Q值
        Variable currentQValues = computeCurrentQValues(experiences);
        
        // 3. 计算TD损失
        Variable loss = lossFunction.loss(targetQValues, currentQValues);
        
        // 4. 反向传播
        model.clearGrads();
        loss.backward();
        optimizer.update();
        
        // 5. 更新统计
        stats.updateLoss(loss.getValue().getNumber().floatValue());
        stats.incrementUpdate();
        incrementTrainingStep();
        
        // 6. 定期更新目标网络
        if (trainingStep % targetUpdateFreq == 0) {
            updateTargetNetwork();
        }
        
        // 7. 衰减探索率
        policy.decayEpsilon(0.995f, 0.01f);
    }
    
    /**
     * 计算目标Q值 - 抽象方法
     * 
     * 【算法差异点】
     * 这是不同Value-Based算法的核心差异所在。
     * 
     * DQN实现:
     *   y = r + γ * max_a' Q_target(s', a')
     * 
     * DoubleDQN实现:
     *   a_best = argmax_a' Q_online(s', a')
     *   y = r + γ * Q_target(s', a_best)
     * 
     * @param experiences 经验批次
     * @return 目标Q值批次 (batchSize, 1)
     */
    protected abstract Variable computeTargetQValues(Experience[] experiences);
    
    /**
     * 计算当前Q值 - 抽象方法
     * 
     * 【实现说明】
     * 对于批次中的每个经验 (s, a, r, s', done):
     * 1. 前向传播获取 Q(s, :) - 所有动作的Q值
     * 2. 提取 Q(s, a) - 实际选择动作的Q值
     * 3. 堆叠成批次Variable
     * 
     * @param experiences 经验批次
     * @return 当前Q值批次 (batchSize, 1)
     */
    protected abstract Variable computeCurrentQValues(Experience[] experiences);
    
    /**
     * 更新目标网络
     * 
     * 【Hard Update】
     * 将在线网络的权重完全复制到目标网络。
     * 大多数DQN变体使用这种方式。
     */
    protected void updateTargetNetwork() {
        ModelUtil.copyWeights(model, targetModel);
    }
    
    /**
     * 检查是否应该进行学习
     * 
     * @return 缓冲区是否有足够经验
     */
    protected boolean shouldLearn() {
        return replayBuffer.canSample(batchSize);
    }
    
    /**
     * 从缓冲区采样批次经验
     * 
     * @return 采样的经验批次
     */
    protected Experience[] sampleBatch() {
        return replayBuffer.sample(batchSize);
    }
    
    // ========== Getter方法 ==========
    
    /**
     * 获取平均损失
     */
    public float getAverageLoss() {
        return stats.getAverageLoss();
    }
    
    /**
     * 获取缓冲区使用率
     */
    public float getBufferUsage() {
        return replayBuffer.getUsageRate();
    }
    
    /**
     * 获取当前探索率
     */
    public float getCurrentEpsilon() {
        return policy.getEpsilon();
    }
    
    /**
     * 设置探索率
     */
    public void setEpsilon(float epsilon) {
        policy.setEpsilon(epsilon);
    }
    
    /**
     * 获取训练统计信息
     */
    public java.util.Map<String, Object> getTrainingStats() {
        java.util.Map<String, Object> statsMap = stats.toMap();
        statsMap.put("training_step", trainingStep);
        statsMap.put("epsilon", getCurrentEpsilon());
        statsMap.put("buffer_usage", getBufferUsage());
        return statsMap;
    }
    
    /**
     * 重置训练统计
     */
    public void resetTrainingStats() {
        stats.reset();
    }
    
    /**
     * 保存模型 - 保存在线网络和目标网络
     */
    @Override
    public void saveModel(String filepath) {
        // 保存在线网络
        model.saveModel(filepath);
        System.out.println(name + " 在线网络已保存到: " + filepath);
        
        // 保存目标网络
        String targetFilepath = filepath.replace(".model", "_target.model");
        targetModel.saveModel(targetFilepath);
        System.out.println(name + " 目标网络已保存到: " + targetFilepath);
    }
    
    /**
     * 加载模型 - 加载在线网络和目标网络
     */
    @Override
    public void loadModel(String filepath) {
        // 加载在线网络
        Model loadedModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(filepath);
        this.model.getModule().loadStateDict(loadedModel.getModule().copyStateDict(), true);
        System.out.println(name + " 在线网络已加载: " + filepath);
        
        // 加载目标网络
        String targetFilepath = filepath.replace(".model", "_target.model");
        Model loadedTargetModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(targetFilepath);
        this.targetModel.getModule().loadStateDict(loadedTargetModel.getModule().copyStateDict(), true);
        System.out.println(name + " 目标网络已加载: " + targetFilepath);
    }
}
