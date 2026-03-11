package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.rl.AbstractAgent;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.util.TrainingStatistics;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于策略的强化学习智能体抽象基类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【算法分类 - Policy-Based RL】
 * 基于策略的强化学习算法直接学习策略函数 π(a|s),
 * 输出给定状态下的动作概率分布,然后从中采样动作。这类算法包括:
 * - REINFORCE (Policy Gradient)
 * - A2C/A3C (Advantage Actor-Critic)
 * - PPO (Proximal Policy Optimization)
 * - TRPO (Trust Region Policy Optimization)
 * 
 * 【核心组件】
 * 1. 策略网络(Policy Network): 输出动作概率分布 π(a|s)
 * 2. 回合缓冲区(Episode Buffer): 存储一个完整回合的经验
 * 3. 对数概率记录: 记录每步动作的 log π(a|s),用于计算策略梯度
 * 4. 回报计算: 蒙特卡洛估计或TD估计
 * 
 * 【与Value-Based的关键区别】
 * - Value-Based: 学习Q(s,a),然后argmax选动作 → 适合离散动作
 * - Policy-Based: 直接学习π(a|s) → 适合连续动作,支持随机策略
 * 
 * 【统一训练流程 - 模板方法模式】
 * learn() 方法定义了Policy-Based算法的标准训练流程:
 * 1. 存储回合内经验和对数概率
 * 2. 检查回合是否结束
 * 3. 回合结束时:
 *    a. 计算回报(蒙特卡罗或TD)
 *    b. 计算策略损失 (子类实现 computePolicyLoss)
 *    c. 反向传播更新策略
 *    d. 清空回合缓冲区
 * 
 * 【算法差异点 - 子类实现】
 * 不同Policy-Based算法的主要差异在于策略损失的计算:
 * - REINFORCE: -log π(a|s) * G_t
 * - A2C: -log π(a|s) * A(s,a) + value_loss
 * - PPO: clip(-log π(a|s) * A(s,a), ...)
 * 
 * 【教学价值】
 * 通过这个基类,学习者可以清晰理解:
 * - Policy-Based算法的通用框架
 * - 为什么需要完整回合数据(蒙特卡罗回报)
 * - 策略梯度定理的应用
 * - 不同算法仅在优势函数估计和损失形式上有差异
 */
public abstract class PolicyBasedAgent extends AbstractAgent {
    
    // ========== 核心组件 ==========
    
    /**
     * 回合经验缓冲区
     * 【作用】存储当前回合的所有经验,回合结束时用于计算回报
     */
    protected List<Experience> episodeBuffer;
    
    /**
     * 回合对数概率缓冲区
     * 【作用】存储每步动作的 log π(a|s),用于计算策略梯度
     */
    protected List<Variable> episodeLogProbs;
    
    /**
     * 回合奖励缓冲区
     * 【作用】存储每步获得的奖励,用于计算回报
     */
    protected List<Float> episodeRewards;
    
    /**
     * 策略网络优化器
     */
    protected Optimizer policyOptimizer;
    
    // ========== 训练统计 ==========
    
    /**
     * 训练统计信息
     */
    protected final TrainingStatistics stats;
    
    /**
     * 回合计数
     */
    protected int episodeCount;
    
    /**
     * 构造函数
     * 
     * @param name 智能体名称
     * @param stateDim 状态空间维度
     * @param actionDim 动作空间维度
     * @param learningRate 学习率
     * @param gamma 折扣因子
     */
    public PolicyBasedAgent(String name, int stateDim, int actionDim,
                           float learningRate, float gamma) {
        super(name, stateDim, actionDim, learningRate, 0.0f, gamma); // Policy-Based不使用epsilon
        
        this.episodeBuffer = new ArrayList<>();
        this.episodeLogProbs = new ArrayList<>();
        this.episodeRewards = new ArrayList<>();
        this.stats = new TrainingStatistics();
        this.episodeCount = 0;
        
        // 注意: model和policyOptimizer由子类在super()后创建
        this.policyOptimizer = null;  // 将在子类构造函数中初始化
    }
    
    /**
     * 选择动作 - 由子类实现
     * 
     * 【策略采样】
     * Policy-Based算法从策略网络输出的概率分布中采样动作:
     * 1. 前向传播: logits = π_θ(s)
     * 2. Softmax: probs = softmax(logits)
     * 3. 采样: a ~ Categorical(probs)
     * 4. 记录 log π(a|s) 用于梯度计算
     */
    @Override
    public abstract Variable selectAction(Variable state);
    
    /**
     * 存储回合经验
     * 
     * 【回合数据】
     * Policy-Based算法需要完整回合数据来计算蒙特卡罗回报。
     * 因此在回合结束前,所有经验都存储在缓冲区中。
     */
    @Override
    public void storeExperience(Experience experience) {
        if (training) {
            episodeBuffer.add(experience);
            episodeRewards.add(experience.getReward());
        }
    }
    
    /**
     * 学习更新 - 模板方法
     * 
     * 【Policy-Based算法统一流程】
     * 这是所有Policy-Based算法的标准学习流程,体现了:
     * - 回合采样: 收集完整回合数据
     * - 蒙特卡罗回报: 从后往前计算累积奖励
     * - 策略梯度更新: 使用 ∇log π(a|s) * G
     * 
     * 子类只需实现 computePolicyLoss
     */
    @Override
    public void learn(Experience experience) {
        // 1. 存储回合经验
        storeExperience(experience);
        
        // 2. 如果回合结束,进行学习
        if (experience.isDone()) {
            learnFromEpisode();
        }
    }
    
    /**
     * 批量学习 - Policy-Based通常不使用批量学习
     * 
     * 【说明】
     * Policy-Based算法通常基于单回合或多回合更新,
     * 而不是像DQN那样从缓冲区随机采样批次。
     * 
     * 【多回合数据处理问题】
     * 如果 batch 包含多个回合的数据,逐个调用 learn(exp) 会导致:
     * - 中间某个 done=true 的经验会提前触发 learnFromEpisode()
     * - 后续属于同一回合的经验会被丢弃
     * - 回合数据不完整,影响策略梯度计算
     * 
     * 因此,Policy-Based 算法应使用 learn() 方法逐步收集完整的回合数据。
     */
    @Override
    public void learnBatch(Experience[] experiences) {
        throw new UnsupportedOperationException(
            "Policy-Based 算法不支持批量学习。请使用 learn() 方法逐步收集完整的回合数据。" +
            "Policy-Based algorithms require complete episode data for Monte Carlo return calculation. " +
            "Use learn() method to collect experiences episode by episode."
        );
    }
    
    /**
     * 从回合数据中学习 - 核心训练逻辑
     * 
     * 【策略梯度更新流程】
     * 1. 计算回报: G_t = Σ γ^k * r_{t+k}
     * 2. 计算策略损失: L = -Σ log π(a_t|s_t) * G_t
     * 3. 反向传播: ∇θ L
     * 4. 参数更新: θ ← θ - α * ∇θ L
     * 5. 清空回合缓冲区
     */
    protected void learnFromEpisode() {
        if (episodeBuffer.isEmpty()) return;
        
        // 1. 计算回报
        List<Float> returns = computeReturns(episodeRewards);
        
        // 2. 计算策略损失 (子类实现算法特定逻辑)
        Variable policyLoss = computePolicyLoss(returns);
        
        // 3. 反向传播更新策略
        updatePolicy(policyLoss);
        
        // 4. 更新统计
        updateStatistics(returns);
        
        // 5. 清空回合数据
        clearEpisodeData();
        
        incrementTrainingStep();
        episodeCount++;
    }
    
    /**
     * 计算回报 - 蒙特卡罗估计
     * 
     * 【蒙特卡罗回报】
     * G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ... + γ^(T-t)*r_T
     * 
     * 【实现技巧】
     * 从后往前计算,使用递推关系:
     * G_t = r_t + γ * G_{t+1}
     * 
     * @param rewards 奖励序列
     * @return 回报序列
     */
    protected List<Float> computeReturns(List<Float> rewards) {
        List<Float> returns = new ArrayList<>();
        float runningReturn = 0.0f;
        
        // 从后往前计算折扣回报
        for (int i = rewards.size() - 1; i >= 0; i--) {
            runningReturn = rewards.get(i) + gamma * runningReturn;
            returns.add(0, runningReturn); // 插入到开头
        }
        
        return returns;
    }
    
    /**
     * 计算策略损失 - 抽象方法
     * 
     * 【算法差异点】
     * 这是不同Policy-Based算法的核心差异所在。
     * 
     * REINFORCE实现:
     *   L = -Σ log π(a_t|s_t) * G_t
     * 
     * A2C实现:
     *   L = -Σ log π(a_t|s_t) * A(s_t,a_t) + α * value_loss
     *   其中 A(s,a) = G_t - V(s_t)
     * 
     * PPO实现:
     *   L = -Σ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
     *   其中 r_t = π_new(a|s) / π_old(a|s)
     * 
     * @param returns 回报序列
     * @return 策略损失
     */
    protected abstract Variable computePolicyLoss(List<Float> returns);
    
    /**
     * 更新策略网络
     * 
     * @param policyLoss 策略损失
     */
    protected void updatePolicy(Variable policyLoss) {
        model.clearGrads();
        policyLoss.backward();
        policyOptimizer.update();
        
        // 更新损失统计
        stats.updateLoss(policyLoss.getValue().getNumber().floatValue());
    }
    
    /**
     * 更新统计信息
     * 
     * @param returns 回报序列
     */
    protected void updateStatistics(List<Float> returns) {
        if (!returns.isEmpty()) {
            float episodeReturn = returns.get(0); // 第一个元素是整个回合的总回报
            stats.updateReward(episodeReturn);
        }
    }
    
    /**
     * 清空回合数据
     */
    protected void clearEpisodeData() {
        episodeBuffer.clear();
        episodeLogProbs.clear();
        episodeRewards.clear();
    }
    
    /**
     * 存储对数概率
     * 
     * 【用于策略梯度】
     * 在 selectAction 中调用,记录 log π(a|s)
     * 
     * @param logProb 对数概率
     */
    protected void storeLogProb(Variable logProb) {
        if (training) {
            episodeLogProbs.add(logProb);
        }
    }
    
    // ========== Getter方法 ==========
    
    /**
     * 获取平均回报
     */
    public float getAverageReturn() {
        return stats.getAverageReward();
    }
    
    /**
     * 获取平均策略损失
     */
    public float getAveragePolicyLoss() {
        return stats.getAverageLoss();
    }
    
    /**
     * 获取回合计数
     */
    public int getEpisodeCount() {
        return episodeCount;
    }
    
    /**
     * 获取训练统计信息
     */
    public java.util.Map<String, Object> getTrainingStats() {
        java.util.Map<String, Object> statsMap = stats.toMap();
        statsMap.put("episode_count", episodeCount);
        statsMap.put("training_step", trainingStep);
        return statsMap;
    }
    
    /**
     * 重置训练统计
     */
    public void resetTrainingStats() {
        stats.reset();
        episodeCount = 0;
        clearEpisodeData();
    }
}
