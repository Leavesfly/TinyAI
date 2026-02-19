package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.util.ModelUtil;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.List;
import java.util.Random;

/**
 * REINFORCE (Policy Gradient) 智能体实现
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【REINFORCE算法】
 * REINFORCEAgent实现了经典的策略梯度算法REINFORCE(蒙特卡罗策略梯度)。
 * 
 * 【核心思想 - 直接优化策略】
 * 与Value-Based方法不同,REINFORCE直接学习策略函数π(a|s):
 * - Value-Based: 学习Q(s,a) → 策略为argmax Q
 * - Policy-Based: 直接学习π(a|s) → 输出动作概率分布
 * 
 * 【策略梯度定理】
 * 目标: 最大化期望回报 J(θ) = E[Σ r_t]
 * 梯度: ∇J(θ) = E[∇log π(a|s) * G_t]
 * 其中 G_t = Σ γ^k * r_{t+k} 是蒙特卡罗回报
 * 
 * 【算法流程】
 * 1. 收集完整回合: 从s0到终止状态
 * 2. 计算回报: G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ...
 * 3. 计算策略损失: L = -Σ log π(a_t|s_t) * G_t
 * 4. 反向传播更新策略参数
 * 
 * 【基线(Baseline)机制】
 * 可选使用基线函数V(s)减少方差:
 * - 优势函数: A(s,a) = G_t - V(s)
 * - 更新公式: L = -Σ log π(a_t|s_t) * A(s,a)
 * - 效果: 方差减小,学习更稳定
 * 
 * 【继承关系】
 * REINFORCEAgent → PolicyBasedAgent → Agent
 * 继承了PolicyBasedAgent的回合采样和回报计算流程
 * 
 * 【教学价值】
 * 通过REINFORCE,学习者可以理解:
 * - Policy-Based vs Value-Based的本质区别
 * - 为什么需要完整回合(蒙特卡罗估计)
 * - 策略梯度定理的实际应用
 * - 基线如何减少方差
 */
public class REINFORCEAgent extends PolicyBasedAgent {
    
    // ========== REINFORCE特有组件 ==========
    
    /**
     * 是否使用基线
     */
    private final boolean useBaseline;
    
    /**
     * 基线网络(价值函数V(s))
     */
    private final Model baselineModel;
    
    /**
     * 基线网络优化器
     */
    private final Adam baselineOptimizer;
    
    /**
     * 随机数生成器(用于策略采样)
     */
    private final Random random;
    
    /**
     * 构造函数
     * 
     * @param name 智能体名称
     * @param stateDim 状态空间维度
     * @param actionDim 动作空间维度
     * @param hiddenSizes 隐藏层尺寸数组
     * @param learningRate 学习率
     * @param gamma 折扣因子
     * @param useBaseline 是否使用基线
     */
    public REINFORCEAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
                         float learningRate, float gamma, boolean useBaseline) {
        super(name, stateDim, actionDim, learningRate, gamma);
        
        this.useBaseline = useBaseline;
        this.random = new Random();
        
        // 创建策略网络 - 使用ModelUtil工具类
        this.model = ModelUtil.createPolicyNetwork(name + "_Policy", stateDim, actionDim, hiddenSizes);
        this.policyOptimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-3f);
        
        // 创建基线网络(如果使用)
        if (useBaseline) {
            this.baselineModel = ModelUtil.createValueNetwork(name + "_Baseline", stateDim, hiddenSizes);
            this.baselineOptimizer = new Adam(baselineModel, learningRate, 0.9f, 0.999f, 1e-2f);
        } else {
            this.baselineModel = null;
            this.baselineOptimizer = null;
        }
    }
    
    /**
     * 选择动作 - 从策略分布中采样
     * 
     * 【策略采样流程】
     * 1. 前向传播: logits = π_θ(s)
     * 2. Softmax: probs = softmax(logits)
     * 3. 采样: a ~ Categorical(probs)
     * 4. 记录 log π(a|s) 用于梯度计算
     * 
     * 【为什么要采样而不是argmax】
     * - 策略梯度需要探索,随机策略自然提供探索
     * - 连续动作空间无法argmax
     * - 随机策略在某些环境更优(如石头剪刀布)
     */
    @Override
    public Variable selectAction(Variable state) {
        // 1. 前向传播获取logits
        Variable logits = model.forward(state);
        
        // 2. 应用Softmax获取概率分布
        Variable probabilities = logits.softMax();
        
        // 3. 从概率分布中采样动作
        int action = sampleFromProbabilities(probabilities);
        
        // 4. 计算并存储对数概率(用于训练)
        if (training) {
            Variable logProb = computeLogProbability(probabilities, action);
            storeLogProb(logProb);
        }
        
        return new Variable(NdArray.of(action));
    }
    
    /**
     * 从概率分布中采样动作
     * 
     * 【分类分布采样】
     * 使用累积概率法:
     * 1. 生成随机数u ~ Uniform(0,1)
     * 2. 找到满足Σ_{i=1}^k p_i >= u的最小k
     * 3. 返回动作k
     */
    private int sampleFromProbabilities(Variable probabilities) {
        NdArray probArray = probabilities.getValue();
        float[] probs = new float[actionDim];
        
        // 提取概率值
        for (int i = 0; i < actionDim; i++) {
            probs[i] = probArray.get(0, i);
        }
        
        // 累积概率采样
        float randomValue = random.nextFloat();
        float cumulativeProb = 0.0f;
        
        for (int i = 0; i < actionDim; i++) {
            cumulativeProb += probs[i];
            if (randomValue <= cumulativeProb) {
                return i;
            }
        }
        
        // 数值误差保护
        return actionDim - 1;
    }
    
    /**
     * 计算特定动作的对数概率
     * 
     * 【对数概率】
     * log π(a|s) 是策略梯度的核心组成部分
     * 
     * 【数值稳定性】
     * 添加小常数ε避免log(0)
     */
    private Variable computeLogProbability(Variable probabilities, int action) {
        // 提取指定动作的概率
        Variable indexVar = new Variable(NdArray.of(new float[]{action}));
        Variable selectedProb = probabilities.indexSelect(1, indexVar);
        
        // 添加小常数避免log(0)
        Variable epsilon = new Variable(NdArray.of(1e-8f));
        Variable clippedProb = selectedProb.add(epsilon);
        
        return clippedProb.log();
    }
    
    /**
     * 计算策略损失 - REINFORCE算法核心
     * 
     * 【策略梯度公式】
     * L = -Σ log π(a_t|s_t) * G_t
     * 
     * 【使用基线时】
     * L = -Σ log π(a_t|s_t) * (G_t - V(s_t))
     * 其中 V(s_t) 是基线网络的输出
     * 
     * 【为什么取负号】
     * 优化器是最小化损失,但我们要最大化期望回报
     * 因此取负将最大化问题转为最小化问题
     * 
     * 【基线的作用】
     * - 减少方差: A(s,a) = G_t - V(s) 的方差更小
     * - 不改变期望: E[V(s)] = E[V(s)] 期望梯度不变
     * - 加速收敛: 方差小意味着学习更稳定
     */
    @Override
    protected Variable computePolicyLoss(List<Float> returns) {
        Variable totalLoss = new Variable(NdArray.of(0.0f));
        
        for (int i = 0; i < episodeLogProbs.size(); i++) {
            Variable logProb = episodeLogProbs.get(i);
            float returnValue = returns.get(i);
            
            // 计算优势函数
            float advantage = returnValue;
            if (useBaseline && baselineModel != null) {
                Variable state = episodeBuffer.get(i).getState();
                Variable baseline = baselineModel.forward(state);
                advantage -= baseline.getValue().getNumber().floatValue();
            }
            
            // 策略损失: -log π(a|s) * A
            Variable advantageVar = new Variable(NdArray.of(-advantage));
            Variable stepLoss = logProb.mul(advantageVar);
            totalLoss = totalLoss.add(stepLoss);
        }
        
        return totalLoss;
    }
    
    /**
     * 从回合数据中学习 - 重写以支持基线更新
     * 
     * 【REINFORCE训练流程】
     * 1. 计算蒙特卡罗回报
     * 2. 更新基线网络(如果使用)
     * 3. 计算策略损失
     * 4. 更新策略网络
     * 5. 清空回合缓冲区
     */
    @Override
    protected void learnFromEpisode() {
        if (episodeBuffer.isEmpty()) return;
        
        // 1. 计算回报
        List<Float> returns = computeReturns(episodeRewards);
        
        // 2. 更新基线网络(如果使用)
        if (useBaseline && baselineModel != null) {
            updateBaseline(returns);
        }
        
        // 3. 计算并更新策略
        Variable policyLoss = computePolicyLoss(returns);
        updatePolicy(policyLoss);
        
        // 4. 更新统计
        updateStatistics(returns);
        
        // 5. 清空回合数据
        clearEpisodeData();
        
        incrementTrainingStep();
        episodeCount++;
    }
    
    /**
     * 更新基线网络
     * 
     * 【基线训练】
     * 最小化均方误差: L = (G_t - V(s_t))^2
     * 即训练价值函数逼近实际回报
     */
    private void updateBaseline(List<Float> returns) {
        for (int i = 0; i < episodeBuffer.size(); i++) {
            Variable state = episodeBuffer.get(i).getState();
            Variable predictedValue = baselineModel.forward(state);
            Variable targetValue = new Variable(NdArray.of(returns.get(i)));
            
            // MSE损失
            Variable diff = predictedValue.sub(targetValue);
            Variable loss = diff.mul(diff);
            
            // 反向传播
            baselineModel.clearGrads();
            loss.backward();
            baselineOptimizer.update();
        }
    }
    
    @Override
    public void saveModel(String filepath) {
        // 保存策略网络
        model.saveModel(filepath);
        System.out.println("REINFORCE策略网络已保存到: " + filepath);

        // 保存基线网络(如果使用)
        if (useBaseline && baselineModel != null) {
            String baselineFilepath = filepath.replace(".model", "_baseline.model");
            baselineModel.saveModel(baselineFilepath);
            System.out.println("REINFORCE基线网络已保存到: " + baselineFilepath);
        }
    }

    @Override
    public void loadModel(String filepath) {
        // 加载策略网络
        Model loadedModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(filepath);
        this.model.getModule().loadStateDict(loadedModel.getModule().copyStateDict(), true);
        System.out.println("REINFORCE策略网络已加载: " + filepath);

        // 加载基线网络(如果使用)
        if (useBaseline && baselineModel != null) {
            String baselineFilepath = filepath.replace(".model", "_baseline.model");
            Model loadedBaselineModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(baselineFilepath);
            this.baselineModel.getModule().loadStateDict(loadedBaselineModel.getModule().copyStateDict(), true);
            System.out.println("REINFORCE基线网络已加载: " + baselineFilepath);
        }
    }
    
    /**
     * 是否使用基线
     */
    public boolean isUsingBaseline() {
        return useBaseline;
    }
}
