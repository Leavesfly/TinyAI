package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * REINFORCE (Policy Gradient) 智能体实现
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * REINFORCEAgent实现了经典的策略梯度算法REINFORCE。
 * 主要特点包括：
 * 1. 直接优化策略网络，输出动作概率分布
 * 2. 使用蒙特卡罗方法计算回报
 * 3. 通过策略梯度定理更新策略参数
 * 4. 支持基线（baseline）减少方差
 */
public class REINFORCEAgent extends Agent {
    
    // REINFORCE特有参数
    private final boolean useBaseline;         // 是否使用基线
    private final Model baselineModel;        // 基线网络（价值函数）
    private final Optimizer baselineOptimizer; // 基线网络优化器
    private final Optimizer policyOptimizer;   // 策略网络优化器
    
    // 回合数据存储
    private List<Experience> episodeExperiences; // 当前回合的经验
    private List<Variable> episodeLogProbs;      // 当前回合的对数概率
    private List<Float> episodeRewards;          // 当前回合的奖励
    
    // 统计信息
    private float averageReturn;                 // 平均回报
    private float totalReturn;                   // 总回报
    private int episodeCount;                    // 回合计数
    private float averagePolicyLoss;             // 平均策略损失
    private float averageBaselineLoss;           // 平均基线损失
    
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
        super(name, stateDim, actionDim, learningRate, 0.0f, gamma); // REINFORCE不使用epsilon
        
        this.useBaseline = useBaseline;
        this.episodeExperiences = new ArrayList<>();
        this.episodeLogProbs = new ArrayList<>();
        this.episodeRewards = new ArrayList<>();
        this.random = new Random();
        
        // 创建策略网络
        this.model = createPolicyNetwork(stateDim, actionDim, hiddenSizes);
        this.policyOptimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-3f); // 使用安全的epsilon值
        
        // 创建基线网络（如果使用）
        if (useBaseline) {
            this.baselineModel = createBaselineNetwork(stateDim, hiddenSizes);
            this.baselineOptimizer = new Adam(baselineModel, learningRate, 0.9f, 0.999f, 1e-2f); // 使用更大的epsilon值确保稳定性
        } else {
            this.baselineModel = null;
            this.baselineOptimizer = null;
        }
        
        // 初始化统计
        this.averageReturn = 0.0f;
        this.totalReturn = 0.0f;
        this.episodeCount = 0;
        this.averagePolicyLoss = 0.0f;
        this.averageBaselineLoss = 0.0f;
    }
    
    /**
     * 创建策略网络
     * 
     * @param stateDim 状态维度
     * @param actionDim 动作维度
     * @param hiddenSizes 隐藏层尺寸
     * @return 策略网络模型
     */
    private Model createPolicyNetwork(int stateDim, int actionDim, int[] hiddenSizes) {
        // 使用Sequential构建MLP网络
        Sequential mlpModule = new Sequential(name + "_PolicyNetwork");
        
        // 输入层
        int inputSize = stateDim;
        
        // 添加隐藏层
        for (int hiddenSize : hiddenSizes) {
            mlpModule.add(new Linear("fc", inputSize, hiddenSize, true));
            mlpModule.add(new ReLU("relu"));
            inputSize = hiddenSize;
        }
        
        // 输出层 (动作概率分布)
        mlpModule.add(new Linear("fc_out", inputSize, actionDim, true));
        
        return new Model(name + "_PolicyModel", mlpModule);
    }
    
    /**
     * 创建基线网络（价值函数）
     * 
     * @param stateDim 状态维度
     * @param hiddenSizes 隐藏层尼寸
     * @return 基线网络模型
     */
    private Model createBaselineNetwork(int stateDim, int[] hiddenSizes) {
        // 使用Sequential构建MLP网络
        Sequential mlpModule = new Sequential(name + "_BaselineNetwork");
        
        // 输入层
        int inputSize = stateDim;
        
        // 添加隐藏层
        for (int hiddenSize : hiddenSizes) {
            mlpModule.add(new Linear("fc", inputSize, hiddenSize, true));
            mlpModule.add(new ReLU("relu"));
            inputSize = hiddenSize;
        }
        
        // 输出层 (价值标量)
        mlpModule.add(new Linear("fc_out", inputSize, 1, true));
        
        return new Model(name + "_BaselineModel", mlpModule);
    }
    
    @Override
    public Variable selectAction(Variable state) {
        // 前向传播获取动作概率分布
        Variable logits = model.forward(state);
        
        // 应用Softmax获取概率分布
        Variable probabilities = applySoftmax(logits);
        
        // 根据概率分布采样动作
        int action = sampleFromProbabilities(probabilities);
        
        // 计算对数概率并存储（用于训练）
        if (training) {
            Variable logProb = computeLogProbability(probabilities, action);
            episodeLogProbs.add(logProb);
        }
        
        return new Variable(NdArray.of(action));
    }
    
    /**
     * 应用Softmax函数
     * 
     * @param logits 网络输出
     * @return 概率分布
     */
    private Variable applySoftmax(Variable logits) {
        // 使用TinyDL的现有函数实现softmax
        return logits.softMax();
    }
    
    /**
     * 从概率分布中采样动作
     * 
     * @param probabilities 概率分布
     * @return 采样的动作
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
        
        // 如果由于数值误差没有采样到，返回最后一个动作
        return actionDim - 1;
    }
    
    /**
     * 计算特定动作的对数概率
     * 
     * @param probabilities 概率分布
     * @param action 选择的动作
     * @return 对数概率
     */
    private Variable computeLogProbability(Variable probabilities, int action) {
        // 使用indexSelect提取指定动作的概率，保持计算图连通
        Variable indexVar = new Variable(NdArray.of(new float[]{action}));
        Variable selectedProb = probabilities.indexSelect(1, indexVar);
        
        // 使用Variable的log操作，保持计算图
        // 为避免log(0)，添加小常数
        Variable epsilon = new Variable(NdArray.of(1e-8f));
        Variable clippedProb = selectedProb.add(epsilon);
        return clippedProb.log();
    }
    
    @Override
    public void storeExperience(Experience experience) {
        if (training) {
            episodeExperiences.add(experience);
            episodeRewards.add(experience.getReward());
        }
    }
    
    @Override
    public void learn(Experience experience) {
        // REINFORCE在回合结束时学习，这里只存储经验
        storeExperience(experience);
    }
    
    @Override
    public void learnBatch(Experience[] experiences) {
        // REINFORCE通常不使用批次学习，但可以实现多回合批次更新
        for (Experience exp : experiences) {
            learn(exp);
        }
    }
    
    /**
     * 回合结束时的学习更新
     */
    public void learnFromEpisode() {
        if (episodeExperiences.isEmpty()) return;
        
        // 计算回报
        List<Float> returns = computeReturns(episodeRewards);
        
        // 计算基线（如果使用）
        List<Variable> baselinesVar = null;
        if (useBaseline) {
            baselinesVar = computeBaselinesVariable();
            updateBaseline(returns);
        }
        
        // 更新策略
        updatePolicyVariable(returns, baselinesVar);
        
        // 更新统计
        updateStatistics(returns);
        
        // 清空回合数据
        clearEpisodeData();
        
        incrementTrainingStep();
    }
    
    /**
     * 计算回报（蒙特卡罗）
     * 
     * @param rewards 奖励序列
     * @return 回报序列
     */
    private List<Float> computeReturns(List<Float> rewards) {
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
     * 计算基线值
     * 
     * @return 基线值序列
     */
    private List<Variable> computeBaselinesVariable() {
        List<Variable> baselines = new ArrayList<>();
        
        for (Experience experience : episodeExperiences) {
            Variable state = experience.getState();
            Variable baselineValue = baselineModel.forward(state);
            // 保持Variable类型，不提取数值
            baselines.add(baselineValue);
        }
        
        return baselines;
    }
    
    /**
     * 计算基线值（数值版本，用于非训练场景）
     * 
     * @return 基线值序列
     */
    private List<Float> computeBaselines() {
        List<Float> baselines = new ArrayList<>();
        
        for (Experience experience : episodeExperiences) {
            Variable state = experience.getState();
            Variable baselineValue = baselineModel.forward(state);
            float baseline = baselineValue.getValue().getNumber().floatValue();
            baselines.add(baseline);
        }
        
        return baselines;
    }
    
    /**
     * 更新基线网络
     * 
     * @param returns 真实回报
     */
    private void updateBaseline(List<Float> returns) {
        float totalBaselineLoss = 0.0f;
        
        for (int i = 0; i < episodeExperiences.size(); i++) {
            Variable state = episodeExperiences.get(i).getState();
            Variable predictedValue = baselineModel.forward(state);
            Variable targetValue = new Variable(NdArray.of(returns.get(i)));
            
            // 计算MSE损失
            Variable loss = computeMSELoss(predictedValue, targetValue);
            
            // 反向传播
            baselineModel.clearGrads();
            loss.backward();
            baselineOptimizer.update();
            
            totalBaselineLoss += loss.getValue().getNumber().floatValue();
        }
        
        // 更新基线损失统计
        averageBaselineLoss = totalBaselineLoss / episodeExperiences.size();
    }
    
    /**
     * 计算均方误差损失
     * 
     * @param predicted 预测值
     * @param target 目标值
     * @return MSE损失
     */
    private Variable computeMSELoss(Variable predicted, Variable target) {
        Variable diff = predicted.sub(target);
        return diff.mul(diff);
    }
    
    /**
     * 更新策略网络（Variable版本，保持计算图）
     * 
     * @param returns 回报序列
     * @param baselines 基线Variable序列（可为null）
     */
    private void updatePolicyVariable(List<Float> returns, List<Variable> baselines) {
        float totalPolicyLoss = 0.0f;
        
        for (int i = 0; i < episodeLogProbs.size(); i++) {
            Variable logProb = episodeLogProbs.get(i);
            Variable returnVar = new Variable(NdArray.of(returns.get(i)));
            
            // 计算优势函数（保持Variable类型）
            Variable advantageVar = returnVar;
            if (baselines != null) {
                advantageVar = returnVar.sub(baselines.get(i));
            }
            
            // 策略梯度：-log(π(a|s)) * A
            // 取负号：因为我们要最大化期望回报，但优化器是最小化损失
            Variable negAdvantage = advantageVar.mul(new Variable(NdArray.of(-1.0f)));
            Variable policyLoss = logProb.mul(negAdvantage);
            
            // 反向传播
            model.clearGrads();
            policyLoss.backward();
            policyOptimizer.update();
            
            totalPolicyLoss += policyLoss.getValue().getNumber().floatValue();
        }
        
        // 更新策略损失统计
        averagePolicyLoss = totalPolicyLoss / episodeLogProbs.size();
    }
    
    /**
     * 更新策略网络
     * 
     * @param returns 回报序列
     * @param baselines 基线序列（可为null）
     */
    private void updatePolicy(List<Float> returns, List<Float> baselines) {
        float totalPolicyLoss = 0.0f;
        
        for (int i = 0; i < episodeLogProbs.size(); i++) {
            Variable logProb = episodeLogProbs.get(i);
            float returnValue = returns.get(i);
            
            // 计算优势函数
            float advantage = returnValue;
            if (baselines != null) {
                advantage -= baselines.get(i);
            }
            
            // 策略梯度：-log(π(a|s)) * A
            Variable advantageVar = new Variable(NdArray.of(-advantage));
            Variable policyLoss = logProb.mul(advantageVar);
            
            // 反向传播
            model.clearGrads();
            policyLoss.backward();
            policyOptimizer.update();
            
            totalPolicyLoss += policyLoss.getValue().getNumber().floatValue();
        }
        
        // 更新策略损失统计
        averagePolicyLoss = totalPolicyLoss / episodeLogProbs.size();
    }
    
    /**
     * 更新统计信息
     * 
     * @param returns 回报序列
     */
    private void updateStatistics(List<Float> returns) {
        if (!returns.isEmpty()) {
            float episodeReturn = returns.get(0); // 第一个元素是整个回合的回报
            totalReturn += episodeReturn;
            episodeCount++;
            averageReturn = totalReturn / episodeCount;
        }
    }
    
    /**
     * 清空回合数据
     */
    private void clearEpisodeData() {
        episodeExperiences.clear();
        episodeLogProbs.clear();
        episodeRewards.clear();
    }
    
    /**
     * 获取平均回报
     * 
     * @return 平均回报
     */
    public float getAverageReturn() {
        return averageReturn;
    }
    
    /**
     * 获取平均策略损失
     * 
     * @return 平均策略损失
     */
    public float getAveragePolicyLoss() {
        return averagePolicyLoss;
    }
    
    /**
     * 获取平均基线损失
     * 
     * @return 平均基线损失
     */
    public float getAverageBaselineLoss() {
        return averageBaselineLoss;
    }
    
    /**
     * 是否使用基线
     * 
     * @return 是否使用基线
     */
    public boolean isUsingBaseline() {
        return useBaseline;
    }
    
    @Override
    public void saveModel(String filepath) {
        // 保存策略网络模型
        model.saveModel(filepath);
        System.out.println("REINFORCE策略网络模型已保存到: " + filepath);

        // 如果使用基线，同时保存基线网络模型
        if (useBaseline) {
            String baselineFilepath = filepath.replace(".model", "_baseline.model");
            baselineModel.saveModel(baselineFilepath);
            System.out.println("REINFORCE基线网络模型已保存到: " + baselineFilepath);
        }
    }

    @Override
    public void loadModel(String filepath) {
        // 加载策略网络模型参数
        Model loadedModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(filepath);
        this.model.getModule().loadStateDict(loadedModel.getModule().copyStateDict(), true);
        System.out.println("REINFORCE策略网络模型已从以下路径加载: " + filepath);

        // 如果使用基线，同时加载基线网络模型
        if (useBaseline) {
            String baselineFilepath = filepath.replace(".model", "_baseline.model");
            Model loadedBaselineModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(baselineFilepath);
            this.baselineModel.getModule().loadStateDict(loadedBaselineModel.getModule().copyStateDict(), true);
            System.out.println("REINFORCE基线网络模型已从以下路径加载: " + baselineFilepath);
        }
    }
    
    /**
     * 获取训练统计信息
     * 
     * @return 统计信息映射
     */
    public java.util.Map<String, Object> getTrainingStats() {
        java.util.Map<String, Object> stats = new java.util.HashMap<>();
        stats.put("episode_count", episodeCount);
        stats.put("average_return", averageReturn);
        stats.put("average_policy_loss", averagePolicyLoss);
        stats.put("use_baseline", useBaseline);
        if (useBaseline) {
            stats.put("average_baseline_loss", averageBaselineLoss);
        }
        return stats;
    }
    
    /**
     * 重置训练统计
     */
    public void resetTrainingStats() {
        totalReturn = 0.0f;
        episodeCount = 0;
        averageReturn = 0.0f;
        averagePolicyLoss = 0.0f;
        averageBaselineLoss = 0.0f;
        clearEpisodeData();
    }
}