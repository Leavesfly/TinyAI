package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;

/**
 * PPO (Proximal Policy Optimization) 智能体实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * PPOAgent实现了PPO算法，这是目前最流行的策略梯度算法之一。
 * 主要特点包括：
 * 1. 使用裁剪目标函数限制策略更新幅度
 * 2. 支持多轮优化（mini-batch更新）
 * 3. 结合价值函数减少方差
 * 4. 训练稳定，适用范围广
 */
public class PPOAgent extends Agent {

    // PPO特有参数
    private final float clipEpsilon;           // PPO裁剪参数
    private final int ppoEpochs;               // PPO更新轮数
    private final int batchSize;               // 小批次大小
    private final Model valueModel;            // 价值网络
    private final Optimizer policyOptimizer;   // 策略网络优化器
    private final Optimizer valueOptimizer;    // 价值网络优化器

    // 回合数据存储
    private List<Experience> episodeExperiences;
    private List<Variable> episodeLogProbs;    // 旧的对数概率
    private List<Float> episodeRewards;
    private List<Variable> episodeStates;

    // 统计信息
    private float averageReturn;
    private float totalReturn;
    private int episodeCount;
    private float averagePolicyLoss;
    private float averageValueLoss;

    private final Random random;

    /**
     * 构造函数
     *
     * @param name         智能体名称
     * @param stateDim     状态空间维度
     * @param actionDim    动作空间维度
     * @param hiddenSizes  隐藏层尺寸数组
     * @param learningRate 学习率
     * @param gamma        折扣因子
     * @param clipEpsilon  PPO裁剪参数
     * @param ppoEpochs    PPO更新轮数
     * @param batchSize    小批次大小
     */
    public PPOAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
                    float learningRate, float gamma, float clipEpsilon,
                    int ppoEpochs, int batchSize) {
        super(name, stateDim, actionDim, learningRate, 0.0f, gamma);

        this.clipEpsilon = clipEpsilon;
        this.ppoEpochs = ppoEpochs;
        this.batchSize = batchSize;
        this.episodeExperiences = new ArrayList<>();
        this.episodeLogProbs = new ArrayList<>();
        this.episodeRewards = new ArrayList<>();
        this.episodeStates = new ArrayList<>();
        this.random = new Random();

        // 创建策略网络
        this.model = createPolicyNetwork(stateDim, actionDim, hiddenSizes);
        this.policyOptimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-3f);

        // 创建价值网络
        this.valueModel = createValueNetwork(stateDim, hiddenSizes);
        this.valueOptimizer = new Adam(valueModel, learningRate, 0.9f, 0.999f, 1e-3f);

        // 初始化统计
        this.averageReturn = 0.0f;
        this.totalReturn = 0.0f;
        this.episodeCount = 0;
        this.averagePolicyLoss = 0.0f;
        this.averageValueLoss = 0.0f;
    }

    /**
     * 创建策略网络
     */
    private Model createPolicyNetwork(int stateDim, int actionDim, int[] hiddenSizes) {
        Sequential mlpModule = new Sequential(name + "_PolicyNetwork");

        int inputSize = stateDim;

        for (int hiddenSize : hiddenSizes) {
            mlpModule.add(new Linear("fc", inputSize, hiddenSize, true));
            mlpModule.add(new ReLU("relu"));
            inputSize = hiddenSize;
        }

        mlpModule.add(new Linear("fc_out", inputSize, actionDim, true));

        return new Model(name + "_PolicyModel", mlpModule);
    }

    /**
     * 创建价值网络
     */
    private Model createValueNetwork(int stateDim, int[] hiddenSizes) {
        Sequential mlpModule = new Sequential(name + "_ValueNetwork");

        int inputSize = stateDim;

        for (int hiddenSize : hiddenSizes) {
            mlpModule.add(new Linear("fc", inputSize, hiddenSize, true));
            mlpModule.add(new ReLU("relu"));
            inputSize = hiddenSize;
        }

        mlpModule.add(new Linear("fc_out", inputSize, 1, true));

        return new Model(name + "_ValueModel", mlpModule);
    }

    @Override
    public Variable selectAction(Variable state) {
        // 前向传播获取动作概率分布
        Variable logits = model.forward(state);
        Variable probabilities = applySoftmax(logits);

        // 根据概率分布采样动作
        int action = sampleFromProbabilities(probabilities);

        // 存储数据用于训练
        if (training) {
            Variable logProb = computeLogProbability(probabilities, action);
            episodeLogProbs.add(logProb);
            episodeStates.add(state);
        }

        return new Variable(NdArray.of(action));
    }

    /**
     * 应用Softmax函数
     */
    private Variable applySoftmax(Variable logits) {
        return logits.softMax();
    }

    /**
     * 从概率分布中采样动作
     */
    private int sampleFromProbabilities(Variable probabilities) {
        NdArray probArray = probabilities.getValue();
        float[] probs = new float[actionDim];

        for (int i = 0; i < actionDim; i++) {
            probs[i] = probArray.get(0, i);
        }

        float randomValue = random.nextFloat();
        float cumulativeProb = 0.0f;

        for (int i = 0; i < actionDim; i++) {
            cumulativeProb += probs[i];
            if (randomValue <= cumulativeProb) {
                return i;
            }
        }

        return actionDim - 1;
    }

    /**
     * 计算特定动作的对数概率
     */
    private Variable computeLogProbability(Variable probabilities, int action) {
        Variable indexVar = new Variable(NdArray.of(new float[]{action}));
        Variable selectedProb = probabilities.indexSelect(1, indexVar);

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
        storeExperience(experience);
    }

    @Override
    public void learnBatch(Experience[] experiences) {
        for (Experience exp : experiences) {
            learn(exp);
        }
    }

    /**
     * 回合结束时的PPO学习更新
     */
    public void learnFromEpisode() {
        if (episodeExperiences.isEmpty()) return;

        // 计算回报和优势函数
        List<Float> returns = computeReturns(episodeRewards);
        List<Float> values = computeValues();
        List<Float> advantages = computeAdvantages(returns, values);

        // 标准化优势函数
        normalizeAdvantages(advantages);

        // PPO多轮更新
        for (int epoch = 0; epoch < ppoEpochs; epoch++) {
            updatePPO(advantages, returns);
        }

        // 更新统计
        updateStatistics(returns);

        // 清空回合数据
        clearEpisodeData();

        incrementTrainingStep();
    }

    /**
     * 计算回报
     */
    private List<Float> computeReturns(List<Float> rewards) {
        List<Float> returns = new ArrayList<>();
        float runningReturn = 0.0f;

        for (int i = rewards.size() - 1; i >= 0; i--) {
            runningReturn = rewards.get(i) + gamma * runningReturn;
            returns.add(0, runningReturn);
        }

        return returns;
    }

    /**
     * 计算价值估计
     */
    private List<Float> computeValues() {
        List<Float> values = new ArrayList<>();

        for (Variable state : episodeStates) {
            Variable value = valueModel.forward(state);
            values.add(value.getValue().getNumber().floatValue());
        }

        return values;
    }

    /**
     * 计算优势函数
     */
    private List<Float> computeAdvantages(List<Float> returns, List<Float> values) {
        List<Float> advantages = new ArrayList<>();

        for (int i = 0; i < returns.size(); i++) {
            float advantage = returns.get(i) - values.get(i);
            advantages.add(advantage);
        }

        return advantages;
    }

    /**
     * 标准化优势函数
     */
    private void normalizeAdvantages(List<Float> advantages) {
        if (advantages.isEmpty()) return;

        // 计算均值和标准差
        float mean = 0.0f;
        for (float adv : advantages) {
            mean += adv;
        }
        mean /= advantages.size();

        float variance = 0.0f;
        for (float adv : advantages) {
            variance += (adv - mean) * (adv - mean);
        }
        variance /= advantages.size();
        float std = (float) Math.sqrt(variance + 1e-8f);

        // 标准化
        for (int i = 0; i < advantages.size(); i++) {
            advantages.set(i, (advantages.get(i) - mean) / std);
        }
    }

    /**
     * PPO更新 - 使用真正的mini-batch更新
     */
    private void updatePPO(List<Float> advantages, List<Float> returns) {
        int dataSize = episodeStates.size();
        float totalPolicyLoss = 0.0f;
        float totalValueLoss = 0.0f;
        int updateCount = 0;

        // 创建索引数组并随机打乱
        Integer[] indices = new Integer[dataSize];
        for (int i = 0; i < dataSize; i++) {
            indices[i] = i;
        }
        java.util.Collections.shuffle(java.util.Arrays.asList(indices), random);

        // 按mini-batch遍历数据
        for (int start = 0; start < dataSize; start += batchSize) {
            int end = Math.min(start + batchSize, dataSize);
            
            // 累积梯度
            float batchPolicyLoss = 0.0f;
            float batchValueLoss = 0.0f;
            
            for (int idx = start; idx < end; idx++) {
                int i = indices[idx];
                Variable state = episodeStates.get(i);
                Variable oldLogProb = episodeLogProbs.get(i);
                float advantage = advantages.get(i);
                float returnValue = returns.get(i);

                // 获取当前动作
                int action = (int) episodeExperiences.get(i).getAction().getValue().getNumber().floatValue();

                // 计算新的对数概率
                Variable logits = model.forward(state);
                Variable probs = applySoftmax(logits);
                Variable newLogProb = computeLogProbability(probs, action);

                // 计算概率比率
                Variable logRatio = newLogProb.sub(oldLogProb);
                Variable ratio = logRatio.exp();

                // PPO裁剪目标
                Variable advantageVar = new Variable(NdArray.of(advantage));
                Variable surr1 = ratio.mul(advantageVar);

                Variable clipRatioLow = new Variable(NdArray.of(1.0f - clipEpsilon));
                Variable clipRatioHigh = new Variable(NdArray.of(1.0f + clipEpsilon));
                Variable clippedRatio = clipVariable(ratio, clipRatioLow, clipRatioHigh);
                Variable surr2 = clippedRatio.mul(advantageVar);

                // 策略损失：-min(surr1, surr2)
                Variable policyLoss = minVariable(surr1, surr2).mul(new Variable(NdArray.of(-1.0f)));
                batchPolicyLoss += policyLoss.getValue().getNumber().floatValue();

                // 反向传播累积梯度
                policyLoss.backward();

                // 价值损失
                Variable predictedValue = valueModel.forward(state);
                Variable targetValue = new Variable(NdArray.of(returnValue));
                Variable valueLoss = computeMSELoss(predictedValue, targetValue);
                batchValueLoss += valueLoss.getValue().getNumber().floatValue();

                // 反向传播累积梯度
                valueLoss.backward();
            }
            
            // 更新策略网络（一个mini-batch后统一更新）
            policyOptimizer.update();
            model.clearGrads();
            
            // 更新价值网络
            valueOptimizer.update();
            valueModel.clearGrads();
            
            totalPolicyLoss += batchPolicyLoss / (end - start);
            totalValueLoss += batchValueLoss / (end - start);
            updateCount++;
        }

        averagePolicyLoss = totalPolicyLoss / updateCount;
        averageValueLoss = totalValueLoss / updateCount;
    }

    /**
     * 裁剪Variable值
     */
    private Variable clipVariable(Variable value, Variable minVal, Variable maxVal) {
        // 简化实现：使用数值裁剪
        float val = value.getValue().getNumber().floatValue();
        float min = minVal.getValue().getNumber().floatValue();
        float max = maxVal.getValue().getNumber().floatValue();
        float clipped = Math.max(min, Math.min(max, val));
        return new Variable(NdArray.of(clipped));
    }

    /**
     * 计算两个Variable的最小值
     */
    private Variable minVariable(Variable v1, Variable v2) {
        float val1 = v1.getValue().getNumber().floatValue();
        float val2 = v2.getValue().getNumber().floatValue();
        return new Variable(NdArray.of(Math.min(val1, val2)));
    }

    /**
     * 计算均方误差损失
     */
    private Variable computeMSELoss(Variable predicted, Variable target) {
        Variable diff = predicted.sub(target);
        return diff.mul(diff);
    }

    /**
     * 更新统计信息
     */
    private void updateStatistics(List<Float> returns) {
        if (!returns.isEmpty()) {
            float episodeReturn = returns.get(0);
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
        episodeStates.clear();
    }

    public float getAverageReturn() {
        return averageReturn;
    }

    public float getAveragePolicyLoss() {
        return averagePolicyLoss;
    }

    public float getAverageValueLoss() {
        return averageValueLoss;
    }

    @Override
    public void saveModel(String filepath) {
        model.saveModel(filepath);
        System.out.println("PPO策略网络模型已保存到: " + filepath);

        String valueFilepath = filepath.replace(".model", "_value.model");
        valueModel.saveModel(valueFilepath);
        System.out.println("PPO价值网络模型已保存到: " + valueFilepath);
    }

    @Override
    public void loadModel(String filepath) {
        Model loadedModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(filepath);
        this.model.getModule().loadStateDict(loadedModel.getModule().copyStateDict(), true);
        System.out.println("PPO策略网络模型已从以下路径加载: " + filepath);

        String valueFilepath = filepath.replace(".model", "_value.model");
        Model loadedValueModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(valueFilepath);
        this.valueModel.getModule().loadStateDict(loadedValueModel.getModule().copyStateDict(), true);
        System.out.println("PPO价值网络模型已从以下路径加载: " + valueFilepath);
    }

    public Map<String, Object> getTrainingStats() {
        Map<String, Object> stats = new java.util.HashMap<>();
        stats.put("episode_count", episodeCount);
        stats.put("average_return", averageReturn);
        stats.put("average_policy_loss", averagePolicyLoss);
        stats.put("average_value_loss", averageValueLoss);
        stats.put("clip_epsilon", clipEpsilon);
        stats.put("ppo_epochs", ppoEpochs);
        return stats;
    }

    public void resetTrainingStats() {
        totalReturn = 0.0f;
        episodeCount = 0;
        averageReturn = 0.0f;
        averagePolicyLoss = 0.0f;
        averageValueLoss = 0.0f;
        clearEpisodeData();
    }
}
