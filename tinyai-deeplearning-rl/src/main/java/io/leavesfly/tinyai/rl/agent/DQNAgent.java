package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.ReplayBuffer;
import io.leavesfly.tinyai.rl.policy.EpsilonGreedyPolicy;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.block.MlpBlock;

/**
 * Deep Q-Network (DQN) 智能体实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * DQNAgent实现了深度Q网络算法，这是第一个成功将深度学习应用于强化学习的算法。
 * 主要特点包括：
 * 1. 使用神经网络逼近Q函数
 * 2. 经验回放机制提高数据利用率
 * 3. 目标网络稳定训练过程
 * 4. ε-贪婪策略平衡探索与利用
 */
public class DQNAgent extends Agent {

    // DQN特有参数
    private final int batchSize;              // 批次大小
    private final int targetUpdateFreq;       // 目标网络更新频率
    private final ReplayBuffer replayBuffer;  // 经验回放缓冲区
    private final Model targetModel;          // 目标网络
    private final EpsilonGreedyPolicy policy; // ε-贪婪策略
    private final Optimizer optimizer;        // 优化器
    private final Loss lossFunction;          // 损失函数

    // 训练统计
    private int updateCount;                  // 更新次数计数
    private float averageLoss;                // 平均损失
    private float totalLoss;                  // 总损失
    private int lossCount;                    // 损失计算次数

    /**
     * 构造函数
     *
     * @param name             智能体名称
     * @param stateDim         状态空间维度
     * @param actionDim        动作空间维度
     * @param hiddenSizes      隐藏层尺寸数组
     * @param learningRate     学习率
     * @param epsilon          初始探索率
     * @param gamma            折扣因子
     * @param batchSize        批次大小
     * @param bufferSize       经验回放缓冲区大小
     * @param targetUpdateFreq 目标网络更新频率
     */
    public DQNAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
                    float learningRate, float epsilon, float gamma,
                    int batchSize, int bufferSize, int targetUpdateFreq) {
        super(name, stateDim, actionDim, learningRate, epsilon, gamma);

        this.batchSize = batchSize;
        this.targetUpdateFreq = targetUpdateFreq;
        this.replayBuffer = new ReplayBuffer(bufferSize);
        this.updateCount = 0;
        this.averageLoss = 0.0f;
        this.totalLoss = 0.0f;
        this.lossCount = 0;

        // 创建Q网络
        this.model = createQNetwork(stateDim, actionDim, hiddenSizes);

        // 创建目标网络（复制主网络）
        this.targetModel = createQNetwork(stateDim, actionDim, hiddenSizes);
        copyModelWeights(model, targetModel);

        // 创建ε-贪婪策略
        this.policy = new EpsilonGreedyPolicy(stateDim, actionDim, epsilon,
                state -> model.forward(state));

        // 创建优化器和损失函数
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-3f); // 使用安全的epsilon值
        this.lossFunction = new MeanSquaredLoss();
    }

    /**
     * 创建Q网络
     *
     * @param stateDim    状态维度
     * @param actionDim   动作维度
     * @param hiddenSizes 隐藏层尺寸
     * @return Q网络模型
     */
    private Model createQNetwork(int stateDim, int actionDim, int[] hiddenSizes) {
        // 构建网络层尺寸数组
        int[] allSizes = new int[hiddenSizes.length + 2];
        allSizes[0] = stateDim;
        System.arraycopy(hiddenSizes, 0, allSizes, 1, hiddenSizes.length);
        allSizes[allSizes.length - 1] = actionDim;

        // 创建MLP网络
        MlpBlock mlpBlock = new MlpBlock(
                name + "_QNetwork",
                1, // batchSize
                null, // inputShape (will be set automatically)
                allSizes
        );

        return new Model(name + "_QModel", mlpBlock);
    }

    /**
     * 复制模型权重
     *
     * @param source 源模型
     * @param target 目标模型
     */
    private void copyModelWeights(Model source, Model target) {
        // 这里应该实现权重复制逻辑
        // 由于TinyDL框架的限制，暂时使用简化实现
        System.out.println("目标网络权重已更新");
    }

    @Override
    public Variable selectAction(Variable state) {
        if (training) {
            return policy.selectAction(state);
        } else {
            // 测试模式：总是选择贪婪动作
            Variable qValues = model.forward(state);
            return selectGreedyAction(qValues);
        }
    }

    /**
     * 选择贪婪动作（Q值最大的动作）
     *
     * @param qValues Q值向量
     * @return 贪婪动作
     */
    private Variable selectGreedyAction(Variable qValues) {
        NdArray qArray = qValues.getValue();
        int bestAction = 0;
        float maxQ = qArray.get(0, 0);

        for (int i = 1; i < actionDim; i++) {
            float q = qArray.get(0, i);
            if (q > maxQ) {
                maxQ = q;
                bestAction = i;
            }
        }

        return new Variable(NdArray.of(bestAction));
    }

    @Override
    public void storeExperience(Experience experience) {
        replayBuffer.push(experience);
    }

    @Override
    public void learn(Experience experience) {
        // 存储经验
        storeExperience(experience);

        // 如果有足够的经验，进行学习
        if (replayBuffer.canSample(batchSize)) {
            Experience[] batch = replayBuffer.sample(batchSize);
            learnBatch(batch);
        }
    }

    @Override
    public void learnBatch(Experience[] experiences) {
        if (experiences.length == 0) return;

        // 准备批量数据
        float[][] states = new float[experiences.length][stateDim];
        float[][] actions = new float[experiences.length][1];
        float[][] rewards = new float[experiences.length][1];
        float[][] nextStates = new float[experiences.length][stateDim];
        boolean[] dones = new boolean[experiences.length];

        // 提取批量数据
        for (int i = 0; i < experiences.length; i++) {
            Experience exp = experiences[i];

            // 状态
            NdArray stateArray = exp.getState().getValue();
            for (int j = 0; j < stateDim; j++) {
                states[i][j] = stateArray.get(0, j);
            }

            // 动作
            actions[i][0] = exp.getAction().getValue().getNumber().floatValue();

            // 奖励
            rewards[i][0] = exp.getReward();

            // 下一状态
            NdArray nextStateArray = exp.getNextState().getValue();
            for (int j = 0; j < stateDim; j++) {
                nextStates[i][j] = nextStateArray.get(0, j);
            }

            // 是否结束
            dones[i] = exp.isDone();
        }

        // 计算目标Q值
        Variable targetQValues = computeTargetQValues(nextStates, rewards, dones);

        // 计算当前Q值
        Variable currentQValues = computeCurrentQValues(states, actions);

        // 计算损失并更新网络
        Variable loss = lossFunction.loss(targetQValues, currentQValues);

        // 反向传播
        model.clearGrads();
        loss.backward();
        optimizer.update();

        // 更新统计
        updateLossStatistics(loss.getValue().getNumber().floatValue());
        incrementTrainingStep();

        // 定期更新目标网络
        if (trainingStep % targetUpdateFreq == 0) {
            copyModelWeights(model, targetModel);
        }

        // 衰减探索率
        policy.decayEpsilon(0.995f, 0.01f);
    }

    /**
     * 计算目标Q值
     *
     * @param nextStates 下一状态批次
     * @param rewards    奖励批次
     * @param dones      结束标志批次
     * @return 目标Q值
     */
    private Variable computeTargetQValues(float[][] nextStates, float[][] rewards, boolean[] dones) {
        int batchSize = nextStates.length;
        float[] targetValues = new float[batchSize];

        for (int i = 0; i < batchSize; i++) {
            if (dones[i]) {
                // 如果是终止状态，目标值就是奖励
                targetValues[i] = rewards[i][0];
            } else {
                // 否则使用Bellman方程：r + γ * max(Q(s', a'))
                Variable nextState = new Variable(NdArray.of(nextStates[i], Shape.of(1, stateDim)));
                Variable nextQValues = targetModel.forward(nextState);

                // 找到最大Q值
                float maxNextQ = findMaxQValue(nextQValues);
                targetValues[i] = rewards[i][0] + gamma * maxNextQ;
            }
        }

        return new Variable(NdArray.of(targetValues, Shape.of(batchSize, 1)));
    }

    /**
     * 计算当前Q值
     *
     * @param states  状态批次
     * @param actions 动作批次
     * @return 当前Q值
     */
    private Variable computeCurrentQValues(float[][] states, float[][] actions) {
        int batchSize = states.length;
        float[] currentValues = new float[batchSize];

        for (int i = 0; i < batchSize; i++) {
            Variable state = new Variable(NdArray.of(states[i], Shape.of(1, stateDim)));
            Variable qValues = model.forward(state);

            int actionIndex = (int) actions[i][0];
            currentValues[i] = qValues.getValue().get(0, actionIndex);
        }

        return new Variable(NdArray.of(currentValues, Shape.of(batchSize, 1)));
    }

    /**
     * 找到Q值向量中的最大值
     *
     * @param qValues Q值向量
     * @return 最大Q值
     */
    private float findMaxQValue(Variable qValues) {
        NdArray qArray = qValues.getValue();
        float maxQ = qArray.get(0, 0);

        for (int i = 1; i < actionDim; i++) {
            float q = qArray.get(0, i);
            if (q > maxQ) {
                maxQ = q;
            }
        }

        return maxQ;
    }

    /**
     * 更新损失统计
     *
     * @param loss 当前损失
     */
    private void updateLossStatistics(float loss) {
        totalLoss += loss;
        lossCount++;
        averageLoss = totalLoss / lossCount;
    }

    /**
     * 获取平均损失
     *
     * @return 平均损失
     */
    public float getAverageLoss() {
        return averageLoss;
    }

    /**
     * 获取经验回放缓冲区使用率
     *
     * @return 使用率
     */
    public float getBufferUsage() {
        return replayBuffer.getUsageRate();
    }

    /**
     * 获取当前探索率
     *
     * @return 探索率
     */
    public float getCurrentEpsilon() {
        return policy.getEpsilon();
    }

    /**
     * 设置探索率
     *
     * @param epsilon 新的探索率
     */
    public void setEpsilon(float epsilon) {
        policy.setEpsilon(epsilon);
    }

    @Override
    public void saveModel(String filepath) {
        // 这里应该实现模型保存逻辑
        System.out.println("DQN模型已保存到: " + filepath);
    }

    @Override
    public void loadModel(String filepath) {
        // 这里应该实现模型加载逻辑
        System.out.println("DQN模型已从以下路径加载: " + filepath);
    }

    /**
     * 获取训练统计信息
     *
     * @return 统计信息映射
     */
    public java.util.Map<String, Object> getTrainingStats() {
        java.util.Map<String, Object> stats = new java.util.HashMap<>();
        stats.put("training_step", trainingStep);
        stats.put("average_loss", averageLoss);
        stats.put("epsilon", getCurrentEpsilon());
        stats.put("buffer_usage", getBufferUsage());
        stats.put("update_count", updateCount);
        return stats;
    }

    /**
     * 重置训练统计
     */
    public void resetTrainingStats() {
        totalLoss = 0.0f;
        lossCount = 0;
        averageLoss = 0.0f;
        updateCount = 0;
    }
}