package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
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
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

import java.util.Map;

/**
 * Double DQN (Double Deep Q-Network) 智能体实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * DoubleDQNAgent实现了Double DQN算法，这是对标准DQN的重要改进。
 * 主要特点包括：
 * 1. 使用两个Q网络解耦动作选择和动作评估
 * 2. 减少Q值过估计问题
 * 3. 训练更稳定，收敛性更好
 * 4. 继承自DQN，保持经验回放和目标网络
 */
public class DoubleDQNAgent extends Agent {

    // Double DQN特有参数
    private final int batchSize;
    private final int targetUpdateFreq;
    private final ReplayBuffer replayBuffer;
    private final Model targetModel;
    private final EpsilonGreedyPolicy policy;
    private final Optimizer optimizer;
    private final Loss lossFunction;

    // 训练统计
    private int updateCount;
    private float averageLoss;
    private float totalLoss;
    private int lossCount;

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
    public DoubleDQNAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
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

        // 创建Q网络（在线网络）
        this.model = createQNetwork(stateDim, actionDim, hiddenSizes);

        // 创建目标网络
        this.targetModel = createQNetwork(stateDim, actionDim, hiddenSizes);
        copyModelWeights(model, targetModel);

        // 创建ε-贪婪策略
        this.policy = new EpsilonGreedyPolicy(stateDim, actionDim, epsilon,
                state -> model.forward(state));

        // 创建优化器和损失函数
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-3f);
        this.lossFunction = new MeanSquaredLoss();
    }

    /**
     * 创建Q网络
     */
    private Model createQNetwork(int stateDim, int actionDim, int[] hiddenSizes) {
        Sequential mlpModule = new Sequential(name + "_QNetwork");

        int inputSize = stateDim;

        for (int hiddenSize : hiddenSizes) {
            mlpModule.add(new Linear("fc", inputSize, hiddenSize, true));
            mlpModule.add(new ReLU("relu"));
            inputSize = hiddenSize;
        }

        mlpModule.add(new Linear("fc_out", inputSize, actionDim, true));

        return new Model(name + "_QModel", mlpModule);
    }

    /**
     * 复制模型权重
     */
    private void copyModelWeights(Model source, Model target) {
        Map<String, NdArray> stateDict = source.getModule().copyStateDict();
        target.getModule().loadStateDict(stateDict, true);
    }

    @Override
    public Variable selectAction(Variable state) {
        if (training) {
            return policy.selectAction(state);
        } else {
            Variable qValues = model.forward(state);
            return selectGreedyAction(qValues);
        }
    }

    /**
     * 选择贪婪动作
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
        storeExperience(experience);

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

        for (int i = 0; i < experiences.length; i++) {
            Experience exp = experiences[i];

            NdArray stateArray = exp.getState().getValue();
            for (int j = 0; j < stateDim; j++) {
                states[i][j] = stateArray.get(0, j);
            }

            actions[i][0] = exp.getAction().getValue().getNumber().floatValue();
            rewards[i][0] = exp.getReward();

            NdArray nextStateArray = exp.getNextState().getValue();
            for (int j = 0; j < stateDim; j++) {
                nextStates[i][j] = nextStateArray.get(0, j);
            }

            dones[i] = exp.isDone();
        }

        // Double DQN: 使用在线网络选择动作，目标网络评估动作
        Variable targetQValues = computeTargetQValuesDoubleDQN(nextStates, rewards, dones);
        Variable currentQValues = computeCurrentQValues(states, actions);

        Variable loss = lossFunction.loss(targetQValues, currentQValues);

        model.clearGrads();
        loss.backward();
        optimizer.update();

        updateLossStatistics(loss.getValue().getNumber().floatValue());
        incrementTrainingStep();

        // 定期更新目标网络
        if (trainingStep % targetUpdateFreq == 0) {
            copyModelWeights(model, targetModel);
        }

        policy.decayEpsilon(0.995f, 0.01f);
    }

    /**
     * Double DQN 目标Q值计算
     * 使用在线网络选择动作，目标网络评估动作
     */
    private Variable computeTargetQValuesDoubleDQN(float[][] nextStates, float[][] rewards, boolean[] dones) {
        int batchSize = nextStates.length;

        Variable[] targetArray = new Variable[batchSize];

        for (int i = 0; i < batchSize; i++) {
            Variable nextState = new Variable(NdArray.of(nextStates[i], Shape.of(1, stateDim)));

            if (dones[i]) {
                targetArray[i] = new Variable(NdArray.of(rewards[i][0]));
            } else {
                // Double DQN核心：在线网络选择动作
                Variable onlineQValues = model.forward(nextState);
                int bestAction = selectBestAction(onlineQValues);

                // 目标网络评估动作
                Variable targetQValues = targetModel.forward(nextState);
                Variable indexVar = new Variable(NdArray.of(new float[]{bestAction}));
                Variable targetQForBestAction = targetQValues.indexSelect(1, indexVar);

                // 计算目标值: r + γ * Q_target(s', a_best_online)
                Variable rewardVar = new Variable(NdArray.of(rewards[i][0]));
                Variable gammaVar = new Variable(NdArray.of(gamma));
                Variable discountedQ = targetQForBestAction.mul(gammaVar);
                targetArray[i] = rewardVar.add(discountedQ);
            }
        }

        return stackVariables(targetArray, batchSize);
    }

    /**
     * 选择Q值最大的动作
     */
    private int selectBestAction(Variable qValues) {
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

        return bestAction;
    }

    /**
     * 计算当前Q值
     */
    private Variable computeCurrentQValues(float[][] states, float[][] actions) {
        int batchSize = states.length;
        Variable[] currentQArray = new Variable[batchSize];

        for (int i = 0; i < batchSize; i++) {
            Variable state = new Variable(NdArray.of(states[i], Shape.of(1, stateDim)));
            Variable qValues = model.forward(state);

            int actionIndex = (int) actions[i][0];
            Variable indexVar = new Variable(NdArray.of(new float[]{actionIndex}));
            currentQArray[i] = qValues.indexSelect(1, indexVar);
        }

        return stackVariables(currentQArray, batchSize);
    }

    /**
     * 将Variable数组堆叠成批次Variable
     */
    private Variable stackVariables(Variable[] variables, int batchSize) {
        float[] values = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            values[i] = variables[i].getValue().getNumber().floatValue();
        }
        return new Variable(NdArray.of(values, Shape.of(batchSize, 1)));
    }

    /**
     * 更新损失统计
     */
    private void updateLossStatistics(float loss) {
        totalLoss += loss;
        lossCount++;
        averageLoss = totalLoss / lossCount;
    }

    public float getAverageLoss() {
        return averageLoss;
    }

    public float getBufferUsage() {
        return replayBuffer.getUsageRate();
    }

    public float getCurrentEpsilon() {
        return policy.getEpsilon();
    }

    public void setEpsilon(float epsilon) {
        policy.setEpsilon(epsilon);
    }

    @Override
    public void saveModel(String filepath) {
        model.saveModel(filepath);
        System.out.println("DoubleDQN在线网络模型已保存到: " + filepath);

        String targetFilepath = filepath.replace(".model", "_target.model");
        targetModel.saveModel(targetFilepath);
        System.out.println("DoubleDQN目标网络模型已保存到: " + targetFilepath);
    }

    @Override
    public void loadModel(String filepath) {
        Model loadedModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(filepath);
        this.model.getModule().loadStateDict(loadedModel.getModule().copyStateDict(), true);
        System.out.println("DoubleDQN在线网络模型已从以下路径加载: " + filepath);

        String targetFilepath = filepath.replace(".model", "_target.model");
        Model loadedTargetModel = io.leavesfly.tinyai.ml.model.ModelSerializer.loadModel(targetFilepath);
        this.targetModel.getModule().loadStateDict(loadedTargetModel.getModule().copyStateDict(), true);
        System.out.println("DoubleDQN目标网络模型已从以下路径加载: " + targetFilepath);
    }

    public Map<String, Object> getTrainingStats() {
        Map<String, Object> stats = new java.util.HashMap<>();
        stats.put("training_step", trainingStep);
        stats.put("average_loss", averageLoss);
        stats.put("epsilon", getCurrentEpsilon());
        stats.put("buffer_usage", getBufferUsage());
        stats.put("update_count", updateCount);
        return stats;
    }

    public void resetTrainingStats() {
        totalLoss = 0.0f;
        lossCount = 0;
        averageLoss = 0.0f;
        updateCount = 0;
    }
}
