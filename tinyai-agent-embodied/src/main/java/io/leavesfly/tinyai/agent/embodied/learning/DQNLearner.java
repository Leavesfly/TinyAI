package io.leavesfly.tinyai.agent.embodied.learning;

import io.leavesfly.tinyai.agent.embodied.memory.EpisodicMemory;
import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;
import io.leavesfly.tinyai.agent.embodied.model.Transition;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;
import java.util.Random;

/**
 * DQN强化学习策略
 * 基于深度Q网络的强化学习算法
 *
 * @author TinyAI Team
 */
public class DQNLearner {
    private static final int STATE_DIM = 10;  // 状态维度
    private static final int ACTION_DIM = 3;  // 动作维度（转向、油门、刹车）
    
    private NdArray qNetwork;      // Q网络参数（简化版）
    private NdArray targetNetwork; // 目标网络参数
    
    private double learningRate;
    private double gamma;          // 折扣因子
    private double epsilon;        // 探索率
    private int batchSize;
    private int updateFrequency;   // 目标网络更新频率
    private int stepCount;
    
    private Random random;

    public DQNLearner() {
        this.learningRate = 0.001;
        this.gamma = 0.99;
        this.epsilon = 0.1;
        this.batchSize = 32;
        this.updateFrequency = 100;
        this.stepCount = 0;
        this.random = new Random();
        
        // 初始化网络（简化版，仅用于演示架构）
        initializeNetworks();
    }

    private void initializeNetworks() {
        // 简化的网络初始化（实际应使用神经网络）
        float[] weights = new float[STATE_DIM * ACTION_DIM];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) (random.nextGaussian() * 0.01);
        }
        qNetwork = NdArray.of(weights);
        // 复制权重
        float[] targetWeights = new float[weights.length];
        System.arraycopy(weights, 0, targetWeights, 0, weights.length);
        targetNetwork = NdArray.of(targetWeights);
    }

    /**
     * 从经验中学习
     */
    public void learn(EpisodicMemory memory) {
        if (memory.getBufferSize() < batchSize) {
            return;
        }

        // 采样一批转移
        List<Transition> batch = memory.sampleBatch(batchSize);
        
        // 计算TD误差并更新Q网络
        double totalLoss = 0.0;
        for (Transition transition : batch) {
            double loss = updateQValue(transition);
            totalLoss += loss;
        }
        
        // 定期更新目标网络
        stepCount++;
        if (stepCount % updateFrequency == 0) {
            updateTargetNetwork();
        }
        
        // 衰减探索率
        epsilon = Math.max(0.01, epsilon * 0.995);
    }

    /**
     * 更新Q值
     */
    private double updateQValue(Transition transition) {
        // 简化的Q学习更新（实际应使用神经网络前向传播）
        PerceptionState state = transition.getState();
        double reward = transition.getReward();
        PerceptionState nextState = transition.getNextState();
        boolean done = transition.isDone();
        
        // 计算当前Q值（简化版）
        double currentQ = estimateQ(state);
        
        // 计算目标Q值
        double targetQ;
        if (done) {
            targetQ = reward;
        } else {
            double maxNextQ = getMaxQ(nextState);
            targetQ = reward + gamma * maxNextQ;
        }
        
        // TD误差
        double tdError = targetQ - currentQ;
        
        // 梯度更新（简化版）
        // 实际应使用反向传播更新网络参数
        
        return Math.abs(tdError);
    }

    /**
     * 估计状态的Q值（简化版）
     */
    private double estimateQ(PerceptionState state) {
        // 简化的Q值估计
        float[] stateFeatures = extractStateFeatures(state);
        
        double q = 0.0;
        int weightSize = qNetwork.getShape().size();
        for (int i = 0; i < STATE_DIM && i < weightSize; i++) {
            q += stateFeatures[i] * qNetwork.get(i);
        }
        
        return q;
    }

    /**
     * 获取状态的最大Q值
     */
    private double getMaxQ(PerceptionState state) {
        return estimateQ(state) * 1.1; // 简化实现
    }

    /**
     * 选择动作（ε-贪心策略）
     */
    public DrivingAction selectAction(PerceptionState state) {
        if (random.nextDouble() < epsilon) {
            // 探索：随机动作
            return randomAction();
        } else {
            // 利用：选择最大Q值的动作
            return greedyAction();
        }
    }

    /**
     * 贪心动作选择
     */
    private DrivingAction greedyAction() {
        // 简化实现：返回保持速度
        return new DrivingAction(0.0, 0.3, 0.0);
    }

    /**
     * 随机动作
     */
    private DrivingAction randomAction() {
        double steering = random.nextDouble() * 2.0 - 1.0;  // [-1, 1]
        double throttle = random.nextDouble();              // [0, 1]
        double brake = random.nextDouble() * 0.3;           // [0, 0.3]
        return new DrivingAction(steering, throttle, brake);
    }

    /**
     * 更新目标网络
     */
    private void updateTargetNetwork() {
        // 复制网络参数
        Shape shape = qNetwork.getShape();
        float[] weights = new float[shape.size()];
        for (int i = 0; i < shape.size(); i++) {
            weights[i] = qNetwork.get(i);
        }
        targetNetwork = NdArray.of(weights);
    }

    /**
     * 提取状态特征（简化版）
     */
    private float[] extractStateFeatures(PerceptionState state) {
        float[] features = new float[STATE_DIM];
        
        // 提取基础特征
        features[0] = (float) state.getVehicleState().getSpeed();
        features[1] = (float) state.getVehicleState().getHeading();
        features[2] = (float) state.getVehicleState().getAcceleration();
        
        if (state.getLaneInfo() != null) {
            features[3] = (float) state.getLaneInfo().getLateralDeviation();
            features[4] = 0.0f; // 简化：没有headingError，使用curvature
        }
        
        // 障碍物特征
        if (state.getObstacleMap() != null && !state.getObstacleMap().isEmpty()) {
            features[5] = (float) state.getObstacleMap().get(0).getDistance();
            features[6] = (float) state.getObstacleMap().get(0).getVelocity().magnitude();
        }
        
        // 其他特征填充
        for (int i = 7; i < STATE_DIM; i++) {
            features[i] = 0.0f;
        }
        
        return features;
    }

    /**
     * 获取动作索引（离散化）
     */
    private int getActionIndex(DrivingAction action) {
        // 简化的动作离散化
        if (action.getBrake() > 0.5) {
            return 0; // 刹车
        } else if (action.getThrottle() > 0.5) {
            return 1; // 加速
        } else {
            return 2; // 保持
        }
    }

    // Getters and Setters
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public int getStepCount() {
        return stepCount;
    }
}
