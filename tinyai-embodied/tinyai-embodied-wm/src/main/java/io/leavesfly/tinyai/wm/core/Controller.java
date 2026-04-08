package io.leavesfly.tinyai.wm.core;

import io.leavesfly.tinyai.wm.model.Action;
import io.leavesfly.tinyai.wm.model.WorldModelState;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.activation.Tanh;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * 控制器（Controller）
 * 基于世界模型的状态选择动作
 * 
 * 架构：
 * - 输入：潜在状态 z_t 和隐藏状态 h_t
 * - 输出：动作 a_t
 * - 可以是简单的线性映射，也可以是复杂的神经网络
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class Controller {
    
    /**
     * 潜在空间维度
     */
    private final int latentSize;
    
    /**
     * 隐藏状态维度
     */
    private final int hiddenSize;
    
    /**
     * 动作空间维度
     */
    private final int actionSize;
    
    /**
     * 策略网络
     */
    private final Sequential policyNetwork;
    
    /**
     * 是否使用确定性策略
     */
    private final boolean deterministic;
    
    /**
     * 构造函数
     *
     * @param latentSize 潜在空间维度
     * @param hiddenSize 隐藏状态维度
     * @param actionSize 动作空间维度
     * @param deterministic 是否使用确定性策略
     */
    public Controller(int latentSize, int hiddenSize, int actionSize, boolean deterministic) {
        this.latentSize = latentSize;
        this.hiddenSize = hiddenSize;
        this.actionSize = actionSize;
        this.deterministic = deterministic;
        
        // 构建策略网络：[z; h] -> action
        int inputSize = latentSize + hiddenSize;
        this.policyNetwork = new Sequential("policy")
            .add(new Linear("fc1", inputSize, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 32))
            .add(new ReLU())
            .add(new Linear("fc3", 32, actionSize))
            .add(new Tanh()); // 限制输出到 [-1, 1]
    }
    
    /**
     * 根据世界模型状态选择动作
     *
     * @param state 世界模型状态（包含潜在状态和隐藏状态）
     * @return 选择的动作
     */
    public Action selectAction(WorldModelState state) {
        // 1. 拼接输入：[z; h]
        NdArray input = concatenate(
            state.getLatentState().getZ(),
            state.getHiddenState().getH()
        );
        
        // 确保是二维数组 (1, features)
        if (input.getShape().getDimNum() == 1) {
            int size = input.getShape().size();
            input = NdArray.of(input.getArray(), Shape.of(1, size));
        }
        
        // 2. 通过策略网络
        Variable output = policyNetwork.forward(new Variable(input));
        NdArray actionVector = output.getValue();
        
        // 如果是二维，取第一行
        if (actionVector.getShape().getDimNum() == 2) {
            int size = actionVector.getShape().getDimension(1);
            float[] data = new float[size];
            for (int i = 0; i < size; i++) {
                data[i] = actionVector.get(0, i);
            }
            actionVector = NdArray.of(data, Shape.of(size));
        }
        
        // 3. 如果是随机策略，添加探索噪声
        if (!deterministic) {
            NdArray noise = NdArray.randn(actionVector.getShape()).mulNum(0.1f);
            actionVector = actionVector.add(noise).clip(-1.0f, 1.0f);
        }
        
        // 转换为 double[]
        float[] floatData = actionVector.getArray();
        double[] doubleData = new double[floatData.length];
        for (int i = 0; i < floatData.length; i++) {
            doubleData[i] = floatData[i];
        }
        
        return Action.createContinuous(doubleData);
    }
    
    /**
     * 评估动作的价值（可选，用于训练）
     *
     * @param state 世界模型状态
     * @param action 动作
     * @return 动作价值
     */
    public double evaluateAction(WorldModelState state, Action action) {
        // 简化实现：计算动作与策略输出的相似度
        Action predictedAction = selectAction(state);
        NdArray diff = action.getActionVector().sub(predictedAction.getActionVector());
        NdArray squared = diff.mul(diff);
        return -squared.sum().getNumber().doubleValue() / squared.getShape().size(); // 负的MSE作为相似度
    }
    
    /**
     * 使用CMA-ES等进化算法优化控制器参数
     * （这里提供接口，具体实现可在训练引擎中完成）
     *
     * @param fitness 适应度函数
     * @return 最优参数
     */
    public NdArray optimizeParameters(FitnessFunction fitness) {
        // TODO: 实现CMA-ES或其他进化算法
        // 这里先返回当前参数
        return getCurrentParameters();
    }
    
    /**
     * 获取当前参数
     */
    private NdArray getCurrentParameters() {
        // 简化实现：返回零向量
        return NdArray.zeros(Shape.of(latentSize + hiddenSize));
    }
    

    
    // 辅助方法：拼接两个NdArray
    private static NdArray concatenate(NdArray a, NdArray b) {
        float[] aData = a.getArray();
        float[] bData = b.getArray();
        float[] result = new float[aData.length + bData.length];
        System.arraycopy(aData, 0, result, 0, aData.length);
        System.arraycopy(bData, 0, result, aData.length, bData.length);
        return NdArray.of(result, Shape.of(result.length));
    }
    
    // Getters
    public int getLatentSize() {
        return latentSize;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public int getActionSize() {
        return actionSize;
    }
    
    public boolean isDeterministic() {
        return deterministic;
    }
    
    /**
     * 适应度函数接口
     */
    @FunctionalInterface
    public interface FitnessFunction {
        /**
         * 评估给定参数的适应度
         *
         * @param parameters 控制器参数
         * @return 适应度值（越高越好）
         */
        double evaluate(NdArray parameters);
    }
}
