package io.leavesfly.tinyai.rl.util;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;

import java.util.Map;

/**
 * 模型操作工具类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【职责】
 * 提供强化学习中常用的神经网络模型操作方法,包括:
 * - 权重复制: 目标网络同步
 * - MLP创建: 标准多层感知机构建
 * - 模型序列化: 保存和加载辅助
 * 
 * 【设计价值】
 * - 消除重复: DQN、DoubleDQN、REINFORCE中的相同网络创建代码
 * - 统一风格: 所有Agent使用相同的网络构建方式
 * - 易于维护: 网络结构修改只需改一处
 * 
 * 【消除的重复代码】
 * - DQNAgent: L103-121 (createQNetwork), L129-133 (copyModelWeights)
 * - DoubleDQNAgent: L97-111 (createQNetwork), L116-119 (copyModelWeights)  
 * - REINFORCEAgent: L103-121 (createPolicyNetwork), L130-148 (createBaselineNetwork)
 */
public class ModelUtil {
    
    /**
     * 私有构造函数 - 工具类不允许实例化
     */
    private ModelUtil() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
    
    /**
     * 复制模型权重
     * 
     * 【目标网络同步】
     * DQN及其变体算法使用目标网络(Target Network)稳定训练。
     * 目标网络的权重定期从在线网络复制,而不是每步更新。
     * 
     * 【算法应用】
     * - DQN: 每N步复制一次
     * - DoubleDQN: 解耦动作选择和评估
     * - DuelingDQN: 分离优势函数和价值函数
     * 
     * 【实现原理】
     * 使用TinyAI的stateDict机制:
     * 1. source.copyStateDict(): 获取源模型的所有参数
     * 2. target.loadStateDict(..., strict=true): 严格加载到目标模型
     * 
     * @param source 源模型(通常是在线网络)
     * @param target 目标模型(通常是目标网络)
     */
    public static void copyWeights(Model source, Model target) {
        if (source == null || target == null) {
            throw new IllegalArgumentException("Source and target models cannot be null");
        }
        
        // 使用 stateDict 和 loadStateDict 实现权重复制
        Map<String, NdArray> stateDict = source.getModule().copyStateDict();
        target.getModule().loadStateDict(stateDict, true);
    }
    
    /**
     * 创建标准MLP(多层感知机)网络
     * 
     * 【网络结构】
     * Input → [Linear → ReLU] × N → Linear(Output)
     * 
     * 【参数说明】
     * - layers: 层尺寸数组,如 [stateDim, 128, 64, actionDim]
     *   表示输入层stateDim → 隐藏层128 → 隐藏层64 → 输出层actionDim
     * - outputActivation: 输出层是否添加激活函数
     *   - false: Q网络(输出原始Q值)
     *   - true: 策略网络可能需要Softmax(但通常在forward时添加)
     * 
     * 【应用场景】
     * - DQN Q网络: createMLP("DQN_Q", [4,128,128,2], false)
     * - REINFORCE策略: createMLP("Policy", [4,64,64,2], false)
     * - REINFORCE基线: createMLP("Baseline", [4,64,64,1], false)
     * 
     * @param name 模型名称
     * @param layers 层尺寸数组(包含输入和输出)
     * @param outputActivation 输出层是否激活
     * @return 创建的模型
     */
    public static Model createMLP(String name, int[] layers, boolean outputActivation) {
        if (layers == null || layers.length < 2) {
            throw new IllegalArgumentException("Layers array must have at least 2 elements (input and output)");
        }
        
        Sequential mlpModule = new Sequential(name + "_Module");
        
        // 添加隐藏层 (从 layers[0] 到 layers[length-2])
        for (int i = 0; i < layers.length - 1; i++) {
            int inputSize = layers[i];
            int outputSize = layers[i + 1];
            boolean isLastLayer = (i == layers.length - 2);
            
            // 添加全连接层
            mlpModule.add(new Linear("fc_" + i, inputSize, outputSize, true));
            
            // 隐藏层添加ReLU,输出层根据参数决定
            if (!isLastLayer || outputActivation) {
                mlpModule.add(new ReLU("relu_" + i));
            }
        }
        
        return new Model(name, mlpModule);
    }
    
    /**
     * 创建Q网络(值函数网络)
     * 
     * 【便捷方法】
     * 为DQN系列算法提供的专用方法,自动处理输出层不激活。
     * 
     * @param name Q网络名称
     * @param stateDim 状态维度
     * @param actionDim 动作维度
     * @param hiddenSizes 隐藏层尺寸数组
     * @return Q网络模型
     */
    public static Model createQNetwork(String name, int stateDim, int actionDim, int[] hiddenSizes) {
        // 构建完整的层尺寸数组: [stateDim, hidden1, hidden2, ..., actionDim]
        int[] layers = new int[hiddenSizes.length + 2];
        layers[0] = stateDim;
        System.arraycopy(hiddenSizes, 0, layers, 1, hiddenSizes.length);
        layers[layers.length - 1] = actionDim;
        
        // Q网络输出层不需要激活函数(输出原始Q值)
        return createMLP(name, layers, false);
    }
    
    /**
     * 创建策略网络(Policy Network)
     * 
     * 【便捷方法】
     * 为REINFORCE、PPO等策略梯度算法提供的专用方法。
     * 输出层不添加激活函数,因为Softmax通常在forward时动态添加。
     * 
     * 【网络结构】
     * Input(stateDim) → Hidden → ... → Output(actionDim)
     * 输出: logits(未归一化的对数概率)
     * 
     * @param name 策略网络名称
     * @param stateDim 状态维度
     * @param actionDim 动作维度
     * @param hiddenSizes 隐藏层尺寸数组
     * @return 策略网络模型
     */
    public static Model createPolicyNetwork(String name, int stateDim, int actionDim, int[] hiddenSizes) {
        int[] layers = new int[hiddenSizes.length + 2];
        layers[0] = stateDim;
        System.arraycopy(hiddenSizes, 0, layers, 1, hiddenSizes.length);
        layers[layers.length - 1] = actionDim;
        
        // 策略网络输出logits,不激活(Softmax在使用时添加)
        return createMLP(name, layers, false);
    }
    
    /**
     * 创建价值网络(Value Network)
     * 
     * 【便捷方法】
     * 为Actor-Critic算法中的价值函数V(s)或基线函数提供的专用方法。
     * 输出一个标量,表示状态价值。
     * 
     * 【网络结构】
     * Input(stateDim) → Hidden → ... → Output(1)
     * 输出: 状态价值 V(s)
     * 
     * 【应用场景】
     * - REINFORCE基线: 减少方差
     * - A2C/PPO的Critic: 价值函数估计
     * - TD3的Q网络: 连续动作空间
     * 
     * @param name 价值网络名称
     * @param stateDim 状态维度
     * @param hiddenSizes 隐藏层尺寸数组
     * @return 价值网络模型
     */
    public static Model createValueNetwork(String name, int stateDim, int[] hiddenSizes) {
        int[] layers = new int[hiddenSizes.length + 2];
        layers[0] = stateDim;
        System.arraycopy(hiddenSizes, 0, layers, 1, hiddenSizes.length);
        layers[layers.length - 1] = 1;  // 输出单个标量值
        
        // 价值网络输出原始数值,不激活
        return createMLP(name, layers, false);
    }
    
    /**
     * 软更新目标网络(Soft Update)
     * 
     * 【Soft Update vs Hard Update】
     * - Hard Update: 每N步完全复制权重 θ_target = θ_online
     * - Soft Update: 每步缓慢更新 θ_target = τ*θ_online + (1-τ)*θ_target
     * 
     * 【算法应用】
     * - DDPG/TD3: 使用soft update (τ=0.001~0.005)
     * - SAC: 使用soft update
     * - DQN: 通常使用hard update
     * 
     * @param source 源模型
     * @param target 目标模型
     * @param tau 更新系数(0~1),越小更新越慢
     */
    public static void softUpdateWeights(Model source, Model target, float tau) {
        if (tau < 0.0f || tau > 1.0f) {
            throw new IllegalArgumentException("Tau must be in range [0, 1]");
        }
        
        Map<String, NdArray> sourceDict = source.getModule().copyStateDict();
        Map<String, NdArray> targetDict = target.getModule().copyStateDict();
        
        // θ_target = τ*θ_source + (1-τ)*θ_target
        for (String key : sourceDict.keySet()) {
            NdArray sourceParam = sourceDict.get(key);
            NdArray targetParam = targetDict.get(key);
            
            // 计算加权平均
            NdArray tauArray = NdArray.of(tau);
            NdArray oneMinusTau = NdArray.of(1.0f - tau);
            NdArray updated = sourceParam.mul(tauArray).add(targetParam.mul(oneMinusTau));
            targetDict.put(key, updated);
        }
        
        target.getModule().loadStateDict(targetDict, true);
    }
}
