package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * Value Network (Critic) - 价值网络
 * 
 * 用于估计状态价值V(s),在PPO算法中扮演Critic角色
 * 
 * 架构:
 * - 输入: 隐藏状态 [batch_size, hidden_dim]
 * - Linear层1: hidden_dim -> hidden_dim
 * - Activation: Tanh
 * - Linear层2: hidden_dim -> 1
 * - 输出: 价值估计 [batch_size, 1]
 * 
 * @author leavesfly
 * @since 2024
 */
public class ValueNetwork extends Module {
    
    private final int hiddenDim;
    private final Linear linear1;
    private final Linear linear2;
    
    private final Map<String, Parameter> parameters;
    
    /**
     * 构造函数
     * 
     * @param hiddenDim 隐藏层维度(应与模型输出维度一致)
     */
    public ValueNetwork(int hiddenDim) {
        super("ValueNetwork");
        this.hiddenDim = hiddenDim;
        
        // 第一层: hidden_dim -> hidden_dim
        this.linear1 = (Linear) registerModule("linear1", new Linear("value_linear1", hiddenDim, hiddenDim, true));
        
        // 第二层: hidden_dim -> 1 (输出价值估计)
        this.linear2 = (Linear) registerModule("linear2", new Linear("value_linear2", hiddenDim, 1, true));
        
        // 收集参数
        this.parameters = new HashMap<>();
        collectParameters();
        
        // 初始化参数
        initializeParameters();
    }
    
    /**
     * 前向传播 - 实现Module抽象方法
     * 
     * @param inputs 输入变量数组
     * @return 输出变量
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("ValueNetwork requires at least one input");
        }
        return forward(inputs[0]);
    }
    
    /**
     * 前向传播
     * 
     * @param hidden 隐藏状态 [batch_size, hidden_dim]
     * @return 价值估计 [batch_size, 1]
     */
    public Variable forward(Variable hidden) {
        // Layer 1 + Tanh激活
        Variable h1 = linear1.forward(hidden);
        Variable activated = h1.tanh();  // 使用 Variable.tanh() 算子
        
        // Layer 2
        Variable value = linear2.forward(activated);
        
        return value;
    }
    
    /**
     * 批量前向传播(处理序列)
     * 
     * @param hiddenStates 隐藏状态序列 [batch_size, seq_len, hidden_dim]
     * @return 价值估计序列 [batch_size, seq_len, 1]
     */
    public Variable forwardSequence(Variable hiddenStates) {
        int[] shape = hiddenStates.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int seqLen = shape[1];
        int dim = shape[2];

        // 使用Variable.reshape()保持计算图连通,梯度可以正确回传
        Variable reshaped = hiddenStates.reshape(Shape.of(batchSize * seqLen, dim));

        // 前向传播，得到 [batch_size*seq_len, 1]
        Variable values = forward(reshaped);

        // 使用Variable.reshape()恢复原始batch维度,保持计算图连通
        return values.reshape(Shape.of(batchSize, seqLen, 1));
    }
    
    /**
     * 估计单个状态的价值
     * 
     * @param hidden 隐藏状态 [hidden_dim]
     * @return 价值标量
     */
    public float estimateValue(Variable hidden) {
        Variable value = forward(hidden);
        return value.getValue().getNumber().floatValue();
    }
    
    // 已删除 tanh 方法，改用 Variable.tanh() 算子
    
    /**
     * Reshape操作(简化实现)
     */
    private Variable reshape(Variable x, int[] newShape) {
        // 简化:返回原Variable
        return x;
    }
    
    /**
     * 收集参数
     */
    private void collectParameters() {
        // Linear1参数
        Map<String, Parameter> linear1Params = linear1.namedParameters();
        for (Map.Entry<String, Parameter> entry : linear1Params.entrySet()) {
            parameters.put("linear1." + entry.getKey(), entry.getValue());
        }
        
        // Linear2参数
        Map<String, Parameter> linear2Params = linear2.namedParameters();
        for (Map.Entry<String, Parameter> entry : linear2Params.entrySet()) {
            parameters.put("linear2." + entry.getKey(), entry.getValue());
        }
    }
    
    /**
     * 初始化参数(正交初始化)
     */
    private void initializeParameters() {
        // 使用小的随机初始化
        for (Parameter param : parameters.values()) {
            NdArray data = param.data();
            float[] buffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) data).buffer;
            
            float scale = (float) Math.sqrt(2.0 / hiddenDim);
            for (int i = 0; i < buffer.length; i++) {
                buffer[i] = (float) ((Math.random() - 0.5) * 2 * scale);
            }
        }
    }
    
    /**
     * 清空梯度
     */
    public void clearGrads() {
        for (Parameter param : parameters.values()) {
            param.clearGrad();
        }
    }
    
    /**
     * 设置训练模式
     */
    public void setTraining(boolean training) {
        // ValueNetwork通常不需要特殊的训练模式
    }
    
    /**
     * 获取所有参数(兼容Module接口)
     */
    public Map<String, Parameter> getAllParams() {
        return new HashMap<>(parameters);
    }
    
    /**
     * 获取隐藏层维度
     */
    public int getHiddenDim() {
        return hiddenDim;
    }
    
    /**
     * 保存模型状态(简化实现)
     */
    public Map<String, float[]> getState() {
        Map<String, float[]> state = new HashMap<>();
        
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            NdArray data = entry.getValue().data();
            float[] buffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) data).buffer;
            state.put(entry.getKey(), buffer.clone());
        }
        
        return state;
    }
    
    /**
     * 加载模型状态(简化实现)
     */
    public void loadState(Map<String, float[]> state) {
        for (Map.Entry<String, float[]> entry : state.entrySet()) {
            Parameter param = parameters.get(entry.getKey());
            if (param != null) {
                NdArray data = param.data();
                float[] buffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) data).buffer;
                System.arraycopy(entry.getValue(), 0, buffer, 0, Math.min(buffer.length, entry.getValue().length));
            }
        }
    }
    
    @Override
    public String toString() {
        return String.format("ValueNetwork(hiddenDim=%d, params=%d)", 
            hiddenDim, parameters.size());
    }
}
