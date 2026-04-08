package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.Map;

/**
 * Expert Network - 专家网络
 * 
 * MoE中的单个专家,采用标准FFN结构:
 * input → Linear(W1) → ReLU → Linear(W2) → output
 * 
 * 架构特点:
 * - 两层全连接网络
 * - ReLU激活函数
 * - 每个专家独立参数
 * - 支持不同隐藏层维度
 * 
 * @author leavesfly
 * @since 2024
 */
public class ExpertNetwork extends Module {
    
    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    private final int expertId;
    
    private final Linear fc1;      // 第一层: input_dim -> hidden_dim
    private final ReLU activation; // 激活函数
    private final Linear fc2;      // 第二层: hidden_dim -> output_dim
    
    /**
     * 构造函数
     * 
     * @param expertId 专家ID
     * @param inputDim 输入维度
     * @param hiddenDim 隐藏层维度
     * @param outputDim 输出维度
     */
    public ExpertNetwork(int expertId, int inputDim, int hiddenDim, int outputDim) {
        super("expert_" + expertId);
        this.expertId = expertId;
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        
        // 创建层
        this.fc1 = new Linear("expert_" + expertId + "_fc1", inputDim, hiddenDim, true);
        this.activation = new ReLU("expert_" + expertId + "_relu");
        this.fc2 = new Linear("expert_" + expertId + "_fc2", hiddenDim, outputDim, true);
        
        // 注册子模块
        registerModule("fc1", fc1);
        registerModule("activation", activation);
        registerModule("fc2", fc2);
    }
    
    /**
     * 前向传播(Variable版本,Module接口要求)
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardVar(inputs[0]);
    }
    
    /**
     * 前向传播(内部调用)
     */
    public Variable forwardVar(Variable input) {
        return fc2.forward(activation.forward(fc1.forward(input)));
    }
    
    /**
     * 前向传播(NdArray接口，仅用于推理)
     * <p>
     * 注意：此方法通过 new Variable 包装输入，不保持上游计算图连通。
     * 训练时应使用 forwardVar(Variable) 方法。
     * 
     * @param inputs 输入数组
     * @return 输出NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        Variable input = new Variable(inputs[0]);
        input.setRequireGrad(false);
        return forwardVar(input).getValue();
    }
    
    /**
     * 获取专家ID
     */
    public int getExpertId() {
        return expertId;
    }
    
    /**
     * 获取输入维度
     */
    public int getInputDim() {
        return inputDim;
    }
    
    /**
     * 获取隐藏层维度
     */
    public int getHiddenDim() {
        return hiddenDim;
    }
    
    /**
     * 获取输出维度
     */
    public int getOutputDim() {
        return outputDim;
    }
    
    /**
     * 获取参数数量
     */
    public int getParameterCount() {
        // fc1: (input_dim + 1) * hidden_dim
        // fc2: (hidden_dim + 1) * output_dim
        return (inputDim + 1) * hiddenDim + (hiddenDim + 1) * outputDim;
    }
    
    /**
     * 克隆专家网络(用于参数共享)
     */
    public ExpertNetwork clone(int newExpertId) {
        ExpertNetwork cloned = new ExpertNetwork(newExpertId, inputDim, hiddenDim, outputDim);
        
        // 复制参数(深拷贝：创建独立的权重副本)
        Map<String, Parameter> params = this.namedParameters();
        Map<String, Parameter> clonedParams = cloned.namedParameters();
        
        for (String key : params.keySet()) {
            Parameter param = params.get(key);
            String newKey = key.replace("expert_" + expertId, "expert_" + newExpertId);
            
            if (clonedParams.containsKey(newKey)) {
                // 深拷贝：复制参数值到新的 NdArray
                NdArray srcData = param.data();
                NdArray dstData = clonedParams.get(newKey).data();
                
                float[] srcBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) srcData).buffer;
                float[] dstBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) dstData).buffer;
                
                // 确保目标缓冲区有足够空间
                int copyLength = Math.min(srcBuffer.length, dstBuffer.length);
                System.arraycopy(srcBuffer, 0, dstBuffer, 0, copyLength);
            }
        }
        
        return cloned;
    }
    
    @Override
    public String toString() {
        return String.format("ExpertNetwork(id=%d, in=%d, hidden=%d, out=%d, params=%d)",
            expertId, inputDim, hiddenDim, outputDim, getParameterCount());
    }
}