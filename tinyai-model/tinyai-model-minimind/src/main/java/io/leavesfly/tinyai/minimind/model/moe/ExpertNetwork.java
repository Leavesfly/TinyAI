package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.activation.SiLU;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.Map;

/**
 * Expert Network - 专家网络（对标 Python FeedForward）
 * 
 * MoE中的单个专家,采用 SwiGLU 结构：
 * output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
 * 
 * 架构特点:
 * - 三层线性层（gate_proj, up_proj, down_proj）
 * - SiLU 激活函数 + 门控机制
 * - 无 bias（对标 Python）
 * - 每个专家独立参数
 * 
 * @author leavesfly
 * @since 2024
 */
public class ExpertNetwork extends Module {
    
    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    private final int expertId;
    
    private final Linear gateProj;   // 门控投影: inputDim -> hiddenDim
    private final SiLU silu;         // SiLU 激活函数
    private final Linear upProj;     // 上投影: inputDim -> hiddenDim
    private final Linear downProj;   // 下投影: hiddenDim -> outputDim
    
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
        
        // SwiGLU 结构（对标 Python FeedForward，无 bias）
        this.gateProj = new Linear("expert_" + expertId + "_gate_proj", inputDim, hiddenDim, false);
        this.silu = new SiLU("expert_" + expertId + "_silu");
        this.upProj = new Linear("expert_" + expertId + "_up_proj", inputDim, hiddenDim, false);
        this.downProj = new Linear("expert_" + expertId + "_down_proj", hiddenDim, outputDim, false);
        
        // 注册子模块
        registerModule("gate_proj", gateProj);
        registerModule("silu", silu);
        registerModule("up_proj", upProj);
        registerModule("down_proj", downProj);
    }
    
    /**
     * 前向传播(Variable版本,Module接口要求)
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardVar(inputs[0]);
    }
    
    /**
     * 前向传播(内部调用) - SwiGLU
     * output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
     */
    public Variable forwardVar(Variable input) {
        Variable gate = silu.forward(gateProj.forward(input));
        Variable up = upProj.forward(input);
        Variable hidden = gate.mul(up);
        return downProj.forward(hidden);
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
     * 获取参数数量（SwiGLU: 3 个无 bias 的 Linear）
     */
    public int getParameterCount() {
        // gate_proj: inputDim * hiddenDim
        // up_proj: inputDim * hiddenDim  
        // down_proj: hiddenDim * outputDim
        return inputDim * hiddenDim * 2 + hiddenDim * outputDim;
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