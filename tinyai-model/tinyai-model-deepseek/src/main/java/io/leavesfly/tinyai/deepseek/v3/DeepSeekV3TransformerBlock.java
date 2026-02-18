package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.MultiHeadAttention;

/**
 * DeepSeek-V3 Transformer块（Pre-LayerNorm + MoE架构）
 * 
 * 采用Pre-LN架构并集成混合专家层，实现参数高效和任务专门化：
 * 1. 多头自注意力层（带因果掩码）
 * 2. 混合专家层(MoE)替代传统FFN
 * 
 * 架构特点：
 * - Pre-LayerNorm提升训练稳定性
 * - MoE层实现参数高效（仅激活Top-K专家）
 * - 支持任务感知的专家路由
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3TransformerBlock extends Module {
    
    private final DeepSeekV3Config config;
    
    // 注意力子层
    private final MultiHeadAttention attention;
    private final LayerNorm layerNorm1;
    private final Dropout attnDropout;
    
    // MoE子层（替代传统FFN）
    private final DeepSeekV3MoELayer moeLayer;
    private final LayerNorm layerNorm2;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config V3配置对象
     */
    public DeepSeekV3TransformerBlock(String name, DeepSeekV3Config config) {
        super(name);
        this.config = config;
        
        int dModel = config.getNEmbd();
        int numHeads = config.getNHead();
        float dropout = (float) config.getResidPdrop();
        float attnDropoutRate = (float) config.getAttnPdrop();
        
        // 初始化注意力子层组件
        this.attention = new MultiHeadAttention("attn", dModel, numHeads, attnDropoutRate);
        this.layerNorm1 = new LayerNorm("ln1", dModel, (float) config.getLayerNormEpsilon());
        this.attnDropout = new Dropout("attn_dropout", dropout);
        
        // 初始化MoE子层
        this.moeLayer = new DeepSeekV3MoELayer(name + "_moe", config);
        this.layerNorm2 = new LayerNorm("ln2", dModel, (float) config.getLayerNormEpsilon());
        
        // 注册所有子模块
        registerModule("attn", attention);
        registerModule("ln1", layerNorm1);
        registerModule("attn_dropout", attnDropout);
        registerModule("moe", moeLayer);
        registerModule("ln2", layerNorm2);
    }
    
    /**
     * 前向传播
     * 
     * Pre-LN + MoE架构流程：
     * 1. 注意力分支: x -> LN -> Attn -> Dropout -> Add(x)
     * 2. MoE分支: x -> LN -> MoE -> Add(x)
     * 
     * @param inputs inputs[0]为输入张量 [batch_size, seq_len, d_model]
     *               inputs[1](可选)为任务类型 TaskType
     * @return 输出张量 [batch_size, seq_len, d_model]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable x = inputs[0];
        int seqLen = x.getValue().getShape().getDimension(1);
        
        // 提取任务类型（如果提供）
        TaskType taskType = null;
        if (inputs.length > 1 && inputs[1] != null) {
            // 实际使用中需要从Variable中提取TaskType
        }
        
        // 生成因果掩码（下三角矩阵）
        Variable causalMask = MultiHeadAttention.generateCausalMaskBatched(seqLen);
        
        // ===== 注意力子层 (Pre-LN) =====
        // LN -> MultiHeadAttention -> Dropout -> Add
        Variable normalized1 = layerNorm1.forward(x);
        Variable attnOutput = attention.forward(normalized1, normalized1, normalized1, causalMask, null);
        attnOutput = attnDropout.forward(attnOutput);
        Variable residual1 = x.add(attnOutput);
        
        // ===== MoE子层 (Pre-LN) =====
        // LN -> MoE -> Add
        Variable normalized2 = layerNorm2.forward(residual1);
        Variable moeOutput;
        if (taskType != null) {
            // 使用任务感知路由
            DeepSeekV3MoELayer.MoEOutput moeResult = moeLayer.computeMoE(normalized2, taskType);
            moeOutput = moeResult.output;
        } else {
            // 普通MoE前向传播
            moeOutput = moeLayer.forward(normalized2);
        }
        Variable output = residual1.add(moeOutput);
        
        return output;
    }
    
    /**
     * 带详细输出的前向传播（包含MoE损失）
     * 
     * @param input 输入张量 [batch_size, seq_len, d_model]
     * @param taskType 任务类型（可选）
     * @return 详细输出结果
     */
    public DetailedForwardResult forwardWithDetails(Variable input, TaskType taskType) {
        int seqLen = input.getValue().getShape().getDimension(1);
        
        // 生成因果掩码
        Variable causalMask = MultiHeadAttention.generateCausalMaskBatched(seqLen);
        
        // ===== 注意力子层 =====
        Variable normalized1 = layerNorm1.forward(input);
        Variable attnOutput = attention.forward(normalized1, normalized1, normalized1, causalMask, null);
        attnOutput = attnDropout.forward(attnOutput);
        Variable residual1 = input.add(attnOutput);
        
        // ===== MoE子层（获取详细结果） =====
        Variable normalized2 = layerNorm2.forward(residual1);
        DeepSeekV3MoELayer.MoEOutput moeResult = moeLayer.computeMoE(normalized2, taskType);
        Variable output = residual1.add(moeResult.output);
        
        return new DetailedForwardResult(output, moeResult);
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekV3Config getConfig() {
        return config;
    }
    
    /**
     * 获取MoE层
     */
    public DeepSeekV3MoELayer getMoeLayer() {
        return moeLayer;
    }
    
    /**
     * 详细前向传播结果类
     */
    public static class DetailedForwardResult {
        /** Transformer块的输出 */
        public final Variable output;
        /** MoE层的详细结果 */
        public final DeepSeekV3MoELayer.MoEOutput moeOutput;
        
        public DetailedForwardResult(Variable output, DeepSeekV3MoELayer.MoEOutput moeOutput) {
            this.output = output;
            this.moeOutput = moeOutput;
        }
        
        /**
         * 获取负载均衡损失
         */
        public double getLoadBalanceLoss() {
            return moeOutput.loadBalanceLoss;
        }
        
        @Override
        public String toString() {
            return String.format(
                "DetailedForwardResult{outputShape=%s, %s}",
                output.getValue().getShape(),
                moeOutput
            );
        }
    }
}
