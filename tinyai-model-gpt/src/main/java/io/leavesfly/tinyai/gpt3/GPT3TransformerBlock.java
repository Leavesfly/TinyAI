package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.transformer.MultiHeadAttention;
import io.leavesfly.tinyai.nnet.layer.transformer.FeedForward;

/**
 * GPT-3 Transformer块实现
 * 
 * 继承自Block类，实现单个GPT-3 Transformer解码器块
 * 采用Pre-LayerNorm架构，支持并行注意力和前馈网络计算
 * 
 * 主要特性：
 * 1. Pre-LayerNorm结构（在子层之前应用层归一化）
 * 2. 并行注意力和MLP计算（GPT-3的优化）
 * 3. 残差连接和Dropout
 * 4. 支持因果掩码的多头自注意力
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3TransformerBlock extends Block {
    
    /** GPT-3配置 */
    private GPT3Config config;
    
    /** 层索引 */
    private int layerIndex;
    
    /** 第一个层归一化（用于注意力） */
    private LayerNorm layerNorm1;
    
    /** 多头自注意力层 */
    private MultiHeadAttention attention;
    
    /** 第二个层归一化（用于前馈网络） */
    private LayerNorm layerNorm2;
    
    /** 前馈网络 */
    private FeedForward feedForward;
    
    /**
     * 构造GPT-3 Transformer块
     * 
     * @param name 块名称
     * @param config GPT-3配置
     * @param layerIndex 层索引
     */
    public GPT3TransformerBlock(String name, GPT3Config config, int layerIndex) {
        super(name, 
              Shape.of(-1, -1, config.getNEmbd()), 
              Shape.of(-1, -1, config.getNEmbd()));
        
        this.config = config;
        this.layerIndex = layerIndex;
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化第一个层归一化（用于注意力）
            layerNorm1 = new LayerNorm(
                name + "_ln_1", 
                config.getNEmbd(), 
                config.getLayerNormEpsilon()
            );
            addLayer(layerNorm1);
            
            // 2. 初始化多头自注意力层（使用因果掩码）
            attention = new MultiHeadAttention(
                name + "_attn", 
                config.getNEmbd(), 
                config.getNHead(), 
                true  // 使用因果掩码（解码器）
            );
            addLayer(attention);
            
            // 3. 初始化第二个层归一化（用于前馈网络）
            layerNorm2 = new LayerNorm(
                name + "_ln_2", 
                config.getNEmbd(), 
                config.getLayerNormEpsilon()
            );
            addLayer(layerNorm2);
            
            // 4. 初始化前馈网络
            feedForward = new FeedForward(
                name + "_mlp", 
                config.getNEmbd(), 
                config.getNInner()
            );
            addLayer(feedForward);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable hiddenStates = inputs[0];  // shape: (batch_size, seq_len, n_embd)
        
        if (config.isParallelAttention()) {
            // GPT-3的并行注意力和MLP计算优化
            return forwardParallel(hiddenStates);
        } else {
            // 传统的串行计算方式
            return forwardSequential(hiddenStates);
        }
    }
    
    /**
     * 并行计算方式（GPT-3的优化）
     * 同时计算注意力和前馈网络，然后合并结果
     */
    private Variable forwardParallel(Variable hiddenStates) {
        // Pre-LayerNorm：在注意力和MLP之前分别应用层归一化
        Variable ln1Output = layerNorm1.layerForward(hiddenStates);
        Variable ln2Output = layerNorm2.layerForward(hiddenStates);
        
        // 并行计算注意力和MLP
        Variable attnOutput = attention.layerForward(ln1Output, ln1Output, ln1Output);
        Variable mlpOutput = feedForward.layerForward(ln2Output);
        
        // 应用dropout（简化版本，实际应用中需要考虑训练/推理模式）
        attnOutput = applyDropout(attnOutput, config.getResidDropout());
        mlpOutput = applyDropout(mlpOutput, config.getResidDropout());
        
        // 残差连接：hidden_states + attention_output + mlp_output
        Variable output = hiddenStates.add(attnOutput).add(mlpOutput);
        
        return output;
    }
    
    /**
     * 串行计算方式（传统方式）
     * 先计算注意力，再计算前馈网络
     */
    private Variable forwardSequential(Variable hiddenStates) {
        // 第一个子层：多头自注意力 + 残差连接
        Variable residual1 = hiddenStates;
        Variable ln1Output = layerNorm1.layerForward(hiddenStates);
        Variable attnOutput = attention.layerForward(ln1Output, ln1Output, ln1Output);
        attnOutput = applyDropout(attnOutput, config.getResidDropout());
        Variable afterAttn = residual1.add(attnOutput);
        
        // 第二个子层：前馈网络 + 残差连接
        Variable residual2 = afterAttn;
        Variable ln2Output = layerNorm2.layerForward(afterAttn);
        Variable mlpOutput = feedForward.layerForward(ln2Output);
        mlpOutput = applyDropout(mlpOutput, config.getResidDropout());
        Variable output = residual2.add(mlpOutput);
        
        return output;
    }
    
    /**
     * 应用Dropout（简化实现）
     * 在实际应用中需要区分训练和推理模式
     */
    private Variable applyDropout(Variable input, double dropoutRate) {
        if (dropoutRate <= 0.0 || dropoutRate >= 1.0) {
            return input;
        }
        
        // 简化的dropout实现
        // 在实际应用中应该：
        // 1. 检查是否在训练模式
        // 2. 生成随机掩码
        // 3. 按比例缩放保留的值
        
        // 这里暂时返回原始输入，实际部署时需要完善
        return input;
    }
    
    /**
     * 获取层的参数数量
     */
    public long getParameterCount() {
        long totalParams = 0;
        var allParams = getAllParams();
        for (var param : allParams.values()) {
            totalParams += param.getValue().getShape().size();
        }
        return totalParams;
    }
    
    /**
     * 打印层信息
     */
    public void printLayerInfo() {
        System.out.println(String.format(
            "GPT3TransformerBlock[%d]: %s\n" +
            "  - 参数数量: %,d\n" +
            "  - 注意力头数: %d\n" +
            "  - 隐藏维度: %d -> %d -> %d\n" +
            "  - 并行计算: %s",
            layerIndex, name, getParameterCount(),
            config.getNHead(),
            config.getNEmbd(), config.getNInner(), config.getNEmbd(),
            config.isParallelAttention() ? "启用" : "禁用"
        ));
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取配置
     */
    public GPT3Config getConfig() {
        return config;
    }
    
    /**
     * 获取层索引
     */
    public int getLayerIndex() {
        return layerIndex;
    }
    
    /**
     * 获取第一个层归一化
     */
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }
    
    /**
     * 获取注意力层
     */
    public MultiHeadAttention getAttention() {
        return attention;
    }
    
    /**
     * 获取第二个层归一化
     */
    public LayerNorm getLayerNorm2() {
        return layerNorm2;
    }
    
    /**
     * 获取前馈网络
     */
    public FeedForward getFeedForward() {
        return feedForward;
    }
}