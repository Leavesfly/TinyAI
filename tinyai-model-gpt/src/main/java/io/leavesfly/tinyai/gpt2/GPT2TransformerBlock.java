package io.leavesfly.tinyai.gpt2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.transformer.MultiHeadAttention;
import io.leavesfly.tinyai.nnet.layer.transformer.FeedForward;

/**
 * GPT-2 Transformer块实现
 * 
 * 实现了单个GPT-2 Transformer解码器块，采用Pre-LayerNorm架构：
 * 1. 输入 → LayerNorm1 → MultiHeadAttention → 残差连接1
 * 2. 残差连接1 → LayerNorm2 → FeedForward → 残差连接2 → 输出
 * 
 * 这是GPT-2相对于原始Transformer的重要改进，将LayerNorm移至每个子层前面
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT2TransformerBlock extends Block {
    
    /** GPT-2配置 */
    private GPT2Config config;
    
    /** 第一个层归一化（注意力前） */
    private LayerNorm layerNorm1;
    
    /** 多头自注意力层 */
    private MultiHeadAttention attention;
    
    /** 第二个层归一化（前馈前） */
    private LayerNorm layerNorm2;
    
    /** 前馈网络 */
    private FeedForward feedForward;
    
    /** 层索引（用于命名和统计） */
    private int layerIdx;
    
    /**
     * 构造GPT-2 Transformer块
     * 
     * @param name 块名称
     * @param config GPT-2配置
     * @param layerIdx 层索引
     */
    public GPT2TransformerBlock(String name, GPT2Config config, int layerIdx) {
        super(name, 
              Shape.of(-1, -1, config.getNEmbd()), 
              Shape.of(-1, -1, config.getNEmbd()));
        
        this.config = config;
        this.layerIdx = layerIdx;
        
        // 验证配置
        config.validate();
        
        init();
    }
    
    /**
     * 兼容构造函数（不需要层索引）
     */
    public GPT2TransformerBlock(String name, GPT2Config config) {
        this(name, config, 0);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 第一个层归一化（注意力前的Pre-LayerNorm）
            layerNorm1 = new LayerNorm(
                name + "_ln_1", 
                config.getNEmbd(), 
                config.getLayerNormEpsilon()
            );
            addLayer(layerNorm1);
            
            // 2. 多头自注意力层（带因果掩码）
            attention = new MultiHeadAttention(
                name + "_attn", 
                config.getNEmbd(), 
                config.getNHead(), 
                true  // 使用因果掩码（解码器）
            );
            addLayer(attention);
            
            // 3. 第二个层归一化（前馈前的Pre-LayerNorm）
            layerNorm2 = new LayerNorm(
                name + "_ln_2", 
                config.getNEmbd(), 
                config.getLayerNormEpsilon()
            );
            addLayer(layerNorm2);
            
            // 4. 前馈网络
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
        Variable x = inputs[0];  // shape: (batch_size, seq_len, n_embd)
        
        // Pre-LayerNorm架构的前向传播
        
        // 1. 注意力子层：LayerNorm → MultiHeadAttention → 残差连接
        Variable ln1Output = layerNorm1.layerForward(x);
        Variable attnOutput = attention.layerForward(ln1Output);
        Variable residual1 = addResidualConnection(x, attnOutput);
        
        // 2. 前馈子层：LayerNorm → FeedForward → 残差连接
        Variable ln2Output = layerNorm2.layerForward(residual1);
        Variable ffnOutput = feedForward.layerForward(ln2Output);
        Variable residual2 = addResidualConnection(residual1, ffnOutput);
        
        // 3. 应用dropout（简化实现）
        Variable result = applyDropout(residual2);
        
        return result;
    }
    
    /**
     * 添加残差连接
     * 
     * @param input 输入变量
     * @param output 子层输出变量
     * @return 残差连接后的变量
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        // 简单的元素加法实现残差连接
        NdArray inputData = input.getValue();
        NdArray outputData = output.getValue();
        
        // 验证形状匹配
        if (!inputData.getShape().equals(outputData.getShape())) {
            throw new IllegalArgumentException(
                String.format("残差连接要求输入和输出形状相同，但得到输入形状%s和输出形状%s",
                            inputData.getShape(), outputData.getShape())
            );
        }
        
        NdArray result = inputData.add(outputData);
        return new Variable(result);
    }
    
    /**
     * 应用Dropout正则化
     * 
     * @param input 输入变量
     * @return 应用dropout后的变量
     */
    private Variable applyDropout(Variable input) {
        // 简化实现：在训练模式下应该实现真正的dropout
        // 这里暂时返回原始输入
        if (config.getResidPdrop() > 0.0) {
            // TODO: 实现训练/推理模式切换和真正的dropout
            return input;
        }
        return input;
    }
    
    /**
     * 获取注意力权重（用于可视化和分析）
     * 
     * @return 注意力权重，如果可用的话
     */
    public NdArray getAttentionWeights() {
        // 这需要MultiHeadAttention层支持返回注意力权重
        // 暂时返回null，实际实现需要修改MultiHeadAttention
        return null;
    }
    
    /**
     * 重置层状态（如果需要）
     */
    public void resetState() {
        // GPT-2 Transformer块通常是无状态的
        // 如果将来需要支持缓存等状态，可以在这里重置
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取GPT-2配置
     * 
     * @return GPT-2配置
     */
    public GPT2Config getConfig() {
        return config;
    }
    
    /**
     * 获取层索引
     * 
     * @return 层索引
     */
    public int getLayerIdx() {
        return layerIdx;
    }
    
    /**
     * 获取第一个层归一化层
     * 
     * @return LayerNorm1
     */
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }
    
    /**
     * 获取多头注意力层
     * 
     * @return MultiHeadAttention
     */
    public MultiHeadAttention getAttention() {
        return attention;
    }
    
    /**
     * 获取第二个层归一化层
     * 
     * @return LayerNorm2
     */
    public LayerNorm getLayerNorm2() {
        return layerNorm2;
    }
    
    /**
     * 获取前馈网络
     * 
     * @return FeedForward
     */
    public FeedForward getFeedForward() {
        return feedForward;
    }
    
    @Override
    public String toString() {
        return String.format("GPT2TransformerBlock{name='%s', layerIdx=%d, nEmbd=%d, nHead=%d, nInner=%d}",
                           name, layerIdx, config.getNEmbd(), config.getNHead(), config.getNInner());
    }
}