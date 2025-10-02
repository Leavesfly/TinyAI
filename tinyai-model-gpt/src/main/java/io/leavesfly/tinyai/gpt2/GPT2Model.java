package io.leavesfly.tinyai.gpt2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * GPT-2模型类
 * 
 * 继承自Model类，实现完整的GPT-2模型封装
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT2Model extends Model {
    
    /** GPT-2配置 */
    private GPT2Config config;
    
    /** GPT2Block实例 */
    private GPT2Block gpt2Block;
    
    /**
     * 构造GPT-2模型
     */
    public GPT2Model(String name, GPT2Config config) {
        super(name, new GPT2Block(name + "_block", config));
        this.config = config;
        this.gpt2Block = (GPT2Block) getBlock();
        
        setDescription("GPT-2语言模型 - " + config.toString());
        updateModelInfo();
    }
    
    /**
     * 使用默认配置的构造函数
     */
    public GPT2Model(String name) {
        this(name, new GPT2Config());
    }
    
    /**
     * 创建小型GPT-2模型
     */
    public static GPT2Model createSmallModel(String name) {
        return new GPT2Model(name, GPT2Config.createSmallConfig());
    }
    
    /**
     * 创建中型GPT-2模型
     */
    public static GPT2Model createMediumModel(String name) {
        return new GPT2Model(name, GPT2Config.createMediumConfig());
    }
    
    /**
     * 创建大型GPT-2模型
     */
    public static GPT2Model createLargeModel(String name) {
        return new GPT2Model(name, GPT2Config.createLargeConfig());
    }
    
    /**
     * 更新模型信息
     */
    private void updateModelInfo() {
        if (getModelInfo() != null) {
            getModelInfo().setArchitectureType("GPT-2");
            addMetric("vocabulary_size", config.getVocabSize());
            addMetric("embedding_dimension", config.getNEmbd());
            addMetric("num_layers", config.getNLayer());
            addMetric("num_heads", config.getNHead());
            
            long totalParams = gpt2Block.getParameterCount();
            getModelInfo().setTotalParameters(totalParams);
        }
    }
    
    /**
     * 模型前向传播
     */
    public Variable predict(NdArray tokenIds) {
        return forward(new Variable(tokenIds));
    }
    
    /**
     * 预测下一个token
     */
    public int predictNextToken(NdArray tokenIds) {
        return gpt2Block.predictNextToken(tokenIds);
    }
    
    /**
     * 生成文本序列
     */
    public NdArray generateSequence(NdArray startTokenIds, int maxLength) {
        return gpt2Block.generateSequence(startTokenIds, maxLength);
    }
    
    /**
     * 验证输入序列的有效性
     */
    public void validateInput(NdArray tokenIds) {
        Shape shape = tokenIds.getShape();
        
        if (shape.getDimNum() != 2) {
            throw new IllegalArgumentException("输入必须是二维数组 (batch_size, seq_len)");
        }
        
        int seqLen = shape.getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大支持长度(%d)", seqLen, config.getNPositions())
            );
        }
    }
    
    /**
     * 获取模型配置信息摘要
     */
    public String getConfigSummary() {
        return String.format(
            "GPT-2模型配置摘要:\n" +
            "- 词汇表大小: %,d\n" +
            "- 嵌入维度: %d\n" +
            "- Transformer层数: %d\n" +
            "- 注意力头数: %d\n" +
            "- 总参数数量: %,d\n",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            gpt2Block.getParameterCount()
        );
    }
    
    @Override
    public void printModelInfo() {
        System.out.println("=== GPT-2 模型详细信息 ===");
        System.out.println(getConfigSummary());
        super.printModelInfo();
        System.out.println("=========================");
    }
    
    // Getter方法
    public GPT2Config getConfig() { return config; }
    public GPT2Block getGPT2Block() { return gpt2Block; }
    public GPT2TokenEmbedding getTokenEmbedding() { return gpt2Block.getTokenEmbedding(); }
    public GPT2TransformerBlock getTransformerBlock(int index) { return gpt2Block.getTransformerBlock(index); }
    public GPT2OutputHead getOutputHead() { return gpt2Block.getOutputHead(); }
}