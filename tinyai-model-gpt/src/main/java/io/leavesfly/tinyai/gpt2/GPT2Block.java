package io.leavesfly.tinyai.gpt2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2主模型块实现
 * 
 * 继承自Block类，实现完整的GPT-2 Transformer解码器架构
 * 根据用户要求，这个类命名为GPT2Block，用于实现完整的GPT-2模型结构
 * 
 * 模型结构：
 * 1. Token嵌入 + 位置嵌入
 * 2. N × GPT2TransformerBlock
 * 3. 最终层归一化
 * 4. 输出头（线性投影到词汇表）
 * 
 * 特点：
 * - 使用仅解码器的Transformer架构
 * - 带因果掩码的多头自注意力
 * - Pre-LayerNorm结构（GPT-2的关键改进）
 * - 支持权重共享和参数高效训练
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT2Block extends Block {
    
    /** GPT-2配置 */
    private GPT2Config config;
    
    /** Token嵌入层 */
    private GPT2TokenEmbedding tokenEmbedding;
    
    /** Transformer块列表 */
    private List<GPT2TransformerBlock> transformerBlocks;
    
    /** 最终层归一化 */
    private LayerNorm finalLayerNorm;
    
    /** 输出头 */
    private GPT2OutputHead outputHead;
    
    /**
     * 构造GPT-2 Block
     * 
     * @param name 模型名称
     * @param config GPT-2配置
     */
    public GPT2Block(String name, GPT2Config config) {
        super(name, 
              Shape.of(-1, config.getNPositions()), 
              Shape.of(-1, config.getNPositions(), config.getVocabSize()));
        
        this.config = config;
        
        // 验证配置
        config.validate();
        
        init();
    }
    
    /**
     * 使用默认配置的构造函数
     */
    public GPT2Block(String name) {
        this(name, new GPT2Config());
    }
    
    /**
     * 创建小型GPT-2模型的静态工厂方法
     */
    public static GPT2Block createSmallModel(String name) {
        return new GPT2Block(name, GPT2Config.createSmallConfig());
    }
    
    /**
     * 创建中型GPT-2模型的静态工厂方法
     */
    public static GPT2Block createMediumModel(String name) {
        return new GPT2Block(name, GPT2Config.createMediumConfig());
    }
    
    /**
     * 创建大型GPT-2模型的静态工厂方法
     */
    public static GPT2Block createLargeModel(String name) {
        return new GPT2Block(name, GPT2Config.createLargeConfig());
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入层
            tokenEmbedding = new GPT2TokenEmbedding(name + "_wte", config);
            addLayer(tokenEmbedding);
            
            // 2. 初始化Transformer块
            transformerBlocks = new ArrayList<>();
            for (int i = 0; i < config.getNLayer(); i++) {
                GPT2TransformerBlock transformerBlock = new GPT2TransformerBlock(
                    name + "_h_" + i, 
                    config, 
                    i
                );
                transformerBlocks.add(transformerBlock);
                addLayer(transformerBlock);
            }
            
            // 3. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(
                name + "_ln_f", 
                config.getNEmbd(), 
                config.getLayerNormEpsilon()
            );
            addLayer(finalLayerNorm);
            
            // 4. 初始化输出头
            outputHead = new GPT2OutputHead(name + "_lm_head", config);
            addLayer(outputHead);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batch_size, seq_len)
        
        // 1. Token嵌入和位置嵌入
        Variable embeddings = tokenEmbedding.layerForward(tokenIds);
        
        // 2. 通过所有Transformer块
        Variable hidden = embeddings;
        for (GPT2TransformerBlock transformerBlock : transformerBlocks) {
            hidden = transformerBlock.layerForward(hidden);
        }
        
        // 3. 最终层归一化
        Variable normalizedHidden = finalLayerNorm.layerForward(hidden);
        
        // 4. 输出头：映射到词汇表
        Variable logits = outputHead.layerForward(normalizedHidden);
        
        return logits;
    }
    
    /**
     * 预测下一个token
     * 
     * @param tokenIds 输入token序列
     * @return 下一个token的预测ID
     */
    public int predictNextToken(NdArray tokenIds) {
        Variable input = new Variable(tokenIds);
        Variable logits = layerForward(input);
        
        // 获取最后一个位置的logits
        NdArray logitsData = logits.getValue();
        int batchSize = logitsData.getShape().getDimension(0);
        int seqLen = logitsData.getShape().getDimension(1);
        int vocabSize = logitsData.getShape().getDimension(2);
        
        // 提取最后一个时间步的logits并找到最大值
        float maxLogit = Float.NEGATIVE_INFINITY;
        int predictedTokenId = 0;
        
        for (int v = 0; v < vocabSize; v++) {
            float logit = logitsData.get(0, seqLen - 1, v);  // 假设batch_size=1
            if (logit > maxLogit) {
                maxLogit = logit;
                predictedTokenId = v;
            }
        }
        
        return predictedTokenId;
    }
    
    /**
     * 生成文本序列
     * 
     * @param startTokenIds 起始token序列
     * @param maxLength 最大生成长度
     * @return 生成的完整序列
     */
    public NdArray generateSequence(NdArray startTokenIds, int maxLength) {
        // 创建副本：先获取原始数据，再创建新的NdArray
        int batchSize = startTokenIds.getShape().getDimension(0);
        int seqLen = startTokenIds.getShape().getDimension(1);
        NdArray currentSequence = NdArray.of(Shape.of(batchSize, seqLen));
        
        // 复制原始数据
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                currentSequence.set(startTokenIds.get(b, s), b, s);
            }
        }
        
        for (int i = 0; i < maxLength; i++) {
            // 预测下一个token
            int nextToken = predictNextToken(currentSequence);
            
            // 扩展序列
            currentSequence = appendToken(currentSequence, nextToken);
            
            // 检查是否达到最大序列长度
            if (currentSequence.getShape().getDimension(1) >= config.getNPositions()) {
                break;
            }
        }
        
        return currentSequence;
    }
    
    /**
     * 向序列追加token
     * 
     * @param sequence 原始序列
     * @param token 要追加的token
     * @return 追加后的序列
     */
    private NdArray appendToken(NdArray sequence, int token) {
        int batchSize = sequence.getShape().getDimension(0);
        int currentLength = sequence.getShape().getDimension(1);
        
        NdArray newSequence = NdArray.of(Shape.of(batchSize, currentLength + 1));
        
        // 复制原有序列
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < currentLength; s++) {
                newSequence.set(sequence.get(b, s), b, s);
            }
            // 追加新token
            newSequence.set(token, b, currentLength);
        }
        
        return newSequence;
    }
    
    /**
     * 获取模型参数数量
     * 
     * @return 总参数数量
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
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("=== GPT-2 模型信息 ===");
        System.out.println("模型名称: " + name);
        System.out.println("配置: " + config);
        System.out.println("参数数量: " + String.format("%,d", getParameterCount()));
        System.out.println("Transformer层数: " + transformerBlocks.size());
        System.out.println("输入形状: " + inputShape);
        System.out.println("输出形状: " + outputShape);
        System.out.println("===================");
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
     * 获取Token嵌入层
     * 
     * @return Token嵌入层
     */
    public GPT2TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    /**
     * 获取所有Transformer块
     * 
     * @return Transformer块列表
     */
    public List<GPT2TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
    
    /**
     * 获取指定索引的Transformer块
     * 
     * @param index 块索引
     * @return 指定的Transformer块
     */
    public GPT2TransformerBlock getTransformerBlock(int index) {
        if (index < 0 || index >= transformerBlocks.size()) {
            throw new IndexOutOfBoundsException("Transformer块索引超出范围: " + index);
        }
        return transformerBlocks.get(index);
    }
    
    /**
     * 获取最终层归一化
     * 
     * @return 最终层归一化
     */
    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }
    
    /**
     * 获取输出头
     * 
     * @return 输出头
     */
    public GPT2OutputHead getOutputHead() {
        return outputHead;
    }
}