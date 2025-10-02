package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.gpt2.GPT2TokenEmbedding;
import io.leavesfly.tinyai.gpt2.GPT2OutputHead;
import io.leavesfly.tinyai.gpt2.GPT2Config;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-3模型主体
 * 
 * 继承自Model类，实现完整的GPT-3语言模型
 * 采用解码器-only的Transformer架构
 * 
 * 主要特性：
 * 1. 更大的模型规模（最大175B参数）
 * 2. 并行注意力和MLP计算
 * 3. 支持稀疏注意力机制
 * 4. 梯度检查点和内存优化
 * 5. 强大的Few-shot学习能力
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3Model extends Model {
    
    /** GPT-3配置 */
    private GPT3Config config;
    
    /** GPT3主块实例 */
    private GPT3MainBlock gpt3Block;
    
    /**
     * 构造GPT-3模型
     * 
     * @param name 模型名称
     * @param config GPT-3配置
     */
    public GPT3Model(String name, GPT3Config config) {
        super(name, new GPT3MainBlock(name + "_transformer", config));
        this.config = config;
        this.gpt3Block = (GPT3MainBlock) getBlock();
        
        setDescription("GPT-3语言模型 - " + config.toString());
        updateModelInfo();
    }
    
    /**
     * 使用默认配置的构造函数
     */
    public GPT3Model(String name) {
        this(name, new GPT3Config());
    }
    
    /**
     * 创建小型GPT-3模型（125M参数）
     */
    public static GPT3Model createSmallModel(String name) {
        return new GPT3Model(name, GPT3Config.createSmallConfig());
    }
    
    /**
     * 创建中型GPT-3模型（350M参数）
     */
    public static GPT3Model createMediumModel(String name) {
        return new GPT3Model(name, GPT3Config.createMediumConfig());
    }
    
    /**
     * 创建大型GPT-3模型（1.3B参数）
     */
    public static GPT3Model createLargeModel(String name) {
        return new GPT3Model(name, GPT3Config.createLargeConfig());
    }
    
    /**
     * 创建超大型GPT-3模型（175B参数）
     */
    public static GPT3Model createXLModel(String name) {
        return new GPT3Model(name, GPT3Config.createXLConfig());
    }
    
    /**
     * 更新模型信息
     */
    private void updateModelInfo() {
        if (getModelInfo() != null) {
            getModelInfo().setArchitectureType("GPT-3");
            addMetric("vocabulary_size", config.getVocabSize());
            addMetric("embedding_dimension", config.getNEmbd());
            addMetric("num_layers", config.getNLayer());
            addMetric("num_heads", config.getNHead());
            addMetric("sparse_attention", config.isSparseAttention() ? 1 : 0);
            addMetric("parallel_attention", config.isParallelAttention() ? 1 : 0);
            
            long totalParams = gpt3Block.getParameterCount();
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
        return gpt3Block.predictNextToken(tokenIds);
    }
    
    /**
     * 生成文本序列
     */
    public NdArray generateSequence(NdArray startTokenIds, int maxLength) {
        return gpt3Block.generateSequence(startTokenIds, maxLength);
    }
    
    /**
     * Few-shot学习生成
     * 基于提供的示例进行上下文学习
     */
    public NdArray fewShotGenerate(NdArray contextTokenIds, int maxNewTokens) {
        return gpt3Block.generateWithContext(contextTokenIds, maxNewTokens);
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
            "GPT-3模型配置摘要:\n" +
            "- 词汇表大小: %,d\n" +
            "- 嵌入维度: %d\n" +
            "- Transformer层数: %d\n" +
            "- 注意力头数: %d\n" +
            "- 前馈网络维度: %d\n" +
            "- 最大序列长度: %d\n" +
            "- 并行注意力: %s\n" +
            "- 稀疏注意力: %s\n" +
            "- 总参数数量: %,d\n",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            config.getNInner(),
            config.getNPositions(),
            config.isParallelAttention() ? "启用" : "禁用",
            config.isSparseAttention() ? "启用" : "禁用",
            gpt3Block.getParameterCount()
        );
    }
    
    @Override
    public void printModelInfo() {
        System.out.println("=== GPT-3 模型详细信息 ===");
        System.out.println(getConfigSummary());
        super.printModelInfo();
        System.out.println("========================");
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取GPT-3配置
     */
    public GPT3Config getConfig() {
        return config;
    }
    
    /**
     * 获取GPT-3主块
     */
    public GPT3MainBlock getGPT3Block() {
        return gpt3Block;
    }
    
    /**
     * 获取Token嵌入层
     */
    public GPT2TokenEmbedding getTokenEmbedding() {
        return gpt3Block.getTokenEmbedding();
    }
    
    /**
     * 获取所有Transformer块
     */
    public List<GPT3TransformerBlock> getTransformerBlocks() {
        return gpt3Block.getTransformerBlocks();
    }
    
    /**
     * 获取指定索引的Transformer块
     */
    public GPT3TransformerBlock getTransformerBlock(int index) {
        return gpt3Block.getTransformerBlock(index);
    }
    
    /**
     * 获取最终层归一化
     */
    public LayerNorm getFinalLayerNorm() {
        return gpt3Block.getFinalLayerNorm();
    }
    
    /**
     * 获取输出头
     */
    public GPT2OutputHead getOutputHead() {
        return gpt3Block.getOutputHead();
    }
}

/**
 * GPT-3主块实现
 * 
 * 继承自Block，包含完整的GPT-3架构：
 * 1. Token嵌入 + 位置嵌入
 * 2. N × GPT3TransformerBlock
 * 3. 最终层归一化
 * 4. 输出头
 */
class GPT3MainBlock extends Block {
    
    /** GPT-3配置 */
    private GPT3Config config;
    
    /** Token嵌入层（复用GPT-2实现） */
    private GPT2TokenEmbedding tokenEmbedding;
    
    /** Transformer块列表 */
    private List<GPT3TransformerBlock> transformerBlocks;
    
    /** 最终层归一化 */
    private LayerNorm finalLayerNorm;
    
    /** 输出头（复用GPT-2实现） */
    private GPT2OutputHead outputHead;
    
    /**
     * 构造GPT-3主块
     */
    public GPT3MainBlock(String name, GPT3Config config) {
        super(name, 
              Shape.of(-1, config.getNPositions()), 
              Shape.of(-1, config.getNPositions(), config.getVocabSize()));
        
        this.config = config;
        config.validate();
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入层（复用GPT-2实现）
            GPT2Config gpt2Config = convertToGPT2Config(config);
            tokenEmbedding = new GPT2TokenEmbedding(name + "_wte", gpt2Config);
            addLayer(tokenEmbedding);
            
            // 2. 初始化所有GPT-3 Transformer块
            transformerBlocks = new ArrayList<>();
            for (int i = 0; i < config.getNLayer(); i++) {
                GPT3TransformerBlock transformerBlock = new GPT3TransformerBlock(
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
            
            // 4. 初始化输出头（复用GPT-2实现）
            outputHead = new GPT2OutputHead(name + "_lm_head", gpt2Config);
            addLayer(outputHead);
            
            alreadyInit = true;
        }
    }
    
    /**
     * 将GPT-3配置转换为GPT-2配置（用于复用嵌入层和输出头）
     */
    private GPT2Config convertToGPT2Config(GPT3Config gpt3Config) {
        GPT2Config gpt2Config = new GPT2Config();
        gpt2Config.setVocabSize(gpt3Config.getVocabSize());
        gpt2Config.setNPositions(gpt3Config.getNPositions());
        gpt2Config.setNEmbd(gpt3Config.getNEmbd());
        gpt2Config.setNLayer(gpt3Config.getNLayer());
        gpt2Config.setNHead(gpt3Config.getNHead());
        gpt2Config.setEmbdPdrop(gpt3Config.getEmbdDropout());
        gpt2Config.setInitializerRange(gpt3Config.getInitializerRange());
        return gpt2Config;
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batch_size, seq_len)
        
        // 1. Token嵌入和位置嵌入
        Variable embeddings = tokenEmbedding.layerForward(tokenIds);
        
        // 2. 通过所有GPT-3 Transformer块
        Variable hidden = embeddings;
        for (GPT3TransformerBlock transformerBlock : transformerBlocks) {
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
     */
    public int predictNextToken(NdArray tokenIds) {
        Variable input = new Variable(tokenIds);
        Variable logits = layerForward(input);
        
        // 获取最后一个位置的logits并找到最大值
        NdArray logitsData = logits.getValue();
        int batchSize = logitsData.getShape().getDimension(0);
        int seqLen = logitsData.getShape().getDimension(1);
        int vocabSize = logitsData.getShape().getDimension(2);
        
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
     */
    public NdArray generateSequence(NdArray startTokenIds, int maxLength) {
        return generateWithContext(startTokenIds, maxLength);
    }
    
    /**
     * 基于上下文生成文本（支持Few-shot学习）
     */
    public NdArray generateWithContext(NdArray contextTokenIds, int maxNewTokens) {
        int batchSize = contextTokenIds.getShape().getDimension(0);
        int contextLength = contextTokenIds.getShape().getDimension(1);
        
        // 创建当前序列的副本
        NdArray currentSequence = NdArray.of(Shape.of(batchSize, contextLength));
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < contextLength; s++) {
                currentSequence.set(contextTokenIds.get(b, s), b, s);
            }
        }
        
        // 逐步生成新token
        for (int i = 0; i < maxNewTokens; i++) {
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
     */
    public long getParameterCount() {
        long totalParams = 0;
        var allParams = getAllParams();
        for (var param : allParams.values()) {
            totalParams += param.getValue().getShape().size();
        }
        return totalParams;
    }
    
    // ==================== Getter方法 ====================
    
    public GPT3Config getConfig() { return config; }
    public GPT2TokenEmbedding getTokenEmbedding() { return tokenEmbedding; }
    public List<GPT3TransformerBlock> getTransformerBlocks() { return transformerBlocks; }
    public GPT3TransformerBlock getTransformerBlock(int index) { return transformerBlocks.get(index); }
    public LayerNorm getFinalLayerNorm() { return finalLayerNorm; }
    public GPT2OutputHead getOutputHead() { return outputHead; }
}