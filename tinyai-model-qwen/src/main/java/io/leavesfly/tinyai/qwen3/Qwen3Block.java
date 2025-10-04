package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;
import io.leavesfly.tinyai.qwen3.block.Qwen3DecoderBlock;
import io.leavesfly.tinyai.qwen3.layer.RMSNormLayer;

/**
 * Qwen3Block - Qwen3模型的核心网络块
 * 
 * 继承自TinyAI的Block类，实现完整的Qwen3 Transformer架构：
 * 1. 词嵌入层 (Token Embedding)
 * 2. 多层解码器层 (Decoder Layers)  
 * 3. 最终层归一化 (Final RMSNorm)
 * 4. 语言模型头 (LM Head) - 可选
 * 
 * 该Block可以单独使用作为特征提取器，也可以配合语言模型头进行文本生成。
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Block extends Block {
    
    /** 配置对象 */
    private Qwen3Config config;
    
    /** 词嵌入层 */
    private Embedding embedTokens;
    
    /** 解码器块列表 */
    private Qwen3DecoderBlock[] decoderBlocks;
    
    /** 最终归一化层 */
    private RMSNormLayer finalNorm;
    
    /** 是否包含语言模型头 */
    private boolean withLMHead;
    
    /** 语言模型头（词汇表投影层） */
    private LinearLayer lmHead;
    
    /**
     * 构造Qwen3Block（不含语言模型头）
     * 
     * @param name Block名称
     * @param config Qwen3配置
     */
    public Qwen3Block(String name, Qwen3Config config) {
        this(name, config, false);
    }
    
    /**
     * 构造Qwen3Block
     * 
     * @param name Block名称  
     * @param config Qwen3配置
     * @param withLMHead 是否包含语言模型头
     */
    public Qwen3Block(String name, Qwen3Config config, boolean withLMHead) {
        super(name, Shape.of(-1, -1), withLMHead ? 
              Shape.of(-1, -1, config.getVocabSize()) : 
              Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.withLMHead = withLMHead;
        
        // 验证配置
        config.validate();
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            initializeComponents();
            addLayersToBlock();
            alreadyInit = true;
        }
    }
    
    /**
     * 初始化模型组件
     */
    private void initializeComponents() {
        // 1. 初始化词嵌入层
        embedTokens = new Embedding(
            name + "_embed_tokens", 
            config.getVocabSize(), 
            config.getHiddenSize()
        );
        
        // 2. 初始化解码器块
        decoderBlocks = new Qwen3DecoderBlock[config.getNumHiddenLayers()];
        for (int i = 0; i < config.getNumHiddenLayers(); i++) {
            decoderBlocks[i] = new Qwen3DecoderBlock(
                name + "_layer_" + i, config);
        }
        
        // 3. 初始化最终归一化层
        finalNorm = new RMSNormLayer(
            name + "_norm", 
            config.getHiddenSize(), 
            config.getRmsNormEps()
        );
        
        // 4. 如果需要，初始化语言模型头
        if (withLMHead) {
            lmHead = new LinearLayer(
                name + "_lm_head", 
                config.getHiddenSize(), 
                config.getVocabSize(), 
                false  // 不使用偏置
            );
            
            // 如果配置要求，共享嵌入权重
            if (config.isTieWordEmbeddings()) {
                // 注意：这里需要共享权重的逻辑，暂时跳过
                // 实际实现中需要将embedTokens的权重与lmHead的权重绑定
            }
        }
    }
    
    /**
     * 将组件添加到Block中
     */
    private void addLayersToBlock() {
        // 添加词嵌入层
        addLayer(embedTokens);
        
        // 添加所有解码器块
        for (Qwen3DecoderBlock decoderBlock : decoderBlocks) {
            addLayer(decoderBlock);
        }
        
        // 添加最终归一化层
        addLayer(finalNorm);
        
        // 如果有语言模型头，添加它
        if (withLMHead && lmHead != null) {
            addLayer(lmHead);
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Qwen3Block需要至少一个输入（input_ids）");
        }
        
        Variable inputIds = inputs[0];
        
        // 可选的注意力掩码
        Variable attentionMask = null;
        if (inputs.length > 1 && inputs[1] != null) {
            attentionMask = inputs[1];
        }
        
        return forwardQwen3(inputIds, attentionMask);
    }
    
    /**
     * Qwen3前向传播
     * 
     * @param inputIds 输入token ID [batch_size, seq_len]
     * @param attentionMask 注意力掩码 [batch_size, seq_len]  
     * @return 模型输出
     */
    private Variable forwardQwen3(Variable inputIds, Variable attentionMask) {
        // 1. 词嵌入
        Variable hiddenStates = embedTokens.layerForward(inputIds);
        
        // 2. 通过所有解码器块
        for (Qwen3DecoderBlock decoderBlock : decoderBlocks) {
            if (attentionMask != null) {
                hiddenStates = decoderBlock.layerForward(hiddenStates, attentionMask);
            } else {
                hiddenStates = decoderBlock.layerForward(hiddenStates);
            }
        }
        
        // 3. 最终归一化
        hiddenStates = finalNorm.layerForward(hiddenStates);
        
        // 4. 如果有语言模型头，进行最终投影
        if (withLMHead && lmHead != null) {
            // 将3D输入重塑为2D进行线性变换
            NdArray hiddenData = hiddenStates.getValue();
            Shape hiddenShape = hiddenData.getShape();
            int batchSize = hiddenShape.getDimension(0);
            int seqLen = hiddenShape.getDimension(1);
            int hiddenSize = hiddenShape.getDimension(2);
            
            NdArray hidden2D = reshape3DTo2D(hiddenData, batchSize, seqLen, hiddenSize);
            Variable logits = lmHead.layerForward(new Variable(hidden2D));
            
            // 重塑回3D：[batch_size, seq_len, vocab_size]
            NdArray logits3D = reshape2DTo3D(logits.getValue(), batchSize, seqLen, config.getVocabSize());
            hiddenStates = new Variable(logits3D);
        }
        
        return hiddenStates;
    }
    
    /**
     * 将3D张量重塑为2D
     */
    private NdArray reshape3DTo2D(NdArray input, int batchSize, int seqLen, int hiddenSize) {
        NdArray result = NdArray.of(Shape.of(batchSize * seqLen, hiddenSize));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < hiddenSize; h++) {
                    result.set(input.get(b, s, h), b * seqLen + s, h);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 将2D张量重塑为3D
     */
    private NdArray reshape2DTo3D(NdArray input, int batchSize, int seqLen, int lastDim) {
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, lastDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int d = 0; d < lastDim; d++) {
                    result.set(input.get(b * seqLen + s, d), b, s, d);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 计算模型参数数量
     * 
     * @return 参数总数
     */
    public long countParameters() {
        long totalParams = 0;
        
        // 词嵌入参数
        totalParams += (long) config.getVocabSize() * config.getHiddenSize();
        
        // 每个解码器块的参数
        long layerParams = 0;
        
        // 注意力层参数
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKeyValueHeads();
        int headDim = config.getHeadDim();
        
        layerParams += (long) hiddenSize * numHeads * headDim;     // query projection
        layerParams += (long) hiddenSize * numKVHeads * headDim;   // key projection  
        layerParams += (long) hiddenSize * numKVHeads * headDim;   // value projection
        layerParams += (long) numHeads * headDim * hiddenSize;     // output projection
        
        // MLP层参数
        int intermediateSize = config.getIntermediateSize();
        layerParams += (long) hiddenSize * intermediateSize;       // gate projection
        layerParams += (long) hiddenSize * intermediateSize;       // up projection
        layerParams += (long) intermediateSize * hiddenSize;       // down projection
        
        // RMSNorm参数
        layerParams += hiddenSize * 2;                             // input + post_attention norm
        
        totalParams += layerParams * config.getNumHiddenLayers();
        
        // 最终归一化参数
        totalParams += hiddenSize;
        
        // 语言模型头参数
        if (withLMHead && !config.isTieWordEmbeddings()) {
            totalParams += (long) hiddenSize * config.getVocabSize();
        }
        
        return totalParams;
    }
    
    // Getter方法
    public Qwen3Config getConfig() { return config; }
    public Embedding getEmbedTokens() { return embedTokens; }
    public Qwen3DecoderBlock[] getDecoderBlocks() { return decoderBlocks; }
    public RMSNormLayer getFinalNorm() { return finalNorm; }
    public boolean isWithLMHead() { return withLMHead; }
    public LinearLayer getLmHead() { return lmHead; }
}