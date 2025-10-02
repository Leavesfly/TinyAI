package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-1 Token嵌入层实现
 * 
 * 负责将离散的token ID转换为连续的向量表示，包括：
 * 1. Token嵌入：将词汇ID映射到向量空间
 * 2. 位置嵌入：为每个序列位置学习位置向量
 * 
 * 与GPT-2的区别：
 * - GPT-1使用学习的位置嵌入而非正弦位置编码
 * - 初始化策略与原论文保持一致
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT1TokenEmbedding extends Layer {
    
    /** Token嵌入参数矩阵 (vocabSize, hiddenSize) */
    private Parameter tokenEmbedding;
    
    /** 位置嵌入参数矩阵 (maxSequenceLength, hiddenSize) */
    private Parameter positionEmbedding;
    
    /** 配置信息 */
    private GPT1Config config;
    
    /**
     * 构造GPT-1 Token嵌入层
     * 
     * @param name 层名称
     * @param config GPT-1配置
     */
    public GPT1TokenEmbedding(String name, GPT1Config config) {
        super(name, 
              Shape.of(-1, config.getMaxSequenceLength()), 
              Shape.of(-1, config.getMaxSequenceLength(), config.getHiddenSize()));
        
        this.config = config;
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入矩阵
            // 使用正态分布初始化，标准差为initializerRange
            tokenEmbedding = new Parameter(
                NdArray.likeRandomN(Shape.of(config.getVocabSize(), config.getHiddenSize()))
                       .mulNum((float) config.getInitializerRange())
            );
            tokenEmbedding.setName(name + "_token_embedding");
            addParam(tokenEmbedding.getName(), tokenEmbedding);
            
            // 2. 初始化位置嵌入矩阵
            // GPT-1使用学习的位置嵌入
            positionEmbedding = new Parameter(
                NdArray.likeRandomN(Shape.of(config.getMaxSequenceLength(), config.getHiddenSize()))
                       .mulNum((float) config.getInitializerRange())
            );
            positionEmbedding.setName(name + "_position_embedding");
            addParam(positionEmbedding.getName(), positionEmbedding);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batchSize, sequenceLength)
        NdArray tokenData = tokenIds.getValue();
        
        int batchSize = tokenData.getShape().getDimension(0);
        int sequenceLength = tokenData.getShape().getDimension(1);
        
        // 验证序列长度
        if (sequenceLength > config.getMaxSequenceLength()) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", 
                            sequenceLength, config.getMaxSequenceLength())
            );
        }
        
        // 1. 获取Token嵌入
        Variable tokenEmbeds = getTokenEmbeddings(tokenData, batchSize, sequenceLength);
        
        // 2. 获取位置嵌入
        Variable positionEmbeds = getPositionEmbeddings(sequenceLength, batchSize);
        
        // 3. 将Token嵌入和位置嵌入相加
        Variable embeddings = tokenEmbeds.add(positionEmbeds);
        
        // 4. 应用Dropout（在实际训练中实现）
        embeddings = applyEmbeddingDropout(embeddings);
        
        return embeddings;
    }
    
    /**
     * 获取Token嵌入
     * 
     * @param tokenIds token ID数组
     * @param batchSize 批次大小
     * @param sequenceLength 序列长度
     * @return Token嵌入变量
     */
    private Variable getTokenEmbeddings(NdArray tokenIds, int batchSize, int sequenceLength) {
        NdArray embeddings = NdArray.of(Shape.of(batchSize, sequenceLength, config.getHiddenSize()));
        
        // 对每个token ID查找对应的嵌入向量
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sequenceLength; s++) {
                int tokenId = (int) tokenIds.get(b, s);
                
                // 验证token ID的有效性
                validateTokenId(tokenId);
                
                // 复制对应的嵌入向量
                copyEmbeddingVector(embeddings, tokenId, b, s);
            }
        }
        
        return new Variable(embeddings);
    }
    
    /**
     * 获取位置嵌入
     * 
     * @param sequenceLength 序列长度
     * @param batchSize 批次大小
     * @return 位置嵌入变量
     */
    private Variable getPositionEmbeddings(int sequenceLength, int batchSize) {
        NdArray posEmbeds = NdArray.of(Shape.of(batchSize, sequenceLength, config.getHiddenSize()));
        
        // 为每个位置添加位置嵌入
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sequenceLength; s++) {
                // 复制对应位置的嵌入向量
                copyPositionVector(posEmbeds, s, b, s);
            }
        }
        
        return new Variable(posEmbeds);
    }
    
    /**
     * 验证Token ID的有效性
     * 
     * @param tokenId 待验证的token ID
     * @throws IllegalArgumentException 如果token ID无效
     */
    private void validateTokenId(int tokenId) {
        if (tokenId < 0 || tokenId >= config.getVocabSize()) {
            throw new IllegalArgumentException(
                String.format("Token ID %d 超出词汇表范围 [0, %d)", 
                            tokenId, config.getVocabSize())
            );
        }
    }
    
    /**
     * 复制嵌入向量到目标位置
     * 
     * @param target 目标数组
     * @param tokenId token ID
     * @param batchIndex 批次索引
     * @param seqIndex 序列索引
     */
    private void copyEmbeddingVector(NdArray target, int tokenId, int batchIndex, int seqIndex) {
        for (int d = 0; d < config.getHiddenSize(); d++) {
            float embeddingValue = tokenEmbedding.getValue().get(tokenId, d);
            target.set(embeddingValue, batchIndex, seqIndex, d);
        }
    }
    
    /**
     * 复制位置向量到目标位置
     * 
     * @param target 目标数组
     * @param positionIndex 位置索引
     * @param batchIndex 批次索引
     * @param seqIndex 序列索引
     */
    private void copyPositionVector(NdArray target, int positionIndex, int batchIndex, int seqIndex) {
        for (int d = 0; d < config.getHiddenSize(); d++) {
            float positionValue = positionEmbedding.getValue().get(positionIndex, d);
            target.set(positionValue, batchIndex, seqIndex, d);
        }
    }
    
    /**
     * 应用嵌入层Dropout
     * 
     * @param embeddings 输入嵌入
     * @return 应用dropout后的嵌入
     */
    private Variable applyEmbeddingDropout(Variable embeddings) {
        // 在简化实现中，暂时返回原始嵌入
        // 实际训练时需要考虑训练/推理模式并实现真正的dropout
        if (config.getEmbeddingDropoutProb() > 0.0) {
            // TODO: 实现真正的dropout
            return embeddings;
        }
        return embeddings;
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 简化的反向传播实现
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取Token嵌入参数
     * 
     * @return Token嵌入参数
     */
    public Parameter getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    /**
     * 获取位置嵌入参数
     * 
     * @return 位置嵌入参数
     */
    public Parameter getPositionEmbedding() {
        return positionEmbedding;
    }
    
    /**
     * 获取配置信息
     * 
     * @return GPT-1配置
     */
    public GPT1Config getConfig() {
        return config;
    }
    
    /**
     * 获取词汇表大小
     * 
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return config.getVocabSize();
    }
    
    /**
     * 获取隐藏层维度
     * 
     * @return 隐藏层维度
     */
    public int getHiddenSize() {
        return config.getHiddenSize();
    }
    
    /**
     * 获取最大序列长度
     * 
     * @return 最大序列长度
     */
    public int getMaxSequenceLength() {
        return config.getMaxSequenceLength();
    }
}