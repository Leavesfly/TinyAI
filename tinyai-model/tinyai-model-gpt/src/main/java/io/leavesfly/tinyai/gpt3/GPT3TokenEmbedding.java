package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * GPT-3 Token嵌入层（完全基于V2 Module）
 * 
 * 负责将离散的token ID转换为连续的向量表示，包括：
 * 1. Token嵌入：将词汇ID映射到向量空间
 * 2. 位置嵌入：为每个序列位置学习位置向量
 * 3. Dropout正则化
 * 
 * @author leavesfly
 * @version 2.0 - 完全基于V2 API
 */
public class GPT3TokenEmbedding extends Module {
    
    private final int vocabSize;
    private final int embeddingDim;
    private final int maxPositions;
    private final float dropoutProb;
    
    // V2 参数
    private Parameter tokenEmbedding;      // Token嵌入矩阵 (vocabSize, embeddingDim)
    private Parameter positionEmbedding;   // 位置嵌入矩阵 (maxPositions, embeddingDim)
    
    // Dropout层
    private Dropout dropout;
    
    /**
     * 构造GPT-3 Token嵌入层
     * 
     * @param name 层名称
     * @param config GPT-3配置
     */
    public GPT3TokenEmbedding(String name, GPT3Config config) {
        super(name);
        
        this.vocabSize = config.getVocabSize();
        this.embeddingDim = config.getNEmbd();
        this.maxPositions = config.getNPositions();
        this.dropoutProb = (float) config.getEmbdPdrop();
        
        // 初始化参数和层
        initializeParameters(config);
    }
    
    /**
     * 初始化参数
     */
    private void initializeParameters(GPT3Config config) {
        // 1. 初始化Token嵌入矩阵
        NdArray tokenEmbedData = NdArray.likeRandomN(Shape.of(vocabSize, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        tokenEmbedding = new Parameter(tokenEmbedData);
        registerParameter("token_embedding", tokenEmbedding);
        
        // 2. 初始化位置嵌入矩阵
        NdArray positionEmbedData = NdArray.likeRandomN(Shape.of(maxPositions, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        positionEmbedding = new Parameter(positionEmbedData);
        registerParameter("position_embedding", positionEmbedding);
        
        // 3. 初始化Dropout
        dropout = new Dropout("embedding_dropout", dropoutProb);
        registerModule("dropout", dropout);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batchSize, sequenceLength)
        NdArray tokenData = tokenIds.getValue();
        
        int batchSize = tokenData.getShape().getDimension(0);
        int sequenceLength = tokenData.getShape().getDimension(1);
        
        // 验证序列长度
        if (sequenceLength > maxPositions) {
            throw new IllegalArgumentException(
                String.format("输入序列长度(%d)超过最大位置数(%d)", sequenceLength, maxPositions)
            );
        }
        
        // 1. 获取Token嵌入
        Variable tokenEmbeds = getTokenEmbeddings(tokenIds, batchSize, sequenceLength);
        
        // 2. 获取位置嵌入
        Variable positionEmbeds = getPositionEmbeddings(sequenceLength, batchSize);
        
        // 3. 相加组合Token和位置嵌入
        Variable combined = tokenEmbeds.add(positionEmbeds);
        
        // 4. 应用Dropout
        Variable result = dropout.forward(combined);
        
        return result;
    }
    
    /**
     * 获取Token嵌入（使用Variable算子）
     * 
     * @param tokenIds Token ID变量
     * @param batchSize 批次大小
     * @param sequenceLength 序列长度
     * @return Token嵌入变量
     */
    private Variable getTokenEmbeddings(Variable tokenIds, int batchSize, int sequenceLength) {
        // 使用IndexSelect算子实现embedding lookup
        // tokenEmbedding: (vocabSize, embeddingDim)
        // tokenIds: (batchSize, sequenceLength)
        // 需要将tokenIds flatten为1D，然后使用indexSelect，最后reshape回(batchSize, sequenceLength, embeddingDim)
        
        // 直接使用Parameter参与计算（Parameter继承自Variable），梯度可以直接传回Parameter
        Variable tokenEmbedVar = tokenEmbedding;
        
        // Flatten tokenIds: (batchSize, sequenceLength) -> (batchSize * sequenceLength)
        Variable flatTokenIds = tokenIds.reshape(Shape.of(batchSize * sequenceLength));
        
        // IndexSelect: 从(vocabSize, embeddingDim)中选择，得到(batchSize * sequenceLength, embeddingDim)
        Variable flatEmbeddings = tokenEmbedVar.indexSelect(0, flatTokenIds);
        
        // Reshape回原始形状: (batchSize, sequenceLength, embeddingDim)
        Variable embeddings = flatEmbeddings.reshape(Shape.of(batchSize, sequenceLength, embeddingDim));
        
        return embeddings;
    }
    
    /**
     * 获取位置嵌入（使用Variable算子）
     * 
     * @param sequenceLength 序列长度
     * @param batchSize 批次大小
     * @return 位置嵌入变量
     */
    private Variable getPositionEmbeddings(int sequenceLength, int batchSize) {
        // 使用IndexSelect算子实现position embedding lookup
        // positionEmbedding: (maxPositions, embeddingDim)
        // 需要选择前sequenceLength个位置，然后扩展到batchSize
        
        // 直接使用Parameter参与计算（Parameter继承自Variable），梯度可以直接传回Parameter
        Variable positionEmbedVar = positionEmbedding;
        
        // 创建位置索引: [0, 1, 2, ..., sequenceLength-1]
        float[] posIndices = new float[sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            posIndices[i] = i;
        }
        Variable posIndexVar = new Variable(NdArray.of(posIndices));
        posIndexVar.setRequireGrad(false);
        
        // IndexSelect: 从(maxPositions, embeddingDim)中选择，得到(sequenceLength, embeddingDim)
        Variable posEmbeds = positionEmbedVar.indexSelect(0, posIndexVar);
        
        // 扩展到batch维度: (sequenceLength, embeddingDim) -> (1, sequenceLength, embeddingDim) -> (batchSize, sequenceLength, embeddingDim)
        posEmbeds = posEmbeds.reshape(Shape.of(1, sequenceLength, embeddingDim));
        posEmbeds = posEmbeds.broadcastTo(Shape.of(batchSize, sequenceLength, embeddingDim));
        
        return posEmbeds;
    }
    
    // ==================== Getter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public int getEmbeddingDim() {
        return embeddingDim;
    }
    
    public int getMaxPositions() {
        return maxPositions;
    }
    
    public float getDropoutProb() {
        return dropoutProb;
    }
    
    public Parameter getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    public Parameter getPositionEmbedding() {
        return positionEmbedding;
    }
    
    @Override
    public String toString() {
        return String.format("GPT3TokenEmbedding{name='%s', vocabSize=%d, embeddingDim=%d, maxPositions=%d, dropout=%.3f}",
            name, vocabSize, embeddingDim, maxPositions, dropoutProb);
    }
}
