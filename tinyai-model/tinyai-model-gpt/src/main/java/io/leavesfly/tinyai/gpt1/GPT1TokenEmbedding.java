package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * GPT-1 Token嵌入层
 */
public class GPT1TokenEmbedding extends Module {
    
    private final int vocabSize;
    private final int embeddingDim;
    private final int maxPositions;
    private final float dropoutProb;
    
    private Parameter tokenEmbedding;
    private Parameter positionEmbedding;
    private Dropout dropout;
    
    public GPT1TokenEmbedding(String name, GPT1Config config) {
        super(name);
        this.vocabSize = config.getVocabSize();
        this.embeddingDim = config.getNEmbd();
        this.maxPositions = config.getNPositions();
        this.dropoutProb = (float) config.getEmbdPdrop();
        initializeParameters(config);
    }
    
    private void initializeParameters(GPT1Config config) {
        NdArray tokenEmbedData = NdArray.likeRandomN(Shape.of(vocabSize, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        tokenEmbedding = new Parameter(tokenEmbedData);
        registerParameter("token_embedding", tokenEmbedding);
        
        NdArray positionEmbedData = NdArray.likeRandomN(Shape.of(maxPositions, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        positionEmbedding = new Parameter(positionEmbedData);
        registerParameter("position_embedding", positionEmbedding);
        
        dropout = new Dropout("embedding_dropout", dropoutProb);
        registerModule("dropout", dropout);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable tokenIds = inputs[0];
        NdArray tokenData = tokenIds.getValue();
        
        int batchSize = tokenData.getShape().getDimension(0);
        int sequenceLength = tokenData.getShape().getDimension(1);
        
        if (sequenceLength > maxPositions) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", sequenceLength, maxPositions));
        }
        
        Variable tokenEmbeds = getTokenEmbeddings(tokenIds, batchSize, sequenceLength);
        Variable positionEmbeds = getPositionEmbeddings(sequenceLength, batchSize);
        return dropout.forward(tokenEmbeds.add(positionEmbeds));
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
    
    public Parameter getTokenEmbedding() { return tokenEmbedding; }
    public Parameter getPositionEmbedding() { return positionEmbedding; }
}
