package io.leavesfly.tinyai.deepseek.base;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * DeepSeek 通用 Token 嵌入基类
 *
 * 统一实现：
 * 1. Token Embedding - 将 token ID 映射到嵌入向量
 * 2. Position Embedding - 为每个位置添加位置信息
 * 3. Dropout - 嵌入层正则化
 *
 * 前向计算完全在 Variable 层面完成，方便自动求导和计算图管理。
 */
public class DeepSeekTokenEmbeddingBase extends Module {

    /** 词汇表大小 */
    protected final int vocabSize;
    /** 嵌入维度 */
    protected final int embeddingDim;
    /** 最大位置数 */
    protected final int maxPositions;
    /** Dropout 概率 */
    protected final float dropoutProb;
    /** 初始化范围（std） */
    protected final double initializerRange;

    /** Token 嵌入参数 [vocabSize, embeddingDim] */
    protected Parameter tokenEmbedding;
    /** 位置嵌入参数 [maxPositions, embeddingDim] */
    protected Parameter positionEmbedding;
    /** Dropout 层 */
    protected Dropout dropout;

    /**
     * 构造函数
     *
     * @param name             模块名称
     * @param vocabSize        词汇表大小
     * @param embeddingDim     嵌入维度
     * @param maxPositions     最大位置数
     * @param dropoutProb      Dropout 概率
     * @param initializerRange 权重初始化范围
     */
    protected DeepSeekTokenEmbeddingBase(String name,
                                         int vocabSize,
                                         int embeddingDim,
                                         int maxPositions,
                                         float dropoutProb,
                                         double initializerRange) {
        super(name);
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.maxPositions = maxPositions;
        this.dropoutProb = dropoutProb;
        this.initializerRange = initializerRange;
        initializeEmbeddings();
    }

    /**
     * 初始化嵌入层参数
     */
    protected void initializeEmbeddings() {
        // 1. 初始化 token 嵌入矩阵 [vocabSize, embeddingDim]
        NdArray tokenEmbedData = NdArray
                .likeRandomN(Shape.of(vocabSize, embeddingDim))
                .mulNum((float) initializerRange);
        tokenEmbedding = new Parameter(tokenEmbedData);
        registerParameter("token_embedding", tokenEmbedding);

        // 2. 初始化位置嵌入矩阵 [maxPositions, embeddingDim]
        NdArray positionEmbedData = NdArray
                .likeRandomN(Shape.of(maxPositions, embeddingDim))
                .mulNum((float) initializerRange);
        positionEmbedding = new Parameter(positionEmbedData);
        registerParameter("position_embedding", positionEmbedding);

        // 3. 初始化 Dropout 层
        dropout = new Dropout("embedding_dropout", dropoutProb);
        registerModule("dropout", dropout);
    }

    /**
     * 前向传播
     *
     * @param inputs 输入变量，inputs[0] 为 token ID 序列 [batch_size, seq_len]
     * @return 嵌入向量 [batch_size, seq_len, embeddingDim]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }

        Variable tokenIds = inputs[0];
        NdArray tokenData = tokenIds.getValue();

        // 验证输入维度
        if (tokenData.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                    String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s",
                            tokenData.getShape())
            );
        }

        int batchSize = tokenData.getShape().getDimension(0);
        int sequenceLength = tokenData.getShape().getDimension(1);

        // 验证序列长度
        if (sequenceLength > maxPositions) {
            throw new IllegalArgumentException(
                    String.format("序列长度(%d)超过最大位置数(%d)", sequenceLength, maxPositions)
            );
        }

        // ✅ 使用 Variable 层面的算子
        Variable tokenEmbedParam = new Variable(tokenEmbedding.data());
        Variable tokenEmbeds = getTokenEmbeddingsV2(tokenIds, tokenEmbedParam, batchSize, sequenceLength);

        Variable posEmbedParam = new Variable(positionEmbedding.data());
        Variable positionEmbeds = getPositionEmbeddingsV2(posEmbedParam, batchSize, sequenceLength);

        // 合并嵌入并应用 dropout
        Variable combined = tokenEmbeds.add(positionEmbeds);
        return dropout.forward(combined);
    }

    /**
     * 获取 token 嵌入向量 (使用 Variable 算子)
     *
     * @param tokenIds       token ID 变量 [batch_size, seq_len]
     * @param tokenEmbedParam token 嵌入参数 [vocabSize, embeddingDim]
     * @param batchSize      批次大小
     * @param sequenceLength 序列长度
     * @return token 嵌入变量 [batch_size, seq_len, embeddingDim]
     */
    protected Variable getTokenEmbeddingsV2(Variable tokenIds,
                                            Variable tokenEmbedParam,
                                            int batchSize,
                                            int sequenceLength) {
        // 1. 展平 tokenIds: [batch_size, seq_len] -> [batch_size * seq_len]
        Variable flatIds = tokenIds.reshape(Shape.of(batchSize * sequenceLength));

        // 2. 使用 indexSelect 选择嵌入: [batch_size * seq_len, embeddingDim]
        Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatIds);

        // 3. Reshape 回 3D: [batch_size, seq_len, embeddingDim]
        return flatEmbeds.reshape(Shape.of(batchSize, sequenceLength, embeddingDim));
    }

    /**
     * 获取位置嵌入向量 (使用 Variable 算子)
     *
     * @param posEmbedParam  位置嵌入参数 [maxPositions, embeddingDim]
     * @param batchSize      批次大小（当前仅用于接口统一，可不参与计算）
     * @param sequenceLength 序列长度
     * @return 位置嵌入变量 [1, seq_len, embeddingDim] - 依赖广播机制自动扩展
     */
    protected Variable getPositionEmbeddingsV2(Variable posEmbedParam,
                                               int batchSize,
                                               int sequenceLength) {
        // 1. 创建位置索引 [0, 1, ..., sequenceLength - 1]
        float[] posIndices = new float[sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            posIndices[i] = i;
        }

        Variable posIds = new Variable(NdArray.of(posIndices));
        posIds.setRequireGrad(false);

        // 2. 使用 indexSelect 选择位置嵌入: [sequenceLength, embeddingDim]
        Variable posEmbeds = posEmbedParam.indexSelect(0, posIds);

        // 3. Reshape 到 [1, seq_len, embeddingDim]，依赖 add 的广播机制自动扩展 batch 维
        return posEmbeds.reshape(Shape.of(1, sequenceLength, embeddingDim));
    }

    /**
     * 获取 token 嵌入参数
     */
    public Parameter getTokenEmbedding() {
        return tokenEmbedding;
    }

    /**
     * 获取位置嵌入参数
     */
    public Parameter getPositionEmbedding() {
        return positionEmbedding;
    }
}
