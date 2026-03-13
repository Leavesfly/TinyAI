package io.leavesfly.tinyai.deepseek.base;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * DeepSeek 通用 Token 嵌入基类（对标 DeepSeek-V3/R1 论文架构）
 *
 * 统一实现：
 * 1. Token Embedding - 将 token ID 映射到嵌入向量
 * 2. Dropout - 嵌入层正则化
 *
 * 注意：位置信息由注意力层中的 RoPE（旋转位置编码）提供，
 * 嵌入层不包含学习式位置嵌入，这与 DeepSeek-V3/R1 论文一致。
 *
 * 输出 = Dropout(TokenEmbed)
 *
 * 前向计算完全在 Variable 层面完成，方便自动求导和计算图管理。
 */
public class DeepSeekTokenEmbeddingBase extends Module {

    /** 词汇表大小 */
    protected final int vocabSize;
    /** 嵌入维度 */
    protected final int embeddingDim;
    /** 最大位置数（用于序列长度校验） */
    protected final int maxPositions;
    /** Dropout 概率 */
    protected final float dropoutProb;
    /** 初始化范围（std） */
    protected final double initializerRange;

    /** Token 嵌入参数 [vocabSize, embeddingDim] */
    protected Parameter tokenEmbedding;
    /** Dropout 层 */
    protected Dropout dropout;

    /**
     * 构造函数
     *
     * @param name             模块名称
     * @param vocabSize        词汇表大小
     * @param embeddingDim     嵌入维度
     * @param maxPositions     最大位置数（用于序列长度校验）
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

        // 2. 初始化 Dropout 层
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

        // Token 嵌入查找
        Variable tokenEmbedParam = new Variable(tokenEmbedding.data());
        Variable flatTokenIds = tokenIds.reshape(Shape.of(batchSize * sequenceLength));
        Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatTokenIds);
        Variable tokenEmbeds = flatEmbeds.reshape(Shape.of(batchSize, sequenceLength, embeddingDim));

        // 应用 Dropout（位置信息由注意力层的 RoPE 提供）
        return dropout.forward(tokenEmbeds);
    }

    /**
     * 获取 token 嵌入参数（供 MTP 头等共享使用）
     */
    public Parameter getTokenEmbedding() {
        return tokenEmbedding;
    }

    /**
     * 获取嵌入维度
     */
    public int getEmbeddingDim() {
        return embeddingDim;
    }

    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 获取最大位置数
     */
    public int getMaxPositions() {
        return maxPositions;
    }
}
