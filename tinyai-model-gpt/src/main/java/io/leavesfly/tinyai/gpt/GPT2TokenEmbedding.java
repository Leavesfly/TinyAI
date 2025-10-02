package io.leavesfly.tinyai.gpt;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2 Token嵌入层实现
 * <p>
 * 将token ID转换为高维向量表示
 * 包含两种嵌入：
 * 1. Token嵌入：将词汇ID映射到向量
 * 2. 位置嵌入：为每个位置学习位置向量（可选，也可以使用PositionalEncoding）
 */
public class GPT2TokenEmbedding extends Layer {

    private Parameter tokenEmbedding;      // token嵌入权重
    private Parameter positionEmbedding;   // 位置嵌入权重（可选）
    private int vocabSize;                 // 词汇表大小
    private int dModel;                    // 嵌入维度
    private int maxSeqLength;              // 最大序列长度
    private boolean usePositionEmbedding;  // 是否使用学习的位置嵌入
    private double dropoutRate;            // dropout比率

    /**
     * 构造GPT-2 Token嵌入层
     *
     * @param name                 层名称
     * @param vocabSize            词汇表大小
     * @param dModel               嵌入维度
     * @param maxSeqLength         最大序列长度
     * @param usePositionEmbedding 是否使用学习的位置嵌入
     * @param dropoutRate          dropout比率
     */
    public GPT2TokenEmbedding(String name, int vocabSize, int dModel, int maxSeqLength,
                              boolean usePositionEmbedding, double dropoutRate) {
        super(name, Shape.of(-1, maxSeqLength), Shape.of(-1, maxSeqLength, dModel));

        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.maxSeqLength = maxSeqLength;
        this.usePositionEmbedding = usePositionEmbedding;
        this.dropoutRate = dropoutRate;

        init();
    }

    /**
     * 使用默认参数的构造函数
     */
    public GPT2TokenEmbedding(String name, int vocabSize, int dModel, int maxSeqLength) {
        this(name, vocabSize, dModel, maxSeqLength, true, 0.1);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化token嵌入矩阵 (vocabSize, dModel)
            // 使用正态分布初始化，标准差为0.02
            tokenEmbedding = new Parameter(NdArray.likeRandomN(Shape.of(vocabSize, dModel)).mulNum(0.02f));
            tokenEmbedding.setName(name + "_token_embedding");
            addParam(tokenEmbedding.getName(), tokenEmbedding);

            // 如果使用位置嵌入，初始化位置嵌入矩阵
            if (usePositionEmbedding) {
                positionEmbedding = new Parameter(NdArray.likeRandomN(Shape.of(maxSeqLength, dModel)).mulNum(0.02f));
                positionEmbedding.setName(name + "_position_embedding");
                addParam(positionEmbedding.getName(), positionEmbedding);
            }

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batch_size, seq_len)
        NdArray tokenData = tokenIds.getValue();

        int batchSize = tokenData.getShape().getDimension(0);
        int seqLen = tokenData.getShape().getDimension(1);

        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                    String.format("Sequence length %d exceeds maximum length %d", seqLen, maxSeqLength)
            );
        }

        // 1. 获取token嵌入
        Variable tokenEmbeds = getTokenEmbeddings(tokenData, batchSize, seqLen);

        // 2. 如果使用位置嵌入，添加位置信息
        if (usePositionEmbedding) {
            Variable posEmbeds = getPositionEmbeddings(seqLen, batchSize);
            tokenEmbeds = tokenEmbeds.add(posEmbeds);
        }

        // 3. 应用dropout（简化版本）
        if (dropoutRate > 0.0) {
            // 在实际实现中需要考虑训练/推理模式
            tokenEmbeds = applyDropout(tokenEmbeds);
        }

        return tokenEmbeds;
    }

    /**
     * 获取token嵌入
     */
    private Variable getTokenEmbeddings(NdArray tokenIds, int batchSize, int seqLen) {
        NdArray embeddings = NdArray.of(Shape.of(batchSize, seqLen, dModel));

        // 对每个token ID查找对应的嵌入向量
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = (int) tokenIds.get(b, s);

                // 检查token ID是否在有效范围内
                if (tokenId < 0 || tokenId >= vocabSize) {
                    throw new IllegalArgumentException(
                            String.format("Token ID %d is out of vocabulary range [0, %d)", tokenId, vocabSize)
                    );
                }

                // 复制对应的嵌入向量
                for (int d = 0; d < dModel; d++) {
                    float embeddingValue = tokenEmbedding.getValue().get(tokenId, d);
                    embeddings.set(embeddingValue, b, s, d);
                }
            }
        }

        return new Variable(embeddings);
    }

    /**
     * 获取位置嵌入
     */
    private Variable getPositionEmbeddings(int seqLen, int batchSize) {
        NdArray posEmbeds = NdArray.of(Shape.of(batchSize, seqLen, dModel));

        // 为每个位置添加位置嵌入
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int d = 0; d < dModel; d++) {
                    float posValue = positionEmbedding.getValue().get(s, d);
                    posEmbeds.set(posValue, b, s, d);
                }
            }
        }

        return new Variable(posEmbeds);
    }

    /**
     * 应用dropout（简化版本）
     */
    private Variable applyDropout(Variable input) {
        // 在实际实现中需要生成随机掩码并考虑训练/推理模式
        return input;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    /**
     * 获取token嵌入参数
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

    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 获取嵌入维度
     */
    public int getDModel() {
        return dModel;
    }

    /**
     * 获取最大序列长度
     */
    public int getMaxSeqLength() {
        return maxSeqLength;
    }

    /**
     * 是否使用位置嵌入
     */
    public boolean isUsePositionEmbedding() {
        return usePositionEmbedding;
    }

    /**
     * 获取dropout比率
     */
    public double getDropoutRate() {
        return dropoutRate;
    }
}