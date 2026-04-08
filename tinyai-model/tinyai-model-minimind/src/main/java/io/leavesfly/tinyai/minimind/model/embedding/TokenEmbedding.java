package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;

/**
 * Token 嵌入层
 * <p>
 * 将 Token IDs 转换为密集的向量表示。
 * 实现词汇表查找功能,支持权重共享(与 LM Head 共享嵌入矩阵)。
 * </p>
 *
 * @author TinyAI Team
 * @version 1.0
 */
public class TokenEmbedding extends Module {

    private final int vocabSize;
    private final int embeddingDim;
    private final float initStd;
    private Parameter weight;

    /**
     * 构造 Token 嵌入层
     *
     * @param vocabSize    词汇表大小
     * @param embeddingDim 嵌入维度
     */
    public TokenEmbedding(int vocabSize, int embeddingDim) {
        this(vocabSize, embeddingDim, 0.02f);
    }

    /**
     * 构造 Token 嵌入层
     *
     * @param vocabSize    词汇表大小
     * @param embeddingDim 嵌入维度
     * @param initStd      初始化标准差
     */
    public TokenEmbedding(int vocabSize, int embeddingDim, float initStd) {
        super("TokenEmbedding");
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.initStd = initStd;

        // 注册嵌入矩阵参数: [vocabSize, embeddingDim]
        NdArray embeddingMatrix = NdArray.likeRandomN(Shape.of(vocabSize, embeddingDim));
        // 缩放到合适的标准差
        embeddingMatrix = embeddingMatrix.mulNum(initStd);
        this.weight = registerParameter("weight", new Parameter(embeddingMatrix, true));
    }

    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("TokenEmbedding requires at least one input");
        }

        Variable tokenIds = inputs[0];

        // Token ID 边界检查
        float[] tokenIdsData = tokenIds.getValue().getArray();
        for (int i = 0; i < tokenIdsData.length; i++) {
            int tokenId = (int) tokenIdsData[i];
            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IllegalArgumentException(
                    String.format("Token ID %d at position %d is out of range [0, %d)", 
                        tokenId, i, vocabSize)
                );
            }
        }

        // 使用 Gather 操作保持计算图连通性，支持自动微分
        // Gather: weight[tokenIds] -> output，反向传播通过 scatter add 回传梯度
        Variable output = weight.gather(tokenIds);

        return output;
    }



    /**
     * 获取嵌入权重参数(用于权重共享)
     *
     * @return 嵌入权重参数
     */
    public Parameter getWeight() {
        return weight;
    }

    @Override
    public String extraRepr() {
        return String.format("vocabSize=%d, embeddingDim=%d", vocabSize, embeddingDim);
    }
}
