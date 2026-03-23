package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

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
        
        // tokenIds shape: [batch_size, seq_len] 或 [batch_size, seq_len, 1]
        int[] shape = tokenIds.getShape().getShapeDims();
        int batchSize = shape[0];
        int seqLen = shape.length > 1 ? shape[1] : 1;

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

        // 使用 Variable 层面的 indexSelect 进行嵌入查找
        // 将 tokenIds reshape 为一维: [batch_size * seq_len]
        Variable flattenedIds = tokenIds.reshape(Shape.of(batchSize * seqLen));
        
        // 使用 indexSelect 从嵌入矩阵中选择对应行
        // weight.data() shape: [vocabSize, embeddingDim]
        Variable weightVar = new Variable(weight.data());
        Variable embedded = weightVar.indexSelect(0, flattenedIds);
        
        // reshape 回原始形状: [batch_size, seq_len, embeddingDim]
        Variable output = embedded.reshape(Shape.of(batchSize, seqLen, embeddingDim));

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
