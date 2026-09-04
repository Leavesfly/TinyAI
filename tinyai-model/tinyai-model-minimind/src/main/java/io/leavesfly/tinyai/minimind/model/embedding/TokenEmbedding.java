package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

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
     * 权重共享：让 TokenEmbedding 使用 lmHead 的 weight（对标 Python）
     * <p>
     * Python: self.model.embed_tokens.weight = self.lm_head.weight
     * Embedding weight shape: [vocabSize, embeddingDim]
     * Linear weight shape:    [vocabSize, hiddenSize]（outFeatures=vocabSize, inFeatures=hiddenSize）
     * 两者形状一致（embeddingDim == hiddenSize），可以共享同一份参数。
     * <p>
     * 注意：除了替换字段引用，还必须同步替换参数注册表里的条目。
     * 否则注册表中会遗留构造时创建的孤儿参数：它不参与前向、永远拿不到梯度，
     * 却会被计入参数量统计并被序列化进每个 checkpoint（白白多占一份 vocab*hidden 内存）。
     *
     * @param lmHead LM Head 线性层
     */
    public void shareWeightWith(Linear lmHead) {
        // 替换为 lmHead 的 weight 参数
        Parameter shared = lmHead.getWeight();
        this.weight = shared;
        // 同步注册表，消除孤儿参数（共享后由 namedParameters 的按实例去重保证只出现一次）
        if (_parameters.containsKey("weight")) {
            _parameters.put("weight", shared);
        }
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
