package io.leavesfly.tinyai.nnet.layer.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * 词嵌入层：将离散索引映射为连续向量表示。
 * <p>
 * 仅支持1D/2D索引输入：
 * - (seq_len,)
 * - (batch_size, seq_len)
 * 输出形状：
 * - 1D输入 -> (seq_len, embedding_dim)
 * - 2D输入 -> (batch_size, seq_len, embedding_dim)，若 seq_len==1 则压缩为 (batch_size, embedding_dim)
 */
public class Embedding extends Module {

    private final int numEmbeddings;
    private final int embeddingDim;
    private final Parameter weight;

    public Embedding(String name, int numEmbeddings, int embeddingDim) {
        super(name);
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        NdArray weightData = NdArray.likeRandomN(Shape.of(numEmbeddings, embeddingDim));
        this.weight = registerParameter("weight", new Parameter(weightData));

        init();
    }

    @Override
    public void resetParameters() {
        // 使用较小方差的正态分布初始化嵌入
        Initializers.normal(weight.data(), 0f, 0.01f);
    }

    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Embedding requires one input indices Variable");
        }
        Variable indices = inputs[0];
        int dim = indices.ndim();

        if (dim != 1 && dim != 2) {
            throw new IllegalArgumentException(
                    String.format("Embedding only supports 1D or 2D index tensors, got %dD", dim));
        }

        // 验证索引范围
        float[] idxData = indices.getValue().getArray();
        for (float idx : idxData) {
            int index = (int) idx;
            if (index < 0 || index >= numEmbeddings) {
                throw new IndexOutOfBoundsException(
                        String.format("Embedding index out of range: %d (expected 0 <= index < %d)",
                                index, numEmbeddings));
            }
        }

        // 使用 Gather 操作保持计算图连通性，支持自动微分
        // Gather: weight[indices] -> output，反向传播通过 scatter add 回传梯度
        Variable result = weight.gather(indices);

        // 2D 输入且 seqLen==1 时压缩为 (batchSize, embeddingDim)
        if (dim == 2 && indices.size(1) == 1) {
            int batchSize = indices.size(0);
            result = result.reshape(Shape.of(batchSize, embeddingDim));
        }

        return result;
    }

    public Parameter getWeight() {
        return weight;
    }

    public int getNumEmbeddings() {
        return numEmbeddings;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    @Override
    public String toString() {
        return "Embedding{" +
                "name='" + name + '\'' +
                ", numEmbeddings=" + numEmbeddings +
                ", embeddingDim=" + embeddingDim +
                '}';
    }
}

