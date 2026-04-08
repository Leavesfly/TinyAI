package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.RotaryEmbedding;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * 旋转位置编码 (Rotary Position Embedding, RoPE)
 * <p>
 * RoPE 通过旋转矩阵对 Q、K 向量进行位置编码，使模型能够捕捉相对位置信息。
 * 相比传统的绝对位置编码，RoPE 具有更好的长度外推能力。
 * </p>
 *
 * @author TinyAI Team
 * @version 1.0
 */
public class RotaryPositionEmbedding extends Module {

    private final int dim;           // 特征维度(必须是偶数)
    private final int maxSeqLen;     // 最大序列长度
    private final float theta;       // 频率基数(默认 10000)
    
    // 使用 RotaryEmbedding Function 实现完整的自动微分
    private final RotaryEmbedding ropeFunction;

    /**
     * 构造 RoPE 位置编码
     *
     * @param dim        特征维度(必须是偶数)
     * @param maxSeqLen  最大序列长度
     * @param theta      频率基数(默认 10000)
     */
    public RotaryPositionEmbedding(int dim, int maxSeqLen, float theta) {
        super("RoPE");
        
        if (dim % 2 != 0) {
            throw new IllegalArgumentException("dim must be even, got: " + dim);
        }
        
        this.dim = dim;
        this.maxSeqLen = maxSeqLen;
        this.theta = theta;
        
        // 创建 RotaryEmbedding Function（支持完整的自动微分）
        this.ropeFunction = new RotaryEmbedding(dim, maxSeqLen, theta);
    }

    /**
     * 构造 RoPE 位置编码(使用默认 theta=10000)
     *
     * @param dim        特征维度
     * @param maxSeqLen  最大序列长度
     */
    public RotaryPositionEmbedding(int dim, int maxSeqLen) {
        this(dim, maxSeqLen, 10000.0f);
    }



    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("RoPE requires at least one input");
        }
        
        Variable x = inputs[0];
        
        // 提取 startPos（如果提供）
        if (inputs.length > 1) {
            // 使用 RotaryEmbedding Function 的完整自动微分实现
            return ropeFunction.call(x, inputs[1]);
        } else {
            // 默认 startPos = 0
            Variable startPosVar = new Variable(NdArray.of(new float[]{0}));
            return ropeFunction.call(x, startPosVar);
        }
    }

    @Override
    public String extraRepr() {
        return String.format("dim=%d, maxSeqLen=%d, theta=%.1f", dim, maxSeqLen, theta);
    }
}
