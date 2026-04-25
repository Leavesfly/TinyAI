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

    // 预计算的 cos/sin 缓存（所有 forward 共享，只建一次）
    private final io.leavesfly.tinyai.ndarr.NdArray cosCache;
    private final io.leavesfly.tinyai.ndarr.NdArray sinCache;

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

        // 通过一个临时 RotaryEmbedding 实例完成 cos/sin 预计算，然后只保留缓存数组
        // 后续每次 forward 都 new 一个新的 RotaryEmbedding，共享这份只读缓存
        // 这样可以避免同一个 Function 实例被 Q、K 多次 call 时 inputShape/startPos 字段互相覆盖的 bug
        RotaryEmbedding bootstrap = new RotaryEmbedding(dim, maxSeqLen, theta);
        this.cosCache = bootstrap.getCosCache();
        this.sinCache = bootstrap.getSinCache();
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

        // 每次 forward 都新建一个 RotaryEmbedding Function 实例（共享 cos/sin 缓存）
        // 关键点：同一 Attention 中 Q 和 K 会先后各调用一次 forward，
        // 若复用同一个 Function 实例，其 inputShape/startPos 字段会被第二次调用覆盖，
        // 导致反向传播时 Q 分支拿到 K 的形状从而报 "数据长度与形状大小不匹配"。
        // 每次 new 一个实例可以让 Q、K 拥有各自独立的计算图节点，互不干扰。
        RotaryEmbedding ropeFunction = new RotaryEmbedding(dim, maxSeqLen, theta, cosCache, sinCache);

        // 提取 startPos（如果提供）
        if (inputs.length > 1) {
            return ropeFunction.call(x, inputs[1]);
        } else {
            // 默认 startPos = 0
            Variable startPosVar = new Variable(NdArray.of(new float[]{0}));
            startPosVar.setRequireGrad(false);
            return ropeFunction.call(x, startPosVar);
        }
    }

    @Override
    public String extraRepr() {
        return String.format("dim=%d, maxSeqLen=%d, theta=%.1f", dim, maxSeqLen, theta);
    }
}
