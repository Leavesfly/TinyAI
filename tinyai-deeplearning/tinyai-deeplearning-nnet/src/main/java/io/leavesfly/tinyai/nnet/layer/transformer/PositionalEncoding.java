package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的PositionalEncoding层
 * <p>
 * 位置编码为序列数据添加位置信息，是Transformer架构的关键组件。
 * <p>
 * 使用正弦和余弦函数生成位置编码：
 * PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 * <p>
 * 其中：
 * - pos: 位置索引
 * - i: 维度索引
 * - d_model: 模型维度
 * <p>
 * 特性：
 * - 编码值在[-1, 1]范围内
 * - 可以处理任意长度的序列
 * - 不同位置的编码是唯一的
 * - 相对位置关系可以通过线性变换学习
 *
 * @author leavesfly
 * @version 2.0
 */
public class PositionalEncoding extends Module {

    private final int dModel;
    private final int maxLen;
    private final float dropout;

    /** 缓存的位置编码 Variable，避免每次 forward 都创建新对象 */
    private transient Variable cachedPeVar;

    /**
     * 构造函数
     *
     * @param name    层名称
     * @param dModel  模型维度
     * @param maxLen  支持的最大序列长度
     * @param dropout dropout比率
     */
    public PositionalEncoding(String name, int dModel, int maxLen, float dropout) {
        super(name);
        this.dModel = dModel;
        this.maxLen = maxLen;
        this.dropout = dropout;

        // 预计算位置编码并注册为缓冲区
        NdArray pe = createPositionalEncoding();
        registerBuffer("pe", pe);

        // 缓存位置编码 Variable（不参与梯度计算）
        this.cachedPeVar = new Variable(pe);
        this.cachedPeVar.setRequireGrad(false);

        init();
    }

    /**
     * 构造函数（默认参数）
     *
     * @param name   层名称
     * @param dModel 模型维度
     */
    public PositionalEncoding(String name, int dModel) {
        this(name, dModel, 5000, 0.1f);
    }

    /**
     * 构造函数
     *
     * @param name   层名称
     * @param dModel 模型维度
     * @param maxLen 支持的最大序列长度
     */
    public PositionalEncoding(String name, int dModel, int maxLen) {
        this(name, dModel, maxLen, 0.1f);
    }

    /**
     * 创建位置编码矩阵
     * <p>
     * 形状: (maxLen, dModel)
     *
     * @return 位置编码矩阵
     */
    private NdArray createPositionalEncoding() {
        float[] peData = new float[maxLen * dModel];

        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < dModel / 2; i++) {
                double angle = pos / Math.pow(10000.0, (2.0 * i) / dModel);

                // 偶数维度使用sin
                peData[pos * dModel + 2 * i] = (float) Math.sin(angle);

                // 奇数维度使用cos
                if (2 * i + 1 < dModel) {
                    peData[pos * dModel + 2 * i + 1] = (float) Math.cos(angle);
                }
            }
        }

        return NdArray.of(peData, Shape.of(maxLen, dModel));
    }

    @Override
    public void resetParameters() {
        // 位置编码是固定的，不需要参数初始化
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        // 使用 Variable 的形状方法进行验证
        int ndim = x.ndim();
        if (ndim != 3) {
            throw new IllegalArgumentException(
                    String.format("Expected 3D input (batch, seq_len, d_model), but got %dD", ndim));
        }

        int seqLen = x.size(1);
        int modelDim = x.size(2);

        if (modelDim != dModel) {
            throw new IllegalArgumentException(
                    String.format("Expected d_model=%d, but got %d", dModel, modelDim));
        }

        if (seqLen > maxLen) {
            throw new IllegalArgumentException(
                    String.format("Sequence length %d exceeds maximum length %d", seqLen, maxLen));
        }

        // 使用 Variable 的 sliceRange 方法切片位置编码，保持计算图连通
        // cachedPeVar 形状: (maxLen, dModel) -> 切片为 (seqLen, dModel)
        Variable peSlice = cachedPeVar.sliceRange(0, 0, seqLen);

        // 添加位置编码到输入（广播到 batch 维度）
        Variable output = x.add(peSlice);

        // 应用dropout（训练模式）
        if (isTraining() && dropout > 0) {
            // 注意：这里需要实现dropout，暂时跳过
            // output = applyDropout(output, dropout);
        }

        return output;
    }

    /**
     * 获取位置编码矩阵
     *
     * @return 位置编码矩阵
     */
    public NdArray getPositionalEncoding() {
        return getBuffer("pe");
    }

    public int getDModel() {
        return dModel;
    }

    public int getMaxLen() {
        return maxLen;
    }

    public float getDropout() {
        return dropout;
    }

    @Override
    public String toString() {
        return "PositionalEncoding{" +
                "name='" + name + '\'' +
                ", dModel=" + dModel +
                ", maxLen=" + maxLen +
                ", dropout=" + dropout +
                '}';
    }
}
