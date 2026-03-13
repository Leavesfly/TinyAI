package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 旋转位置编码 (Rotary Position Embedding, RoPE) Function
 * <p>
 * 实现 RoPE 的前向和反向传播，使其成为完整的可微分算子
 * 
 * RoPE 通过旋转矩阵对向量进行位置编码：
 * - 将向量的每对相邻元素视为2D平面上的点
 * - 根据位置和频率进行旋转
 * 
 * 旋转公式：
 * x_rotated[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
 * x_rotated[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
 * 
 * 其中 θ = position * freq[i]
 * 
 * @author TinyAI Team
 * @version 1.0
 */
public class RotaryEmbedding extends Function {

    private final int dim;           // 特征维度(必须是偶数)
    private final int maxSeqLen;     // 最大序列长度
    private final float theta;       // 频率基数(默认 10000)
    
    // 预计算的 cos 和 sin 缓存
    private NdArray cosCache;
    private NdArray sinCache;
    
    // 缓存用于反向传播的信息
    private Shape inputShape;
    private int startPos;

    /**
     * 构造函数
     *
     * @param dim        特征维度(必须是偶数)
     * @param maxSeqLen  最大序列长度
     * @param theta      频率基数(默认 10000)
     */
    public RotaryEmbedding(int dim, int maxSeqLen, float theta) {
        if (dim % 2 != 0) {
            throw new IllegalArgumentException("dim must be even, got: " + dim);
        }
        
        this.dim = dim;
        this.maxSeqLen = maxSeqLen;
        this.theta = theta;
        
        // 预计算 cos 和 sin 值
        precomputeFreqsCis();
    }

    /**
     * 默认构造函数(theta=10000)
     */
    public RotaryEmbedding(int dim, int maxSeqLen) {
        this(dim, maxSeqLen, 10000.0f);
    }

    /**
     * 预计算频率的 cos 和 sin 值
     */
    private void precomputeFreqsCis() {
        int halfDim = dim / 2;
        
        // 计算频率: freq[i] = 1.0 / (theta ^ (2i / dim))
        float[] freqs = new float[halfDim];
        for (int i = 0; i < halfDim; i++) {
            freqs[i] = (float) (1.0 / Math.pow(theta, (2.0 * i) / dim));
        }
        
        // 为每个位置计算 cos 和 sin
        float[] cosData = new float[maxSeqLen * halfDim];
        float[] sinData = new float[maxSeqLen * halfDim];
        
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < halfDim; i++) {
                float angle = pos * freqs[i];
                cosData[pos * halfDim + i] = (float) Math.cos(angle);
                sinData[pos * halfDim + i] = (float) Math.sin(angle);
            }
        }
        
        // 缓存为 NdArray: [maxSeqLen, halfDim]
        this.cosCache = NdArray.of(cosData, Shape.of(maxSeqLen, halfDim));
        this.sinCache = NdArray.of(sinData, Shape.of(maxSeqLen, halfDim));
    }

    /**
     * 前向传播
     * 
     * @param inputs inputs[0]: 输入张量 [batch_size, seq_len, dim] 或 [batch_size, num_heads, seq_len, head_dim]
     *               inputs[1]: 起始位置 scalar (可选，默认0)
     * @return 应用 RoPE 后的张量
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        this.startPos = inputs.length > 1 ? (int) inputs[1].getNumber().floatValue() : 0;
        this.inputShape = x.getShape();
        
        int[] shape = inputShape.getShapeDims();
        
        // 支持两种输入形状
        boolean is3D = (shape.length == 3);  // [B, L, D]
        boolean is4D = (shape.length == 4);  // [B, H, L, D]
        
        if (!is3D && !is4D) {
            throw new IllegalArgumentException(
                "Input must be 3D [B,L,D] or 4D [B,H,L,D], got: " + Arrays.toString(shape)
            );
        }
        
        int batchSize, seqLen, featureDim, numHeads = 1;
        
        if (is3D) {
            batchSize = shape[0];
            seqLen = shape[1];
            featureDim = shape[2];
        } else {  // is4D
            batchSize = shape[0];
            numHeads = shape[1];
            seqLen = shape[2];
            featureDim = shape[3];
        }
        
        if (featureDim != dim) {
            throw new IllegalArgumentException(
                "Feature dim mismatch: expected " + dim + ", got " + featureDim
            );
        }
        
        if (startPos + seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                "Sequence too long: startPos=" + startPos + ", seqLen=" + seqLen + 
                ", maxSeqLen=" + maxSeqLen
            );
        }
        
        return applyRotation(x, startPos, is3D, batchSize, numHeads, seqLen);
    }

    /**
     * 应用旋转变换
     */
    private NdArray applyRotation(NdArray x, int startPos, boolean is3D, 
                                  int batchSize, int numHeads, int seqLen) {
        float[] xData = x.getArray();
        float[] cosData = cosCache.getArray();
        float[] sinData = sinCache.getArray();
        
        int halfDim = dim / 2;
        float[] output = new float[xData.length];
        
        if (is3D) {
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int pos = startPos + s;
                    applyRotationToVector(xData, output, b * seqLen * dim + s * dim,
                                        cosData, sinData, pos * halfDim, halfDim);
                }
            }
            return NdArray.of(output, Shape.of(batchSize, seqLen, dim));
        } else {  // is4D
            for (int b = 0; b < batchSize; b++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int s = 0; s < seqLen; s++) {
                        int pos = startPos + s;
                        int offset = ((b * numHeads + h) * seqLen + s) * dim;
                        applyRotationToVector(xData, output, offset,
                                            cosData, sinData, pos * halfDim, halfDim);
                    }
                }
            }
            return NdArray.of(output, Shape.of(batchSize, numHeads, seqLen, dim));
        }
    }

    /**
     * 对单个向量应用旋转
     */
    private void applyRotationToVector(float[] xData, float[] output, int xOffset,
                                      float[] cosData, float[] sinData, int freqOffset, int halfDim) {
        for (int i = 0; i < halfDim; i++) {
            float x0 = xData[xOffset + 2 * i];
            float x1 = xData[xOffset + 2 * i + 1];
            float cos = cosData[freqOffset + i];
            float sin = sinData[freqOffset + i];
            
            output[xOffset + 2 * i]     = x0 * cos - x1 * sin;
            output[xOffset + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    /**
     * 反向传播
     * 
     * RoPE 的反向传播是旋转的逆操作：
     * 如果前向是旋转 θ，反向就是旋转 -θ
     * 
     * @param yGrad 输出梯度
     * @return 输入梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        int[] shape = inputShape.getShapeDims();
        boolean is3D = (shape.length == 3);
        
        int batchSize, seqLen, numHeads = 1;
        
        if (is3D) {
            batchSize = shape[0];
            seqLen = shape[1];
        } else {
            batchSize = shape[0];
            numHeads = shape[1];
            seqLen = shape[2];
        }
        
        // 反向旋转：使用 -θ（即交换 sin 的符号）
        NdArray xGrad = applyInverseRotation(yGrad, startPos, is3D, batchSize, numHeads, seqLen);
        
        // 根据实际输入数量返回对应数量的梯度
        if (inputs.length == 1) {
            return Collections.singletonList(xGrad);
        }
        return Arrays.asList(xGrad, null);
    }

    /**
     * 应用逆旋转（用于反向传播）
     */
    private NdArray applyInverseRotation(NdArray yGrad, int startPos, boolean is3D,
                                         int batchSize, int numHeads, int seqLen) {
        float[] gradData = yGrad.getArray();
        float[] cosData = cosCache.getArray();
        float[] sinData = sinCache.getArray();
        
        int halfDim = dim / 2;
        float[] xGrad = new float[gradData.length];
        
        if (is3D) {
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int pos = startPos + s;
                    applyInverseRotationToVector(gradData, xGrad, b * seqLen * dim + s * dim,
                                                cosData, sinData, pos * halfDim, halfDim);
                }
            }
            return NdArray.of(xGrad, Shape.of(batchSize, seqLen, dim));
        } else {
            for (int b = 0; b < batchSize; b++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int s = 0; s < seqLen; s++) {
                        int pos = startPos + s;
                        int offset = ((b * numHeads + h) * seqLen + s) * dim;
                        applyInverseRotationToVector(gradData, xGrad, offset,
                                                    cosData, sinData, pos * halfDim, halfDim);
                    }
                }
            }
            return NdArray.of(xGrad, Shape.of(batchSize, numHeads, seqLen, dim));
        }
    }

    /**
     * 对单个向量应用逆旋转
     * 逆旋转公式（旋转 -θ）：
     * x[2i]   = x_rot[2i] * cos + x_rot[2i+1] * sin
     * x[2i+1] = -x_rot[2i] * sin + x_rot[2i+1] * cos
     */
    private void applyInverseRotationToVector(float[] gradData, float[] xGrad, int offset,
                                             float[] cosData, float[] sinData, int freqOffset, int halfDim) {
        for (int i = 0; i < halfDim; i++) {
            float grad0 = gradData[offset + 2 * i];
            float grad1 = gradData[offset + 2 * i + 1];
            float cos = cosData[freqOffset + i];
            float sin = sinData[freqOffset + i];
            
            // 逆旋转
            xGrad[offset + 2 * i]     = grad0 * cos + grad1 * sin;
            xGrad[offset + 2 * i + 1] = -grad0 * sin + grad1 * cos;
        }
    }

    @Override
    public int requireInputNum() {
        return -1;  // 返回-1表示支持可变参数（1个或2个），第二个参数 startPos 是可选的
    }
}
