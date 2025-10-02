package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * 旋转位置编码层（Rotary Position Embedding, RoPE）
 * 
 * RoPE是一种相对位置编码方法，通过旋转变换将位置信息直接编码到注意力计算中。
 * 相比传统的绝对位置编码，RoPE能够更好地处理任意长度的序列。
 * 
 * 核心思想：
 * 1. 将注意力向量分成多个二维子空间
 * 2. 在每个子空间中应用旋转变换，旋转角度与位置相关
 * 3. 使得相对位置信息能够自然地在注意力计算中体现
 * 
 * @author 山泽
 * @version 1.0
 */
public class RotaryPositionalEmbedding extends Layer {
    
    /**
     * 特征维度
     */
    private int dim;
    
    /**
     * 最大位置编码长度
     */
    private int maxPositionEmbeddings;
    
    /**
     * 基础频率参数
     */
    private float base;
    
    /**
     * 预计算的逆频率
     */
    private NdArray invFreq;

    /**
     * 构造旋转位置编码层
     * 
     * @param name 层名称
     * @param dim 特征维度（通常是注意力头的维度）
     * @param maxPositionEmbeddings 最大位置编码长度
     * @param base RoPE的基础频率参数
     */
    public RotaryPositionalEmbedding(String name, int dim, int maxPositionEmbeddings, float base) {
        super(name, Shape.of(-1, -1, -1, dim), Shape.of(-1, -1, -1, dim));
        this.dim = dim;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.base = base;
        init();
    }

    /**
     * 使用默认参数的构造函数
     * 
     * @param name 层名称
     * @param dim 特征维度
     * @param maxPositionEmbeddings 最大位置编码长度
     */
    public RotaryPositionalEmbedding(String name, int dim, int maxPositionEmbeddings) {
        this(name, dim, maxPositionEmbeddings, 10000.0f);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 计算逆频率：1.0 / (base^(2i/dim)) for i in [0, dim/2)
            int halfDim = dim / 2;
            float[] invFreqData = new float[halfDim];
            
            for (int i = 0; i < halfDim; i++) {
                float exponent = 2.0f * i / dim;
                invFreqData[i] = 1.0f / (float) Math.pow(base, exponent);
            }
            
            invFreq = NdArray.of(invFreqData).reshape(Shape.of(halfDim));
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        int seqLen = inputs.length > 1 ? (int) inputs[1].getValue().get(0) : 
                     input.getValue().getShape().getDimension(2); // 默认从输入推断序列长度
        
        return new Variable(applyRotaryPosEmb(input.getValue(), seqLen));
    }
    
    /**
     * 应用旋转位置编码到输入张量
     * 
     * @param x 输入张量，形状为 [batch_size, num_heads, seq_len, head_dim]
     * @param seqLen 序列长度
     * @return 应用RoPE后的张量
     */
    private NdArray applyRotaryPosEmb(NdArray x, int seqLen) {
        Shape inputShape = x.getShape();
        
        // 验证输入形状
        if (inputShape.getDimNum() != 4) {
            throw new IllegalArgumentException("RoPE期望4D输入 [batch_size, num_heads, seq_len, head_dim]，实际: " + inputShape);
        }
        
        int batchSize = inputShape.getDimension(0);
        int numHeads = inputShape.getDimension(1);
        int actualSeqLen = inputShape.getDimension(2);
        int headDim = inputShape.getDimension(3);
        
        // 使用实际序列长度
        seqLen = Math.min(seqLen, actualSeqLen);
        
        if (headDim != dim) {
            throw new IllegalArgumentException("头维度不匹配: 期望 " + dim + "，实际 " + headDim);
        }
        
        // 计算cos和sin值
        RopeCoseSin cosSin = computeCosSin(seqLen);
        
        // 应用旋转变换
        return applyRotation(x, cosSin.cos, cosSin.sin, batchSize, numHeads, seqLen, headDim);
    }
    
    /**
     * 计算RoPE的cos和sin值
     */
    private RopeCoseSin computeCosSin(int seqLen) {
        int halfDim = dim / 2;
        
        // 生成位置索引
        float[] positions = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            positions[i] = i;
        }
        
        // 计算频率矩阵：positions × inv_freq
        NdArray freqs = NdArray.of(Shape.of(seqLen, halfDim));
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < halfDim; j++) {
                freqs.set(positions[i] * invFreq.get(j), i, j);
            }
        }
        
        // 将频率复制一倍以匹配完整维度
        NdArray embFreqs = NdArray.of(Shape.of(seqLen, dim));
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < halfDim; j++) {
                float freq = freqs.get(i, j);
                embFreqs.set(freq, i, j);           // 前半部分
                embFreqs.set(freq, i, j + halfDim); // 后半部分
            }
        }
        
        // 计算cos和sin
        NdArray cos = computeCos(embFreqs);
        NdArray sin = computeSin(embFreqs);
        
        return new RopeCoseSin(cos, sin);
    }
    
    /**
     * 计算cos值
     */
    private NdArray computeCos(NdArray freqs) {
        NdArray cos = NdArray.of(freqs.getShape());
        int seqLen = freqs.getShape().getDimension(0);
        
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < dim; j++) {
                cos.set((float) Math.cos(freqs.get(i, j)), i, j);
            }
        }
        
        return cos;
    }
    
    /**
     * 计算sin值
     */
    private NdArray computeSin(NdArray freqs) {
        NdArray sin = NdArray.of(freqs.getShape());
        int seqLen = freqs.getShape().getDimension(0);
        
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < dim; j++) {
                sin.set((float) Math.sin(freqs.get(i, j)), i, j);
            }
        }
        
        return sin;
    }
    
    /**
     * 应用旋转变换到输入张量
     */
    private NdArray applyRotation(NdArray x, NdArray cos, NdArray sin, 
                                  int batchSize, int numHeads, int seqLen, int headDim) {
        NdArray output = NdArray.of(x.getShape());
        int halfDim = headDim / 2;
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    // 应用旋转变换：每对维度进行旋转
                    for (int i = 0; i < halfDim; i++) {
                        float x1 = x.get(b, h, s, i);
                        float x2 = x.get(b, h, s, i + halfDim);
                        
                        float cosVal = cos.get(s, i);
                        float sinVal = sin.get(s, i);
                        
                        // 旋转变换：
                        // [x1'] = [cos -sin] [x1]
                        // [x2']   [sin  cos] [x2]
                        float rotated1 = x1 * cosVal - x2 * sinVal;
                        float rotated2 = x1 * sinVal + x2 * cosVal;
                        
                        output.set(rotated1, b, h, s, i);
                        output.set(rotated2, b, h, s, i + halfDim);
                    }
                }
            }
        }
        
        return output;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        Variable[] variables = new Variable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            variables[i] = new Variable(inputs[i]);
        }
        return layerForward(variables).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // RoPE的反向传播需要计算旋转的逆变换
        // 简化实现返回原梯度
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1; // 基本输入，序列长度可以从输入推断
    }
    
    /**
     * 内部类：存储cos和sin值
     */
    private static class RopeCoseSin {
        final NdArray cos;
        final NdArray sin;
        
        RopeCoseSin(NdArray cos, NdArray sin) {
            this.cos = cos;
            this.sin = sin;
        }
    }
    
    /**
     * 获取特征维度
     * 
     * @return 特征维度
     */
    public int getDim() {
        return dim;
    }
    
    /**
     * 获取最大位置编码长度
     * 
     * @return 最大位置编码长度
     */
    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }
    
    /**
     * 获取基础频率参数
     * 
     * @return 基础频率参数
     */
    public float getBase() {
        return base;
    }
}