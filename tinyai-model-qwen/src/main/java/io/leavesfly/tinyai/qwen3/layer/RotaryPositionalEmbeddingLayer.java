package io.leavesfly.tinyai.qwen3.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

/**
 * 旋转位置编码 (RoPE) 层
 * 
 * 通过旋转查询和键向量来编码位置信息，相比传统的绝对位置编码：
 * 1. 提供相对位置信息
 * 2. 支持长序列外推
 * 3. 旋转矩阵保持向量模长不变
 * 4. 计算高效，无需额外参数
 * 
 * 核心思想：将高维向量分成若干个二维子向量，对每个子向量应用旋转变换。
 * 
 * @author 山泽
 * @version 1.0
 */
public class RotaryPositionalEmbeddingLayer extends Layer {
    
    /** 头维度 */
    private int headDim;
    
    /** 最大位置编码长度 */
    private int maxPositionEmbeddings;
    
    /** RoPE基础频率 */
    private double base;
    
    /** 逆频率数组，缓存计算结果 */
    private NdArray invFreq;
    
    /**
     * 构造RoPE层
     * 
     * @param name 层名称
     * @param headDim 头维度
     * @param maxPositionEmbeddings 最大位置编码长度
     * @param base 基础频率
     */
    public RotaryPositionalEmbeddingLayer(String name, int headDim, int maxPositionEmbeddings, double base) {
        super(name, Shape.of(-1, -1, -1, headDim), Shape.of(-1, -1, -1, headDim));
        this.headDim = headDim;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.base = base;
        init();
    }
    
    /**
     * 构造RoPE层（使用默认参数）
     * 
     * @param name 层名称
     * @param headDim 头维度
     */
    public RotaryPositionalEmbeddingLayer(String name, int headDim) {
        this(name, headDim, 2048, 10000.0);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 计算逆频率：1.0 / (base ^ (2i / headDim)) for i in [0, headDim/2)
            int freqDim = headDim / 2;
            NdArray invFreqData = NdArray.of(Shape.of(freqDim));
            
            for (int i = 0; i < freqDim; i++) {
                double exponent = (2.0 * i) / headDim;
                double freq = 1.0 / Math.pow(base, exponent);
                invFreqData.set((float) freq, i);
            }
            
            this.invFreq = invFreqData;
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // RoPE通常不作为独立层使用，而是在注意力计算中应用
        // 这里提供基础的前向传播实现
        throw new UnsupportedOperationException("RoPE层应该通过applyRotaryPosEmb方法使用");
    }
    
    /**
     * 应用旋转位置编码到查询和键向量
     * 
     * @param query 查询向量 [batch_size, num_heads, seq_len, head_dim]
     * @param key 键向量 [batch_size, num_heads, seq_len, head_dim]
     * @param seqLen 序列长度
     * @return 应用RoPE后的查询和键向量数组 [rotated_query, rotated_key]
     */
    public NdArray[] applyRotaryPosEmb(NdArray query, NdArray key, int seqLen) {
        // 计算位置编码
        NdArray[] cosAndSin = computePositionalEncoding(seqLen);
        NdArray cos = cosAndSin[0];
        NdArray sin = cosAndSin[1];
        
        // 应用旋转
        NdArray rotatedQuery = applyRotation(query, cos, sin);
        NdArray rotatedKey = applyRotation(key, cos, sin);
        
        return new NdArray[]{rotatedQuery, rotatedKey};
    }
    
    /**
     * 计算位置编码的cos和sin值
     * 
     * @param seqLen 序列长度
     * @return [cos, sin]数组
     */
    private NdArray[] computePositionalEncoding(int seqLen) {
        int freqDim = headDim / 2;
        
        // 生成位置索引 [0, 1, 2, ..., seqLen-1]
        NdArray positions = NdArray.of(Shape.of(seqLen));
        for (int i = 0; i < seqLen; i++) {
            positions.set(i, i);
        }
        
        // 计算频率矩阵：pos * inv_freq
        // freqs shape: [seq_len, freq_dim]
        NdArray freqs = NdArray.of(Shape.of(seqLen, freqDim));
        for (int pos = 0; pos < seqLen; pos++) {
            for (int f = 0; f < freqDim; f++) {
                float freq = positions.get(pos) * invFreq.get(f);
                freqs.set(freq, pos, f);
            }
        }
        
        // 复制频率以匹配头部维度：[seq_len, head_dim]
        NdArray embFreqs = NdArray.of(Shape.of(seqLen, headDim));
        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < freqDim; i++) {
                float freq = freqs.get(pos, i);
                embFreqs.set(freq, pos, i);           // 前半部分
                embFreqs.set(freq, pos, i + freqDim); // 后半部分
            }
        }
        
        // 计算cos和sin
        NdArray cos = NdArray.of(Shape.of(seqLen, headDim));
        NdArray sin = NdArray.of(Shape.of(seqLen, headDim));
        
        for (int pos = 0; pos < seqLen; pos++) {
            for (int d = 0; d < headDim; d++) {
                float freq = embFreqs.get(pos, d);
                cos.set((float) Math.cos(freq), pos, d);
                sin.set((float) Math.sin(freq), pos, d);
            }
        }
        
        return new NdArray[]{cos, sin};
    }
    
    /**
     * 应用旋转变换
     * 
     * @param x 输入向量 [batch_size, num_heads, seq_len, head_dim]
     * @param cos cos值 [seq_len, head_dim]
     * @param sin sin值 [seq_len, head_dim]
     * @return 旋转后的向量
     */
    private NdArray applyRotation(NdArray x, NdArray cos, NdArray sin) {
        Shape xShape = x.getShape();
        int batchSize = xShape.getDimension(0);
        int numHeads = xShape.getDimension(1);
        int seqLen = xShape.getDimension(2);
        int headDim = xShape.getDimension(3);
        
        NdArray result = NdArray.of(xShape);
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    // 获取旋转的一半向量
                    NdArray rotatedHalf = rotateHalf(x, b, h, s, headDim);
                    
                    for (int d = 0; d < headDim; d++) {
                        float originalValue = x.get(b, h, s, d);
                        float rotatedValue = rotatedHalf.get(d);
                        float cosValue = cos.get(s, d);
                        float sinValue = sin.get(s, d);
                        
                        // 应用旋转：x * cos + rotate_half(x) * sin
                        float rotatedResult = originalValue * cosValue + rotatedValue * sinValue;
                        result.set(rotatedResult, b, h, s, d);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 旋转输入的一半特征
     * 将 [x1, x2, x3, x4, ...] 转换为 [-x2, x1, -x4, x3, ...]
     * 
     * @param x 输入向量
     * @param b batch索引
     * @param h head索引
     * @param s 序列索引
     * @param headDim 头维度
     * @return 旋转的一半向量
     */
    private NdArray rotateHalf(NdArray x, int b, int h, int s, int headDim) {
        NdArray rotated = NdArray.of(Shape.of(headDim));
        
        int halfDim = headDim / 2;
        
        // 前半部分：x[..., :headDim//2] -> -x[..., headDim//2:]
        for (int i = 0; i < halfDim; i++) {
            float value = x.get(b, h, s, i + halfDim);
            rotated.set(-value, i);
        }
        
        // 后半部分：x[..., headDim//2:] -> x[..., :headDim//2]
        for (int i = 0; i < halfDim; i++) {
            float value = x.get(b, h, s, i);
            rotated.set(value, i + halfDim);
        }
        
        return rotated;
    }
    
    /**
     * 获取头维度
     */
    public int getHeadDim() {
        return headDim;
    }
    
    /**
     * 获取最大位置编码长度
     */
    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }
    
    /**
     * 获取基础频率
     */
    public double getBase() {
        return base;
    }
    
    /**
     * 获取逆频率数组
     */
    public NdArray getInvFreq() {
        return invFreq;
    }
}