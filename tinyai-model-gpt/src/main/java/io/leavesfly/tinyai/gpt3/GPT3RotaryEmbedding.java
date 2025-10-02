package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

/**
 * GPT-3旋转位置编码（RoPE）实现
 * 
 * 旋转位置编码是GPT-3的一个重要改进，相比传统的绝对位置编码：
 * 1. 更好地处理长序列
 * 2. 具有相对位置编码的特性
 * 3. 不需要学习位置参数
 * 4. 在推理时可以处理训练时未见过的序列长度
 * 
 * 原理：
 * - 对查询(Q)和键(K)向量应用旋转变换
 * - 旋转角度与位置相关，频率递减
 * - 保持向量的模长不变
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3RotaryEmbedding extends Layer {
    
    /** 旋转维度 */
    private int rotaryDim;
    
    /** 最大序列长度 */
    private int maxSeqLength;
    
    /** 基础频率 */
    private double base;
    
    /** 预计算的频率倒数 */
    private NdArray invFreq;
    
    /**
     * 构造旋转位置编码层
     * 
     * @param name 层名称
     * @param rotaryDim 旋转维度（通常是head_dim的一部分）
     * @param maxSeqLength 最大序列长度
     * @param base 基础频率（默认10000）
     */
    public GPT3RotaryEmbedding(String name, int rotaryDim, int maxSeqLength, double base) {
        super(name, Shape.of(-1, maxSeqLength, rotaryDim), Shape.of(-1, maxSeqLength, rotaryDim));
        
        if (rotaryDim % 2 != 0) {
            throw new IllegalArgumentException("旋转维度必须是偶数");
        }
        
        this.rotaryDim = rotaryDim;
        this.maxSeqLength = maxSeqLength;
        this.base = base;
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public GPT3RotaryEmbedding(String name, int rotaryDim, int maxSeqLength) {
        this(name, rotaryDim, maxSeqLength, 10000.0);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 预计算频率倒数：1 / (base^(2i/d)) for i in [0, d/2)
            int halfDim = rotaryDim / 2;
            invFreq = NdArray.of(Shape.of(halfDim));
            
            for (int i = 0; i < halfDim; i++) {
                double freq = 1.0 / Math.pow(base, (2.0 * i) / rotaryDim);
                invFreq.set((float) freq, i);
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 这个方法通常不直接调用，而是通过applyRotaryPositionEmbedding使用
        throw new UnsupportedOperationException("请使用applyRotaryPositionEmbedding方法");
    }
    
    /**
     * 生成旋转位置编码的cos和sin值
     * 
     * @param seqLength 序列长度
     * @return 包含cos和sin的数组：[cos, sin]
     */
    public NdArray[] generateRotaryEmbedding(int seqLength) {
        if (seqLength > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大支持长度(%d)", seqLength, maxSeqLength)
            );
        }
        
        int halfDim = rotaryDim / 2;
        
        // 生成位置索引：[0, 1, 2, ..., seqLength-1]
        NdArray positions = NdArray.of(Shape.of(seqLength));
        for (int i = 0; i < seqLength; i++) {
            positions.set(i, i);
        }
        
        // 计算频率矩阵：positions * invFreq
        // 结果形状：(seqLength, halfDim)
        NdArray freqs = NdArray.of(Shape.of(seqLength, halfDim));
        for (int pos = 0; pos < seqLength; pos++) {
            for (int dim = 0; dim < halfDim; dim++) {
                float freq = positions.get(pos) * invFreq.get(dim);
                freqs.set(freq, pos, dim);
            }
        }
        
        // 将频率矩阵扩展到完整的旋转维度
        // 每个频率对应cos和sin，所以要重复
        NdArray embFreqs = NdArray.of(Shape.of(seqLength, rotaryDim));
        for (int pos = 0; pos < seqLength; pos++) {
            for (int dim = 0; dim < halfDim; dim++) {
                float freq = freqs.get(pos, dim);
                embFreqs.set(freq, pos, dim);           // 前半部分
                embFreqs.set(freq, pos, dim + halfDim); // 后半部分
            }
        }
        
        // 计算cos和sin
        NdArray cos = NdArray.of(Shape.of(seqLength, rotaryDim));
        NdArray sin = NdArray.of(Shape.of(seqLength, rotaryDim));
        
        for (int pos = 0; pos < seqLength; pos++) {
            for (int dim = 0; dim < rotaryDim; dim++) {
                float freq = embFreqs.get(pos, dim);
                cos.set((float) Math.cos(freq), pos, dim);
                sin.set((float) Math.sin(freq), pos, dim);
            }
        }
        
        return new NdArray[]{cos, sin};
    }
    
    /**
     * 对查询和键向量应用旋转位置编码
     * 
     * @param query 查询向量 (batch_size, seq_len, num_heads, head_dim)
     * @param key 键向量 (batch_size, seq_len, num_heads, head_dim)
     * @param seqLength 序列长度
     * @return 应用RoPE后的[query, key]
     */
    public Variable[] applyRotaryPositionEmbedding(Variable query, Variable key, int seqLength) {
        NdArray queryData = query.getValue();
        NdArray keyData = key.getValue();
        
        // 验证输入形状
        validateInputShape(queryData, "query");
        validateInputShape(keyData, "key");
        
        // 生成cos和sin
        NdArray[] cosAndSin = generateRotaryEmbedding(seqLength);
        NdArray cos = cosAndSin[0];
        NdArray sin = cosAndSin[1];
        
        // 应用旋转变换
        NdArray rotatedQuery = applyRotaryTransform(queryData, cos, sin);
        NdArray rotatedKey = applyRotaryTransform(keyData, cos, sin);
        
        return new Variable[]{new Variable(rotatedQuery), new Variable(rotatedKey)};
    }
    
    /**
     * 验证输入形状
     */
    private void validateInputShape(NdArray input, String name) {
        Shape shape = input.getShape();
        if (shape.getDimNum() != 4) {
            throw new IllegalArgumentException(
                String.format("%s必须是4维张量 (batch_size, seq_len, num_heads, head_dim)", name)
            );
        }
        
        int headDim = shape.getDimension(3);
        if (rotaryDim > headDim) {
            throw new IllegalArgumentException(
                String.format("旋转维度(%d)不能大于头维度(%d)", rotaryDim, headDim)
            );
        }
    }
    
    /**
     * 应用旋转变换
     */
    private NdArray applyRotaryTransform(NdArray input, NdArray cos, NdArray sin) {
        Shape inputShape = input.getShape();
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int numHeads = inputShape.getDimension(2);
        int headDim = inputShape.getDimension(3);
        
        NdArray result = NdArray.of(inputShape);
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    // 应用旋转到前rotaryDim维度
                    for (int d = 0; d < rotaryDim; d++) {
                        float x = input.get(b, s, h, d);
                        float cosVal = cos.get(s, d);
                        float sinVal = sin.get(s, d);
                        
                        // 旋转变换：计算配对维度
                        int pairIdx = (d < rotaryDim / 2) ? d + rotaryDim / 2 : d - rotaryDim / 2;
                        float y = input.get(b, s, h, pairIdx);
                        
                        float rotatedVal;
                        if (d < rotaryDim / 2) {
                            // 前半部分：x*cos - y*sin
                            rotatedVal = x * cosVal - y * sinVal;
                        } else {
                            // 后半部分：x*cos + y*sin
                            rotatedVal = x * cosVal + y * sinVal;
                        }
                        
                        result.set(rotatedVal, b, s, h, d);
                    }
                    
                    // 复制未旋转的维度
                    for (int d = rotaryDim; d < headDim; d++) {
                        result.set(input.get(b, s, h, d), b, s, h, d);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 旋转向量的一半维度（用于计算旋转变换）
     */
    public static NdArray rotateHalf(NdArray x) {
        Shape shape = x.getShape();
        int lastDim = shape.getDimension(shape.getDimNum() - 1);
        int halfDim = lastDim / 2;
        
        NdArray result = NdArray.of(shape);
        
        // 根据输入的维度数处理
        if (shape.getDimNum() == 4) {
            int batchSize = shape.getDimension(0);
            int seqLen = shape.getDimension(1);
            int numHeads = shape.getDimension(2);
            
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int h = 0; h < numHeads; h++) {
                        // 前半部分取负号并移到后半部分
                        for (int d = 0; d < halfDim; d++) {
                            result.set(-x.get(b, s, h, d + halfDim), b, s, h, d);
                        }
                        // 后半部分移到前半部分
                        for (int d = 0; d < halfDim; d++) {
                            result.set(x.get(b, s, h, d), b, s, h, d + halfDim);
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    // ==================== Getter方法 ====================
    
    public int getRotaryDim() { return rotaryDim; }
    public int getMaxSeqLength() { return maxSeqLength; }
    public double getBase() { return base; }
    public NdArray getInvFreq() { return invFreq; }
}