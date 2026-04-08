package io.leavesfly.tinyai.nl.block;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nl.core.AssociativeMemory;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;

/**
 * 多频率注意力块（MultiFrequencyAttention）
 * 实现支持多个时间尺度的注意力机制
 * 
 * <p>该块维护多个不同更新频率的注意力头，使模型能够同时
 * 关注短期和长期的依赖关系。</p>
 * 
 * @author TinyAI Team
 */
public class MultiFrequencyAttention extends Module {
    
    /**
     * 不同频率的记忆模块
     */
    private AssociativeMemory[] frequencyMemories;
    
    /**
     * 频率数量
     */
    private int numFrequencies;
    
    /**
     * 注意力头维度
     */
    private int headDim;
    
    /**
     * 频率权重参数（可学习），用于加权不同频率的注意力输出
     */
    private Parameter frequencyWeights;
    
    /**
     * 查询投影权重
     */
    private Parameter queryProjection;
    
    public MultiFrequencyAttention(String name, int numFrequencies, int headDim, Shape inputShape) {
        super(name);
        this.numFrequencies = numFrequencies;
        this.headDim = headDim;
        this.frequencyMemories = new AssociativeMemory[numFrequencies];
        
        for (int i = 0; i < numFrequencies; i++) {
            // 低频记忆容量更大（存储更多长期信息）
            int capacity = 100 * (i + 1);
            frequencyMemories[i] = new AssociativeMemory(capacity);
        }
        
        // 初始化频率权重（均匀分布）
        float[] weightData = new float[numFrequencies];
        float uniformWeight = 1.0f / numFrequencies;
        for (int i = 0; i < numFrequencies; i++) {
            weightData[i] = uniformWeight;
        }
        frequencyWeights = new Parameter(
            NdArray.of(weightData, Shape.of(1, numFrequencies)));
        registerParameter("frequency_weights", frequencyWeights);
    }
    
    public MultiFrequencyAttention(String name, int numFrequencies, int headDim) {
        this(name, numFrequencies, headDim, null);
    }
    
    @Override
    public void resetParameters() {
        float uniformWeight = 1.0f / numFrequencies;
        float[] weightData = new float[numFrequencies];
        for (int i = 0; i < numFrequencies; i++) {
            weightData[i] = uniformWeight;
        }
        frequencyWeights = new Parameter(
            NdArray.of(weightData, Shape.of(1, numFrequencies)));
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            return null;
        }
        
        Variable query = inputs[0];
        
        // 从不同频率的记忆中检索
        Variable[] retrievedValues = new Variable[numFrequencies];
        int validCount = 0;
        for (int i = 0; i < numFrequencies; i++) {
            retrievedValues[i] = frequencyMemories[i].retrieve(query);
            if (retrievedValues[i] != null) {
                validCount++;
            }
        }
        
        // 如果没有任何记忆可检索，直接返回查询
        if (validCount == 0) {
            return query;
        }
        
        // 使用频率权重对检索结果进行加权求和
        // 先对权重做 softmax 归一化
        NdArray weightsData = frequencyWeights.getValue();
        float[] rawWeights = new float[numFrequencies];
        for (int i = 0; i < numFrequencies; i++) {
            rawWeights[i] = weightsData.get(new int[]{0, i});
        }
        float[] normalizedWeights = softmaxWeights(rawWeights);
        
        // 加权求和
        Variable result = null;
        for (int i = 0; i < numFrequencies; i++) {
            if (retrievedValues[i] != null) {
                Variable weighted = retrievedValues[i].mul(new Variable(normalizedWeights[i]));
                if (result == null) {
                    result = weighted;
                } else {
                    // 检查形状兼容性
                    int[] resultShape = result.getValue().getShape().getShapeDims();
                    int[] weightedShape = weighted.getValue().getShape().getShapeDims();
                    if (java.util.Arrays.equals(resultShape, weightedShape)) {
                        result = result.add(weighted);
                    }
                }
            }
        }
        
        return result != null ? result : query;
    }
    
    /**
     * 对权重数组应用 softmax 归一化
     */
    private float[] softmaxWeights(float[] weights) {
        float[] result = new float[weights.length];
        float maxVal = Float.NEGATIVE_INFINITY;
        for (float w : weights) {
            maxVal = Math.max(maxVal, w);
        }
        float sumExp = 0.0f;
        for (int i = 0; i < weights.length; i++) {
            result[i] = (float) Math.exp(weights[i] - maxVal);
            sumExp += result[i];
        }
        if (sumExp > 0) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= sumExp;
            }
        }
        return result;
    }
    
    /**
     * 更新指定频率的记忆
     */
    public void updateMemory(int frequencyIndex, Variable key, Variable value) {
        if (frequencyIndex >= 0 && frequencyIndex < numFrequencies) {
            frequencyMemories[frequencyIndex].store(key, value);
        }
    }
    
    public int getNumFrequencies() {
        return numFrequencies;
    }
    
    public AssociativeMemory getMemory(int index) {
        if (index >= 0 && index < numFrequencies) {
            return frequencyMemories[index];
        }
        return null;
    }
}
