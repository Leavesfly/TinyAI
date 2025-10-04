package io.leavesfly.tinyai.qwen3.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

/**
 * RMS归一化层
 * 
 * 相比LayerNorm，RMSNorm去除了重新中心化的步骤，只进行重新缩放。
 * 计算公式：RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
 * 
 * 优势：
 * 1. 计算更高效，减少了均值计算
 * 2. 训练更稳定
 * 3. 广泛应用于现代大语言模型
 * 
 * @author 山泽
 * @version 1.0
 */
public class RMSNormLayer extends Layer {
    
    /** 缩放权重参数 */
    private Parameter weight;
    
    /** 方差的小常数，防止除零 */
    private double eps;
    
    /** 特征维度 */
    private int hiddenSize;
    
    /**
     * 构造RMSNorm层
     * 
     * @param name 层名称
     * @param hiddenSize 特征维度
     * @param eps epsilon值，默认1e-6
     */
    public RMSNormLayer(String name, int hiddenSize, double eps) {
        super(name, Shape.of(-1, hiddenSize), Shape.of(-1, hiddenSize));
        this.hiddenSize = hiddenSize;
        this.eps = eps;
        init();
    }
    
    /**
     * 构造RMSNorm层（使用默认eps）
     * 
     * @param name 层名称
     * @param hiddenSize 特征维度
     */
    public RMSNormLayer(String name, int hiddenSize) {
        this(name, hiddenSize, 1e-6);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化权重为1
            NdArray weightData = NdArray.ones(Shape.of(hiddenSize));
            weight = new Parameter(weightData);
            weight.setName("weight");
            addParam(weight.getName(), weight);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        NdArray inputData = input.getValue();
        
        // 获取输入形状
        Shape inputShape = inputData.getShape();
        
        // 计算RMSNorm
        NdArray result = computeRMSNorm(inputData);
        
        return new Variable(result);
    }
    
    /**
     * 计算RMS归一化
     * 
     * @param input 输入数据
     * @return 归一化后的数据
     */
    private NdArray computeRMSNorm(NdArray input) {
        Shape inputShape = input.getShape();
        int dimNum = inputShape.getDimNum();
        
        // 创建输出数组
        NdArray output = NdArray.of(inputShape);
        
        if (dimNum == 2) {
            // 2D输入：(batch_size, hidden_size)
            int batchSize = inputShape.getDimension(0);
            
            for (int i = 0; i < batchSize; i++) {
                // 计算当前样本的均方根
                float sumSquare = 0.0f;
                for (int j = 0; j < hiddenSize; j++) {
                    float value = input.get(i, j);
                    sumSquare += value * value;
                }
                
                // 计算RMS
                float rms = (float) Math.sqrt(sumSquare / hiddenSize + eps);
                
                // 应用归一化和缩放
                for (int j = 0; j < hiddenSize; j++) {
                    float normalizedValue = input.get(i, j) / rms;
                    float scaledValue = normalizedValue * weight.getValue().get(j);
                    output.set(scaledValue, i, j);
                }
            }
        } else if (dimNum == 3) {
            // 3D输入：(batch_size, seq_len, hidden_size)
            int batchSize = inputShape.getDimension(0);
            int seqLen = inputShape.getDimension(1);
            
            for (int i = 0; i < batchSize; i++) {
                for (int s = 0; s < seqLen; s++) {
                    // 计算当前位置的均方根
                    float sumSquare = 0.0f;
                    for (int j = 0; j < hiddenSize; j++) {
                        float value = input.get(i, s, j);
                        sumSquare += value * value;
                    }
                    
                    // 计算RMS
                    float rms = (float) Math.sqrt(sumSquare / hiddenSize + eps);
                    
                    // 应用归一化和缩放
                    for (int j = 0; j < hiddenSize; j++) {
                        float normalizedValue = input.get(i, s, j) / rms;
                        float scaledValue = normalizedValue * weight.getValue().get(j);
                        output.set(scaledValue, i, s, j);
                    }
                }
            }
        } else {
            throw new IllegalArgumentException(
                String.format("RMSNorm不支持%dD输入，只支持2D和3D", dimNum));
        }
        
        return output;
    }
    
    /**
     * 获取epsilon值
     */
    public double getEps() {
        return eps;
    }
    
    /**
     * 获取特征维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    /**
     * 获取权重参数
     */
    public Parameter getWeight() {
        return weight;
    }
}