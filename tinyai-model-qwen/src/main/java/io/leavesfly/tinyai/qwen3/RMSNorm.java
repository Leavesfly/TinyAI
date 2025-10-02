package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * RMS归一化层（Root Mean Square Layer Normalization）
 * 
 * RMSNorm是一种简化的层归一化方法，不使用偏移参数，
 * 只使用缩放参数。公式为：
 * 
 * RMSNorm(x) = x / RMS(x) * weight
 * 其中 RMS(x) = sqrt(mean(x^2) + eps)
 * 
 * 相比LayerNorm，RMSNorm计算更简单，在许多大语言模型中表现良好。
 * 
 * @author 山泽
 * @version 1.0
 */
public class RMSNorm extends Layer {
    
    /**
     * 缩放权重参数
     */
    private Parameter weight;
    
    /**
     * 防止除零的小数值
     */
    private float eps;
    
    /**
     * 隐藏层维度
     */
    private int hiddenSize;

    /**
     * 构造RMSNorm层
     * 
     * @param name 层名称
     * @param hiddenSize 隐藏层维度
     * @param eps 防止除零的小数值
     */
    public RMSNorm(String name, int hiddenSize, float eps) {
        super(name, Shape.of(-1, -1, hiddenSize), Shape.of(-1, -1, hiddenSize));
        this.hiddenSize = hiddenSize;
        this.eps = eps;
        init();
    }

    /**
     * 使用默认eps值的构造函数
     * 
     * @param name 层名称  
     * @param hiddenSize 隐藏层维度
     */
    public RMSNorm(String name, int hiddenSize) {
        this(name, hiddenSize, 1e-6f);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化权重为全1
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
        
        // 输入可能是2D (batch_size, hidden_size) 或 3D (batch_size, seq_len, hidden_size)
        if (inputShape.getDimNum() == 2) {
            return forward2D(input);
        } else if (inputShape.getDimNum() == 3) {
            return forward3D(input);
        } else {
            throw new IllegalArgumentException("RMSNorm只支持2D或3D输入，当前输入维度: " + inputShape.getDimNum());
        }
    }
    
    /**
     * 处理2D输入的前向传播
     * 输入形状: (batch_size, hidden_size)
     */
    private Variable forward2D(Variable input) {
        NdArray inputData = input.getValue();
        int batchSize = inputData.getShape().getDimension(0);
        
        NdArray output = NdArray.of(inputData.getShape());
        
        for (int b = 0; b < batchSize; b++) {
            // 计算每个样本的RMS
            float sumSquare = 0.0f;
            for (int h = 0; h < hiddenSize; h++) {
                float val = inputData.get(b, h);
                sumSquare += val * val;
            }
            
            float rms = (float) Math.sqrt(sumSquare / hiddenSize + eps);
            
            // 归一化并应用权重
            for (int h = 0; h < hiddenSize; h++) {
                float normalized = inputData.get(b, h) / rms;
                float weighted = normalized * weight.getValue().get(h);
                output.set(weighted, b, h);
            }
        }
        
        return new Variable(output);
    }
    
    /**
     * 处理3D输入的前向传播
     * 输入形状: (batch_size, seq_len, hidden_size)
     */
    private Variable forward3D(Variable input) {
        NdArray inputData = input.getValue();
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        NdArray output = NdArray.of(inputData.getShape());
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 计算每个位置的RMS
                float sumSquare = 0.0f;
                for (int h = 0; h < hiddenSize; h++) {
                    float val = inputData.get(b, s, h);
                    sumSquare += val * val;
                }
                
                float rms = (float) Math.sqrt(sumSquare / hiddenSize + eps);
                
                // 归一化并应用权重
                for (int h = 0; h < hiddenSize; h++) {
                    float normalized = inputData.get(b, s, h) / rms;
                    float weighted = normalized * weight.getValue().get(h);
                    output.set(weighted, b, s, h);
                }
            }
        }
        
        return new Variable(output);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 简化的反向传播实现
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 获取权重参数
     * 
     * @return 权重参数
     */
    public Parameter getWeight() {
        return weight;
    }
    
    /**
     * 获取eps值
     * 
     * @return eps值
     */
    public float getEps() {
        return eps;
    }
    
    /**
     * 获取隐藏层维度
     * 
     * @return 隐藏层维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
}