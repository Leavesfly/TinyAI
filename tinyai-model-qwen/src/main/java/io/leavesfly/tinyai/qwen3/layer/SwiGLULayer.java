package io.leavesfly.tinyai.qwen3.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

/**
 * SwiGLU激活函数层
 * 
 * SwiGLU是一种门控激活函数，结合了Swish激活和门控机制：
 * SwiGLU(x) = Swish(xW1) ⊙ (xW2)
 * 其中：
 * - Swish(x) = x * sigmoid(x)
 * - ⊙ 表示元素级别的乘法
 * - W1和W2是两个独立的线性变换权重
 * 
 * 优势：
 * 1. 比传统的ReLU具有更好的表达能力
 * 2. 门控机制允许网络学习信息的选择性传递
 * 3. 现代大语言模型的标准选择
 * 
 * @author 山泽
 * @version 1.0
 */
public class SwiGLULayer extends Layer {
    
    /** 输入维度 */
    private int inputDim;
    
    /** 输出维度 */
    private int outputDim;
    
    /**
     * 构造SwiGLU激活层
     * 
     * @param name 层名称
     * @param inputDim 输入维度
     * @param outputDim 输出维度
     */
    public SwiGLULayer(String name, int inputDim, int outputDim) {
        super(name, Shape.of(-1, inputDim), Shape.of(-1, outputDim));
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // SwiGLU不需要参数，激活函数逻辑在MLP层中实现
            // 这里主要是为了保持Layer的一致性接口
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        if (inputs.length != 2) {
            throw new IllegalArgumentException("SwiGLU需要两个输入：gate和up");
        }
        
        Variable gate = inputs[0];  // 门控输入
        Variable up = inputs[1];    // 上投影输入
        
        // 应用SwiGLU激活
        NdArray result = applySwiGLU(gate.getValue(), up.getValue());
        return new Variable(result);
    }
    
    /**
     * 单独的SwiGLU计算方法，供其他层调用
     * 
     * @param gate 门控输入
     * @param up 上投影输入
     * @return SwiGLU激活结果
     */
    public static NdArray applySwiGLU(NdArray gate, NdArray up) {
        // 验证输入形状一致
        if (!gate.getShape().equals(up.getShape())) {
            throw new IllegalArgumentException(
                String.format("gate形状 %s 必须与up形状 %s 一致", 
                    gate.getShape(), up.getShape()));
        }
        
        Shape inputShape = gate.getShape();
        NdArray result = NdArray.of(inputShape);
        
        // 应用SwiGLU：Swish(gate) ⊙ up
        if (inputShape.getDimNum() == 2) {
            // 2D输入：(batch_size, hidden_size)
            int batchSize = inputShape.getDimension(0);
            int hiddenSize = inputShape.getDimension(1);
            
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    float gateValue = gate.get(i, j);
                    float upValue = up.get(i, j);
                    
                    // Swish激活：x * sigmoid(x)
                    float swishValue = gateValue * sigmoid(gateValue);
                    
                    // 门控乘法
                    float swiGLUValue = swishValue * upValue;
                    
                    result.set(swiGLUValue, i, j);
                }
            }
        } else if (inputShape.getDimNum() == 3) {
            // 3D输入：(batch_size, seq_len, hidden_size)
            int batchSize = inputShape.getDimension(0);
            int seqLen = inputShape.getDimension(1);
            int hiddenSize = inputShape.getDimension(2);
            
            for (int i = 0; i < batchSize; i++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        float gateValue = gate.get(i, s, j);
                        float upValue = up.get(i, s, j);
                        
                        // Swish激活：x * sigmoid(x)
                        float swishValue = gateValue * sigmoid(gateValue);
                        
                        // 门控乘法
                        float swiGLUValue = swishValue * upValue;
                        
                        result.set(swiGLUValue, i, s, j);
                    }
                }
            }
        } else {
            throw new IllegalArgumentException(
                String.format("SwiGLU不支持%dD输入", inputShape.getDimNum()));
        }
        
        return result;
    }
    
    /**
     * Sigmoid激活函数
     * 
     * @param x 输入值
     * @return sigmoid(x) = 1 / (1 + exp(-x))
     */
    private static float sigmoid(float x) {
        // 数值稳定性处理
        if (x > 20) {
            return 1.0f;
        } else if (x < -20) {
            return 0.0f;
        } else {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
        }
    }
    
    /**
     * Swish激活函数
     * 
     * @param x 输入值
     * @return swish(x) = x * sigmoid(x)
     */
    public static float swish(float x) {
        return x * sigmoid(x);
    }
    
    /**
     * 独立的Swish激活函数实现，供其他地方使用
     * 
     * @param input 输入数组
     * @return Swish激活后的数组
     */
    public static NdArray applySwish(NdArray input) {
        Shape inputShape = input.getShape();
        NdArray result = NdArray.of(inputShape);
        
        if (inputShape.getDimNum() == 2) {
            int rows = inputShape.getDimension(0);
            int cols = inputShape.getDimension(1);
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float value = input.get(i, j);
                    result.set(swish(value), i, j);
                }
            }
        } else if (inputShape.getDimNum() == 3) {
            int dim0 = inputShape.getDimension(0);
            int dim1 = inputShape.getDimension(1);
            int dim2 = inputShape.getDimension(2);
            
            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    for (int k = 0; k < dim2; k++) {
                        float value = input.get(i, j, k);
                        result.set(swish(value), i, j, k);
                    }
                }
            }
        } else {
            throw new IllegalArgumentException(
                String.format("Swish不支持%dD输入", inputShape.getDimNum()));
        }
        
        return result;
    }
    
    /**
     * 获取输入维度
     */
    public int getInputDim() {
        return inputDim;
    }
    
    /**
     * 获取输出维度
     */
    public int getOutputDim() {
        return outputDim;
    }
}