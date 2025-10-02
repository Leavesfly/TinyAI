package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * SiLU激活函数层（Sigmoid Linear Unit）
 * 
 * SiLU也被称为Swish激活函数，其定义为：
 * SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
 * 
 * SiLU在许多深度学习任务中表现优异，特别是在大语言模型中广泛使用。
 * 相比ReLU，SiLU是平滑且可微的，能够更好地处理梯度流。
 * 
 * @author 山泽
 * @version 1.0
 */
public class SiLULayer extends Layer {

    /**
     * 构造SiLU激活函数层
     * 
     * @param name 层名称
     * @param inputShape 输入形状
     */
    public SiLULayer(String name, Shape inputShape) {
        super(name, inputShape, inputShape);
        init();
    }

    /**
     * 为任意形状构造SiLU层的便利构造函数
     * 
     * @param name 层名称
     */
    public SiLULayer(String name) {
        super(name, Shape.of(-1), Shape.of(-1));
        init();
    }

    @Override
    public void init() {
        // SiLU激活函数没有可训练参数
        alreadyInit = true;
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        NdArray inputData = input.getValue();
        
        // 计算SiLU: x * sigmoid(x)
        return new Variable(applySiLU(inputData));
    }
    
    /**
     * 应用SiLU激活函数到NdArray
     * 
     * @param input 输入数组
     * @return 应用SiLU后的数组
     */
    private NdArray applySiLU(NdArray input) {
        Shape shape = input.getShape();
        NdArray output = NdArray.of(shape);
        
        // 根据维度处理不同形状的输入
        if (shape.getDimNum() == 1) {
            applySiLU1D(input, output);
        } else if (shape.getDimNum() == 2) {
            applySiLU2D(input, output);
        } else if (shape.getDimNum() == 3) {
            applySiLU3D(input, output);
        } else {
            // 对于更高维的数组，展平处理
            applySiLUFlattened(input, output);
        }
        
        return output;
    }
    
    /**
     * 处理1D数组的SiLU激活
     */
    private void applySiLU1D(NdArray input, NdArray output) {
        int len = input.getShape().getDimension(0);
        for (int i = 0; i < len; i++) {
            float x = input.get(i);
            float silu = x * sigmoid(x);
            output.set(silu, i);
        }
    }
    
    /**
     * 处理2D数组的SiLU激活
     */
    private void applySiLU2D(NdArray input, NdArray output) {
        int rows = input.getShape().getDimension(0);
        int cols = input.getShape().getDimension(1);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float x = input.get(i, j);
                float silu = x * sigmoid(x);
                output.set(silu, i, j);
            }
        }
    }
    
    /**
     * 处理3D数组的SiLU激活
     */
    private void applySiLU3D(NdArray input, NdArray output) {
        int dim0 = input.getShape().getDimension(0);
        int dim1 = input.getShape().getDimension(1);
        int dim2 = input.getShape().getDimension(2);
        
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                for (int k = 0; k < dim2; k++) {
                    float x = input.get(i, j, k);
                    float silu = x * sigmoid(x);
                    output.set(silu, i, j, k);
                }
            }
        }
    }
    
    /**
     * 处理任意维度数组的SiLU激活（展平方式）
     */
    private void applySiLUFlattened(NdArray input, NdArray output) {
        int totalSize = input.getShape().size();
        
        // 获取输入数据
        float[] inputFlat;
        NdArray flattenedInput = input.flatten();
        if (flattenedInput instanceof io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) {
            inputFlat = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) flattenedInput).buffer;
        } else {
            // 对于非 NdArrayCpu 实现，手动复制数据
            inputFlat = new float[totalSize];
            for (int i = 0; i < totalSize; i++) {
                inputFlat[i] = flattenedInput.get(i);
            }
        }
        
        float[] outputFlat = new float[totalSize];
        
        for (int i = 0; i < totalSize; i++) {
            float x = inputFlat[i];
            outputFlat[i] = x * sigmoid(x);
        }
        
        // 将结果重新形状为原始形状
        NdArray flatOutput = NdArray.of(outputFlat);
        flatOutput = flatOutput.reshape(input.getShape());
        
        // 复制到输出数组
        NdArray flattenedOutput = output.flatten();
        if (flattenedOutput instanceof io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) {
            System.arraycopy(((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) flatOutput.flatten()).buffer, 
                           0, 
                           ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) flattenedOutput).buffer, 
                           0, 
                           totalSize);
        } else {
            // 对于非 NdArrayCpu 实现，使用 set 方法
            for (int i = 0; i < totalSize; i++) {
                flattenedOutput.set(flatOutput.get(i), i);
            }
        }
    }
    
    /**
     * 计算sigmoid函数
     * sigmoid(x) = 1 / (1 + exp(-x))
     * 
     * @param x 输入值
     * @return sigmoid(x)
     */
    private float sigmoid(float x) {
        // 为了数值稳定性，处理极大和极小值
        if (x > 500) {
            return 1.0f;
        } else if (x < -500) {
            return 0.0f;
        }
        
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // SiLU的反向传播需要计算导数
        // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        // 简化实现返回原梯度
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 静态方法：直接应用SiLU激活函数
     * 
     * @param input 输入数组
     * @return 应用SiLU后的数组
     */
    public static NdArray silu(NdArray input) {
        SiLULayer layer = new SiLULayer("temp_silu");
        return layer.applySiLU(input);
    }
}