package io.leavesfly.tinyai.qwen3.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.qwen3.Qwen3Config;

/**
 * Qwen3 多层感知机 (MLP) 层
 * 
 * 使用SwiGLU激活函数的前馈网络：
 * MLP(x) = down_proj(SwiGLU(gate_proj(x), up_proj(x)))
 * 其中：
 * - gate_proj: 门控投影层
 * - up_proj: 上投影层  
 * - down_proj: 下投影层
 * - SwiGLU(gate, up) = Swish(gate) ⊙ up
 * 
 * 网络结构：
 * input[hidden_size] -> gate_proj[intermediate_size] -> Swish激活
 *                    -> up_proj[intermediate_size] -> 元素级乘法
 *                    -> down_proj[hidden_size] -> output[hidden_size]
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3MLPLayer extends Layer {
    
    /** 配置对象 */
    private Qwen3Config config;
    
    /** 隐藏层维度 */
    private int hiddenSize;
    
    /** 中间层维度 */
    private int intermediateSize;
    
    /** 门控投影层：hidden_size -> intermediate_size */
    private LinearLayer gateProjection;
    
    /** 上投影层：hidden_size -> intermediate_size */
    private LinearLayer upProjection;
    
    /** 下投影层：intermediate_size -> hidden_size */
    private LinearLayer downProjection;
    
    /**
     * 构造Qwen3 MLP层
     * 
     * @param name 层名称
     * @param config Qwen3配置
     */
    public Qwen3MLPLayer(String name, Qwen3Config config) {
        super(name, Shape.of(-1, -1, config.getHiddenSize()), 
              Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.hiddenSize = config.getHiddenSize();
        this.intermediateSize = config.getIntermediateSize();
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化三个线性投影层
            gateProjection = new LinearLayer(
                name + "_gate", hiddenSize, intermediateSize, false);
            upProjection = new LinearLayer(
                name + "_up", hiddenSize, intermediateSize, false);
            downProjection = new LinearLayer(
                name + "_down", intermediateSize, hiddenSize, false);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        NdArray inputData = input.getValue();
        
        return forwardMLP(inputData);
    }
    
    /**
     * MLP前向传播
     * 
     * @param input 输入数据 [batch_size, seq_len, hidden_size]
     * @return MLP输出
     */
    private Variable forwardMLP(NdArray input) {
        Shape inputShape = input.getShape();
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        
        // 1. 将3D输入重塑为2D用于线性变换
        NdArray input2D = reshape3DTo2D(input, batchSize, seqLen, hiddenSize);
        
        // 2. 门控投影和上投影
        Variable gateOutput = gateProjection.layerForward(new Variable(input2D));
        Variable upOutput = upProjection.layerForward(new Variable(input2D));
        
        // 3. 应用SwiGLU激活：Swish(gate) ⊙ up
        NdArray swiGLUOutput = applySwiGLU(gateOutput.getValue(), upOutput.getValue());
        
        // 4. 下投影
        Variable downOutput = downProjection.layerForward(new Variable(swiGLUOutput));
        
        // 5. 重塑回3D
        NdArray result = reshape2DTo3D(downOutput.getValue(), batchSize, seqLen, hiddenSize);
        
        return new Variable(result);
    }
    
    /**
     * 应用SwiGLU激活函数
     * 
     * @param gate 门控输入
     * @param up 上投影输入
     * @return SwiGLU激活结果
     */
    private NdArray applySwiGLU(NdArray gate, NdArray up) {
        // 验证输入形状一致
        if (!gate.getShape().equals(up.getShape())) {
            throw new IllegalArgumentException(
                String.format("gate形状 %s 必须与up形状 %s 一致", 
                    gate.getShape(), up.getShape()));
        }
        
        Shape shape = gate.getShape();
        NdArray result = NdArray.of(shape);
        
        // SwiGLU计算：Swish(gate) ⊙ up
        int rows = shape.getDimension(0);
        int cols = shape.getDimension(1);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float gateValue = gate.get(i, j);
                float upValue = up.get(i, j);
                
                // Swish激活：x * sigmoid(x)
                float swishValue = swish(gateValue);
                
                // 门控乘法
                float swiGLUValue = swishValue * upValue;
                
                result.set(swiGLUValue, i, j);
            }
        }
        
        return result;
    }
    
    /**
     * Swish激活函数：swish(x) = x * sigmoid(x)
     * 
     * @param x 输入值
     * @return swish激活结果
     */
    private float swish(float x) {
        return x * sigmoid(x);
    }
    
    /**
     * Sigmoid激活函数：sigmoid(x) = 1 / (1 + exp(-x))
     * 
     * @param x 输入值
     * @return sigmoid激活结果
     */
    private float sigmoid(float x) {
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
     * 将3D张量重塑为2D用于线性变换
     * 
     * @param input 输入3D张量 [batch_size, seq_len, hidden_size]
     * @param batchSize 批次大小
     * @param seqLen 序列长度
     * @param hiddenSize 隐藏维度
     * @return 2D张量 [batch_size * seq_len, hidden_size]
     */
    private NdArray reshape3DTo2D(NdArray input, int batchSize, int seqLen, int hiddenSize) {
        NdArray result = NdArray.of(Shape.of(batchSize * seqLen, hiddenSize));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < hiddenSize; h++) {
                    result.set(input.get(b, s, h), b * seqLen + s, h);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 将2D张量重塑回3D
     * 
     * @param input 输入2D张量 [batch_size * seq_len, hidden_size]
     * @param batchSize 批次大小
     * @param seqLen 序列长度
     * @param hiddenSize 隐藏维度
     * @return 3D张量 [batch_size, seq_len, hidden_size]
     */
    private NdArray reshape2DTo3D(NdArray input, int batchSize, int seqLen, int hiddenSize) {
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, hiddenSize));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < hiddenSize; h++) {
                    result.set(input.get(b * seqLen + s, h), b, s, h);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 获取隐藏维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    /**
     * 获取中间维度
     */
    public int getIntermediateSize() {
        return intermediateSize;
    }
    
    /**
     * 获取门控投影层
     */
    public LinearLayer getGateProjection() {
        return gateProjection;
    }
    
    /**
     * 获取上投影层
     */
    public LinearLayer getUpProjection() {
        return upProjection;
    }
    
    /**
     * 获取下投影层
     */
    public LinearLayer getDownProjection() {
        return downProjection;
    }
}