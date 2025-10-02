package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen3前馈神经网络层（MLP）
 * 
 * 实现了Qwen3模型的前馈神经网络，使用SwiGLU激活函数。
 * SwiGLU是一种门控激活函数，结合了Swish（SiLU）激活和门控机制，
 * 在大语言模型中表现出色。
 * 
 * 网络结构：
 * 1. gate_proj: 门控投影 (hidden_size -> intermediate_size)
 * 2. up_proj: 上投影 (hidden_size -> intermediate_size) 
 * 3. SiLU激活函数
 * 4. down_proj: 下投影 (intermediate_size -> hidden_size)
 * 
 * 计算公式：
 * SwiGLU(x) = SiLU(gate_proj(x)) ⊙ up_proj(x)
 * output = down_proj(SwiGLU(x))
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3MLP extends Layer {
    
    /**
     * 配置信息
     */
    private Qwen3Config config;
    
    /**
     * 隐藏层维度
     */
    private int hiddenSize;
    
    /**
     * 中间层维度
     */
    private int intermediateSize;
    
    /**
     * 门控投影层
     */
    private LinearLayer gateProj;
    
    /**
     * 上投影层
     */
    private LinearLayer upProj;
    
    /**
     * 下投影层
     */
    private LinearLayer downProj;
    
    /**
     * SiLU激活函数层
     */
    private SiLULayer actFn;

    /**
     * 构造Qwen3前馈网络层
     * 
     * @param name 层名称
     * @param config 配置信息
     */
    public Qwen3MLP(String name, Qwen3Config config) {
        super(name, Shape.of(-1, -1, config.getHiddenSize()), Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.hiddenSize = config.getHiddenSize();
        this.intermediateSize = config.getIntermediateSize();
        
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 门控线性投影层（不使用偏置）
            gateProj = new LinearLayer(name + "_gate_proj", hiddenSize, intermediateSize, false);
            
            // 上投影层（不使用偏置）
            upProj = new LinearLayer(name + "_up_proj", hiddenSize, intermediateSize, false);
            
            // 下投影层（不使用偏置）
            downProj = new LinearLayer(name + "_down_proj", intermediateSize, hiddenSize, false);
            
            // SiLU激活函数
            actFn = new SiLULayer(name + "_activation", Shape.of(-1, -1, intermediateSize));
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        
        // 获取输入形状
        NdArray inputData = input.getValue();
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 将三维输入重塑为二维以进行矩阵乘法
        NdArray inputReshaped = inputData.reshape(Shape.of(batchSize * seqLen, hiddenSize));
        Variable reshapedInput = new Variable(inputReshaped);
        
        // 门控路径：gate_proj + SiLU激活
        Variable gateOutput = gateProj.layerForward(reshapedInput);
        Variable gateActivated = applySiLUActivation(gateOutput, batchSize, seqLen);
        
        // 上投影路径
        Variable upOutput = upProj.layerForward(reshapedInput);
        
        // SwiGLU：门控激活 * 上投影
        Variable swiGLUOutput = elementwiseMultiply(gateActivated, upOutput, batchSize, seqLen);
        
        // 下投影
        Variable downOutput = downProj.layerForward(swiGLUOutput);
        
        // 重塑回三维
        NdArray result = downOutput.getValue().reshape(Shape.of(batchSize, seqLen, hiddenSize));
        
        return new Variable(result);
    }
    
    /**
     * 应用SiLU激活函数
     */
    private Variable applySiLUActivation(Variable input, int batchSize, int seqLen) {
        // 重塑为三维以使用SiLU层
        NdArray inputData = input.getValue();
        NdArray reshaped3D = inputData.reshape(Shape.of(batchSize, seqLen, intermediateSize));
        
        // 应用SiLU激活
        Variable activated = actFn.layerForward(new Variable(reshaped3D));
        
        // 重塑回二维
        NdArray result = activated.getValue().reshape(Shape.of(batchSize * seqLen, intermediateSize));
        return new Variable(result);
    }
    
    /**
     * 逐元素乘法（SwiGLU的关键步骤）
     */
    private Variable elementwiseMultiply(Variable gate, Variable up, int batchSize, int seqLen) {
        NdArray gateData = gate.getValue();
        NdArray upData = up.getValue();
        
        // 验证形状匹配
        if (!gateData.getShape().equals(upData.getShape())) {
            throw new IllegalArgumentException("门控输出和上投影输出形状不匹配");
        }
        
        NdArray result = NdArray.of(gateData.getShape());
        int totalSize = batchSize * seqLen;
        
        // 逐元素相乘
        for (int i = 0; i < totalSize; i++) {
            for (int j = 0; j < intermediateSize; j++) {
                float gateVal = gateData.get(i, j);
                float upVal = upData.get(i, j);
                result.set(gateVal * upVal, i, j);
            }
        }
        
        return new Variable(result);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 前馈网络的反向传播需要依次通过各层
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
     * 获取配置信息
     * 
     * @return 配置信息
     */
    public Qwen3Config getConfig() {
        return config;
    }
    
    /**
     * 获取隐藏层维度
     * 
     * @return 隐藏层维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    /**
     * 获取中间层维度
     * 
     * @return 中间层维度
     */
    public int getIntermediateSize() {
        return intermediateSize;
    }
    
    /**
     * 获取门控投影层
     * 
     * @return 门控投影层
     */
    public LinearLayer getGateProj() {
        return gateProj;
    }
    
    /**
     * 获取上投影层
     * 
     * @return 上投影层
     */
    public LinearLayer getUpProj() {
        return upProj;
    }
    
    /**
     * 获取下投影层
     * 
     * @return 下投影层
     */
    public LinearLayer getDownProj() {
        return downProj;
    }
    
    /**
     * 获取激活函数层
     * 
     * @return 激活函数层
     */
    public SiLULayer getActFn() {
        return actFn;
    }
}