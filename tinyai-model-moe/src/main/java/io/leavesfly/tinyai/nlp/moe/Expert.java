package io.leavesfly.tinyai.nlp.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * MoE专家网络类
 * <p>
 * 专家网络是Mixture of Experts (MoE)架构中的核心组件，
 * 每个专家都是一个独立的前馈神经网络，专门处理特定类型的输入。
 * <p>
 * 结构：
 * Input → Linear1 → ReLU → Linear2 → Output
 * <p>
 * 特点：
 * 1. 每个专家都有相同的网络结构但不同的参数
 * 2. 可以通过增加专家数量来增加模型容量
 * 3. 只有被激活的专家才会参与计算，实现稀疏计算
 * 4. 不同专家可以学习处理不同类型的语言模式
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Expert extends Layer {
    
    private LinearLayer firstLinear;    // 第一个线性层
    private ReLuLayer activation;       // 激活函数层
    private LinearLayer secondLinear;   // 第二个线性层
    
    private int dModel;                 // 输入/输出维度
    private int dExpert;                // 专家隐藏层维度
    private int expertId;               // 专家ID标识
    
    /**
     * 构造专家网络
     * 
     * @param name 专家网络名称
     * @param expertId 专家ID（用于标识不同专家）
     * @param dModel 输入和输出维度
     * @param dExpert 隐藏层维度（专家容量）
     */
    public Expert(String name, int expertId, int dModel, int dExpert) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        this.expertId = expertId;
        this.dModel = dModel;
        this.dExpert = dExpert;
        init();
    }
    
    /**
     * 使用默认隐藏维度的构造函数（4倍dModel）
     */
    public Expert(String name, int expertId, int dModel) {
        this(name, expertId, dModel, dModel * 4);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 第一个线性层：dModel -> dExpert
            firstLinear = new LinearLayer(
                name + "_expert" + expertId + "_linear1", 
                dModel, 
                dExpert, 
                true
            );
            
            // ReLU激活函数
            activation = new ReLuLayer(
                name + "_expert" + expertId + "_relu", 
                Shape.of(-1, -1, dExpert)
            );
            
            // 第二个线性层：dExpert -> dModel
            secondLinear = new LinearLayer(
                name + "_expert" + expertId + "_linear2", 
                dExpert, 
                dModel, 
                true
            );
            
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
        
        // 验证输入维度
        if (inputData.getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException(
                String.format("专家%d输入维度不匹配。期望%d，实际%d", 
                             expertId, dModel, inputData.getShape().getDimension(2))
            );
        }
        
        // 将三维输入重塑为二维进行线性变换
        NdArray inputReshaped = inputData.reshape(Shape.of(batchSize * seqLen, dModel));
        Variable reshapedInput = new Variable(inputReshaped);
        
        // 第一个线性变换: dModel -> dExpert
        Variable hidden = firstLinear.layerForward(reshapedInput);
        
        // ReLU激活
        Variable activated = activation.layerForward(hidden);
        
        // 第二个线性变换: dExpert -> dModel
        Variable output2D = secondLinear.layerForward(activated);
        
        // 重塑回三维
        NdArray output3D = output2D.getValue().reshape(Shape.of(batchSize, seqLen, dModel));
        
        return new Variable(output3D);
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 专家网络的反向传播需要依次通过各层
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 获取专家ID
     */
    public int getExpertId() {
        return expertId;
    }
    
    /**
     * 获取输入/输出维度
     */
    public int getDModel() {
        return dModel;
    }
    
    /**
     * 获取隐藏层维度
     */
    public int getDExpert() {
        return dExpert;
    }
    
    /**
     * 获取第一个线性层
     */
    public LinearLayer getFirstLinear() {
        return firstLinear;
    }
    
    /**
     * 获取激活函数层
     */
    public ReLuLayer getActivation() {
        return activation;
    }
    
    /**
     * 获取第二个线性层
     */
    public LinearLayer getSecondLinear() {
        return secondLinear;
    }
    
    /**
     * 计算专家的参数数量
     */
    public long getParameterCount() {
        long params = 0;
        
        // 第一个线性层: (dModel + 1) * dExpert (包含偏置)
        params += (dModel + 1) * dExpert;
        
        // 第二个线性层: (dExpert + 1) * dModel (包含偏置)
        params += (dExpert + 1) * dModel;
        
        return params;
    }
    
    @Override
    public String toString() {
        return String.format("Expert(id=%d, dModel=%d, dExpert=%d, params=%d)", 
                           expertId, dModel, dExpert, getParameterCount());
    }
}