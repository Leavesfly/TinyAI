package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * 深度可分离卷积层（简化版本）
 * 实现MobileNet中的Depthwise Separable Convolution
 * 包含深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）
 * 
 * Depthwise Separable Convolution = Depthwise Convolution + Pointwise Convolution
 * - Depthwise: 对每个输入通道分别应用一个卷积核
 * - Pointwise: 使用1x1卷积来组合特征
 */
public class DepthwiseSeparableConvLayer extends Layer {

    private ConvLayer depthwiseConv;  // 深度卷积层
    private ConvLayer pointwiseConv;  // 逐点卷积层
    
    private int inChannels;          // 输入通道数
    private int outChannels;         // 输出通道数
    private int kernelSize;          // 深度卷积核尺寸
    private int stride;              // 步长
    private int padding;             // 填充
    private boolean useBias;         // 是否使用偏置
    
    /**
     * 构造深度可分离卷积层
     * 
     * @param name 层名称
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize 深度卷积核尺寸
     * @param stride 步长
     * @param padding 填充
     * @param useBias 是否使用偏置
     */
    public DepthwiseSeparableConvLayer(String name, int inChannels, int outChannels, int kernelSize,
                                       int stride, int padding, boolean useBias) {
        super(name, null, null);
        
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;
        
        init();
    }
    
    /**
     * 简化构造函数
     */
    public DepthwiseSeparableConvLayer(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        
        // 从输入形状推断参数
        if (_inputShape != null && _inputShape.size() == 4) {
            this.inChannels = _inputShape.getDimension(1);
        } else {
            this.inChannels = 32;
        }
        
        this.outChannels = inChannels * 2;  // 默认输出通道数为输入的两倍
        this.kernelSize = 3;
        this.stride = 1;
        this.padding = 1;
        this.useBias = true;
        
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化深度卷积层
            // 深度卷积：每个输入通道对应一个输出通道
            // 每个通道使用一个独立的卷积核
            depthwiseConv = new ConvLayer(
                name + "_depthwise",
                inChannels,    // 输入通道数
                inChannels,    // 输出通道数(与输入相同)
                kernelSize,    // 卷积核尺寸
                stride,        // 步长
                padding,       // 填充
                useBias        // 是否使用偏置
            );
            
            // 2. 初始化逐点卷积层
            // 逐点卷积：使用1x1卷积核组合特征
            pointwiseConv = new ConvLayer(
                name + "_pointwise",
                inChannels,    // 输入通道数(深度卷积的输出)
                outChannels,   // 期望的输出通道数
                1,             // 1x1卷积核
                1,             // 步长为1
                0,             // 无填充
                useBias        // 是否使用偏置
            );
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        
        // 检查输入形状
        NdArray inputData = x.getValue();
        if (inputData.getShape().size() != 4) {
            throw new RuntimeException("深度可分离卷积层输入必须是4维的: (batch_size, channels, height, width)");
        }
        
        // 步骤1：深度卷积
        Variable depthwiseOutput = performDepthwiseConv(x);
        
        // 步骤2：逐点卷积
        Variable pointwiseOutput = pointwiseConv.layerForward(depthwiseOutput);
        
        return pointwiseOutput;
    }
    
    /**
     * 执行深度卷积操作(简化实现)
     * 在实际实现中，深度卷积需要对每个通道分别应用卷积核
     */
    private Variable performDepthwiseConv(Variable x) {
        NdArray inputData = x.getValue();
        
        int batchSize = inputData.getShape().getDimension(0);
        int channels = inputData.getShape().getDimension(1);
        int height = inputData.getShape().getDimension(2);
        int width = inputData.getShape().getDimension(3);
        
        // 计算输出尺寸
        int outputHeight = (height + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (width + 2 * padding - kernelSize) / stride + 1;
        
        // 简化实现：直接使用普通卷积作为深度卷积的近似
        // 在实际实现中，应该对每个通道分别应用卷积核
        return depthwiseConv.layerForward(x);
    }
    
    /**
     * 获取深度卷积层
     */
    public ConvLayer getDepthwiseConv() {
        return depthwiseConv;
    }
    
    /**
     * 获取逐点卷积层
     */
    public ConvLayer getPointwiseConv() {
        return pointwiseConv;
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 深度可分离卷积的反向传播比较复杂，这里提供简化版本
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 清理梯度
     */
    @Override
    public void clearGrads() {
        if (depthwiseConv != null) {
            depthwiseConv.clearGrads();
        }
        if (pointwiseConv != null) {
            pointwiseConv.clearGrads();
        }
    }
}