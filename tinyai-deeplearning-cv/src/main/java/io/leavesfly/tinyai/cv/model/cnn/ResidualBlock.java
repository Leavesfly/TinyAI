package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationType;
import io.leavesfly.tinyai.nnet.layer.norm.BatchNormLayer;

/**
 * ResNet残差块实现
 * 
 * 残差块是ResNet网络的核心组件，通过引入残差连接解决了深度网络训练中的梯度消失问题。
 * 
 * 残差块结构：
 * INPUT -> CONV1 -> BN1 -> RELU -> CONV2 -> BN2 -> ADD -> RELU -> OUTPUT
 *            |                                            ^
 *            |____________________________________________|
 *                           残差连接
 * 
 * 两种残差块类型：
 * 1. 基本块(Basic Block): 两个3x3卷积层
 * 2. 瓶颈块(Bottleneck Block): 1x1 -> 3x3 -> 1x1卷积层
 * 
 * 本实现为基本块版本
 */
public class ResidualBlock {
    
    // 卷积层
    private ConvLayer conv1;
    private ConvLayer conv2;
    
    // 批归一化层
    private BatchNormLayer bn1;
    private BatchNormLayer bn2;
    
    // 激活层
    private ActivationLayer activation1;
    private ActivationLayer activation2;
    
    // 用于维度匹配的1x1卷积层（当输入输出维度不匹配时使用）
    private ConvLayer shortcutConv;
    private BatchNormLayer shortcutBn;
    
    private final int inChannels;
    private final int outChannels;
    private final int stride;
    
    /**
     * 构造残差块
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     * @param stride 步长（用于下采样）
     */
    public ResidualBlock(int inChannels, int outChannels, int stride) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.stride = stride;
        
        // 第一个卷积层
        conv1 = new ConvLayer("conv1", inChannels, outChannels, 3, stride, 1, true);
        
        // 第一个批归一化层
        bn1 = new BatchNormLayer("bn1", outChannels);
        
        // 第一个激活层: ReLU
        activation1 = new ActivationLayer("activation1", ActivationType.RELU);
        
        // 第二个卷积层
        conv2 = new ConvLayer("conv2", outChannels, outChannels, 3, 1, 1, true);
        
        // 第二个批归一化层
        bn2 = new BatchNormLayer("bn2", outChannels);
        
        // 第二个激活层: ReLU
        activation2 = new ActivationLayer("activation2", ActivationType.RELU);
        
        // 当输入输出维度不匹配时，需要添加1x1卷积层进行维度匹配
        if (inChannels != outChannels || stride != 1) {
            shortcutConv = new ConvLayer("shortcut_conv", inChannels, outChannels, 1, stride, 0, true);
            shortcutBn = new BatchNormLayer("shortcut_bn", outChannels);
        }
    }
    
    /**
     * 前向传播
     * @param input 输入特征图
     * @return 输出特征图
     */
    public Variable forward(Variable input) {
        // 主路径
        Variable residual = input;
        
        // 第一个卷积块: 卷积 -> 批归一化 -> 激活
        Variable x = conv1.layerForward(input);
        x = bn1.layerForward(x);
        x = activation1.layerForward(x);
        
        // 第二个卷积块: 卷积 -> 批归一化
        x = conv2.layerForward(x);
        x = bn2.layerForward(x);
        
        // 残差连接
        Variable shortcut;
        if (shortcutConv != null) {
            // 当输入输出维度不匹配时，使用1x1卷积进行维度匹配
            shortcut = shortcutConv.layerForward(residual);
            shortcut = shortcutBn.layerForward(shortcut);
        } else {
            // 当输入输出维度匹配时，直接使用输入作为捷径连接
            shortcut = residual;
        }
        
        // 将主路径和捷径连接相加
        Variable output = x.add(shortcut);
        
        // 最后的激活函数
        output = activation2.layerForward(output);
        
        return output;
    }
    
    /**
     * 获取残差块的所有层
     * @return 层列表
     */
    public Layer[] getLayers() {
        if (shortcutConv != null) {
            return new Layer[] {
                conv1, bn1, activation1,
                conv2, bn2,
                shortcutConv, shortcutBn,
                activation2
            };
        } else {
            return new Layer[] {
                conv1, bn1, activation1,
                conv2, bn2,
                activation2
            };
        }
    }
    
    /**
     * 获取残差块的参数数量
     * @return 参数数量
     */
    public int getParameterCount() {
        int count = 0;
        Layer[] layers = getLayers();
        for (Layer layer : layers) {
            count += layer.getParamNum();
        }
        return count;
    }
}