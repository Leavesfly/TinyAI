package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationType;
import io.leavesfly.tinyai.nnet.layer.norm.BatchNormLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.PoolingLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * VGG网络块实现
 * 
 * VGG网络由牛津大学的Visual Geometry Group提出，以其简洁统一的架构而闻名。
 * VGG的核心思想是使用多个小卷积核(3x3)堆叠来替代大卷积核，从而增加网络深度。
 * 
 * VGG块结构：
 * (CONV -> BN -> RELU) x N -> POOL
 * 
 * 其中N通常为2或3，表示连续的卷积层数量
 */
public class VGGBlock {
    
    // 卷积层列表
    private List<ConvLayer> convLayers;
    
    // 批归一化层列表
    private List<BatchNormLayer> bnLayers;
    
    // 激活层列表
    private List<ActivationLayer> activationLayers;
    
    // 池化层
    private PoolingLayer poolLayer;
    
    private final int numConvs; // 卷积层数量
    private final int inChannels; // 输入通道数
    private final int outChannels; // 输出通道数
    
    /**
     * 构造VGG块
     * @param numConvs 卷积层数量
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     */
    public VGGBlock(int numConvs, int inChannels, int outChannels) {
        this.numConvs = numConvs;
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        
        // 初始化各层
        initLayers();
    }
    
    /**
     * 初始化网络层
     */
    private void initLayers() {
        convLayers = new ArrayList<>();
        bnLayers = new ArrayList<>();
        activationLayers = new ArrayList<>();
        
        // 添加指定数量的卷积层
        int currentChannels = inChannels;
        for (int i = 0; i < numConvs; i++) {
            // 3x3卷积层，填充1以保持尺寸不变
            ConvLayer conv = new ConvLayer("conv" + (i + 1), currentChannels, outChannels, 3, 1, 1, true);
            convLayers.add(conv);
            
            // 批归一化层
            BatchNormLayer bn = new BatchNormLayer("bn" + (i + 1), outChannels);
            bnLayers.add(bn);
            
            // ReLU激活层
            ActivationLayer activation = new ActivationLayer("activation" + (i + 1), ActivationType.RELU);
            activationLayers.add(activation);
            
            currentChannels = outChannels;
        }
        
        // 2x2最大池化层，步长2
        poolLayer = new PoolingLayer("pool", PoolingLayer.PoolingType.MAX, 2, 2, 0);
    }
    
    /**
     * 前向传播
     * @param input 输入特征图
     * @return 输出特征图
     */
    public Variable forward(Variable input) {
        Variable x = input;
        
        // 依次通过所有卷积层、批归一化层和激活层
        for (int i = 0; i < numConvs; i++) {
            x = convLayers.get(i).layerForward(x);
            x = bnLayers.get(i).layerForward(x);
            x = activationLayers.get(i).layerForward(x);
        }
        
        // 最后通过池化层
        x = poolLayer.layerForward(x);
        
        return x;
    }
    
    /**
     * 获取VGG块的所有层
     * @return 层列表
     */
    public Layer[] getLayers() {
        List<Layer> layers = new ArrayList<>();
        
        for (int i = 0; i < numConvs; i++) {
            layers.add(convLayers.get(i));
            layers.add(bnLayers.get(i));
            layers.add(activationLayers.get(i));
        }
        
        layers.add(poolLayer);
        
        return layers.toArray(new Layer[0]);
    }
    
    /**
     * 获取VGG块的参数数量
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