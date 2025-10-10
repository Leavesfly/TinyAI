package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.PoolingLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationType;
import io.leavesfly.tinyai.nnet.layer.norm.BatchNormLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 简化版ResNet实现
 * 
 * ResNet(Residual Network)由微软研究院的何恺明等人在2015年提出，
 * 通过引入残差连接解决了深度网络训练中的梯度消失问题，使得训练极深的网络成为可能。
 * 
 * 本实现是一个简化版的ResNet-18结构：
 * INPUT -> CONV1 -> BN1 -> RELU -> POOL1 ->
 * RES_LAYER1 -> RES_LAYER2 -> RES_LAYER3 -> RES_LAYER4 ->
 * AVG_POOL -> FLATTEN -> FC -> OUTPUT
 * 
 * 详细结构：
 * 1. 输入层: 224x224x3彩色图像
 * 2. 初始卷积层: 64个7x7卷积核，步长2，输出64@112x112
 * 3. 批归一化层1
 * 4. 激活层1: ReLU
 * 5. 池化层1: 3x3最大池化，步长2，输出64@56x56
 * 6. 残差层1: 2个基本残差块，输出64@56x56
 * 7. 残差层2: 2个基本残差块，输出128@28x28
 * 8. 残差层3: 2个基本残差块，输出256@14x14
 * 9. 残差层4: 2个基本残差块，输出512@7x7
 * 10. 全局平均池化层: 输出512@1x1
 * 11. 展平层
 * 12. 全连接层: 输出numClasses
 */
public class ResNet {
    
    // 初始层
    private ConvLayer conv1;
    private BatchNormLayer bn1;
    private ActivationLayer activation1;
    private PoolingLayer pool1;
    
    // 残差层
    private List<ResidualBlock> resLayers;
    
    // 最终层
    private PoolingLayer globalAvgPool;
    private LinearLayer fc;
    
    private final int numClasses;
    private final int[] layerConfig; // 每个残差层中的残差块数量
    
    /**
     * 构造ResNet模型
     * @param numClasses 分类数量
     * @param layerConfig 每个残差层中的残差块数量，如[2, 2, 2, 2]表示ResNet-18
     */
    public ResNet(int numClasses, int[] layerConfig) {
        this.numClasses = numClasses;
        this.layerConfig = layerConfig != null ? layerConfig : new int[]{2, 2, 2, 2};
        
        // 初始化各层
        initLayers();
    }
    
    /**
     * 构造ResNet-18模型
     * @param numClasses 分类数量
     */
    public ResNet(int numClasses) {
        this(numClasses, new int[]{2, 2, 2, 2}); // ResNet-18
    }
    
    /**
     * 构造ResNet-18模型（默认1000分类）
     */
    public ResNet() {
        this(1000, new int[]{2, 2, 2, 2}); // ResNet-18
    }
    
    /**
     * 初始化网络层
     */
    private void initLayers() {
        // 初始卷积层: 输入3通道，输出64通道，7x7卷积核，步长2
        conv1 = new ConvLayer("conv1", 3, 64, 7, 2, 3, true);
        
        // 批归一化层1
        bn1 = new BatchNormLayer("bn1", 64);
        
        // 激活层1: ReLU
        activation1 = new ActivationLayer("activation1", ActivationType.RELU);
        
        // 池化层1: 3x3最大池化，步长2
        pool1 = new PoolingLayer("pool1", PoolingLayer.PoolingType.MAX, 3, 2, 1);
        
        // 残差层
        resLayers = new ArrayList<>();
        
        // 构建残差层
        int inChannels = 64;
        int outChannels = 64;
        int stride = 1;
        
        for (int i = 0; i < layerConfig.length; i++) {
            // 对于每个残差层
            if (i > 0) {
                // 除了第一个残差层，其他层的第一个残差块需要下采样
                outChannels *= 2;
                stride = 2;
            } else {
                stride = 1;
            }
            
            // 添加指定数量的残差块
            for (int j = 0; j < layerConfig[i]; j++) {
                ResidualBlock block = new ResidualBlock(inChannels, outChannels, j == 0 ? stride : 1);
                resLayers.add(block);
                inChannels = outChannels;
            }
        }
        
        // 全局平均池化层
        globalAvgPool = new PoolingLayer("global_avg_pool", PoolingLayer.PoolingType.ADAPTIVE, 1, 1, 0);
        
        // 全连接层: 输入512，输出numClasses
        fc = new LinearLayer("fc", 512, numClasses, true);
    }
    
    /**
     * 前向传播
     * @param input 输入图像 (batch_size, 3, 224, 224)
     * @return 输出结果 (batch_size, numClasses)
     */
    public Variable forward(Variable input) {
        // 初始层: 卷积 -> 批归一化 -> 激活 -> 池化
        Variable x = conv1.layerForward(input);
        x = bn1.layerForward(x);
        x = activation1.layerForward(x);
        x = pool1.layerForward(x);
        
        // 残差层
        for (ResidualBlock block : resLayers) {
            x = block.forward(x);
        }
        
        // 全局平均池化
        x = globalAvgPool.layerForward(x);
        
        // 展平
        x = x.flatten();
        
        // 全连接层
        x = fc.layerForward(x);
        
        return x;
    }
    
    /**
     * 获取模型的所有层
     * @return 层列表
     */
    public Layer[] getLayers() {
        List<Layer> layers = new ArrayList<>();
        
        layers.add(conv1);
        layers.add(bn1);
        layers.add(activation1);
        layers.add(pool1);
        
        for (ResidualBlock block : resLayers) {
            for (Layer layer : block.getLayers()) {
                layers.add(layer);
            }
        }
        
        layers.add(globalAvgPool);
        layers.add(fc);
        
        return layers.toArray(new Layer[0]);
    }
    
    /**
     * 获取模型的参数数量
     * @return 参数数量
     */
    public int getParameterCount() {
        int count = 0;
        Layer[] layers = getLayers();
        for (Layer layer : layers) {
            count += layer.getParamNum();
        }
        
        // 加上残差块的参数
        for (ResidualBlock block : resLayers) {
            count += block.getParameterCount();
        }
        
        return count;
    }
}