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
 * 简化版VGG网络实现
 * 
 * VGG网络由牛津大学的Visual Geometry Group提出，在2014年的ImageNet竞赛中取得了优异成绩。
 * VGG网络以其简洁统一的架构而闻名，全部使用3x3的小卷积核和2x2的最大池化。
 * 
 * 本实现是一个简化版的VGG-11结构：
 * INPUT -> CONV1 -> BN1 -> RELU -> POOL1 ->
 * VGG_BLOCK1 -> VGG_BLOCK2 -> VGG_BLOCK3 -> VGG_BLOCK4 -> VGG_BLOCK5 ->
 * ADAPTIVE_POOL -> FLATTEN -> FC1 -> RELU -> FC2 -> RELU -> OUTPUT
 * 
 * 详细结构：
 * 1. 输入层: 224x224x3彩色图像
 * 2. 初始卷积层: 64个3x3卷积核，输出64@224x224
 * 3. 批归一化层1
 * 4. 激活层1: ReLU
 * 5. 池化层1: 2x2最大池化，输出64@112x112
 * 6. VGG块1: 1个卷积层，输出128@56x56
 * 7. VGG块2: 1个卷积层，输出256@28x28
 * 8. VGG块3: 2个卷积层，输出512@14x14
 * 9. VGG块4: 2个卷积层，输出512@7x7
 * 10. 自适应平均池化层: 输出512@1x1
 * 11. 展平层
 * 12. 全连接层1: 4096个神经元
 * 13. 激活层2: ReLU
 * 14. 全连接层2: 4096个神经元
 * 15. 激活层3: ReLU
 * 16. 输出层: numClasses个神经元
 */
public class VGG {
    
    // 初始层
    private ConvLayer conv1;
    private BatchNormLayer bn1;
    private ActivationLayer activation1;
    private PoolingLayer pool1;
    
    // VGG块列表
    private List<VGGBlock> vggBlocks;
    
    // 最终层
    private PoolingLayer adaptivePool;
    private LinearLayer fc1;
    private ActivationLayer activation2;
    private LinearLayer fc2;
    private ActivationLayer activation3;
    private LinearLayer output;
    
    private final int numClasses;
    private final int[] blockConfig; // 每个VGG块中的卷积层数量
    
    /**
     * 构造VGG模型
     * @param numClasses 分类数量
     * @param blockConfig 每个VGG块中的卷积层数量，如[1, 1, 2, 2, 2]表示VGG-11
     */
    public VGG(int numClasses, int[] blockConfig) {
        this.numClasses = numClasses;
        this.blockConfig = blockConfig != null ? blockConfig : new int[]{1, 1, 2, 2, 2};
        
        // 初始化各层
        initLayers();
    }
    
    /**
     * 构造VGG-11模型
     * @param numClasses 分类数量
     */
    public VGG(int numClasses) {
        this(numClasses, new int[]{1, 1, 2, 2, 2}); // VGG-11
    }
    
    /**
     * 构造VGG-11模型（默认1000分类）
     */
    public VGG() {
        this(1000, new int[]{1, 1, 2, 2, 2}); // VGG-11
    }
    
    /**
     * 初始化网络层
     */
    private void initLayers() {
        // 初始卷积层: 输入3通道，输出64通道，3x3卷积核
        conv1 = new ConvLayer("conv1", 3, 64, 3, 1, 1, true);
        
        // 批归一化层1
        bn1 = new BatchNormLayer("bn1", 64);
        
        // 激活层1: ReLU
        activation1 = new ActivationLayer("activation1", ActivationType.RELU);
        
        // 池化层1: 2x2最大池化
        pool1 = new PoolingLayer("pool1", PoolingLayer.PoolingType.MAX, 2, 2, 0);
        
        // VGG块
        vggBlocks = new ArrayList<>();
        
        // 构建VGG块
        int inChannels = 64;
        int outChannels = 64;
        
        for (int i = 0; i < blockConfig.length; i++) {
            // 每个VGG块的输出通道数翻倍（除了第一个块）
            if (i > 0) {
                outChannels *= 2;
            }
            
            // 添加VGG块
            VGGBlock block = new VGGBlock(blockConfig[i], inChannels, outChannels);
            vggBlocks.add(block);
            inChannels = outChannels;
        }
        
        // 自适应平均池化层: 输出512@1x1
        adaptivePool = new PoolingLayer("adaptive_pool", PoolingLayer.PoolingType.ADAPTIVE, 1, 1, 0);
        
        // 全连接层1: 输入512，输出4096
        fc1 = new LinearLayer("fc1", 512, 4096, true);
        
        // 激活层2: ReLU
        activation2 = new ActivationLayer("activation2", ActivationType.RELU);
        
        // 全连接层2: 输入4096，输出4096
        fc2 = new LinearLayer("fc2", 4096, 4096, true);
        
        // 激活层3: ReLU
        activation3 = new ActivationLayer("activation3", ActivationType.RELU);
        
        // 输出层: 输入4096，输出numClasses
        output = new LinearLayer("output", 4096, numClasses, true);
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
        
        // VGG块
        for (VGGBlock block : vggBlocks) {
            x = block.forward(x);
        }
        
        // 自适应平均池化
        x = adaptivePool.layerForward(x);
        
        // 展平
        x = x.flatten();
        
        // 全连接层
        x = fc1.layerForward(x);
        x = activation2.layerForward(x);
        
        x = fc2.layerForward(x);
        x = activation3.layerForward(x);
        
        x = output.layerForward(x);
        
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
        
        for (VGGBlock block : vggBlocks) {
            for (Layer layer : block.getLayers()) {
                layers.add(layer);
            }
        }
        
        layers.add(adaptivePool);
        layers.add(fc1);
        layers.add(activation2);
        layers.add(fc2);
        layers.add(activation3);
        layers.add(output);
        
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
        
        // 加上VGG块的参数
        for (VGGBlock block : vggBlocks) {
            count += block.getParameterCount();
        }
        
        return count;
    }
}