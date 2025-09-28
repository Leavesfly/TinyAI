package io.leavesfly.tinyai.modality.cv;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.LayerAble;
import io.leavesfly.tinyai.nnet.block.SequentialBlock;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.PoolingLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.AffineLayer;
import io.leavesfly.tinyai.nnet.layer.norm.BatchNorm;
import io.leavesfly.tinyai.nnet.layer.norm.Dropout;
import io.leavesfly.tinyai.nnet.layer.norm.Flatten;

/**
 * 增强的深度卷积神经网络实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * SimpleConvNet类实现了增强的深度卷积神经网络，包含多个卷积层、池化层、全连接层和正则化层的深度架构。
 * 支持批量归一化、残差连接和灵活配置，适用于图像分类等计算机视觉任务。
 * <p>
 * 网络结构：
 * 1. 第一个卷积块: Conv -> BatchNorm -> ReLU -> MaxPool
 * 2. 第二个卷积块: Conv -> BatchNorm -> ReLU -> MaxPool
 * 3. 第三个卷积块: Conv -> BatchNorm -> ReLU -> MaxPool
 * 4. 展平层: Flatten
 * 5. 第一个全连接层: Linear -> BatchNorm -> ReLU -> Dropout
 * 6. 第二个全连接层: Linear -> BatchNorm -> ReLU -> Dropout
 * 7. 输出层: Linear
 */
public class SimpleConvNet extends SequentialBlock {

    // 网络超参数
    private final int numClasses;
    private final boolean useBatchNorm;
    private final float dropoutRate;
    
    /**
     * 构造函数，创建一个标准的CNN网络
     *
     * @param _name         块的名称
     * @param _xInputShape  输入数据的形状 (batch_size, channels, height, width)
     * @param numClasses    输出类别数
     */
    public SimpleConvNet(String _name, Shape _xInputShape, int numClasses) {
        this(_name, _xInputShape, numClasses, true, 0.5f);
    }
    
    /**
     * 构造函数，创建一个可配置的CNN网络
     *
     * @param _name         块的名称
     * @param _xInputShape  输入数据的形状 (batch_size, channels, height, width)
     * @param numClasses    输出类别数
     * @param useBatchNorm  是否使用批量归一化
     * @param dropoutRate   Dropout比例
     */
    public SimpleConvNet(String _name, Shape _xInputShape, int numClasses, 
                        boolean useBatchNorm, float dropoutRate) {
        super(_name, _xInputShape, Shape.of(-1, numClasses));
        
        this.numClasses = numClasses;
        this.useBatchNorm = useBatchNorm;
        this.dropoutRate = dropoutRate;
        
        // 构建网络架构
        buildNetwork();
    }
    
    /**
     * 构建完整的CNN网络架构
     */
    private void buildNetwork() {
        if (inputShape == null || inputShape.getDimNum() != 4) {
            throw new IllegalArgumentException("输入形状必须是4维的 (batch_size, channels, height, width)");
        }
        
        int inputChannels = inputShape.getDimension(1);
        int inputHeight = inputShape.getDimension(2);
        int inputWidth = inputShape.getDimension(3);
        
        // 第一个卷积块 (32个3x3卷积核)
        addConvBlock("conv1", inputChannels, 32, 3, 1, 1);
        addPoolingLayer("pool1", PoolingLayer.PoolingType.MAX, 2, 2, 0);
        
        // 第二个卷积块 (64个3x3卷积核)
        addConvBlock("conv2", 32, 64, 3, 1, 1);
        addPoolingLayer("pool2", PoolingLayer.PoolingType.MAX, 2, 2, 0);
        
        // 第三个卷积块 (128个3x3卷积核)
        addConvBlock("conv3", 64, 128, 3, 1, 1);
        addPoolingLayer("pool3", PoolingLayer.PoolingType.MAX, 2, 2, 0);
        
        // 展平层
        addLayer(new Flatten("flatten", null));
        
        // 计算展平后的特征数
        int flattenedSize = calculateFlattenedSize(inputHeight, inputWidth, 128);
        
        // 第一个全连接层 (512个神经元)
        addFullyConnectedBlock("fc1", flattenedSize, 512);
        
        // 第二个全连接层 (256个神经元)
        addFullyConnectedBlock("fc2", 512, 256);
        
        // 输出层 (不使用激活函数)
        addLayer(new AffineLayer("output", Shape.of(-1, 256), numClasses, true));
    }
    
    /**
     * 添加卷积块 (Conv -> BatchNorm -> ReLU)
     */
    private void addConvBlock(String prefix, int inChannels, int outChannels, 
                             int kernelSize, int stride, int padding) {
        // 卷积层
        addLayer(new ConvLayer(prefix + "_conv", inChannels, outChannels, 
                              kernelSize, stride, padding, true));
        
        // 批量归一化层（可选）
        if (useBatchNorm) {
            addLayer(new BatchNorm(prefix + "_bn", null, outChannels));
        }
        
        // ReLU激活函数
        addLayer(new ReLuLayer(prefix + "_relu"));
    }
    
    /**
     * 添加池化层
     */
    private void addPoolingLayer(String name, PoolingLayer.PoolingType type, 
                                int poolSize, int stride, int padding) {
        addLayer(new PoolingLayer(name, type, poolSize, stride, padding));
    }
    
    /**
     * 添加全连接块 (Linear -> BatchNorm -> ReLU -> Dropout)
     */
    private void addFullyConnectedBlock(String prefix, int inputSize, int outputSize) {
        // 全连接层
        addLayer(new AffineLayer(prefix + "_linear", Shape.of(-1, inputSize), outputSize, true));
        
        // 批量归一化层（可选）
        if (useBatchNorm) {
            addLayer(new BatchNorm(prefix + "_bn", null, outputSize));
        }
        
        // ReLU激活函数
        addLayer(new ReLuLayer(prefix + "_relu"));
        
        // Dropout层（可选）
        if (dropoutRate > 0.0f) {
            addLayer(new Dropout(prefix + "_dropout", dropoutRate));
        }
    }
    
    /**
     * 计算经过卷积和池化后的展平特征数
     * 假设经过3次最大池化，每次池化步长为2
     */
    private int calculateFlattenedSize(int inputHeight, int inputWidth, int finalChannels) {
        // 每次池化后尺寸减半
        int finalHeight = inputHeight;
        int finalWidth = inputWidth;
        
        // 经过3次池化
        for (int i = 0; i < 3; i++) {
            finalHeight = finalHeight / 2;
            finalWidth = finalWidth / 2;
        }
        
        return finalChannels * finalHeight * finalWidth;
    }
    
    @Override
    public void init() {
        // 初始化所有层
        for (LayerAble layer : layers) {
            if (layer != null) {
                layer.init();
            }
        }
    }
    
    /**
     * 获取网络配置信息
     */
    public String getNetworkInfo() {
        StringBuilder info = new StringBuilder();
        info.append("SimpleConvNet配置:\n");
        info.append("- 输入形状: ").append(inputShape).append("\n");
        info.append("- 输出类别数: ").append(numClasses).append("\n");
        info.append("- 使用批量归一化: ").append(useBatchNorm).append("\n");
        info.append("- Dropout比例: ").append(dropoutRate).append("\n");
        info.append("- 网络层数: ").append(layers.size()).append("\n");
        return info.toString();
    }
    
    /**
     * 打印网络架构
     */
    public void printArchitecture() {
        System.out.println("=== SimpleConvNet 网络架构 ===");
        System.out.println(getNetworkInfo());
        System.out.println("\n层次结构:");
        for (int i = 0; i < layers.size(); i++) {
            LayerAble layer = layers.get(i);
            System.out.printf("%2d. %s (%s)\n", i + 1, layer.getClass().getSimpleName(), layer.getClass().getSimpleName());
        }
        System.out.println("==============================");
    }
    
    /**
     * 获取网络层数
     * 
     * @return 网络中的层数
     */
    public int getLayersCount() {
        return layers.size();
    }
    
    /**
     * 获取输出类别数
     * 
     * @return 输出类别数
     */
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * 是否使用批量归一化
     * 
     * @return 是否使用批量归一化
     */
    public boolean isUseBatchNorm() {
        return useBatchNorm;
    }
    
    /**
     * 获取Dropout比例
     * 
     * @return Dropout比例
     */
    public float getDropoutRate() {
        return dropoutRate;
    }
    
    /**
     * 获取指定索引的层
     * 
     * @param index 层的索引
     * @return 对应的层对象
     * @throws IndexOutOfBoundsException 如果索引超出范围
     */
    public LayerAble getLayer(int index) {
        if (index < 0 || index >= layers.size()) {
            throw new IndexOutOfBoundsException("层索引超出范围: " + index);
        }
        return layers.get(index);
    }
    
    /**
     * 获取网络的简要摘要信息
     * 
     * @return 网络摘要字符串
     */
    public String getSummary() {
        return String.format("SimpleConvNet[类别数=%d, 层数=%d, 批量归一化=%s, Dropout=%.2f]",
                numClasses, layers.size(), useBatchNorm, dropoutRate);
    }

}