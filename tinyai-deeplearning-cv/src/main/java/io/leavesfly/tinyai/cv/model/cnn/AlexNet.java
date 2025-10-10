package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.PoolingLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationType;
import io.leavesfly.tinyai.nnet.layer.norm.BatchNormLayer;

/**
 * 简化版AlexNet卷积神经网络实现
 * 
 * 原始AlexNet由Alex Krizhevsky等人在2012年提出，在ImageNet竞赛中取得了突破性成果。
 * 本实现是简化版本，适用于较小的数据集。
 * 
 * 网络结构：
 * INPUT -> CONV1 -> RELU -> POOL1 -> BN1 ->
 * CONV2 -> RELU -> POOL2 -> BN2 ->
 * CONV3 -> RELU -> BN3 ->
 * CONV4 -> RELU -> BN4 ->
 * CONV5 -> RELU -> POOL5 -> BN5 ->
 * FC1 -> RELU -> DROPOUT1 ->
 * FC2 -> RELU -> DROPOUT2 ->
 * OUTPUT
 * 
 * 详细结构：
 * 1. 输入层: 224x224x3彩色图像
 * 2. 卷积层1: 64个11x11卷积核，步长4，输出64@55x55
 * 3. 激活层1: ReLU激活函数
 * 4. 池化层1: 3x3最大池化，步长2，输出64@27x27
 * 5. 批归一化层1
 * 6. 卷积层2: 192个5x5卷积核，输出192@27x27
 * 7. 激活层2: ReLU激活函数
 * 8. 池化层2: 3x3最大池化，步长2，输出192@13x13
 * 9. 批归一化层2
 * 10. 卷积层3: 384个3x3卷积核，输出384@13x13
 * 11. 激活层3: ReLU激活函数
 * 12. 批归一化层3
 * 13. 卷积层4: 256个3x3卷积核，输出256@13x13
 * 14. 激活层4: ReLU激活函数
 * 15. 批归一化层4
 * 16. 卷积层5: 256个3x3卷积核，输出256@13x13
 * 17. 激活层5: ReLU激活函数
 * 18. 池化层5: 3x3最大池化，步长2，输出256@6x6
 * 19. 批归一化层5
 * 20. 全连接层1: 4096个神经元
 * 21. 激活层6: ReLU激活函数
 * 22. Dropout层1: 丢弃率0.5
 * 23. 全连接层2: 4096个神经元
 * 24. 激活层7: ReLU激活函数
 * 25. Dropout层2: 丢弃率0.5
 * 26. 输出层: 1000个神经元（ImageNet分类）
 */
public class AlexNet {
    
    // 卷积层
    private ConvLayer conv1;
    private ConvLayer conv2;
    private ConvLayer conv3;
    private ConvLayer conv4;
    private ConvLayer conv5;
    
    // 池化层
    private PoolingLayer pool1;
    private PoolingLayer pool2;
    private PoolingLayer pool5;
    
    // 批归一化层
    private BatchNormLayer bn1;
    private BatchNormLayer bn2;
    private BatchNormLayer bn3;
    private BatchNormLayer bn4;
    private BatchNormLayer bn5;
    
    // 全连接层
    private LinearLayer fc1;
    private LinearLayer fc2;
    private LinearLayer output;
    
    // 激活层
    private ActivationLayer activation1;
    private ActivationLayer activation2;
    private ActivationLayer activation3;
    private ActivationLayer activation4;
    private ActivationLayer activation5;
    private ActivationLayer activation6;
    private ActivationLayer activation7;
    
    // Dropout层（简化实现，这里用激活层代替）
    private ActivationLayer dropout1;
    private ActivationLayer dropout2;
    
    /**
     * 构造简化版AlexNet模型
     * @param numClasses 分类数量（默认1000）
     */
    public AlexNet(int numClasses) {
        // 初始化各层
        
        // 第一个卷积层: 输入3通道，输出64通道，11x11卷积核，步长4
        conv1 = new ConvLayer("conv1", 3, 64, 11, 4, 2, true);
        
        // 第一个激活层: ReLU
        activation1 = new ActivationLayer("activation1", ActivationType.RELU);
        
        // 第一个池化层: 3x3最大池化，步长2
        pool1 = new PoolingLayer("pool1", PoolingLayer.PoolingType.MAX, 3, 2, 0);
        
        // 第一个批归一化层
        bn1 = new BatchNormLayer("bn1", 64);
        
        // 第二个卷积层: 输入64通道，输出192通道，5x5卷积核
        conv2 = new ConvLayer("conv2", 64, 192, 5, 1, 2, true);
        
        // 第二个激活层: ReLU
        activation2 = new ActivationLayer("activation2", ActivationType.RELU);
        
        // 第二个池化层: 3x3最大池化，步长2
        pool2 = new PoolingLayer("pool2", PoolingLayer.PoolingType.MAX, 3, 2, 0);
        
        // 第二个批归一化层
        bn2 = new BatchNormLayer("bn2", 192);
        
        // 第三个卷积层: 输入192通道，输出384通道，3x3卷积核
        conv3 = new ConvLayer("conv3", 192, 384, 3, 1, 1, true);
        
        // 第三个激活层: ReLU
        activation3 = new ActivationLayer("activation3", ActivationType.RELU);
        
        // 第三个批归一化层
        bn3 = new BatchNormLayer("bn3", 384);
        
        // 第四个卷积层: 输入384通道，输出256通道，3x3卷积核
        conv4 = new ConvLayer("conv4", 384, 256, 3, 1, 1, true);
        
        // 第四个激活层: ReLU
        activation4 = new ActivationLayer("activation4", ActivationType.RELU);
        
        // 第四个批归一化层
        bn4 = new BatchNormLayer("bn4", 256);
        
        // 第五个卷积层: 输入256通道，输出256通道，3x3卷积核
        conv5 = new ConvLayer("conv5", 256, 256, 3, 1, 1, true);
        
        // 第五个激活层: ReLU
        activation5 = new ActivationLayer("activation5", ActivationType.RELU);
        
        // 第五个池化层: 3x3最大池化，步长2
        pool5 = new PoolingLayer("pool5", PoolingLayer.PoolingType.MAX, 3, 2, 0);
        
        // 第五个批归一化层
        bn5 = new BatchNormLayer("bn5", 256);
        
        // 第一个全连接层: 输入256*6*6=9216，输出4096
        fc1 = new LinearLayer("fc1", 9216, 4096, true);
        
        // 第六个激活层: ReLU
        activation6 = new ActivationLayer("activation6", ActivationType.RELU);
        
        // 第一个Dropout层（简化实现）
        dropout1 = new ActivationLayer("dropout1", ActivationType.RELU); // 实际应用中应使用专门的Dropout层
        
        // 第二个全连接层: 输入4096，输出4096
        fc2 = new LinearLayer("fc2", 4096, 4096, true);
        
        // 第七个激活层: ReLU
        activation7 = new ActivationLayer("activation7", ActivationType.RELU);
        
        // 第二个Dropout层（简化实现）
        dropout2 = new ActivationLayer("dropout2", ActivationType.RELU); // 实际应用中应使用专门的Dropout层
        
        // 输出层: 输入4096，输出numClasses
        output = new LinearLayer("output", 4096, numClasses, true);
    }
    
    /**
     * 构造简化版AlexNet模型（默认1000分类）
     */
    public AlexNet() {
        this(1000);
    }
    
    /**
     * 前向传播
     * @param input 输入图像 (batch_size, 3, 224, 224)
     * @return 输出结果 (batch_size, numClasses)
     */
    public Variable forward(Variable input) {
        // 第一组: 卷积 -> 激活 -> 池化 -> 批归一化
        Variable x = conv1.layerForward(input);
        x = activation1.layerForward(x);
        x = pool1.layerForward(x);
        x = bn1.layerForward(x);
        
        // 第二组: 卷积 -> 激活 -> 池化 -> 批归一化
        x = conv2.layerForward(x);
        x = activation2.layerForward(x);
        x = pool2.layerForward(x);
        x = bn2.layerForward(x);
        
        // 第三组: 卷积 -> 激活 -> 批归一化
        x = conv3.layerForward(x);
        x = activation3.layerForward(x);
        x = bn3.layerForward(x);
        
        // 第四组: 卷积 -> 激活 -> 批归一化
        x = conv4.layerForward(x);
        x = activation4.layerForward(x);
        x = bn4.layerForward(x);
        
        // 第五组: 卷积 -> 激活 -> 池化 -> 批归一化
        x = conv5.layerForward(x);
        x = activation5.layerForward(x);
        x = pool5.layerForward(x);
        x = bn5.layerForward(x);
        
        // 将4D特征图展平为2D向量
        x = x.flatten();
        
        // 全连接层
        x = fc1.layerForward(x);
        x = activation6.layerForward(x);
        x = dropout1.layerForward(x); // 简化实现
        
        x = fc2.layerForward(x);
        x = activation7.layerForward(x);
        x = dropout2.layerForward(x); // 简化实现
        
        x = output.layerForward(x);
        
        return x;
    }
    
    /**
     * 获取模型的所有层
     * @return 层列表
     */
    public Layer[] getLayers() {
        return new Layer[] {
            conv1, activation1, pool1, bn1,
            conv2, activation2, pool2, bn2,
            conv3, activation3, bn3,
            conv4, activation4, bn4,
            conv5, activation5, pool5, bn5,
            fc1, activation6, dropout1,
            fc2, activation7, dropout2,
            output
        };
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
        return count;
    }
}