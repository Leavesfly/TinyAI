package io.leavesfly.tinyai.cv.model.lenet;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.PoolingLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ActivationType;

/**
 * LeNet-5卷积神经网络实现
 * 
 * LeNet-5是Yann LeCun在1998年提出的一个经典CNN架构，主要用于手写数字识别。
 * 网络结构：
 * INPUT -> CONV1 -> TANH -> POOL1 -> CONV2 -> TANH -> POOL2 -> 
 * FC1 -> TANH -> FC2 -> TANH -> OUTPUT
 * 
 * 详细结构：
 * 1. 输入层: 32x32单通道图像
 * 2. 卷积层1: 6个5x5卷积核，输出6@28x28
 * 3. 激活层1: Tanh激活函数
 * 4. 池化层1: 2x2平均池化，输出6@14x14
 * 5. 卷积层2: 16个5x5卷积核，输出16@10x10
 * 6. 激活层2: Tanh激活函数
 * 7. 池化层2: 2x2平均池化，输出16@5x5
 * 8. 全连接层1: 120个神经元
 * 9. 激活层3: Tanh激活函数
 * 10. 全连接层2: 84个神经元
 * 11. 激活层4: Tanh激活函数
 * 12. 输出层: 10个神经元（对应0-9数字）
 */
public class LeNet5 {
    
    // 卷积层
    private ConvLayer conv1;
    private ConvLayer conv2;
    
    // 池化层
    private PoolingLayer pool1;
    private PoolingLayer pool2;
    
    // 全连接层
    private LinearLayer fc1;
    private LinearLayer fc2;
    private LinearLayer output;
    
    // 激活层
    private ActivationLayer activation1;
    private ActivationLayer activation2;
    private ActivationLayer activation3;
    private ActivationLayer activation4;
    
    /**
     * 构造LeNet-5模型
     */
    public LeNet5() {
        // 初始化各层
        
        // 第一个卷积层: 输入1通道，输出6通道，5x5卷积核
        conv1 = new ConvLayer("conv1", 1, 6, 5, 1, 0, true);
        
        // 第一个激活层: Tanh
        activation1 = new ActivationLayer("activation1", ActivationType.TANH);
        
        // 第一个池化层: 2x2平均池化
        pool1 = new PoolingLayer("pool1", PoolingLayer.PoolingType.AVERAGE, 2, 2, 0);
        
        // 第二个卷积层: 输入6通道，输出16通道，5x5卷积核
        conv2 = new ConvLayer("conv2", 6, 16, 5, 1, 0, true);
        
        // 第二个激活层: Tanh
        activation2 = new ActivationLayer("activation2", ActivationType.TANH);
        
        // 第二个池化层: 2x2平均池化
        pool2 = new PoolingLayer("pool2", PoolingLayer.PoolingType.AVERAGE, 2, 2, 0);
        
        // 第一个全连接层: 输入16*5*5=400，输出120
        fc1 = new LinearLayer("fc1", 400, 120, true);
        
        // 第三个激活层: Tanh
        activation3 = new ActivationLayer("activation3", ActivationType.TANH);
        
        // 第二个全连接层: 输入120，输出84
        fc2 = new LinearLayer("fc2", 120, 84, true);
        
        // 第四个激活层: Tanh
        activation4 = new ActivationLayer("activation4", ActivationType.TANH);
        
        // 输出层: 输入84，输出10
        output = new LinearLayer("output", 84, 10, true);
    }
    
    /**
     * 前向传播
     * @param input 输入图像 (batch_size, 1, 32, 32)
     * @return 输出结果 (batch_size, 10)
     */
    public Variable forward(Variable input) {
        // 第一层: 卷积 -> 激活 -> 池化
        Variable x = conv1.layerForward(input);
        x = activation1.layerForward(x);
        x = pool1.layerForward(x);
        
        // 第二层: 卷积 -> 激活 -> 池化
        x = conv2.layerForward(x);
        x = activation2.layerForward(x);
        x = pool2.layerForward(x);
        
        // 将4D特征图展平为2D向量
        x = x.flatten();
        
        // 全连接层
        x = fc1.layerForward(x);
        x = activation3.layerForward(x);
        
        x = fc2.layerForward(x);
        x = activation4.layerForward(x);
        
        x = output.layerForward(x);
        
        return x;
    }
    
    /**
     * 获取模型的所有层
     * @return 层列表
     */
    public Layer[] getLayers() {
        return new Layer[] {
            conv1, activation1, pool1,
            conv2, activation2, pool2,
            fc1, activation3,
            fc2, activation4,
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