package io.leavesfly.tinyai.cv;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.func.Variable;

/**
 * SimpleConvNet演示程序
 * 展示如何创建、配置和使用卷积神经网络
 * 
 * @author leavesfly
 * @version 0.01
 */
public class SimpleConvNetDemo {
    
    public static void main(String[] args) {
        System.out.println("=== SimpleConvNet演示程序 ===\n");
        
        // 演示1: 创建标准的CNN网络
        demonstrateStandardCNN();
        
        // 演示2: 前向传播
        demonstrateForwardPass();
    }
    
    /**
     * 演示创建标准的CNN网络
     */
    private static void demonstrateStandardCNN() {
        System.out.println("1. 创建标准CNN网络 (CIFAR-10风格)");
        
        // 创建输入形状 (批次大小, 通道数, 高度, 宽度)
        Shape inputShape = Shape.of(8, 3, 32, 32);
        int numClasses = 10;
        
        // 创建网络
        SimpleConvNet network = new SimpleConvNet("cifar10_cnn", inputShape, numClasses);
        
        // 打印网络信息
        network.printArchitecture();
        System.out.println();
    }
    
    /**
     * 演示前向传播过程
     */
    private static void demonstrateForwardPass() {
        System.out.println("2. 前向传播演示");
        
        // 创建网络
        Shape inputShape = Shape.of(2, 3, 32, 32);
        SimpleConvNet network = new SimpleConvNet("demo_network", inputShape, 5);
        
        // 初始化网络
        System.out.println("初始化网络...");
        network.init();
        
        // 创建随机输入数据
        NdArray inputData = NdArray.likeRandomN(inputShape);
        Variable input = new Variable(inputData);
        
        System.out.println("输入形状: " + inputData.getShape());
        
        // 执行前向传播
        long startTime = System.currentTimeMillis();
        Variable output = network.layerForward(input);
        long endTime = System.currentTimeMillis();
        
        // 显示结果
        System.out.println("输出形状: " + output.getValue().getShape());
        System.out.println("前向传播耗时: " + (endTime - startTime) + "ms");
        System.out.println("参数数量: " + network.getAllParams().size());
    }
}