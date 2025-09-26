package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.ndarr.NDArray;

/**
 * TinyAI演示程序 - 展示N维数组和函数计算功能
 * 
 * @author TinyAI Team
 * @version 1.0
 */
public class TinyAIDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 欢迎使用 TinyAI 深度学习框架 ===\n");
        
        // 演示N维数组功能
        demonstrateNDArray();
        
        System.out.println();
        
        // 演示数学函数功能
        demonstrateMathFunctions();
        
        System.out.println("\n=== TinyAI 演示完成 ===");
    }
    
    /**
     * 演示NDArray的基本功能
     */
    private static void demonstrateNDArray() {
        System.out.println("📊 N维数组演示:");
        
        // 创建一个2x3的矩阵
        NDArray matrix = new NDArray(2, 3);
        System.out.println("创建2x3矩阵: " + matrix);
        
        // 填充数据
        matrix.set(1.0, 0, 0);
        matrix.set(2.0, 0, 1); 
        matrix.set(3.0, 0, 2);
        matrix.set(4.0, 1, 0);
        matrix.set(5.0, 1, 1);
        matrix.set(6.0, 1, 2);
        
        System.out.println("填充数据后: " + matrix);
        
        // 矩阵运算
        NDArray doubled = matrix.multiply(2.0);
        System.out.println("矩阵乘以2: " + doubled);
        
        NDArray added = matrix.add(doubled);
        System.out.println("矩阵相加: " + added);
        
        // 创建三维数组
        NDArray tensor = new NDArray(2, 2, 2);
        tensor.fill(3.14);
        System.out.println("创建2x2x2张量并填充π: " + tensor);
    }
    
    /**
     * 演示数学函数功能
     */
    private static void demonstrateMathFunctions() {
        System.out.println("🧮 数学函数演示:");
        
        // 创建测试数据
        NDArray testData = new NDArray(2, 2);
        testData.set(-2.0, 0, 0);
        testData.set(-0.5, 0, 1);
        testData.set(0.5, 1, 0);
        testData.set(2.0, 1, 1);
        
        System.out.println("原始数据: " + testData);
        
        // 应用不同的激活函数
        NDArray sigmoidResult = MathFunction.applyFunction(testData, MathFunction.SIGMOID);
        System.out.println("Sigmoid激活: " + sigmoidResult);
        
        NDArray reluResult = MathFunction.applyFunction(testData, MathFunction.RELU);
        System.out.println("ReLU激活: " + reluResult);
        
        NDArray tanhResult = MathFunction.applyFunction(testData, MathFunction.TANH);
        System.out.println("Tanh激活: " + tanhResult);
        
        // 计算统计信息
        MathFunction.Statistics stats = MathFunction.calculateStatistics(testData);
        System.out.println("统计信息: " + stats);
        
        // 演示单个函数值
        System.out.println("\n单个函数值演示:");
        double x = 1.0;
        System.out.printf("输入值: %.2f%n", x);
        System.out.printf("Sigmoid(%.2f) = %.6f%n", x, MathFunction.SIGMOID.apply(x));
        System.out.printf("ReLU(%.2f) = %.6f%n", x, MathFunction.RELU.apply(x));
        System.out.printf("Tanh(%.2f) = %.6f%n", x, MathFunction.TANH.apply(x));
        
        // 演示导数计算
        System.out.println("\n导数计算演示:");
        System.out.printf("Sigmoid'(%.2f) = %.6f%n", x, MathFunction.SIGMOID.derivative(x));
        System.out.printf("ReLU'(%.2f) = %.6f%n", x, MathFunction.RELU.derivative(x));
        System.out.printf("Tanh'(%.2f) = %.6f%n", x, MathFunction.TANH.derivative(x));
    }
}