package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * MaskedSoftmaxCELoss功能演示和测试
 */
public class MaskedSoftmaxCELossDemo {
    
    public static void main(String[] args) {
        System.out.println("=== MaskedSoftmaxCELoss 功能演示 ===");
        
        testBasicFunctionality();
        testMaskFunctionality();
        testLossStats();
        
        System.out.println("\n=== 所有测试完成 ===");
    }
    
    private static void testBasicFunctionality() {
        System.out.println("\n1. 基本功能测试");
        
        try {
            // 创建测试数据: 批次=2, 序列长度=3, 词汇表=4
            float[][][] predictData = {
                {{0.1f, 0.2f, 0.3f, 0.4f}, {0.2f, 0.3f, 0.4f, 0.1f}, {0.3f, 0.4f, 0.1f, 0.2f}},
                {{0.4f, 0.3f, 0.2f, 0.1f}, {0.1f, 0.4f, 0.3f, 0.2f}, {0.2f, 0.1f, 0.4f, 0.3f}}
            };
            
            float[][] labelData = {{3, 2, 1}, {0, 1, 2}};
            
            Variable predict = new Variable(NdArray.of(predictData));
            Variable labels = new Variable(NdArray.of(labelData));
            
            MaskedSoftmaxCELoss lossFunction = new MaskedSoftmaxCELoss();
            Variable loss = lossFunction.loss(labels, predict);
            
            System.out.println("损失值: " + loss.getValue().getNumber().floatValue());
            
        } catch (Exception e) {
            System.out.println("基本功能测试失败: " + e.getMessage());
        }
    }
    
    private static void testMaskFunctionality() {
        System.out.println("\n2. 掩码功能测试");
        
        try {
            // 包含填充的数据
            float[][][] predictData = {
                {{0.1f, 0.2f, 0.3f, 0.4f}, {0.2f, 0.3f, 0.4f, 0.1f}, {0.0f, 0.0f, 0.0f, 0.0f}},
                {{0.4f, 0.3f, 0.2f, 0.1f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}}
            };
            
            float[][] labelData = {{3, 2, 0}, {1, 0, 0}};  // 0表示填充
            
            Variable predict = new Variable(NdArray.of(predictData));
            Variable labels = new Variable(NdArray.of(labelData));
            
            MaskedSoftmaxCELoss lossFunction = new MaskedSoftmaxCELoss();
            Variable maskedLoss = lossFunction.maskedSoftmaxCrossEntropy(labels, predict, 0);
            
            System.out.println("掩码损失值: " + maskedLoss.getValue().getNumber().floatValue());
            
        } catch (Exception e) {
            System.out.println("掩码功能测试失败: " + e.getMessage());
        }
    }
    
    private static void testLossStats() {
        System.out.println("\n3. 统计信息测试");
        
        try {
            float[][][] predictData = {
                {{0.1f, 0.2f, 0.3f, 0.4f}, {0.2f, 0.3f, 0.4f, 0.1f}, {0.3f, 0.4f, 0.1f, 0.2f}},
                {{0.4f, 0.3f, 0.2f, 0.1f}, {0.1f, 0.4f, 0.3f, 0.2f}, {0.0f, 0.0f, 0.0f, 0.0f}}
            };
            
            float[][] labelData = {{3, 2, 1}, {0, 1, 0}};
            
            Variable predict = new Variable(NdArray.of(predictData));
            Variable labels = new Variable(NdArray.of(labelData));
            
            MaskedSoftmaxCELoss lossFunction = new MaskedSoftmaxCELoss();
            MaskedSoftmaxCELoss.LossStats stats = lossFunction.computeLossStats(labels, predict, 0);
            
            System.out.println("损失统计: " + stats.toString());
            
        } catch (Exception e) {
            System.out.println("统计信息测试失败: " + e.getMessage());
        }
    }
}