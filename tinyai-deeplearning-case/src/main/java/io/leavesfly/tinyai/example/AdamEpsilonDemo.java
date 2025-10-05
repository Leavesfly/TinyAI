package io.leavesfly.tinyai.example;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Adam优化器Epsilon数值稳定性演示
 * 
 * 本演示用于验证和调试Adam优化器中的epsilon参数设置，
 * 确保在各种数值条件下都能避免除零异常。
 * 
 * @author 山泽
 * @version 1.0
 */
public class AdamEpsilonDemo {
    
    private static final float EPSILON_NDARRAY = 1e-7f; // NdArrayCpu的EPSILON值
    
    public static void main(String[] args) {
        System.out.println("=== Adam优化器Epsilon数值稳定性演示 ===\n");
        
        // 测试不同的epsilon值
        float[] epsilonValues = {1e-8f, 1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f};
        
        System.out.println("NdArrayCpu的除零检查阈值: " + EPSILON_NDARRAY);
        System.out.println("测试不同的Adam epsilon值:\n");
        
        for (float epsilon : epsilonValues) {
            testEpsilonValue(epsilon);
        }
        
        // 演示实际的Adam计算
        demonstrateAdamCalculation();
    }
    
    /**
     * 测试指定epsilon值的数值稳定性
     */
    private static void testEpsilonValue(float epsilon) {
        System.out.printf("测试 epsilon = %.0e:\n", epsilon);
        
        // 创建极小的v值来模拟训练早期的情况
        float[] testValues = {0f, 1e-12f, 1e-10f, 1e-8f, 1e-6f};
        
        for (float vValue : testValues) {
            NdArray v = NdArray.of(new float[]{vValue}, Shape.of(1));
            
            try {
                // 计算 sqrt(v) + epsilon
                NdArray denominator = v.pow(0.5f).add(NdArray.like(v.getShape(), epsilon));
                float result = denominator.getNumber().floatValue();
                
                // 检查是否会引发除零异常
                if (result < EPSILON_NDARRAY) {
                    System.out.printf("  v=%.0e: 分母=%.2e (危险！< %.0e)\n", vValue, result, EPSILON_NDARRAY);
                } else {
                    System.out.printf("  v=%.0e: 分母=%.2e (安全)\n", vValue, result);
                }
                
                // 尝试执行除法操作
                NdArray numerator = NdArray.of(new float[]{1.0f}, Shape.of(1));
                numerator.div(denominator);
                
            } catch (ArithmeticException e) {
                System.out.printf("  v=%.0e: 除零异常！\n", vValue);
            }
        }
        System.out.println();
    }
    
    /**
     * 演示实际的Adam优化器计算过程
     */
    private static void demonstrateAdamCalculation() {
        System.out.println("=== 实际Adam计算演示 ===\n");
        
        // 使用安全的epsilon值
        float safeEpsilon = 1e-4f;
        System.out.printf("使用安全的epsilon值: %.0e\n\n", safeEpsilon);
        
        // 模拟梯度下降的几个步骤
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float learningRate = 0.001f;
        
        // 初始化动量
        NdArray m = NdArray.zeros(Shape.of(2, 2));
        NdArray v = NdArray.zeros(Shape.of(2, 2));
        
        // 模拟几个不同的梯度情况
        float[][] gradients = {
            {1e-8f, 1e-6f, 1e-4f, 1e-2f},  // 极小梯度
            {0.1f, 0.2f, 0.3f, 0.4f},       // 正常梯度
            {1.0f, 2.0f, 3.0f, 4.0f}        // 大梯度
        };
        
        for (int step = 0; step < gradients.length; step++) {
            System.out.printf("步骤 %d - 梯度情况: %s\n", step + 1, getGradientDescription(step));
            
            NdArray grad = NdArray.of(gradients[step], Shape.of(2, 2));
            
            // Adam更新
            m = m.add(grad.sub(m).mulNum(1 - beta1));
            v = v.add(grad.mul(grad).sub(v).mulNum(1 - beta2));
            
            // 偏差校正
            int t = step + 1;
            float fix1 = (float) (1.0 - Math.pow(beta1, t));
            float fix2 = (float) (1.0 - Math.pow(beta2, t));
            float correctedLr = (float) (learningRate * Math.sqrt(fix2) / fix1);
            
            try {
                // 安全的Adam更新计算
                NdArray denominator = v.pow(0.5f).add(NdArray.like(v.getShape(), safeEpsilon));
                NdArray delta = m.mulNum(correctedLr).div(denominator);
                
                System.out.printf("  更新成功 - 最大更新值: %.6f\n", getMaxValue(delta));
                System.out.printf("  分母最小值: %.2e\n", getMinValue(denominator));
                
            } catch (ArithmeticException e) {
                System.out.println("  更新失败: " + e.getMessage());
            }
            
            System.out.println();
        }
        
        System.out.println("建议使用的Adam epsilon值: 1e-4f 或更大");
        System.out.println("这确保了在各种梯度条件下的数值稳定性。");
    }
    
    private static String getGradientDescription(int step) {
        switch (step) {
            case 0: return "极小梯度";
            case 1: return "正常梯度";  
            case 2: return "大梯度";
            default: return "未知";
        }
    }
    
    private static float getMaxValue(NdArray array) {
        // 简单实现：获取数组的最大绝对值
        float max = 0f;
        for (int i = 0; i < array.getShape().size(); i++) {
            float val = Math.abs(array.get(0, i));
            if (val > max) max = val;
        }
        return max;
    }
    
    private static float getMinValue(NdArray array) {
        // 简单实现：获取数组的最小值
        float min = Float.MAX_VALUE;
        for (int i = 0; i < array.getShape().size(); i++) {
            float val = array.get(0, i);
            if (val < min) min = val;
        }
        return min;
    }
}