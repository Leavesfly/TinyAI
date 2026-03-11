package io.leavesfly.tinyai.ml.model.parameter;

import io.leavesfly.tinyai.ml.exception.ParameterMismatchException;
import io.leavesfly.tinyai.ml.visual.ValidationUtils;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

/**
 * 参数操作接口，提供统一的参数复制、比较等操作，支持任意维度的参数。
 *
 * @author TinyAI
 * @version 1.0
 */
public class ParameterOperator {
    
    /**
     * 复制参数值（支持任意维度）
     * 
     * @param source 源参数
     * @param target 目标参数
     * @throws ParameterMismatchException 如果形状不匹配
     */
    public static void copyParameter(Parameter source, Parameter target) {
        ValidationUtils.requireNonNull(source, "source parameter");
        ValidationUtils.requireNonNull(target, "target parameter");
        
        NdArray sourceData = source.getValue();
        NdArray targetData = target.getValue();
        
        ValidationUtils.requireShapeMatch(sourceData.getShape(), targetData.getShape(), 
                                         "Parameter shape mismatch");
        
        // 使用底层数组进行批量复制（最高效）
        float[] sourceArray = sourceData.getArray();
        float[] targetArray = targetData.getArray();
        
        System.arraycopy(sourceArray, 0, targetArray, 0, sourceArray.length);
    }
    
    /**
     * 比较两个参数是否相等（支持任意维度）
     * 
     * @param param1 参数1
     * @param param2 参数2
     * @param tolerance 容忍度（用于浮点数比较）
     * @return true 如果参数相等
     */
    public static boolean compareParameter(Parameter param1, Parameter param2, double tolerance) {
        if (param1 == null || param2 == null) {
            return param1 == param2;
        }
        
        NdArray data1 = param1.getValue();
        NdArray data2 = param2.getValue();
        
        // 检查形状
        if (!data1.getShape().equals(data2.getShape())) {
            return false;
        }
        
        // 使用底层数组进行批量比较
        float[] array1 = data1.getArray();
        float[] array2 = data2.getArray();
        
        for (int i = 0; i < array1.length; i++) {
            if (Math.abs(array1[i] - array2[i]) > tolerance) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 比较两个参数是否相等（使用默认容忍度）
     * 
     * @param param1 参数1
     * @param param2 参数2
     * @return true 如果参数相等
     */
    public static boolean compareParameter(Parameter param1, Parameter param2) {
        return compareParameter(param1, param2, 1e-6);
    }
    
    /**
     * 获取参数的统计信息
     * 
     * @param param 参数
     * @return 统计信息
     */
    public static ParameterStatistics getStatistics(Parameter param) {
        ValidationUtils.requireNonNull(param, "parameter");
        
        NdArray data = param.getValue();
        float[] array = data.getArray();
        
        if (array.length == 0) {
            return new ParameterStatistics();
        }
        
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        double sum = 0.0;
        
        for (float value : array) {
            min = Math.min(min, value);
            max = Math.max(max, value);
            sum += value;
        }
        
        return new ParameterStatistics(
            data.getShape(),
            array.length,
            min,
            max,
            sum / array.length,
            sum
        );
    }
    
    /**
     * 参数统计信息类
     */
    public static class ParameterStatistics {
        private final Shape shape;
        private final long elementCount;
        private final float minValue;
        private final float maxValue;
        private final double meanValue;
        private final double sumValue;
        
        public ParameterStatistics() {
            this(null, 0, 0, 0, 0, 0);
        }
        
        public ParameterStatistics(Shape shape, long elementCount, 
                                  float minValue, float maxValue, 
                                  double meanValue, double sumValue) {
            this.shape = shape;
            this.elementCount = elementCount;
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.meanValue = meanValue;
            this.sumValue = sumValue;
        }
        
        public Shape getShape() {
            return shape;
        }
        
        public long getElementCount() {
            return elementCount;
        }
        
        public float getMinValue() {
            return minValue;
        }
        
        public float getMaxValue() {
            return maxValue;
        }
        
        public double getMeanValue() {
            return meanValue;
        }
        
        public double getSumValue() {
            return sumValue;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ParameterStatistics{shape=%s, elements=%d, min=%.6f, max=%.6f, mean=%.6f}",
                shape, elementCount, minValue, maxValue, meanValue
            );
        }
    }
}

