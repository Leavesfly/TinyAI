package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.ndarr.NDArray;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.util.function.Function;

/**
 * 函数计算类 - TinyAI项目的数学函数处理模块
 * 
 * @author TinyAI Team
 * @version 1.0
 */
public class MathFunction {
    
    /**
     * 激活函数接口
     */
    @FunctionalInterface
    public interface ActivationFunction {
        /**
         * 计算激活函数值
         * 
         * @param x 输入值
         * @return 激活函数输出
         */
        double apply(double x);
        
        /**
         * 计算激活函数的导数
         * 
         * @param x 输入值
         * @return 导数值
         */
        default double derivative(double x) {
            // 数值微分近似
            double h = 1e-7;
            return (apply(x + h) - apply(x - h)) / (2 * h);
        }
    }
    
    // 常用激活函数
    public static final ActivationFunction SIGMOID = new ActivationFunction() {
        @Override
        public double apply(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        
        @Override
        public double derivative(double x) {
            double sigmoid = apply(x);
            return sigmoid * (1 - sigmoid);
        }
    };
    
    public static final ActivationFunction TANH = new ActivationFunction() {
        @Override
        public double apply(double x) {
            return Math.tanh(x);
        }
        
        @Override
        public double derivative(double x) {
            double tanh = apply(x);
            return 1 - tanh * tanh;
        }
    };
    
    public static final ActivationFunction RELU = new ActivationFunction() {
        @Override
        public double apply(double x) {
            return Math.max(0, x);
        }
        
        @Override
        public double derivative(double x) {
            return x > 0 ? 1.0 : 0.0;
        }
    };
    
    /**
     * 对NDArray应用激活函数
     * 
     * @param array 输入数组
     * @param function 激活函数
     * @return 应用函数后的新数组
     */
    public static NDArray applyFunction(NDArray array, ActivationFunction function) {
        int[] shape = array.getShape();
        NDArray result = new NDArray(shape);
        
        // 遍历所有元素并应用函数
        applyFunctionRecursive(array, result, function, new int[shape.length], 0);
        
        return result;
    }
    
    /**
     * 递归应用函数到多维数组的每个元素
     */
    private static void applyFunctionRecursive(NDArray input, NDArray output, 
                                             ActivationFunction function,
                                             int[] indices, int dimension) {
        if (dimension == input.getDimensions()) {
            // 到达最底层，应用函数
            double value = input.get(indices);
            output.set(function.apply(value), indices);
            return;
        }
        
        // 递归处理当前维度
        int[] shape = input.getShape();
        for (int i = 0; i < shape[dimension]; i++) {
            indices[dimension] = i;
            applyFunctionRecursive(input, output, function, indices, dimension + 1);
        }
    }
    
    /**
     * 计算数组的统计信息
     * 
     * @param array 输入数组
     * @return 统计信息对象
     */
    public static Statistics calculateStatistics(NDArray array) {
        return new Statistics(array);
    }
    
    /**
     * 统计信息类
     */
    public static class Statistics {
        private final double mean;
        private final double variance;
        private final double min;
        private final double max;
        private final double sum;
        
        public Statistics(NDArray array) {
            int[] shape = array.getShape();
            double[] values = new double[array.getSize()];
            
            // 提取所有值
            extractValues(array, values, new int[shape.length], 0, new int[]{0});
            
            // 计算统计量
            this.sum = calculateSum(values);
            this.mean = sum / values.length;
            this.variance = calculateVariance(values, mean);
            this.min = findMin(values);
            this.max = findMax(values);
        }
        
        private void extractValues(NDArray array, double[] values, int[] indices, 
                                 int dimension, int[] valueIndex) {
            if (dimension == array.getDimensions()) {
                values[valueIndex[0]++] = array.get(indices);
                return;
            }
            
            int[] shape = array.getShape();
            for (int i = 0; i < shape[dimension]; i++) {
                indices[dimension] = i;
                extractValues(array, values, indices, dimension + 1, valueIndex);
            }
        }
        
        private double calculateSum(double[] values) {
            double sum = 0;
            for (double value : values) {
                sum += value;
            }
            return sum;
        }
        
        private double calculateVariance(double[] values, double mean) {
            double variance = 0;
            for (double value : values) {
                double diff = value - mean;
                variance += diff * diff;
            }
            return variance / values.length;
        }
        
        private double findMin(double[] values) {
            double min = values[0];
            for (double value : values) {
                if (value < min) min = value;
            }
            return min;
        }
        
        private double findMax(double[] values) {
            double max = values[0];
            for (double value : values) {
                if (value > max) max = value;
            }
            return max;
        }
        
        // Getters
        public double getMean() { return mean; }
        public double getVariance() { return variance; }
        public double getStandardDeviation() { return Math.sqrt(variance); }
        public double getMin() { return min; }
        public double getMax() { return max; }
        public double getSum() { return sum; }
        
        @Override
        public String toString() {
            return String.format("Statistics{mean=%.4f, std=%.4f, min=%.4f, max=%.4f, sum=%.4f}",
                    mean, getStandardDeviation(), min, max, sum);
        }
    }
    
    /**
     * 创建函数图表
     * 
     * @param function 要绘制的函数
     * @param start 起始值
     * @param end 结束值
     * @param points 采样点数
     * @param title 图表标题
     * @return JFreeChart对象
     */
    public static JFreeChart createFunctionChart(Function<Double, Double> function,
                                               double start, double end, int points,
                                               String title) {
        XYSeries series = new XYSeries(title);
        
        double step = (end - start) / (points - 1);
        for (int i = 0; i < points; i++) {
            double x = start + i * step;
            double y = function.apply(x);
            series.add(x, y);
        }
        
        XYSeriesCollection dataset = new XYSeriesCollection(series);
        
        return ChartFactory.createXYLineChart(
                title,
                "X",
                "Y",
                dataset,
                org.jfree.chart.plot.PlotOrientation.VERTICAL,
                true,  // legend
                true,  // tooltips
                false  // URLs
        );
    }
    
    /**
     * 显示函数图表
     * 
     * @param chart JFreeChart对象
     */
    public static void displayChart(JFreeChart chart) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("TinyAI 函数图表");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(800, 600));
            
            frame.setContentPane(chartPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
    
    /**
     * 演示方法 - 展示各种功能
     */
    public static void demo() {
        System.out.println("=== TinyAI 函数计算模块演示 ===");
        
        // 创建测试数组
        NDArray array = new NDArray(2, 3);
        array.set(1.0, 0, 0);
        array.set(-0.5, 0, 1);
        array.set(2.0, 0, 2);
        array.set(-1.0, 1, 0);
        array.set(0.5, 1, 1);
        array.set(1.5, 1, 2);
        
        System.out.println("原始数组: " + array);
        
        // 应用不同的激活函数
        NDArray sigmoidResult = applyFunction(array, SIGMOID);
        NDArray reluResult = applyFunction(array, RELU);
        NDArray tanhResult = applyFunction(array, TANH);
        
        System.out.println("Sigmoid结果: " + sigmoidResult);
        System.out.println("ReLU结果: " + reluResult);
        System.out.println("Tanh结果: " + tanhResult);
        
        // 计算统计信息
        Statistics stats = calculateStatistics(array);
        System.out.println("统计信息: " + stats);
        
        // 创建并显示Sigmoid函数图表
        JFreeChart sigmoidChart = createFunctionChart(
                x -> SIGMOID.apply(x), -10, 10, 100, "Sigmoid函数"
        );
        displayChart(sigmoidChart);
        
        System.out.println("=== 演示完成 ===");
    }
}