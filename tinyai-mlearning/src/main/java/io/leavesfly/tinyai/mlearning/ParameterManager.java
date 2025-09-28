package io.leavesfly.tinyai.mlearning;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * 参数管理器 - 专门处理模型参数的保存、加载和管理
 * <p>
 * 提供更灵活的参数操作功能，包括：
 * 1. 参数的保存和加载
 * 2. 参数在不同模型间的复制
 * 3. 参数的比较和统计
 * 4. 参数的筛选和深拷贝
 *
 * @author TinyDL
 * @version 1.0
 */
public class ParameterManager {

    /**
     * 保存参数到文件
     *
     * @param parameters 参数映射
     * @param filePath   保存路径
     */
    public static void saveParameters(Map<String, Parameter> parameters, String filePath) {
        try {
            File file = new File(filePath);
            if (file.getParentFile() != null && !file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }

            try (FileOutputStream fos = new FileOutputStream(file);
                 ObjectOutputStream oos = new ObjectOutputStream(fos)) {
                oos.writeObject(parameters);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save parameters: " + e.getMessage(), e);
        }
    }

    /**
     * 从文件加载参数
     *
     * @param filePath 文件路径
     * @return 参数映射
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Parameter> loadParameters(String filePath) {
        try {
            File file = new File(filePath);
            if (!file.exists()) {
                throw new RuntimeException("Parameters file does not exist: " + filePath);
            }

            try (FileInputStream fis = new FileInputStream(file);
                 ObjectInputStream ois = new ObjectInputStream(fis)) {
                return (Map<String, Parameter>) ois.readObject();
            }
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Failed to load parameters: " + e.getMessage(), e);
        }
    }

    /**
     * 将参数从一个模型复制到另一个模型
     *
     * @param sourceModel 源模型
     * @param targetModel 目标模型
     * @param strict      是否严格模式（所有参数都必须匹配）
     * @return 成功复制的参数数量
     */
    public static int copyParameters(Model sourceModel, Model targetModel, boolean strict) {
        //todo
        return 0;
    }

    /**
     * 复制参数（非严格模式）
     *
     * @param sourceModel 源模型
     * @param targetModel 目标模型
     * @return 成功复制的参数数量
     */
    public static int copyParameters(Model sourceModel, Model targetModel) {
        return copyParameters(sourceModel, targetModel, false);
    }

    /**
     * 比较两个模型的参数
     *
     * @param model1    模型1
     * @param model2    模型2
     * @param tolerance 容忍度
     * @return 参数是否相同
     */
    public static boolean compareParameters(Model model1, Model model2, double tolerance) {
        //todo
        return false;
    }

    /**
     * 比较两个模型的参数（默认容忍度）
     *
     * @param model1 模型1
     * @param model2 模型2
     * @return 参数是否相同
     */
    public static boolean compareParameters(Model model1, Model model2) {
        return compareParameters(model1, model2, 1e-6);
    }

    /**
     * 获取参数统计信息
     *
     * @param parameters 参数映射
     * @return 统计信息
     */
    public static ParameterStats getParameterStats(Map<String, Parameter> parameters) {
        //todo
        return null;
    }

    /**
     * 创建参数映射的深拷贝
     *
     * @param original 原始参数映射
     * @return 深拷贝的参数映射
     */
    public static Map<String, Parameter> deepCopyParameters(Map<String, Parameter> original) {
        //todo
        return null;
    }

    /**
     * 筛选参数（根据名称模式）
     *
     * @param parameters  参数映射
     * @param namePattern 名称模式（支持通配符*）
     * @return 筛选后的参数映射
     */
    public static Map<String, Parameter> filterParameters(Map<String, Parameter> parameters, String namePattern) {
        Map<String, Parameter> filtered = new HashMap<>();

        // 将通配符模式转换为正则表达式
        String regex = namePattern.replace("*", ".*");

        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            if (entry.getKey().matches(regex)) {
                filtered.put(entry.getKey(), entry.getValue());
            }
        }

        return filtered;
    }

    /**
     * 参数统计信息类
     * <p>
     * 用于存储和表示模型参数的统计信息，包括：
     * 1. 总参数数量
     * 2. 参数组数量
     * 3. 参数值的最小值、最大值和平均值
     */
    public static class ParameterStats {
        public long totalParameters = 0;
        public int parameterCount = 0;
        public float minValue = Float.MAX_VALUE;
        public float maxValue = Float.MIN_VALUE;
        public double sum = 0.0;
        public double meanValue = 0.0;

        @Override
        public String toString() {
            return String.format(
                    "ParameterStats{totalParams=%d, paramCount=%d, min=%.6f, max=%.6f, mean=%.6f}",
                    totalParameters, parameterCount, minValue, maxValue, meanValue
            );
        }
    }

    /**
     * 保存参数统计信息到文本文件
     *
     * @param parameters 参数映射
     * @param filePath   文件路径
     */
    public static void saveParameterStats(Map<String, Parameter> parameters, String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            ParameterStats stats = getParameterStats(parameters);

            writer.println("=== 模型参数统计 ===");
            writer.println("总参数数量: " + stats.totalParameters);
            writer.println("参数组数量: " + stats.parameterCount);
            writer.println("最小值: " + String.format("%.6f", stats.minValue));
            writer.println("最大值: " + String.format("%.6f", stats.maxValue));
            writer.println("平均值: " + String.format("%.6f", stats.meanValue));
            writer.println();

            writer.println("=== 参数详细信息 ===");
            for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
                String name = entry.getKey();
                Parameter param = entry.getValue();
                Shape shape = param.getValue().getShape();

                writer.println(String.format("%-40s %s (%d个参数)",
                        name, shape.toString(), shape.size()));
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save parameter stats: " + e.getMessage(), e);
        }
    }
}