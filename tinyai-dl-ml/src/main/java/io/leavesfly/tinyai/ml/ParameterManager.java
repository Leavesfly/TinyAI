package io.leavesfly.tinyai.ml;

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
        if (sourceModel == null || targetModel == null) {
            throw new IllegalArgumentException("模型不能为空");
        }

        Map<String, Parameter> sourceParams = sourceModel.getAllParams();
        Map<String, Parameter> targetParams = targetModel.getAllParams();
        
        int copiedCount = 0;
        int skippedCount = 0;
        
        for (Map.Entry<String, Parameter> sourceEntry : sourceParams.entrySet()) {
            String paramName = sourceEntry.getKey();
            Parameter sourceParam = sourceEntry.getValue();
            
            if (targetParams.containsKey(paramName)) {
                Parameter targetParam = targetParams.get(paramName);
                
                // 检查形状是否匹配
                if (sourceParam.getValue().getShape().equals(targetParam.getValue().getShape())) {
                    // 复制参数值
                    try {
                        // 尝试按维度复制
                        if (sourceParam.getValue().getShape().getDimNum() == 2) {
                            // 2D数组处理
                            float[][] sourceMatrix = sourceParam.getValue().getMatrix();
                            float[][] targetMatrix = targetParam.getValue().getMatrix();
                            for (int i = 0; i < sourceMatrix.length; i++) {
                                for (int j = 0; j < sourceMatrix[i].length; j++) {
                                    targetParam.getValue().set(sourceMatrix[i][j], i, j);
                                }
                            }
                            copiedCount++;
                        } else if (sourceParam.getValue().getShape().getDimNum() == 1) {
                            // 1D数组处理
                            float value = sourceParam.getValue().getNumber().floatValue();
                            targetParam.getValue().set(value, 0);
                            copiedCount++;
                        } else {
                            // 标量处理
                            float value = sourceParam.getValue().getNumber().floatValue();
                            targetParam.getValue().set(value);
                            copiedCount++;
                        }
                    } catch (Exception e) {
                        System.out.println("警告: 无法复制参数 " + paramName + ": " + e.getMessage());
                        skippedCount++;
                    }
                } else {
                    String message = "参数 " + paramName + " 形状不匹配: 源=" + 
                                   sourceParam.getValue().getShape() + ", 目标=" + 
                                   targetParam.getValue().getShape();
                    if (strict) {
                        throw new RuntimeException(message);
                    } else {
                        System.out.println("警告: " + message + "，跳过复制");
                        skippedCount++;
                    }
                }
            } else {
                String message = "目标模型中不存在参数: " + paramName;
                if (strict) {
                    throw new RuntimeException(message);
                } else {
                    System.out.println("警告: " + message + "，跳过复制");
                    skippedCount++;
                }
            }
        }
        
        // 在严格模式下，检查目标模型是否有额外的参数
        if (strict) {
            for (String targetParamName : targetParams.keySet()) {
                if (!sourceParams.containsKey(targetParamName)) {
                    throw new RuntimeException("源模型中不存在参数: " + targetParamName);
                }
            }
        }
        
        System.out.println("参数复制完成: 成功 " + copiedCount + " 个，跳过 " + skippedCount + " 个");
        return copiedCount;
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
        if (model1 == null || model2 == null) {
            return false;
        }

        Map<String, Parameter> params1 = model1.getAllParams();
        Map<String, Parameter> params2 = model2.getAllParams();

        // 检查参数数量是否相同
        if (params1.size() != params2.size()) {
            return false;
        }

        // 检查每个参数是否相同
        for (Map.Entry<String, Parameter> entry : params1.entrySet()) {
            String paramName = entry.getKey();
            Parameter param1 = entry.getValue();

            if (!params2.containsKey(paramName)) {
                return false;
            }

            Parameter param2 = params2.get(paramName);
            
            // 检查形状是否相同
            if (!param1.getValue().getShape().equals(param2.getValue().getShape())) {
                return false;
            }

                    // 检查数值是否相同
                    if (param1.getValue().getShape().getDimNum() == 2) {
                        float[][] matrix1 = param1.getValue().getMatrix();
                        float[][] matrix2 = param2.getValue().getMatrix();
                        
                        for (int i = 0; i < matrix1.length; i++) {
                            for (int j = 0; j < matrix1[i].length; j++) {
                                if (Math.abs(matrix1[i][j] - matrix2[i][j]) > tolerance) {
                                    return false;
                                }
                            }
                        }
                    } else {
                        // 如果不是矩阵，直接比较数值
                        if (Math.abs(param1.getValue().getNumber().doubleValue() - 
                                   param2.getValue().getNumber().doubleValue()) > tolerance) {
                            return false;
                        }
                    }
        }

        return true;
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
        if (parameters == null || parameters.isEmpty()) {
            return new ParameterStats();
        }
        
        ParameterStats stats = new ParameterStats();
        stats.parameterCount = parameters.size();
        
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            Parameter param = entry.getValue();
            Shape shape = param.getValue().getShape();
            long paramSize = shape.size();
            stats.totalParameters += paramSize;
            
            // 计算数值统计
            try {
                if (shape.getDimNum() == 0 || paramSize == 1) {
                    // 标量参数
                    float value = param.getValue().getNumber().floatValue();
                    stats.minValue = Math.min(stats.minValue, value);
                    stats.maxValue = Math.max(stats.maxValue, value);
                    stats.sum += value;
                } else {
                    // 矩阵参数
                    float[][] matrix = param.getValue().getMatrix();
                    for (int i = 0; i < matrix.length; i++) {
                        for (int j = 0; j < matrix[i].length; j++) {
                            float value = matrix[i][j];
                            stats.minValue = Math.min(stats.minValue, value);
                            stats.maxValue = Math.max(stats.maxValue, value);
                            stats.sum += value;
                        }
                    }
                }
            } catch (Exception e) {
                // 如果无法获取数值，跳过该参数
                System.out.println("警告: 无法获取参数 " + entry.getKey() + " 的数值: " + e.getMessage());
            }
        }
        
        // 计算平均值
        if (stats.totalParameters > 0) {
            stats.meanValue = stats.sum / stats.totalParameters;
        }
        
        // 如果没有找到任何数值，重置最小值和最大值
        if (stats.minValue == Float.MAX_VALUE) {
            stats.minValue = 0;
        }
        if (stats.maxValue == Float.MIN_VALUE) {
            stats.maxValue = 0;
        }
        
        return stats;
    }

    /**
     * 创建参数映射的深拷贝
     *
     * @param original 原始参数映射
     * @return 深拷贝的参数映射
     */
    public static Map<String, Parameter> deepCopyParameters(Map<String, Parameter> original) {
        if (original == null) {
            return null;
        }
        
        Map<String, Parameter> copy = new HashMap<>();
        
        for (Map.Entry<String, Parameter> entry : original.entrySet()) {
            String paramName = entry.getKey();
            Parameter originalParam = entry.getValue();
            
            try {
                // 创建新的NdArray拷贝
                Shape shape = originalParam.getValue().getShape();
                
                if (shape.getDimNum() == 0 || shape.size() == 1) {
                    // 标量参数
                    float value = originalParam.getValue().getNumber().floatValue();
                    NdArray newArray = NdArray.of(value);
                    copy.put(paramName, new Parameter(newArray));
                } else {
                    // 矩阵参数
                    float[][] matrix = originalParam.getValue().getMatrix();
                    
                    // 创建新的矩阵拷贝
                    float[][] newMatrix = new float[matrix.length][];
                    for (int i = 0; i < matrix.length; i++) {
                        newMatrix[i] = matrix[i].clone();
                    }
                    
                    // 创建新的NdArray
                    NdArray newArray = NdArray.of(newMatrix);
                    copy.put(paramName, new Parameter(newArray));
                }
            } catch (Exception e) {
                // 如果拷贝失败，记录错误但继续处理其他参数
                System.out.println("警告: 无法拷贝参数 " + paramName + ": " + e.getMessage());
                
                // 尝试使用序列化的方式拷贝
                try {
                    ByteArrayOutputStream bos = new ByteArrayOutputStream();
                    ObjectOutputStream oos = new ObjectOutputStream(bos);
                    oos.writeObject(originalParam);
                    oos.close();
                    
                    ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
                    ObjectInputStream ois = new ObjectInputStream(bis);
                    Parameter clonedParam = (Parameter) ois.readObject();
                    ois.close();
                    
                    copy.put(paramName, clonedParam);
                } catch (Exception e2) {
                    System.out.println("错误: 无法拷贝参数 " + paramName + "，跳过");
                }
            }
        }
        
        return copy;
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

    /**
     * 将二维数组展平为一维数组
     *
     * @param matrix 二维数组
     * @return 一维数组
     */
    private static float[] flatten2D(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] result = new float[rows * cols];
        
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[index++] = matrix[i][j];
            }
        }
        
        return result;
    }
}