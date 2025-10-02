package io.leavesfly.tinyai.ml;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;

/**
 * JSON格式的模型信息导出器
 * <p>
 * 提供模型信息的JSON格式导出功能，便于查看和分析模型的详细信息。
 * 支持多种导出模式：
 * 1. 完整信息导出：包含模型的所有元数据
 * 2. 简化报告导出：仅包含关键信息
 * 3. 比较报告导出：用于比较两个模型的差异
 *
 * @author TinyDL
 * @version 1.0
 */
public class ModelInfoExporter {

    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    /**
     * 将模型信息导出为JSON格式（默认包含参数详细信息）
     *
     * @param model    模型
     * @param filePath 保存路径
     */
    public static void exportToJson(Model model, String filePath) {
        exportToJson(model, filePath, true);
    }

    /**
     * 将模型信息导出为JSON格式
     *
     * @param model             模型
     * @param filePath          保存路径
     * @param includeParameters 是否包含参数详细信息
     */
    public static void exportToJson(Model model, String filePath, boolean includeParameters) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            ModelInfo info = model.getModelInfo();

            writer.println("{");

            // 基本信息
            writer.println("  \"basicInfo\": {");
            writeJsonField(writer, "modelName", info.getModelName(), true);
            writeJsonField(writer, "modelVersion", info.getModelVersion(), true);
            writeJsonField(writer, "frameworkVersion", info.getFrameworkVersion(), true);
            writeJsonField(writer, "createdTime", formatDate(info.getCreatedTime()), true);
            writeJsonField(writer, "lastModifiedTime", formatDate(info.getLastModifiedTime()), true);
            writeJsonField(writer, "description", info.getDescription(), false);
            writer.println("  },");

            // 架构信息
            writer.println("  \"architecture\": {");
            writeJsonField(writer, "type", info.getArchitectureType(), true);
            writeJsonField(writer, "inputShape", shapeToString(info.getInputShape()), true);
            writeJsonField(writer, "outputShape", shapeToString(info.getOutputShape()), true);
            writeJsonField(writer, "totalLayers", info.getTotalLayers(), true);
            writeJsonField(writer, "totalParameters", info.getTotalParameters(), true);

            // 层统计
            if (!info.getLayerCounts().isEmpty()) {
                writer.println("    \"layerCounts\": {");
                int count = 0;
                for (Map.Entry<String, Integer> entry : info.getLayerCounts().entrySet()) {
                    boolean isLast = (++count == info.getLayerCounts().size());
                    writeJsonField(writer, entry.getKey(), entry.getValue(), !isLast, "      ");
                }
                writer.println("    }");
            } else {
                writer.println("    \"layerCounts\": {}");
            }
            writer.println("  },");

            // 训练信息
            writer.println("  \"training\": {");
            writeJsonField(writer, "trainedEpochs", info.getTrainedEpochs(), true);
            writeJsonField(writer, "finalLoss", info.getFinalLoss(), true);
            writeJsonField(writer, "bestLoss", info.getBestLoss(), true);
            writeJsonField(writer, "optimizerType", info.getOptimizerType(), true);
            writeJsonField(writer, "learningRate", info.getLearningRate(), true);
            writeJsonField(writer, "batchSize", info.getBatchSize(), true);
            writeJsonField(writer, "lossFunction", info.getLossFunction(), true);
            writeJsonField(writer, "trainingTimeMs", info.getTrainingTimeMs(), false);
            writer.println("  },");

            // 性能指标
            writer.println("  \"metrics\": {");
            if (!info.getMetrics().isEmpty()) {
                int count = 0;
                for (Map.Entry<String, Double> entry : info.getMetrics().entrySet()) {
                    boolean isLast = (++count == info.getMetrics().size());
                    writeJsonField(writer, entry.getKey(), entry.getValue(), !isLast);
                }
            }
            writer.println("  },");

            // 自定义属性
            writer.println("  \"customProperties\": {");
            if (!info.getCustomProperties().isEmpty()) {
                int count = 0;
                for (Map.Entry<String, Object> entry : info.getCustomProperties().entrySet()) {
                    boolean isLast = (++count == info.getCustomProperties().size());
                    writeJsonField(writer, entry.getKey(), entry.getValue(), !isLast);
                }
            }
            writer.println("  }");

            // 参数详细信息（可选）
            if (includeParameters) {
                writer.println(",");
                writer.println("  \"parameters\": {");
                Map<String, Parameter> params = model.getAllParams();
                if (!params.isEmpty()) {
                    int count = 0;
                    for (Map.Entry<String, Parameter> entry : params.entrySet()) {
                        boolean isLast = (++count == params.size());
                        Parameter param = entry.getValue();
                        Shape shape = param.getValue().getShape();

                        writer.println("    \"" + entry.getKey() + "\": {");
                        writeJsonField(writer, "shape", shapeToString(shape), true, "      ");
                        writeJsonField(writer, "size", shape.size(), false, "      ");
                        writer.print("    }");
                        if (!isLast) writer.print(",");
                        writer.println();
                    }
                }
                writer.println("  }");
            }

            writer.println("}");

        } catch (IOException e) {
            throw new RuntimeException("Failed to export model info to JSON: " + e.getMessage(), e);
        }
    }

    /**
     * 将ModelInfo对象导出为JSON
     *
     * @param modelInfo 模型信息
     * @param filePath  保存路径
     */
    public static void exportModelInfoToJson(ModelInfo modelInfo, String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("{");

            // 基本信息
            writer.println("  \"basicInfo\": {");
            writeJsonField(writer, "modelName", modelInfo.getModelName(), true);
            writeJsonField(writer, "modelVersion", modelInfo.getModelVersion(), true);
            writeJsonField(writer, "frameworkVersion", modelInfo.getFrameworkVersion(), true);
            writeJsonField(writer, "createdTime", formatDate(modelInfo.getCreatedTime()), true);
            writeJsonField(writer, "lastModifiedTime", formatDate(modelInfo.getLastModifiedTime()), true);
            writeJsonField(writer, "description", modelInfo.getDescription(), false);
            writer.println("  },");

            // 架构信息
            writer.println("  \"architecture\": {");
            writeJsonField(writer, "type", modelInfo.getArchitectureType(), true);
            writeJsonField(writer, "inputShape", shapeToString(modelInfo.getInputShape()), true);
            writeJsonField(writer, "outputShape", shapeToString(modelInfo.getOutputShape()), true);
            writeJsonField(writer, "totalLayers", modelInfo.getTotalLayers(), true);
            writeJsonField(writer, "totalParameters", modelInfo.getTotalParameters(), false);
            writer.println("  },");

            // 训练信息
            writer.println("  \"training\": {");
            writeJsonField(writer, "trainedEpochs", modelInfo.getTrainedEpochs(), true);
            writeJsonField(writer, "finalLoss", modelInfo.getFinalLoss(), true);
            writeJsonField(writer, "bestLoss", modelInfo.getBestLoss(), true);
            writeJsonField(writer, "optimizerType", modelInfo.getOptimizerType(), true);
            writeJsonField(writer, "learningRate", modelInfo.getLearningRate(), true);
            writeJsonField(writer, "batchSize", modelInfo.getBatchSize(), true);
            writeJsonField(writer, "lossFunction", modelInfo.getLossFunction(), false);
            writer.println("  },");

            // 性能指标
            writer.println("  \"metrics\": {");
            if (!modelInfo.getMetrics().isEmpty()) {
                int count = 0;
                for (Map.Entry<String, Double> entry : modelInfo.getMetrics().entrySet()) {
                    boolean isLast = (++count == modelInfo.getMetrics().size());
                    writeJsonField(writer, entry.getKey(), entry.getValue(), !isLast);
                }
            }
            writer.println("  }");

            writer.println("}");

        } catch (IOException e) {
            throw new RuntimeException("Failed to export ModelInfo to JSON: " + e.getMessage(), e);
        }
    }

    /**
     * 生成简化的JSON报告
     *
     * @param model    模型
     * @param filePath 保存路径
     */
    public static void exportSimpleReport(Model model, String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            ModelInfo info = model.getModelInfo();

            writer.println("{");
            writeJsonField(writer, "modelName", info.getModelName(), true);
            writeJsonField(writer, "architectureType", info.getArchitectureType(), true);
            writeJsonField(writer, "totalParameters", info.getTotalParameters(), true);
            writeJsonField(writer, "trainedEpochs", info.getTrainedEpochs(), true);
            writeJsonField(writer, "finalLoss", info.getFinalLoss(), true);
            writeJsonField(writer, "inputShape", shapeToString(info.getInputShape()), true);
            writeJsonField(writer, "outputShape", shapeToString(info.getOutputShape()), false);
            writer.println("}");

        } catch (IOException e) {
            throw new RuntimeException("Failed to export simple report: " + e.getMessage(), e);
        }
    }

    /**
     * 生成比较报告（比较两个模型）
     *
     * @param model1   模型1
     * @param model2   模型2
     * @param filePath 保存路径
     */
    public static void exportComparisonReport(Model model1, Model model2, String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("{");
            writer.println("  \"comparison\": {");

            // 模型1信息
            writer.println("    \"model1\": {");
            writeModelSummary(writer, model1, "      ");
            writer.println("    },");

            // 模型2信息
            writer.println("    \"model2\": {");
            writeModelSummary(writer, model2, "      ");
            writer.println("    },");

            // 比较结果
            writer.println("    \"differences\": {");
            ModelInfo info1 = model1.getModelInfo();
            ModelInfo info2 = model2.getModelInfo();

            writeJsonField(writer, "parameterCountDiff",
                    info1.getTotalParameters() - info2.getTotalParameters(), true, "      ");
            writeJsonField(writer, "layerCountDiff",
                    info1.getTotalLayers() - info2.getTotalLayers(), true, "      ");

            boolean sameArchitecture = (info1.getArchitectureType() != null &&
                    info1.getArchitectureType().equals(info2.getArchitectureType()));
            writeJsonField(writer, "sameArchitecture", sameArchitecture, true, "      ");

            boolean parametersEqual = ParameterManager.compareParameters(model1, model2);
            writeJsonField(writer, "parametersEqual", parametersEqual, false, "      ");

            writer.println("    }");
            writer.println("  }");
            writer.println("}");

        } catch (IOException e) {
            throw new RuntimeException("Failed to export comparison report: " + e.getMessage(), e);
        }
    }

    // 辅助方法

    /**
     * 写入模型摘要信息
     *
     * @param writer PrintWriter对象
     * @param model  模型
     * @param indent 缩进字符串
     */
    private static void writeModelSummary(PrintWriter writer, Model model, String indent) {
        ModelInfo info = model.getModelInfo();
        writeJsonField(writer, "name", info.getModelName(), true, indent);
        writeJsonField(writer, "architecture", info.getArchitectureType(), true, indent);
        writeJsonField(writer, "totalParameters", info.getTotalParameters(), true, indent);
        writeJsonField(writer, "totalLayers", info.getTotalLayers(), true, indent);
        writeJsonField(writer, "trainedEpochs", info.getTrainedEpochs(), true, indent);
        writeJsonField(writer, "finalLoss", info.getFinalLoss(), false, indent);
    }

    /**
     * 写入JSON字段
     *
     * @param writer  PrintWriter对象
     * @param key     字段键
     * @param value   字段值
     * @param hasNext 是否有后续字段
     */
    private static void writeJsonField(PrintWriter writer, String key, Object value, boolean hasNext) {
        writeJsonField(writer, key, value, hasNext, "    ");
    }

    /**
     * 写入JSON字段（带缩进）
     *
     * @param writer  PrintWriter对象
     * @param key     字段键
     * @param value   字段值
     * @param hasNext 是否有后续字段
     * @param indent  缩进字符串
     */
    private static void writeJsonField(PrintWriter writer, String key, Object value, boolean hasNext, String indent) {
        writer.print(indent + "\"" + key + "\": ");

        if (value == null) {
            writer.print("null");
        } else if (value instanceof String) {
            writer.print("\"" + escapeJsonString((String) value) + "\"");
        } else if (value instanceof Number) {
            writer.print(value.toString());
        } else if (value instanceof Boolean) {
            writer.print(value.toString());
        } else {
            writer.print("\"" + escapeJsonString(value.toString()) + "\"");
        }

        if (hasNext) {
            writer.print(",");
        }
        writer.println();
    }

    /**
     * 转义JSON字符串
     *
     * @param str 原始字符串
     * @return 转义后的字符串
     */
    private static String escapeJsonString(String str) {
        if (str == null) return "";
        return str.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    /**
     * 格式化日期
     *
     * @param date 日期对象
     * @return 格式化后的日期字符串
     */
    private static String formatDate(Date date) {
        return date != null ? DATE_FORMAT.format(date) : null;
    }

    /**
     * 形状转字符串
     *
     * @param shape 形状对象
     * @return 形状字符串表示
     */
    private static String shapeToString(Shape shape) {
        return shape != null ? shape.toString() : null;
    }
}