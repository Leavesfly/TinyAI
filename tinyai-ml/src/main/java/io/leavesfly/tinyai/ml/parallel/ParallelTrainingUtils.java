package io.leavesfly.tinyai.ml.parallel;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Parameter;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;

/**
 * 并行训练工具类
 * <p>
 * 提供模型深拷贝、梯度同步等并行训练所需的工具方法
 */
public class ParallelTrainingUtils {

    /**
     * 创建模型的深拷贝，用于多线程训练
     * 每个线程需要独立的模型实例来避免参数冲突
     *
     * @param originalModel 原始模型
     * @return 深拷贝的模型实例
     * @throws RuntimeException 如果拷贝失败
     */
    public static Model deepCopyModel(Model originalModel) {
        try {
            // 使用序列化进行深拷贝
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(originalModel);
            oos.close();

            ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bais);
            Model copiedModel = (Model) ois.readObject();
            ois.close();

            return copiedModel;

        } catch (Exception e) {
            throw new RuntimeException("模型深拷贝失败: " + e.getMessage(), e);
        }
    }

    /**
     * 将聚合后的梯度应用到主模型的参数上
     *
     * @param model               主模型
     * @param aggregatedGradients 聚合后的梯度
     */
    public static void applyAggregatedGradients(Model model, Map<String, NdArray> aggregatedGradients) {
        Map<String, Parameter> modelParams = model.getAllParams();

        for (Map.Entry<String, Parameter> entry : modelParams.entrySet()) {
            String paramName = entry.getKey();
            Parameter parameter = entry.getValue();

            // 获取对应的聚合梯度
            NdArray aggregatedGrad = aggregatedGradients.get(paramName);
            if (aggregatedGrad != null) {
                // 将聚合梯度设置到参数上
                parameter.setGrad(aggregatedGrad);
            }
        }
    }

    /**
     * 计算并行训练的推荐线程数
     * 基于CPU核心数和数据批次数量
     *
     * @param batchCount 数据批次总数
     * @return 推荐的线程数
     */
    public static int getRecommendedThreadCount(int batchCount) {
        int availableCores = Runtime.getRuntime().availableProcessors();

        // 线程数不应超过可用核心数的75%，也不应超过批次数量
        int maxThreads = Math.max(1, (int) (availableCores * 0.75));
        return Math.min(maxThreads, batchCount);
    }

    /**
     * 检查模型是否支持并行训练
     * 主要检查模型是否可序列化
     *
     * @param model 要检查的模型
     * @return true 如果支持并行训练
     */
    public static boolean isModelParallelizable(Model model) {
        try {
            // 尝试序列化测试
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(model);
            oos.close();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 格式化并行训练的统计信息
     *
     * @param threadCount       线程数
     * @param successfulBatches 成功处理的批次数
     * @param totalBatches      总批次数
     * @param averageLoss       平均损失
     * @param processingTimeMs  处理时间（毫秒）
     * @return 格式化的统计信息字符串
     */
    public static String formatParallelStats(int threadCount, int successfulBatches,
                                             int totalBatches, float averageLoss, long processingTimeMs) {
        return String.format(
                "并行训练统计 [线程数: %d, 成功批次: %d/%d, 平均损失: %.6f, 处理时间: %dms]",
                threadCount, successfulBatches, totalBatches, averageLoss, processingTimeMs
        );
    }
}