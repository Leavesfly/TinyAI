package io.leavesfly.tinyai.agent.rag;

import java.util.List;

/**
 * 向量相似度计算类
 * 提供多种向量相似度和距离计算方法
 */
public class VectorSimilarity {

    /**
     * 计算余弦相似度
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 余弦相似度值，范围[0, 1]
     */
    public static double cosineSimilarity(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.size() != vec2.size()) {
            return 0.0;
        }

        int size = vec1.size();
        if (size == 0) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        // 计算点积和各自的范数
        for (int i = 0; i < size; i++) {
            double v1 = vec1.get(i);
            double v2 = vec2.get(i);
            
            dotProduct += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        // 计算范数的平方根
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);

        // 避免除零错误
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }

        return dotProduct / (norm1 * norm2);
    }

    /**
     * 计算欧几里得距离
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 欧几里得距离
     */
    public static double euclideanDistance(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.size() != vec2.size()) {
            return Double.POSITIVE_INFINITY;
        }

        int size = vec1.size();
        if (size == 0) {
            return 0.0;
        }

        double sumSquaredDiffs = 0.0;

        for (int i = 0; i < size; i++) {
            double diff = vec1.get(i) - vec2.get(i);
            sumSquaredDiffs += diff * diff;
        }

        return Math.sqrt(sumSquaredDiffs);
    }

    /**
     * 计算曼哈顿距离（L1距离）
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 曼哈顿距离
     */
    public static double manhattanDistance(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.size() != vec2.size()) {
            return Double.POSITIVE_INFINITY;
        }

        int size = vec1.size();
        if (size == 0) {
            return 0.0;
        }

        double sumAbsDiffs = 0.0;

        for (int i = 0; i < size; i++) {
            double diff = Math.abs(vec1.get(i) - vec2.get(i));
            sumAbsDiffs += diff;
        }

        return sumAbsDiffs;
    }

    /**
     * 计算切比雪夫距离（L∞距离）
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 切比雪夫距离
     */
    public static double chebyshevDistance(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.size() != vec2.size()) {
            return Double.POSITIVE_INFINITY;
        }

        int size = vec1.size();
        if (size == 0) {
            return 0.0;
        }

        double maxDiff = 0.0;

        for (int i = 0; i < size; i++) {
            double diff = Math.abs(vec1.get(i) - vec2.get(i));
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }

        return maxDiff;
    }

    /**
     * 计算点积（内积）
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 点积值
     */
    public static double dotProduct(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.size() != vec2.size()) {
            return 0.0;
        }

        int size = vec1.size();
        double product = 0.0;

        for (int i = 0; i < size; i++) {
            product += vec1.get(i) * vec2.get(i);
        }

        return product;
    }

    /**
     * 计算向量的L2范数（欧几里得范数）
     * @param vec 向量
     * @return L2范数
     */
    public static double l2Norm(List<Double> vec) {
        if (vec == null || vec.isEmpty()) {
            return 0.0;
        }

        double sumSquares = 0.0;
        for (double v : vec) {
            sumSquares += v * v;
        }

        return Math.sqrt(sumSquares);
    }

    /**
     * 计算向量的L1范数（曼哈顿范数）
     * @param vec 向量
     * @return L1范数
     */
    public static double l1Norm(List<Double> vec) {
        if (vec == null || vec.isEmpty()) {
            return 0.0;
        }

        double sumAbs = 0.0;
        for (double v : vec) {
            sumAbs += Math.abs(v);
        }

        return sumAbs;
    }

    /**
     * 向量标准化（单位化）
     * @param vec 输入向量
     * @return 标准化后的向量
     */
    public static List<Double> normalize(List<Double> vec) {
        if (vec == null || vec.isEmpty()) {
            return vec;
        }

        double norm = l2Norm(vec);
        if (norm == 0.0) {
            return vec; // 零向量无法标准化
        }

        return vec.stream()
                .map(v -> v / norm)
                .collect(java.util.ArrayList::new, java.util.ArrayList::add, java.util.ArrayList::addAll);
    }

    /**
     * 将欧几里得距离转换为相似度
     * 使用公式：similarity = 1 / (1 + distance)
     * @param distance 欧几里得距离
     * @return 相似度值，范围(0, 1]
     */
    public static double distanceToSimilarity(double distance) {
        return 1.0 / (1.0 + distance);
    }

    /**
     * 计算皮尔逊相关系数
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 皮尔逊相关系数，范围[-1, 1]
     */
    public static double pearsonCorrelation(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.size() != vec2.size() || vec1.size() < 2) {
            return 0.0;
        }

        int n = vec1.size();
        
        // 计算均值
        double mean1 = vec1.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double mean2 = vec2.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        double numerator = 0.0;
        double sumSq1 = 0.0;
        double sumSq2 = 0.0;
        
        for (int i = 0; i < n; i++) {
            double diff1 = vec1.get(i) - mean1;
            double diff2 = vec2.get(i) - mean2;
            
            numerator += diff1 * diff2;
            sumSq1 += diff1 * diff1;
            sumSq2 += diff2 * diff2;
        }
        
        double denominator = Math.sqrt(sumSq1 * sumSq2);
        if (denominator == 0.0) {
            return 0.0;
        }
        
        return numerator / denominator;
    }
}