package io.leavesfly.tinyai.minimind.model;

import java.util.Arrays;

/**
 * 文本采样工具类
 * <p>
 * 提供统一的文本生成采样方法，支持多种采样策略：
 * - 贪婪采样（temperature = 0）
 * - 温度采样
 * - Top-K 采样
 * - Top-P（nucleus）采样
 * - 多项式采样
 * <p>
 * 该工具类为 MiniMindModel 提供统一的采样接口，
 * 避免采样逻辑在多处重复。
 *
 * @author leavesfly
 * @version 1.0
 */
public class TextSampler {

    /**
     * 统一采样入口
     * <p>
     * 根据给定的参数从 logits 中采样一个 token ID
     *
     * @param logits      Logits 数组，形状 [vocab_size]
     * @param temperature 温度参数（控制随机性，0.0 = 贪婪，1.0 = 原始分布）
     * @param topK        Top-K 采样参数（0 表示不使用）
     * @param topP        Top-P 采样参数（0.0 表示不使用）
     * @return 采样的 token ID
     */
    public static int sample(float[] logits, float temperature, int topK, float topP) {
        // 应用温度缩放（创建副本避免修改原始数组）
        float[] workingLogits = logits;
        if (temperature > 0 && temperature != 1.0f) {
            workingLogits = new float[logits.length];
            for (int i = 0; i < logits.length; i++) {
                workingLogits[i] = logits[i] / temperature;
            }
        }

        // Softmax 转换为概率
        float[] probs = softmax(workingLogits);

        // 贪婪采样（temperature = 0）
        if (temperature == 0.0f) {
            return argmax(probs);
        }

        // Top-K 采样
        if (topK > 0) {
            probs = applyTopK(probs, topK);
        }

        // Top-P 采样
        if (topP > 0 && topP < 1.0f) {
            probs = applyTopP(probs, topP);
        }

        // 多项式采样
        return multinomialSample(probs);
    }

    /**
     * Softmax 函数
     * <p>
     * 将 logits 转换为概率分布，包含 sum==0 保护逻辑
     *
     * @param logits Logits 数组
     * @return 概率分布数组
     */
    public static float[] softmax(float[] logits) {
        // 找到最大值（数值稳定性）
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }

        // 计算指数和
        float sum = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }

        // 添加 sum == 0 的保护，避免 NaN
        if (sum == 0.0f) {
            // 如果所有概率都为 0，则均匀分布
            for (int i = 0; i < probs.length; i++) {
                probs[i] = 1.0f / probs.length;
            }
        } else {
            // 归一化
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
        }

        return probs;
    }

    /**
     * Argmax 函数
     * <p>
     * 返回数组中最大值的索引
     *
     * @param array 输入数组
     * @return 最大值的索引
     */
    public static int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Top-K 采样过滤
     * <p>
     * 保留概率最大的 K 个 token，其余置零并重新归一化
     *
     * @param probs 概率分布数组
     * @param k     保留的 token 数量
     * @return 过滤后的概率分布数组
     */
    public static float[] applyTopK(float[] probs, int k) {
        // 降序排序索引
        int[] indices = argsort(probs);

        // 创建结果数组
        float[] result = new float[probs.length];
        for (int i = 0; i < Math.min(k, indices.length); i++) {
            result[indices[i]] = probs[indices[i]];
        }

        // 重新归一化
        float sum = 0.0f;
        for (float v : result) {
            sum += v;
        }
        if (sum > 0) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= sum;
            }
        }

        return result;
    }

    /**
     * Top-P（nucleus）采样过滤
     * <p>
     * 保留累积概率达到 P 的最小 token 集合，其余置零并重新归一化
     *
     * @param probs 概率分布数组
     * @param p     累积概率阈值（0.0 < p < 1.0）
     * @return 过滤后的概率分布数组
     */
    public static float[] applyTopP(float[] probs, float p) {
        // 降序排序索引
        int[] indices = argsort(probs);

        // 创建结果数组
        float[] result = new float[probs.length];
        float cumSum = 0.0f;

        for (int idx : indices) {
            if (cumSum >= p) {
                break;
            }
            result[idx] = probs[idx];
            cumSum += probs[idx];
        }

        // 重新归一化
        float sum = 0.0f;
        for (float v : result) {
            sum += v;
        }
        if (sum > 0) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= sum;
            }
        }

        return result;
    }

    /**
     * 多项式采样
     * <p>
     * 根据概率分布随机采样一个 token ID
     *
     * @param probs 概率分布数组（应已归一化）
     * @return 采样的 token ID
     */
    public static int multinomialSample(float[] probs) {
        float rand = (float) Math.random();
        float cumSum = 0.0f;

        for (int i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) {
                return i;
            }
        }

        // 如果由于浮点精度问题未命中，返回最后一个
        return probs.length - 1;
    }

    /**
     * 降序排序索引
     * <p>
     * 返回按值降序排列的索引数组
     *
     * @param array 输入数组
     * @return 降序排列的索引数组
     */
    private static int[] argsort(float[] array) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }

        // 降序排序
        Arrays.sort(indices, (a, b) -> Float.compare(array[b], array[a]));

        // 转换为 int[]
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = indices[i];
        }

        return result;
    }
}