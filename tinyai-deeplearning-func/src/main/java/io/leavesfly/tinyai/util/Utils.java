package io.leavesfly.tinyai.util;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 通用工具类
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * Util类提供了各种通用的工具方法，包括数值微分、数组转换、格式化等功能。
 */
public class Utils {

    /**
     * 数字微分函数的通用求导
     * 使用数值微分方法计算函数在指定点的导数
     *
     * @param func  要求导的函数
     * @param input 输入值
     * @param eps   微小增量，默认为0.00001
     * @return 导数结果
     */
    public static NdArray numericalDiff(Function func, NdArray input, float eps) {

        if (eps == 0) {
            eps = 0.00001F;
        }

        NdArray x1 = input.add(NdArray.of(eps));
        NdArray x0 = input.sub(NdArray.of(eps));

        NdArray y1 = func.forward(x1);
        NdArray y0 = func.forward(x0);

        return y1.sub(y0).divNum(eps * 2);
    }


    /**
     * 生成从0开始的连续整数序列
     *
     * @param size 序列长度
     * @return 连续整数数组
     */
    public static int[] getSeq(int size) {
        int[] seq = new int[size];
        for (int i = 0; i < size; i++) {
            seq[i] = i;
        }
        return seq;
    }

    /**
     * 生成从0开始的连续整数对象序列
     *
     * @param size 序列长度
     * @return 连续整数对象数组
     */
    public static Integer[] getSeqIndex(int size) {
        Integer[] seq = new Integer[size];
        for (int i = 0; i < size; i++) {
            seq[i] = i;
        }
        return seq;
    }

    /**
     * 将浮点数组转换为整型数组
     *
     * @param src 浮点数组
     * @return 整型数组
     */
    public static int[] toInt(float[] src) {
        int[] res = new int[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = (int) src[i];
        }
        return res;
    }

    /**
     * 将整型数组转换为整型对象数组
     *
     * @param src 整型数组
     * @return 整型对象数组
     */
    public static Integer[] toIntObject(int[] src) {
        Integer[] res = new Integer[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = src[i];
        }
        return res;
    }

    /**
     * 将整型数组转换为浮点数组
     *
     * @param src 整型数组
     * @return 浮点数组
     */
    public static float[] toFloat(int[] src) {
        float[] res = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = src[i];
        }
        return res;
    }

    /**
     * 将浮点对象数组转换为浮点数组
     *
     * @param src 浮点对象数组
     * @return 浮点数组
     */
    public static float[] toFloat(Float[] src) {
        float[] res = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = src[i];
        }
        return res;
    }

    /**
     * 格式化浮点数，保留适当的小数位数
     *
     * @param num 要格式化的数字
     * @return 格式化后的字符串
     */
    public static String format(float num) {
        return String.format("%,.7g", num);
    }

    /**
     * 返回数组中最大值的索引
     *
     * @param array 输入数组
     * @return 最大值的索引
     */
    public static int argMax(float[] array) {
        float maxValue = Float.MIN_VALUE;
        int maxIndex = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}