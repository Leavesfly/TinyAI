package io.leavesfly.tinyai.ndarr.cpu.utils;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;

/**
 * 数组格式化工具类，提供数组的字符串表示。
 */
public class ArrayFormatter {

    /**
     * 将数组转换为字符串，小数组显示全部元素，大数组显示部分元素。
     *
     * @param array 数组
     * @return 字符串表示
     */
    public static String toString(NdArrayCpu array) {
        StringBuilder sb = new StringBuilder();
        sb.append("NdArray{");
        sb.append("shape=").append(array.shape);
        sb.append(", data=");

        if (array.shape.size() <= 10) {
            // 小数组直接显示所有元素
            toStringHelper(sb, array, 0, new int[array.shape.dimension.length]);
        } else {
            // 大数组只显示前几个元素
            sb.append("[");
            for (int i = 0; i < Math.min(5, array.buffer.length); i++) {
                sb.append(String.format("%.4f", array.buffer[i]));
                if (i < Math.min(4, array.buffer.length - 1)) {
                    sb.append(", ");
                }
            }
            if (array.buffer.length > 5) {
                sb.append(", ..., ").append(String.format("%.4f", array.buffer[array.buffer.length - 1]));
            }
            sb.append("]");
        }

        sb.append("}");
        return sb.toString();
    }

    /**
     * 递归构建多维数组的字符串表示
     *
     * @param sb       字符串构建器
     * @param array    数组
     * @param dimIndex 当前维度索引
     * @param indices  多维索引数组
     */
    private static void toStringHelper(StringBuilder sb, NdArrayCpu array, int dimIndex, int[] indices) {
        if (dimIndex == array.shape.dimension.length) {
            sb.append(String.format("%.4f", array.get(indices)));
            return;
        }

        sb.append("[");
        for (int i = 0; i < array.shape.dimension[dimIndex]; i++) {
            indices[dimIndex] = i;
            toStringHelper(sb, array, dimIndex + 1, indices);
            if (i < array.shape.dimension[dimIndex] - 1) {
                sb.append(", ");
                if (dimIndex == array.shape.dimension.length - 2) {
                    sb.append("\n ");
                }
            }
        }
        sb.append("]");

        if (dimIndex == 0) {
            sb.append("\n");
        }
    }
}

