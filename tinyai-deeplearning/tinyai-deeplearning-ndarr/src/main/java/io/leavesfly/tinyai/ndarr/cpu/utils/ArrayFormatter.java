package io.leavesfly.tinyai.ndarr.cpu.utils;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;

/**
 * 数组格式化工具类，提供数组的字符串表示。
 */
public class ArrayFormatter {

    /** 元素总数不超过此阈值时，显示全部元素 */
    private static final int SMALL_ARRAY_THRESHOLD = 10;

    /** 大数组预览时显示的元素数量 */
    private static final int PREVIEW_ELEMENT_COUNT = 5;

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

        if (array.shape.size() <= SMALL_ARRAY_THRESHOLD) {
            // 小数组直接显示所有元素
            toStringHelper(sb, array, 0, new int[array.shape.dimension.length]);
        } else {
            // 大数组只显示前几个元素
            sb.append("[");
            int previewCount = Math.min(PREVIEW_ELEMENT_COUNT, array.buffer.length);
            for (int i = 0; i < previewCount; i++) {
                sb.append(String.format("%.4f", array.buffer[i]));
                if (i < previewCount - 1) {
                    sb.append(", ");
                }
            }
            if (array.buffer.length > PREVIEW_ELEMENT_COUNT) {
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

