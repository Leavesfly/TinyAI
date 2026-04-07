package io.leavesfly.tinyai.ndarr.cpu.utils;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;

/**
 * 数组格式转换工具类，提供 NdArray 与 Java 数组格式之间的转换。
 */
public class ArrayConverter {

    /**
     * 将NdArray转换为二维数组（矩阵）返回
     *
     * @param ndArray NdArray实例
     * @return 二维数组表示
     * @throws IllegalArgumentException 当数组维度大于2时抛出
     */
    public static float[][] toMatrix(NdArrayCpu ndArray) {
        ShapeCpu shape = ndArray.shape;
        if (shape.isMatrix()) {
            int rows = shape.dimension[0];
            int cols = shape.dimension[1];
            float[][] matrix = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                System.arraycopy(ndArray.buffer, i * cols, matrix[i], 0, cols);
            }
            return matrix;
        } else if (shape.dimension.length == 1) {
            float[][] matrix = new float[1][shape.dimension[0]];
            System.arraycopy(ndArray.buffer, 0, matrix[0], 0, shape.dimension[0]);
            return matrix;
        } else {
            throw new IllegalArgumentException("不支持维度大于2");
        }
    }

    /**
     * 将NdArray转换为三维数组返回
     *
     * @param ndArray NdArray实例
     * @return 三维数组表示
     * @throws IllegalArgumentException 当数组不是三维时抛出
     */
    public static float[][][] to3dArray(NdArrayCpu ndArray) {
        ShapeCpu shape = ndArray.shape;
        if (shape.dimension.length == 3) {
            float[][][] result = new float[shape.dimension[0]][shape.dimension[1]][shape.dimension[2]];
            int index = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    for (int k = 0; k < shape.dimension[2]; k++) {
                        result[i][j][k] = ndArray.buffer[index];
                        index++;
                    }
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("仅支持三维数组，当前维度: " + shape.dimension.length);
        }
    }

    /**
     * 将 NdArray 转换为四维数组返回
     *
     * @param ndArray NdArray实例
     * @return 四维数组表示
     * @throws IllegalArgumentException 当数组不是四维时抛出
     */
    public static float[][][][] to4dArray(NdArrayCpu ndArray) {
        ShapeCpu shape = ndArray.shape;
        if (shape.dimension.length == 4) {
            float[][][][] result = new float[shape.dimension[0]][shape.dimension[1]][shape.dimension[2]][shape.dimension[3]];
            int index = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    for (int k = 0; k < shape.dimension[2]; k++) {
                        for (int l = 0; l < shape.dimension[3]; l++) {
                            result[i][j][k][l] = ndArray.buffer[index];
                            index++;
                        }
                    }
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("仅支持四维数组，当前维度: " + shape.dimension.length);
        }
    }

    /**
     * 通用数组展平方法
     *
     * @param array 多维数组对象
     * @param buffer 目标一维数组
     * @param index 起始索引
     * @return 下一个可用索引
     */
    public static int flattenArray(Object array, float[] buffer, int index) {
        if (array instanceof float[]) {
            float[] arr = (float[]) array;
            System.arraycopy(arr, 0, buffer, index, arr.length);
            return index + arr.length;
        } else if (array.getClass().isArray()) {
            Object[] arr = (Object[]) array;
            for (Object subArray : arr) {
                index = flattenArray(subArray, buffer, index);
            }
        }
        return index;
    }
}

