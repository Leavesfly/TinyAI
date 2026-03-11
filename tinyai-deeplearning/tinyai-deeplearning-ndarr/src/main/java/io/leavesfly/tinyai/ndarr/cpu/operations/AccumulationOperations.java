package io.leavesfly.tinyai.ndarr.cpu.operations;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.IndexConverter;

import java.util.Arrays;

/**
 * 累加操作类，提供在指定位置累加数组元素等功能。
 */
public class AccumulationOperations {

    /**
     * 在指定行、列位置累加另一数组元素，常用于反向传播梯度累积。
     *
     * @param array     目标数组
     * @param rowSlices 行索引数组
     * @param colSlices 列索引数组
     * @param other     要累加的数组
     * @return 累加结果数组
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     * @throws RuntimeException         当数组不是矩阵时抛出
     */
    public static NdArrayCpu addAt(NdArrayCpu array, int[] rowSlices, int[] colSlices, NdArray other) {
        NdArrayCpu otherArray = (NdArrayCpu) other;

        // 验证当前数组是否为矩阵
        if (!array.shape.isMatrix()) {
            throw new RuntimeException("当前数组不是矩阵！");
        }

        // 处理null参数
        if (rowSlices == null) {
            rowSlices = NdArrayUtil.getSeq(array.shape.getRow());
        }
        if (colSlices == null) {
            colSlices = NdArrayUtil.getSeq(array.shape.getColumn());
        }

        // 创建结果数组的副本
        NdArrayCpu result = new NdArrayCpu(Arrays.copyOf(array.buffer, array.buffer.length), array.shape);

        // 验证输入参数
        validateAddAtParameters(rowSlices, colSlices, other);

        // 执行累加操作
        if (rowSlices.length == colSlices.length) {
            // 当行索引和列索引数量相等时，按对应位置累加
            addAtEqualLength(result, rowSlices, colSlices, otherArray);
        } else {
            // 当行索引和列索引数量不等时，对所有组合进行累加
            addAtDifferentLength(result, rowSlices, colSlices, otherArray);
        }

        return result;
    }

    /**
     * 当行索引和列索引数量相等时的累加操作
     */
    private static void addAtEqualLength(NdArrayCpu result, int[] rowSlices, int[] colSlices, NdArrayCpu other) {
        for (int i = 0; i < rowSlices.length; i++) {
            int row = rowSlices[i];
            int col = colSlices[i];

            // 边界检查
            if (row < 0 || row >= result.shape.getRow() || col < 0 || col >= result.shape.getColumn()) {
                throw new IllegalArgumentException(
                        String.format("索引超出范围：位置(%d, %d)，数组形状%s", row, col, result.shape)
                );
            }

            // 计算要累加的值
            float valueToAdd;
            if (other.shape.isMatrix() && other.shape.getRow() == 1) {
                valueToAdd = other.buffer[i % other.buffer.length];
            } else if (other.shape.isMatrix() && other.shape.getColumn() == 1) {
                valueToAdd = other.buffer[i % other.buffer.length];
            } else if (other.shape.isMatrix()) {
                if (i < other.shape.getRow() * other.shape.getColumn()) {
                    valueToAdd = other.buffer[i];
                } else {
                    valueToAdd = other.buffer[i % other.buffer.length];
                }
            } else {
                valueToAdd = other.buffer[i % other.buffer.length];
            }

            result.buffer[row * result.shape.getColumn() + col] += valueToAdd;
        }
    }

    /**
     * 当行索引和列索引数量不等时的累加操作
     */
    private static void addAtDifferentLength(NdArrayCpu result, int[] rowSlices, int[] colSlices, NdArrayCpu other) {
        for (int i = 0; i < rowSlices.length; i++) {
            int row = rowSlices[i];

            // 边界检查
            if (row < 0 || row >= result.shape.getRow()) {
                throw new IllegalArgumentException(
                        String.format("行索引超出范围：%d，最大行索引%d", row, result.shape.getRow() - 1)
                );
            }

            for (int j = 0; j < colSlices.length; j++) {
                int col = colSlices[j];

                // 边界检查
                if (col < 0 || col >= result.shape.getColumn()) {
                    throw new IllegalArgumentException(
                            String.format("列索引超出范围：%d，最大列索引%d", col, result.shape.getColumn() - 1)
                    );
                }

                // 计算other数组中的对应位置
                int otherIndex;
                if (other.shape.isMatrix()) {
                    otherIndex = i * colSlices.length + j;
                    if (otherIndex >= other.buffer.length) {
                        otherIndex = otherIndex % other.buffer.length;
                    }
                } else {
                    otherIndex = i * colSlices.length + j;
                    if (otherIndex >= other.buffer.length) {
                        otherIndex = otherIndex % other.buffer.length;
                    }
                }

                result.buffer[row * result.shape.getColumn() + col] += other.buffer[otherIndex];
            }
        }
    }

    /**
     * 验证addAt方法的输入参数
     *
     * @param rowSlices 行索引数组
     * @param colSlices 列索引数组
     * @param other     要累加的数组
     * @throws IllegalArgumentException 当参数不合法时抛出
     */
    private static void validateAddAtParameters(int[] rowSlices, int[] colSlices, NdArray other) {
        if (rowSlices.length == 0 || colSlices.length == 0) {
            throw new IllegalArgumentException("行索引数组和列索引数组不能为空");
        }

        if (other == null) {
            throw new IllegalArgumentException("要累加的数组不能为null");
        }

        if (((NdArrayCpu) other).buffer.length == 0) {
            throw new IllegalArgumentException("要累加的数组不能为空");
        }
    }

    /**
     * 将另一个数组累加到当前数组的指定位置
     *
     * @param array 目标数组
     * @param i     起始行索引
     * @param j     起始列索引
     * @param other 要累加的数组
     * @return 目标数组实例
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    public static NdArrayCpu addTo(NdArrayCpu array, int i, int j, NdArray other) {
        if (array.shape.getDimNum() < 2 || other.getShape().getDimNum() < 2) {
            throw new IllegalArgumentException("操作需要至少二维数组");
        }

        // 仅支持维度数量一致，且前置维度完全匹配
        if (array.shape.getDimNum() != other.getShape().getDimNum()) {
            throw new IllegalArgumentException("源和目标的维度数量必须一致");
        }
        for (int dim = 0; dim < array.shape.getDimNum() - 2; dim++) {
            if (array.shape.getDimension(dim) != other.getShape().getDimension(dim)) {
                throw new IllegalArgumentException(
                        String.format("维度%d不匹配：%d vs %d", dim,
                                array.shape.getDimension(dim), other.getShape().getDimension(dim))
                );
            }
        }

        int rowOffset = i;
        int colOffset = j;
        int targetRows = array.shape.getDimension(array.shape.getDimNum() - 2);
        int targetCols = array.shape.getDimension(array.shape.getDimNum() - 1);

        if (rowOffset < 0 || colOffset < 0) {
            throw new IllegalArgumentException("偏移不能为负数");
        }

        ShapeCpu otherShape = (ShapeCpu) other.getShape();
        int dimNum = otherShape.getDimNum();
        int[] otherIdx = new int[dimNum];
        int[] targetIdx = new int[dimNum];

        NdArrayCpu otherArray = (NdArrayCpu) other;
        for (int flat = 0; flat < otherArray.buffer.length; flat++) {
            IndexConverter.flatToMultiIndex(flat, otherIdx, otherShape);
            System.arraycopy(otherIdx, 0, targetIdx, 0, dimNum);
            targetIdx[dimNum - 2] += rowOffset;
            targetIdx[dimNum - 1] += colOffset;

            if (targetIdx[dimNum - 2] >= targetRows || targetIdx[dimNum - 1] >= targetCols) {
                throw new IllegalArgumentException("累加位置超出目标数组范围");
            }

            int dstIndex = array.shape.getIndex(targetIdx);
            array.buffer[dstIndex] += otherArray.buffer[flat];
        }
        return array;
    }
}

