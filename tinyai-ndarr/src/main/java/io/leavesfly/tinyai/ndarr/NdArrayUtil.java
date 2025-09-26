package io.leavesfly.tinyai.ndarr;

import java.util.Arrays;

/**
 * NdArray工具类，提供数组操作的辅助方法
 */
public class NdArrayUtil {

    /**
     * 按照指定轴对多个NdArray进行合并
     *
     * @param axis 合并的轴向，0表示按第一个维度合并，1表示按第二个维度合并，以此类推
     * @param ndArrays 需要合并的NdArray数组
     * @return 合并后的NdArray
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     */
    public static NdArray merge(int axis, NdArray... ndArrays) {
        // 边界检查
        if (ndArrays == null || ndArrays.length == 0) {
            throw new IllegalArgumentException("输入数组不能为空");
        }

        if (ndArrays.length == 1) {
            return ndArrays[0];
        }

        // 获取第一个数组作为参考
        NdArray first = ndArrays[0];
        if (first == null) {
            throw new IllegalArgumentException("输入数组不能包含null");
        }

        // 验证所有数组的形状兼容性
        Shape firstShape = first.getShape();
        if (axis < 0 || axis >= firstShape.dimension.length) {
            throw new IllegalArgumentException("axis参数超出范围: " + axis);
        }

        // 验证所有数组除了指定轴外的其他维度都相同
        for (int i = 1; i < ndArrays.length; i++) {
            if (ndArrays[i] == null) {
                throw new IllegalArgumentException("输入数组不能包含null");
            }

            Shape shape = ndArrays[i].getShape();
            if (shape.dimension.length != firstShape.dimension.length) {
                throw new IllegalArgumentException("所有数组的维度数必须相同");
            }

            // 检查除了合并轴之外的所有维度是否一致
            for (int dim = 0; dim < shape.dimension.length; dim++) {
                if (dim != axis && shape.dimension[dim] != firstShape.dimension[dim]) {
                    throw new IllegalArgumentException("除了合并轴外，所有数组的维度必须一致");
                }
            }
        }

        // 计算合并后的形状
        int[] newDimensions = Arrays.copyOf(firstShape.dimension, firstShape.dimension.length);
        newDimensions[axis] = 0;
        for (NdArray array : ndArrays) {
            newDimensions[axis] += array.getShape().dimension[axis];
        }

        Shape newShape = new Shape(newDimensions);
        NdArray result = new NdArray(newShape);

        // 执行合并操作
        if (firstShape.isMatrix() && axis == 0) {
            // 优化处理：按第一个维度合并矩阵（行合并）
            int offset = 0;
            for (NdArray array : ndArrays) {
                System.arraycopy(array.buffer, 0, result.buffer, offset, array.buffer.length);
                offset += array.buffer.length;
            }
        } else if (firstShape.isMatrix() && axis == 1) {
            // 优化处理：按第二个维度合并矩阵（列合并）
            mergeMatrixAxis1(result, ndArrays);
        } else {
            // 通用处理：沿任意轴合并
            mergeAlongAxis(result, axis, ndArrays);
        }

        return result;
    }

    /**
     * 优化的矩阵按列合并方法
     *
     * @param result 结果数组
     * @param ndArrays 待合并数组
     */
    private static void mergeMatrixAxis1(NdArray result, NdArray... ndArrays) {
        int rowIndex = 0;
        for (int i = 0; i < ndArrays[0].getShape().getRow(); i++) {
            int colIndex = 0;
            for (NdArray array : ndArrays) {
                Shape shape = array.getShape();
                for (int j = 0; j < shape.getColumn(); j++) {
                    result.buffer[rowIndex * result.getShape().getColumn() + colIndex + j] =
                            array.buffer[i * shape.getColumn() + j];
                }
                colIndex += shape.getColumn();
            }
            rowIndex++;
        }
    }

    /**
     * 沿指定轴合并多个数组（通用方法）
     *
     * @param result 结果数组
     * @param axis 合并轴
     * @param ndArrays 待合并数组
     */
    private static void mergeAlongAxis(NdArray result, int axis, NdArray... ndArrays) {
        int[] indices = new int[result.getShape().dimension.length];
        int[] resultMultipliers = result.getShape().multipliers;

        int offset = 0;
        for (NdArray array : ndArrays) {
            Shape arrayShape = array.getShape();
            int axisSize = arrayShape.dimension[axis];

            // 为当前数组设置偏移量
            indices[axis] = offset;

            // 复制数据
            for (int i = 0; i < array.buffer.length; i++) {
                // 将一维索引转换为多维索引
                int[] arrayIndices = new int[arrayShape.dimension.length];
                int remaining = i;
                for (int dim = arrayShape.dimension.length - 1; dim >= 0; dim--) {
                    if (arrayShape.multipliers[dim] == 0) {
                        arrayIndices[dim] = 0;
                    } else {
                        arrayIndices[dim] = remaining / arrayShape.multipliers[dim];
                        remaining %= arrayShape.multipliers[dim];
                    }
                }

                // 更新结果索引
                for (int dim = 0; dim < arrayShape.dimension.length; dim++) {
                    if (dim != axis) {
                        indices[dim] = arrayIndices[dim];
                    }
                }

                // 设置结果值
                int resultIndex = 0;
                for (int dim = 0; dim < indices.length; dim++) {
                    resultIndex += indices[dim] * resultMultipliers[dim];
                }
                result.buffer[resultIndex] = array.buffer[i];
            }

            offset += axisSize;
        }
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
}