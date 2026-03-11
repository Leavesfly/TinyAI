package io.leavesfly.tinyai.ndarr.cpu.aggregations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;
import io.leavesfly.tinyai.ndarr.cpu.utils.IndexConverter;

/**
 * 聚合操作类，提供求和、均值、方差、最大/最小值等运算。
 * 直接访问底层 buffer，减少索引转换和方法调用开销。
 */
public class ReductionOperations {

    /**
     * 元素累和运算，计算数组所有元素的总和
     *
     * @param array 数组
     * @return 所有元素的总和（标量）
     */
    public static NdArrayCpu sum(NdArrayCpu array) {
        float sum = 0f;
        float[] buffer = array.buffer;
        for (int i = 0; i < buffer.length; i++) {
            sum += buffer[i];
        }
        return new NdArrayCpu(sum);
    }

    /**
     * 矩阵均值运算，沿指定轴计算均值
     *
     * @param array 数组
     * @param axis  聚合轴，axis=0表示按列计算均值，axis=1表示按行计算均值
     * @return 均值运算结果数组
     */
    public static NdArrayCpu mean(NdArrayCpu array, int axis) {
        return axisSum(array, axis, true);
    }

    /**
     * 矩阵方差运算，沿指定轴计算方差
     *
     * @param array 数组
     * @param axis  聚合轴，axis=0表示按列计算方差，axis=1表示按行计算方差
     * @return 方差运算结果数组
     */
    public static NdArrayCpu var(NdArrayCpu array, int axis) {
        ArrayValidator.validateAxis(axis, array.shape.getDimNum());
        
        ShapeCpu newShape = computeReducedShape(array.shape, axis);
        NdArrayCpu result = new NdArrayCpu(newShape);
        
        int axisSize = array.shape.getDimension(axis);
        int[] indices = new int[array.shape.getDimNum()];
        int[] resultIndices = new int[newShape.getDimNum()];
        
        float[] buffer = array.buffer;
        ShapeCpu shape = array.shape;
        
        for (int i = 0; i < newShape.size(); i++) {
            // 将结果索引转换为多维索引
            IndexConverter.flatToMultiIndex(i, resultIndices, newShape);
            
            // 构建完整的索引数组（排除axis维度）
            buildIndicesExcludingAxis(indices, resultIndices, axis, shape.getDimNum());
            
            // 计算均值和方差（单次遍历）
            float sum = 0f;
            for (int j = 0; j < axisSize; j++) {
                indices[axis] = j;
                sum += buffer[shape.getIndex(indices)];
            }
            float mean = sum / axisSize;
            
            // 计算方差
            float variance = 0f;
            for (int j = 0; j < axisSize; j++) {
                indices[axis] = j;
                float diff = buffer[shape.getIndex(indices)] - mean;
                variance += diff * diff;
            }
            result.buffer[i] = variance / axisSize;
        }
        return result;
    }

    /**
     * 矩阵累和运算，沿指定轴计算累和
     *
     * @param array 数组
     * @param axis  聚合轴，axis=0表示按列累和，axis=1表示按行累和
     * @return 累和运算结果数组
     */
    public static NdArrayCpu sum(NdArrayCpu array, int axis) {
        return axisSum(array, axis, false);
    }

    /**
     * 沿指定轴查找最大值
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最大值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu max(NdArrayCpu array, int axis) {
        return axisMinMax(array, axis, true);
    }

    /**
     * 沿指定轴查找最小值
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最小值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu min(NdArrayCpu array, int axis) {
        return axisMinMax(array, axis, false);
    }

    /**
     * 查找数组中的最大值（全局最大值）
     *
     * @param array 数组
     * @return 数组中的最大值
     */
    public static float max(NdArrayCpu array) {
        float max = Float.NEGATIVE_INFINITY;
        float[] buffer = array.buffer;
        for (int i = 0; i < buffer.length; i++) {
            if (buffer[i] > max) {
                max = buffer[i];
            }
        }
        return max;
    }

    // =============================================================================
    // 私有辅助方法
    // =============================================================================

    /**
     * 计算沿轴求和或均值（优化版本，避免创建中间数组）
     *
     * @param array   数组
     * @param axis    聚合轴
     * @param computeMean 是否计算均值（true）还是求和（false）
     * @return 聚合结果数组
     */
    private static NdArrayCpu axisSum(NdArrayCpu array, int axis, boolean computeMean) {
        ArrayValidator.validateAxis(axis, array.shape.getDimNum());
        
        ShapeCpu newShape = computeReducedShape(array.shape, axis);
        NdArrayCpu result = new NdArrayCpu(newShape);
        
        int axisSize = array.shape.getDimension(axis);
        int[] indices = new int[array.shape.getDimNum()];
        int[] resultIndices = new int[newShape.getDimNum()];
        
        float[] buffer = array.buffer;
        ShapeCpu shape = array.shape;
        
        for (int i = 0; i < newShape.size(); i++) {
            // 将结果索引转换为多维索引
            IndexConverter.flatToMultiIndex(i, resultIndices, newShape);
            
            // 构建完整的索引数组（排除axis维度）
            buildIndicesExcludingAxis(indices, resultIndices, axis, shape.getDimNum());
            
            // 计算沿axis维度的累和
            float sum = 0f;
            for (int j = 0; j < axisSize; j++) {
                indices[axis] = j;
                sum += buffer[shape.getIndex(indices)];
            }
            result.buffer[i] = computeMean ? sum / axisSize : sum;
        }
        return result;
    }

    /**
     * 计算沿轴的最大值或最小值（优化版本）
     *
     * @param array   数组
     * @param axis    聚合轴
     * @param findMax true表示查找最大值，false表示查找最小值
     * @return 聚合结果数组
     */
    private static NdArrayCpu axisMinMax(NdArrayCpu array, int axis, boolean findMax) {
        ArrayValidator.validateAxis(axis, array.shape.getDimNum());
        
        ShapeCpu newShape = computeReducedShape(array.shape, axis);
        NdArrayCpu result = new NdArrayCpu(newShape);
        
        int axisSize = array.shape.getDimension(axis);
        int[] indices = new int[array.shape.getDimNum()];
        int[] resultIndices = new int[newShape.getDimNum()];
        
        float[] buffer = array.buffer;
        ShapeCpu shape = array.shape;
        
        float initialValue = findMax ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
        
        for (int i = 0; i < newShape.size(); i++) {
            // 将结果索引转换为多维索引
            IndexConverter.flatToMultiIndex(i, resultIndices, newShape);
            
            // 构建完整的索引数组（排除axis维度）
            buildIndicesExcludingAxis(indices, resultIndices, axis, shape.getDimNum());
            
            // 查找沿axis维度的最值
            float extremum = initialValue;
            for (int j = 0; j < axisSize; j++) {
                indices[axis] = j;
                float value = buffer[shape.getIndex(indices)];
                if ((findMax && value > extremum) || (!findMax && value < extremum)) {
                    extremum = value;
                }
            }
            result.buffer[i] = extremum;
        }
        return result;
    }

    /**
     * 计算去除指定轴后的新形状
     *
     * @param shape 原始形状
     * @param axis  要移除的轴
     * @return 新的形状
     */
    private static ShapeCpu computeReducedShape(ShapeCpu shape, int axis) {
        int dimNum = shape.getDimNum();
        int[] newDimensions = new int[dimNum - 1];
        int newIndex = 0;
        for (int i = 0; i < dimNum; i++) {
            if (i != axis) {
                newDimensions[newIndex++] = shape.getDimension(i);
            }
        }
        return ShapeCpu.of(newDimensions);
    }

    /**
     * 构建完整的索引数组，将结果索引映射到原始数组索引（排除axis维度）
     *
     * @param indices       输出的完整索引数组
     * @param resultIndices 结果数组的多维索引
     * @param axis          要排除的轴
     * @param totalDims     总维度数
     */
    private static void buildIndicesExcludingAxis(int[] indices, int[] resultIndices, int axis, int totalDims) {
        int resultIndex = 0;
        for (int dim = 0; dim < totalDims; dim++) {
            if (dim == axis) {
                indices[dim] = 0; // axis维度将在后续循环中变化
            } else {
                indices[dim] = resultIndices[resultIndex++];
            }
        }
    }
}

