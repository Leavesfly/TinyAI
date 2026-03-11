package io.leavesfly.tinyai.ndarr.cpu.operations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;

/**
 * 轴操作类，提供沿指定轴的最大值、最小值、argMax 等操作。
 * 当前实现主要优化最后两个轴。
 */
public class AxisOperations {

    /**
     * 沿指定轴查找最大值的索引
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最大值索引数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu argMax(NdArrayCpu array, int axis) {
        int normalizedAxis = normalizeAxis(axis, array.shape.getDimNum());
        int dimNum = array.shape.getDimNum();

        if (normalizedAxis == dimNum - 2) {
            return argMaxSecondLastAxis(array);
        } else if (normalizedAxis == dimNum - 1) {
            return argMaxLastAxis(array);
        }
        throw new IllegalArgumentException(
                String.format("不支持的轴参数: %d，仅支持 %d(列) 或 %d(行)", axis, dimNum - 2, dimNum - 1)
        );
    }

    /**
     * 沿倒数第二个轴查找最大值的索引（按行查找每列的最大值索引）
     */
    private static NdArrayCpu argMaxSecondLastAxis(NdArrayCpu array) {
        AxisContext ctx = new AxisContext(array);
        int[] newDims = createResultShape(array.shape, array.shape.getDimNum() - 2, 1);
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        float[] srcBuffer = array.buffer;
        float[] dstBuffer = result.buffer;

        // 预计算步长以提高性能
        final int matrixSize = ctx.lastDimSize * ctx.secondLastDimSize;
        final int rowStride = ctx.lastDimSize;

        for (int batch = 0; batch < ctx.batchSize; batch++) {
            int batchOffset = batch * matrixSize;
            int resultBatchOffset = batch * ctx.lastDimSize;

            for (int j = 0; j < ctx.lastDimSize; j++) {
                float maxValue = Float.NEGATIVE_INFINITY;
                int maxIndex = -1;

                for (int i = 0; i < ctx.secondLastDimSize; i++) {
                    int index = batchOffset + i * rowStride + j;
                    float value = srcBuffer[index];
                    if (maxValue < value) {
                        maxValue = value;
                        maxIndex = i;
                    }
                }

                dstBuffer[resultBatchOffset + j] = maxIndex;
            }
        }
        return result;
    }

    /**
     * 沿最后一个轴查找最大值的索引（按列查找每行的最大值索引）
     */
    private static NdArrayCpu argMaxLastAxis(NdArrayCpu array) {
        AxisContext ctx = new AxisContext(array);
        int[] newDims = createResultShape(array.shape, array.shape.getDimNum() - 1, 1);
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        float[] srcBuffer = array.buffer;
        float[] dstBuffer = result.buffer;

        // 预计算步长以提高性能
        final int matrixSize = ctx.lastDimSize * ctx.secondLastDimSize;
        final int rowStride = ctx.lastDimSize;

        for (int batch = 0; batch < ctx.batchSize; batch++) {
            int batchOffset = batch * matrixSize;
            int resultBatchOffset = batch * ctx.secondLastDimSize;

            for (int i = 0; i < ctx.secondLastDimSize; i++) {
                float maxValue = Float.NEGATIVE_INFINITY;
                int maxIndex = -1;
                int rowOffset = batchOffset + i * rowStride;

                for (int j = 0; j < ctx.lastDimSize; j++) {
                    float value = srcBuffer[rowOffset + j];
                    if (maxValue < value) {
                        maxValue = value;
                        maxIndex = j;
                    }
                }

                dstBuffer[resultBatchOffset + i] = maxIndex;
            }
        }
        return result;
    }

    /**
     * 沿指定轴查找最大值（优化版本，仅支持最后两个轴）
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最大值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu max(NdArrayCpu array, int axis) {
        int normalizedAxis = normalizeAxis(axis, array.shape.getDimNum());
        int dimNum = array.shape.getDimNum();

        if (normalizedAxis == dimNum - 1) {
            return maxLastAxis(array);
        } else if (normalizedAxis == dimNum - 2) {
            return maxSecondLastAxis(array);
        }
        throw new IllegalArgumentException(
                String.format("不支持的轴参数: %d，仅支持 %d(列) 或 %d(行)", axis, dimNum - 2, dimNum - 1)
        );
    }

    /**
     * 沿最后一个轴查找最大值（按列查找每行的最大值）
     */
    private static NdArrayCpu maxLastAxis(NdArrayCpu array) {
        AxisContext ctx = new AxisContext(array);
        int[] newDims = createResultShape(array.shape, array.shape.getDimNum() - 1, 1);
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        float[] srcBuffer = array.buffer;
        float[] dstBuffer = result.buffer;

        final int matrixSize = ctx.lastDimSize * ctx.secondLastDimSize;
        final int rowStride = ctx.lastDimSize;

        for (int batch = 0; batch < ctx.batchSize; batch++) {
            int batchOffset = batch * matrixSize;
            int resultBatchOffset = batch * ctx.secondLastDimSize;

            for (int i = 0; i < ctx.secondLastDimSize; i++) {
                float max = Float.NEGATIVE_INFINITY;
                int rowOffset = batchOffset + i * rowStride;

                for (int j = 0; j < ctx.lastDimSize; j++) {
                    float value = srcBuffer[rowOffset + j];
                    if (max < value) {
                        max = value;
                    }
                }

                dstBuffer[resultBatchOffset + i] = max;
            }
        }
        return result;
    }

    /**
     * 沿倒数第二个轴查找最大值（按行查找每列的最大值）
     */
    private static NdArrayCpu maxSecondLastAxis(NdArrayCpu array) {
        AxisContext ctx = new AxisContext(array);
        int[] newDims = createResultShape(array.shape, array.shape.getDimNum() - 2, 1);
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        float[] srcBuffer = array.buffer;
        float[] dstBuffer = result.buffer;

        final int matrixSize = ctx.lastDimSize * ctx.secondLastDimSize;
        final int rowStride = ctx.lastDimSize;

        for (int batch = 0; batch < ctx.batchSize; batch++) {
            int batchOffset = batch * matrixSize;
            int resultBatchOffset = batch * ctx.lastDimSize;

            for (int j = 0; j < ctx.lastDimSize; j++) {
                float max = Float.NEGATIVE_INFINITY;

                for (int i = 0; i < ctx.secondLastDimSize; i++) {
                    int index = batchOffset + i * rowStride + j;
                    float value = srcBuffer[index];
                    if (max < value) {
                        max = value;
                    }
                }

                dstBuffer[resultBatchOffset + j] = max;
            }
        }
        return result;
    }

    /**
     * 沿指定轴查找最小值（优化版本，仅支持最后两个轴）
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最小值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu min(NdArrayCpu array, int axis) {
        int normalizedAxis = normalizeAxis(axis, array.shape.getDimNum());
        int dimNum = array.shape.getDimNum();

        if (normalizedAxis == dimNum - 1) {
            return minLastAxis(array);
        } else if (normalizedAxis == dimNum - 2) {
            return minSecondLastAxis(array);
        }
        throw new IllegalArgumentException(
                String.format("不支持的轴参数: %d，仅支持 %d(列) 或 %d(行)", axis, dimNum - 2, dimNum - 1)
        );
    }

    /**
     * 沿最后一个轴查找最小值（按列查找每行的最小值）
     */
    private static NdArrayCpu minLastAxis(NdArrayCpu array) {
        AxisContext ctx = new AxisContext(array);
        int[] newDims = createResultShape(array.shape, array.shape.getDimNum() - 1, 1);
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        float[] srcBuffer = array.buffer;
        float[] dstBuffer = result.buffer;

        final int matrixSize = ctx.lastDimSize * ctx.secondLastDimSize;
        final int rowStride = ctx.lastDimSize;

        for (int batch = 0; batch < ctx.batchSize; batch++) {
            int batchOffset = batch * matrixSize;
            int resultBatchOffset = batch * ctx.secondLastDimSize;

            for (int i = 0; i < ctx.secondLastDimSize; i++) {
                float min = Float.MAX_VALUE;
                int rowOffset = batchOffset + i * rowStride;

                for (int j = 0; j < ctx.lastDimSize; j++) {
                    float value = srcBuffer[rowOffset + j];
                    if (min > value) {
                        min = value;
                    }
                }

                dstBuffer[resultBatchOffset + i] = min;
            }
        }
        return result;
    }

    /**
     * 沿倒数第二个轴查找最小值（按行查找每列的最小值）
     */
    private static NdArrayCpu minSecondLastAxis(NdArrayCpu array) {
        AxisContext ctx = new AxisContext(array);
        int[] newDims = createResultShape(array.shape, array.shape.getDimNum() - 2, 1);
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        float[] srcBuffer = array.buffer;
        float[] dstBuffer = result.buffer;

        final int matrixSize = ctx.lastDimSize * ctx.secondLastDimSize;
        final int rowStride = ctx.lastDimSize;

        for (int batch = 0; batch < ctx.batchSize; batch++) {
            int batchOffset = batch * matrixSize;
            int resultBatchOffset = batch * ctx.lastDimSize;

            for (int j = 0; j < ctx.lastDimSize; j++) {
                float min = Float.MAX_VALUE;

                for (int i = 0; i < ctx.secondLastDimSize; i++) {
                    int index = batchOffset + i * rowStride + j;
                    float value = srcBuffer[index];
                    if (min > value) {
                        min = value;
                    }
                }

                dstBuffer[resultBatchOffset + j] = min;
            }
        }
        return result;
    }

    /**
     * 归一化轴索引（支持负数索引）
     *
     * @param axis   原始轴索引
     * @param dimNum 维度数量
     * @return 归一化后的轴索引
     */
    private static int normalizeAxis(int axis, int dimNum) {
        ArrayValidator.validateAxis(axis, dimNum);
        return axis < 0 ? axis + dimNum : axis;
    }

    /**
     * 创建结果数组的形状，将指定轴的大小设置为1
     *
     * @param shape     原始形状
     * @param targetAxis 目标轴索引
     * @param newSize   目标轴的新大小
     * @return 新的维度数组
     */
    private static int[] createResultShape(ShapeCpu shape, int targetAxis, int newSize) {
        int dimNum = shape.getDimNum();
        int[] newDims = new int[dimNum];
        for (int i = 0; i < dimNum; i++) {
            newDims[i] = (i == targetAxis) ? newSize : shape.getDimension(i);
        }
        return newDims;
    }

    /**
     * 轴操作上下文类，用于缓存频繁访问的维度信息以提高性能
     */
    private static class AxisContext {
        final int dimNum;
        final int lastDimSize;
        final int secondLastDimSize;
        final int batchSize;

        AxisContext(NdArrayCpu array) {
            this.dimNum = array.shape.getDimNum();
            this.lastDimSize = array.shape.getDimension(dimNum - 1);
            this.secondLastDimSize = array.shape.getDimension(dimNum - 2);
            this.batchSize = array.shape.size() / (lastDimSize * secondLastDimSize);
        }
    }
}

