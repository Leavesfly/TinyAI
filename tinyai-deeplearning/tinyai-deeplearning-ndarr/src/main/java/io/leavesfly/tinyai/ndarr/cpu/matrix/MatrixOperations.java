package io.leavesfly.tinyai.ndarr.cpu.matrix;

import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;

/**
 * 矩阵运算操作类。
 * 提供矩阵乘法、切片等操作，采用缓存友好的循环顺序和预计算优化。
 */
public class MatrixOperations {

    /**
     * 矩阵内积运算（矩阵乘法）。
     * 要求左矩阵列数等于右矩阵行数，支持多维批量乘法和广播。
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 矩阵乘法结果
     * @throws IllegalArgumentException 当数组不是矩阵或维度不匹配时抛出
     */
    public static NdArrayCpu dot(NdArrayCpu left, NdArrayCpu right) {
        // 提前验证维度要求
        if (left.shape.getDimNum() < 2 || right.shape.getDimNum() < 2) {
            throw new IllegalArgumentException("矩阵乘法操作需要至少二维数组");
        }

        // 缓存维度信息，避免重复调用
        int leftDimNum = left.shape.getDimNum();
        int rightDimNum = right.shape.getDimNum();
        int leftCols = left.shape.getDimension(leftDimNum - 1);  // 左矩阵列数（K）
        int leftRows = left.shape.getDimension(leftDimNum - 2);   // 左矩阵行数（M）
        int rightCols = right.shape.getDimension(rightDimNum - 1); // 右矩阵列数（N）
        int rightRows = right.shape.getDimension(rightDimNum - 2); // 右矩阵行数（K）

        // 验证矩阵乘法维度匹配：左矩阵列数 = 右矩阵行数
        if (leftCols != rightRows) {
            throw new IllegalArgumentException(
                    String.format("矩阵乘法维度不匹配：%s × %s，第一个矩阵的列数(%d)必须等于第二个矩阵的行数(%d)",
                            left.shape, right.shape, leftCols, rightRows));
        }

        // 特化处理：2D矩阵乘法（最常见情况，性能最优）
        if (leftDimNum == 2 && rightDimNum == 2) {
            return dot2D(left, right, leftRows, leftCols, rightCols);
        }

        // 通用多维矩阵乘法
        return dotMultiDim(left, right, leftDimNum, rightDimNum, leftRows, leftCols, rightRows, rightCols);
    }

    /**
     * 2D 矩阵乘法特化实现，使用 i-k-j 循环顺序提高缓存命中率。
     *
     * @param left      左矩阵
     * @param right     右矩阵
     * @param leftRows  左矩阵行数（M）
     * @param leftCols  左矩阵列数（K）
     * @param rightCols 右矩阵列数（N）
     * @return 矩阵乘法结果 (M × N)
     */
    private static NdArrayCpu dot2D(NdArrayCpu left, NdArrayCpu right, int leftRows, int leftCols, int rightCols) {
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(leftRows, rightCols));
        float[] leftBuf = left.buffer;
        float[] rightBuf = right.buffer;
        float[] resultBuf = result.buffer;

        // 优化循环顺序：i-k-j 提高缓存友好性
        // 外层循环：遍历左矩阵的行
        for (int i = 0; i < leftRows; i++) {
            int leftRowOffset = i * leftCols;  // 预计算左矩阵行偏移
            int resultRowOffset = i * rightCols; // 预计算结果矩阵行偏移

            // 中层循环：遍历公共维度K
            for (int k = 0; k < leftCols; k++) {
                float leftVal = leftBuf[leftRowOffset + k]; // 左矩阵元素 A[i][k]
                int rightRowOffset = k * rightCols; // 预计算右矩阵行偏移

                // 内层循环：遍历右矩阵的列（连续访问，缓存友好）
                for (int j = 0; j < rightCols; j++) {
                    // C[i][j] += A[i][k] * B[k][j]
                    resultBuf[resultRowOffset + j] += leftVal * rightBuf[rightRowOffset + j];
                }
            }
        }

        return result;
    }

    /**
     * 多维数组矩阵乘法实现（支持批量矩阵乘法和广播）
     *
     * @param left      左操作数数组
     * @param right     右操作数数组
     * @param leftDimNum 左数组维度数
     * @param rightDimNum 右数组维度数
     * @param leftRows  左矩阵行数（M）
     * @param leftCols  左矩阵列数（K）
     * @param rightRows 右矩阵行数（K）
     * @param rightCols 右矩阵列数（N）
     * @return 矩阵乘法结果
     */
    private static NdArrayCpu dotMultiDim(NdArrayCpu left, NdArrayCpu right,
                                          int leftDimNum, int rightDimNum,
                                          int leftRows, int leftCols, int rightRows, int rightCols) {
        // 计算结果形状
        int maxDimNum = Math.max(leftDimNum, rightDimNum);
        int[] newDims = new int[maxDimNum];

        // 计算前置维度（广播维度）
        for (int i = 0; i < maxDimNum - 2; i++) {
            int leftDim = (i < leftDimNum - 2) ? left.shape.getDimension(i) : 1;
            int rightDim = (i < rightDimNum - 2) ? right.shape.getDimension(i) : 1;
            newDims[i] = Math.max(leftDim, rightDim);
        }

        // 设置最后两个维度：结果矩阵形状为 (..., M, N)
        newDims[maxDimNum - 2] = leftRows;
        newDims[maxDimNum - 1] = rightCols;

        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        // 预计算批量大小和步长，避免循环内重复计算
        int matrixSize = leftRows * rightCols;  // 单个矩阵的元素数
        int batchSize = result.shape.size() / matrixSize;

        int leftMatrixSize = leftRows * leftCols;   // 左矩阵单个批次的元素数
        int rightMatrixSize = rightRows * rightCols; // 右矩阵单个批次的元素数

        int leftBatchSize = left.shape.size() / leftMatrixSize;
        int rightBatchSize = right.shape.size() / rightMatrixSize;

        // 预计算步长，用于索引计算
        int leftBatchStride = leftMatrixSize;
        int rightBatchStride = rightMatrixSize;
        int resultBatchStride = matrixSize;

        float[] leftBuf = left.buffer;
        float[] rightBuf = right.buffer;
        float[] resultBuf = result.buffer;

        // 批量矩阵乘法
        for (int batch = 0; batch < batchSize; batch++) {
            // 计算当前批次的索引（支持广播）
            int leftBatchIdx = (leftBatchSize == 1) ? 0 : batch % leftBatchSize;
            int rightBatchIdx = (rightBatchSize == 1) ? 0 : batch % rightBatchSize;

            // 预计算批次偏移量
            int leftBatchOffset = leftBatchIdx * leftBatchStride;
            int rightBatchOffset = rightBatchIdx * rightBatchStride;
            int resultBatchOffset = batch * resultBatchStride;

            // 使用缓存友好的循环顺序进行矩阵乘法
            for (int i = 0; i < leftRows; i++) {
                int leftRowOffset = leftBatchOffset + i * leftCols;
                int resultRowOffset = resultBatchOffset + i * rightCols;

                for (int k = 0; k < leftCols; k++) {
                    float leftVal = leftBuf[leftRowOffset + k];
                    int rightRowOffset = rightBatchOffset + k * rightCols;

                    for (int j = 0; j < rightCols; j++) {
                        resultBuf[resultRowOffset + j] += leftVal * rightBuf[rightRowOffset + j];
                    }
                }
            }
        }

        return result;
    }

    /**
     * 获取数组子集（切片操作）。
     * 支持对矩阵最后两维切片，预计算索引并批量复制以优化连续访问。
     *
     * @param array      数组
     * @param rowSlices  行索引数组，null 表示选择所有行
     * @param colSlices  列索引数组，null 表示选择所有列
     * @return 切片结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public static NdArrayCpu getItem(NdArrayCpu array, int[] rowSlices, int[] colSlices) {
        // 提前验证维度要求
        if (array.shape.getDimNum() < 2) {
            throw new IllegalArgumentException("切片操作需要至少二维数组");
        }

        // 缓存维度信息
        int dimNum = array.shape.getDimNum();
        int cols = array.shape.getDimension(dimNum - 1);  // 列数
        int rows = array.shape.getDimension(dimNum - 2);  // 行数

        // 处理点索引模式：rowSlices和colSlices都不为null，且长度相等
        if (rowSlices != null && colSlices != null) {
            validateSliceIndices(rowSlices, colSlices, rows, cols);
            return getItemPointIndices(array, rowSlices, colSlices, cols);
        }

        // 处理矩形切片模式：至少有一个为null，或长度不等
        return getItemRectangularSlice(array, rowSlices, colSlices, rows, cols);
    }

    /**
     * 点索引模式：rowSlices[i] 与 colSlices[i] 组成坐标点，获取对应值。
     *
     * @param array      源数组
     * @param rowSlices  行索引数组
     * @param colSlices  列索引数组
     * @param cols       源数组列数
     * @return 结果数组（形状 1×N，N 为索引数组长度）
     */
    private static NdArrayCpu getItemPointIndices(NdArrayCpu array, int[] rowSlices, int[] colSlices, int cols) {
        int count = colSlices.length;
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(1, count));
        float[] srcBuf = array.buffer;
        float[] dstBuf = result.buffer;

        // 预计算索引，避免循环内重复计算
        for (int i = 0; i < count; i++) {
            int index = rowSlices[i] * cols + colSlices[i];
            dstBuf[i] = srcBuf[index];
        }

        return result;
    }

    /**
     * 矩形切片模式：获取行、列切片组合而成的子矩阵。
     *
     * @param array      源数组
     * @param rowSlices  行索引数组，null 表示选择所有行
     * @param colSlices  列索引数组，null 表示选择所有列
     * @param rows       源数组行数
     * @param cols       源数组列数
     * @return 切片结果数组
     */
    private static NdArrayCpu getItemRectangularSlice(NdArrayCpu array, int[] rowSlices, int[] colSlices,
                                                      int rows, int cols) {
        // 处理null切片：生成完整的索引序列
        if (colSlices == null) {
            colSlices = NdArrayUtil.getSeq(cols);
        }
        if (rowSlices == null) {
            rowSlices = NdArrayUtil.getSeq(rows);
        }

        // 验证索引范围
        validateSliceRange(rowSlices, rows, "行");
        validateSliceRange(colSlices, cols, "列");

        int resultRows = rowSlices.length;
        int resultCols = colSlices.length;
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(resultRows, resultCols));

        float[] srcBuf = array.buffer;
        float[] dstBuf = result.buffer;

        // 优化：对于连续的行切片，可以使用System.arraycopy批量复制
        if (isConsecutive(rowSlices) && isConsecutive(colSlices) && colSlices.length == cols) {
            // 连续行且完整列：可以使用批量复制
            int startRow = rowSlices[0];
            int rowCount = resultRows;
            int srcOffset = startRow * cols;
            int dstOffset = 0;
            System.arraycopy(srcBuf, srcOffset, dstBuf, dstOffset, rowCount * cols);
        } else {
            // 非连续切片：逐个元素复制
            for (int i = 0; i < resultRows; i++) {
                int srcRowOffset = rowSlices[i] * cols;
                int dstRowOffset = i * resultCols;

                for (int j = 0; j < resultCols; j++) {
                    dstBuf[dstRowOffset + j] = srcBuf[srcRowOffset + colSlices[j]];
                }
            }
        }

        return result;
    }

    /**
     * 验证切片索引的有效性（点索引模式）
     */
    private static void validateSliceIndices(int[] rowSlices, int[] colSlices, int maxRows, int maxCols) {
        if (rowSlices.length != colSlices.length) {
            throw new IllegalArgumentException(
                    String.format("点索引模式：行索引数组长度(%d)必须等于列索引数组长度(%d)",
                            rowSlices.length, colSlices.length));
        }

        // 验证索引范围
        validateSliceRange(rowSlices, maxRows, "行");
        validateSliceRange(colSlices, maxCols, "列");
    }

    /**
     * 验证切片索引范围
     */
    private static void validateSliceRange(int[] indices, int maxValue, String dimension) {
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= maxValue) {
                throw new IllegalArgumentException(
                        String.format("%s索引超出范围：索引[%d]=%d，有效范围[0, %d)",
                                dimension, i, indices[i], maxValue - 1));
            }
        }
    }

    /**
     * 检查索引数组是否连续（用于优化批量复制）
     */
    private static boolean isConsecutive(int[] indices) {
        if (indices.length <= 1) {
            return true;
        }
        for (int i = 1; i < indices.length; i++) {
            if (indices[i] != indices[i - 1] + 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * 设置数组子集（切片赋值）。
     * 支持点索引和矩形切片，预计算索引并批量复制以优化写入。
     *
     * @param array      数组
     * @param rowSlices  行索引数组，null 表示选择所有行
     * @param colSlices  列索引数组，null 表示选择所有列
     * @param data       要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public static NdArrayCpu setItem(NdArrayCpu array, int[] rowSlices, int[] colSlices, float[] data) {
        // 提前验证维度要求
        if (array.shape.getDimNum() < 2) {
            throw new IllegalArgumentException("切片赋值操作需要至少二维数组");
        }

        // 缓存维度信息
        int dimNum = array.shape.getDimNum();
        int cols = array.shape.getDimension(dimNum - 1);  // 列数
        int rows = array.shape.getDimension(dimNum - 2);  // 行数

        // 验证数据长度
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("数据数组不能为空");
        }

        // 处理点索引模式：rowSlices和colSlices都不为null
        if (rowSlices != null && colSlices != null) {
            validateSetItemIndices(rowSlices, colSlices, data.length, rows, cols);
            setItemPointIndices(array, rowSlices, colSlices, data, cols);
            return array;
        }

        // 处理矩形切片模式
        if (rowSlices == null && colSlices == null) {
            // 完整矩阵赋值
            if (data.length != array.shape.size()) {
                throw new IllegalArgumentException(
                        String.format("完整矩阵赋值：数据长度(%d)必须等于数组大小(%d)",
                                data.length, array.shape.size()));
            }
            System.arraycopy(data, 0, array.buffer, 0, data.length);
            return array;
        }

        // 部分矩形切片赋值（需要更复杂的实现）
        throw new IllegalArgumentException(
                "当前仅支持点索引模式（rowSlices和colSlices都不为null）和完整矩阵赋值（两者都为null）");
    }

    /**
     * 点索引模式：设置指定坐标点的值
     *
     * @param array      目标数组
     * @param rowSlices  行索引数组
     * @param colSlices  列索引数组
     * @param data       要设置的数据
     * @param cols       目标数组列数
     */
    private static void setItemPointIndices(NdArrayCpu array, int[] rowSlices, int[] colSlices,
                                           float[] data, int cols) {
        float[] buf = array.buffer;
        int count = data.length;

        // 预计算索引，避免循环内重复计算
        for (int i = 0; i < count; i++) {
            int index = rowSlices[i] * cols + colSlices[i];
            buf[index] = data[i];
        }
    }

    /**
     * 验证setItem操作的索引有效性
     */
    private static void validateSetItemIndices(int[] rowSlices, int[] colSlices, int dataLength,
                                              int maxRows, int maxCols) {
        if (rowSlices.length != colSlices.length) {
            throw new IllegalArgumentException(
                    String.format("点索引模式：行索引数组长度(%d)必须等于列索引数组长度(%d)",
                            rowSlices.length, colSlices.length));
        }

        if (rowSlices.length != dataLength) {
            throw new IllegalArgumentException(
                    String.format("索引数组长度(%d)必须等于数据数组长度(%d)",
                            rowSlices.length, dataLength));
        }

        // 验证索引范围
        validateSliceRange(rowSlices, maxRows, "行");
        validateSliceRange(colSlices, maxCols, "列");
    }

    // =============================================================================
    // 增强方法：高性能切片赋值操作（新增）
    // =============================================================================

    /**
     * 高性能连续区域赋值，使用 System.arraycopy 批量复制，性能优于逐点赋值。
     *
     * @param array    目标数组
     * @param startRow 起始行索引（包含）
     * @param endRow   结束行索引（不包含）
     * @param startCol 起始列索引（包含）
     * @param endCol   结束列索引（不包含）
     * @param data     要设置的数据
     * @return 当前数组实例
     */
    public static NdArrayCpu setBlock(NdArrayCpu array,
                                      int startRow, int endRow, int startCol, int endCol, float[] data) {

        if (array.shape.getDimNum() < 2) {
            throw new IllegalArgumentException("setBlock需要至少二维数组");
        }

        int cols = array.shape.getColumn();
        int blockRows = endRow - startRow;
        int blockCols = endCol - startCol;

        // 验证数据长度
        if (data.length != blockRows * blockCols) {
            throw new IllegalArgumentException(
                    String.format("数据长度(%d)必须等于区域大小(%d×%d=%d)",
                            data.length, blockRows, blockCols, blockRows * blockCols));
        }

        // 验证边界
        if (startRow < 0 || endRow > array.shape.getRow() ||
                startCol < 0 || endCol > cols) {
            throw new IllegalArgumentException(
                    String.format("区域索引超出边界: [%d:%d, %d:%d], 数组大小: [%d, %d]",
                            startRow, endRow, startCol, endCol, array.shape.getRow(), cols));
        }

        // 高效批量复制
        for (int i = 0; i < blockRows; i++) {
            int srcOffset = i * blockCols;
            int dstOffset = (startRow + i) * cols + startCol;
            System.arraycopy(data, srcOffset, array.buffer, dstOffset, blockCols);
        }

        return array;
    }

    /**
     * 行切片赋值，高效设置指定行的数据。
     *
     * @param array      目标数组
     * @param rowIndices 行索引数组
     * @param data       要设置的数据
     * @return 当前数组实例
     */
    public static NdArrayCpu setRows(NdArrayCpu array, int[] rowIndices, float[] data) {
        if (array.shape.getDimNum() < 2) {
            throw new IllegalArgumentException("setRows需要至少二维数组");
        }

        int cols = array.shape.getColumn();

        if (data.length != rowIndices.length * cols) {
            throw new IllegalArgumentException(
                    String.format("数据长度(%d)必须等于%d行×%d列=%d",
                            data.length, rowIndices.length, cols, rowIndices.length * cols));
        }

        // 逐行复制
        for (int i = 0; i < rowIndices.length; i++) {
            int row = rowIndices[i];
            if (row < 0 || row >= array.shape.getRow()) {
                throw new IllegalArgumentException("行索引超出范围: " + row);
            }
            System.arraycopy(data, i * cols, array.buffer, row * cols, cols);
        }

        return array;
    }

    /**
     * 列切片赋值，高效设置指定列的数据。
     *
     * @param array      目标数组
     * @param colIndices 列索引数组
     * @param data       要设置的数据
     * @return 当前数组实例
     */
    public static NdArrayCpu setCols(NdArrayCpu array, int[] colIndices, float[] data) {
        if (array.shape.getDimNum() < 2) {
            throw new IllegalArgumentException("setCols需要至少二维数组");
        }

        int rows = array.shape.getRow();
        int cols = array.shape.getColumn();

        if (data.length != rows * colIndices.length) {
            throw new IllegalArgumentException(
                    String.format("数据长度(%d)必须等于%d行×%d列=%d",
                            data.length, rows, colIndices.length, rows * colIndices.length));
        }

        // 逐列赋值（注意：列不连续，无法使用arraycopy）
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < colIndices.length; c++) {
                int col = colIndices[c];
                if (col < 0 || col >= cols) {
                    throw new IllegalArgumentException("列索引超出范围: " + col);
                }
                array.buffer[r * cols + col] = data[r * colIndices.length + c];
            }
        }

        return array;
    }
}

