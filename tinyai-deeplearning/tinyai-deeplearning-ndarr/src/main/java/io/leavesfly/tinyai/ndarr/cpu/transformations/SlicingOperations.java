package io.leavesfly.tinyai.ndarr.cpu.transformations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;

/**
 * 切片操作类，提供获取子数组等切片功能。
 */
public class SlicingOperations {

    /**
     * 获取子数组（矩阵的子区域）
     *
     * @param array    源数组
     * @param startRow 起始行索引（包含）
     * @param endRow   结束行索引（不包含）
     * @param startCol 起始列索引（包含）
     * @param endCol   结束列索引（不包含）
     * @return 子数组
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    public static NdArrayCpu subNdArray(NdArrayCpu array, int startRow, int endRow, int startCol, int endCol) {
        // 对于多维数组，我们只支持最后两个维度的子区域提取
        if (array.shape.getDimNum() >= 2) {
            int lastDimSize = array.shape.getDimension(array.shape.getDimNum() - 1);
            int secondLastDimSize = array.shape.getDimension(array.shape.getDimNum() - 2);

            // 确保索引在有效范围内
            startRow = Math.max(0, startRow);
            endRow = Math.min(secondLastDimSize, endRow);
            startCol = Math.max(0, startCol);
            endCol = Math.min(lastDimSize, endCol);

            NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(endRow - startRow, endCol - startCol));
            for (int i = startRow; i < endRow; i++) {
                for (int j = startCol; j < endCol; j++) {
                    // 计算多维索引（简化处理）
                    int srcIndex = i * lastDimSize + j;
                    int dstIndex = result.shape.getColumn() * (i - startRow) + j - startCol;
                    result.buffer[dstIndex] = array.buffer[srcIndex];
                }
            }
            return result;
        }

        throw new IllegalArgumentException("操作需要至少二维数组");
    }
}

