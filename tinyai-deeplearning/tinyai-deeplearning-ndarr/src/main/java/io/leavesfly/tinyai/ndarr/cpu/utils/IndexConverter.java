package io.leavesfly.tinyai.ndarr.cpu.utils;

import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;

/**
 * 索引转换工具类，提供多维索引与一维线性索引的相互转换。
 */
public class IndexConverter {

    /**
     * 将一维线性索引转换为多维索引
     *
     * @param linearIndex 一维线性索引
     * @param indices     多维索引数组（输出参数）
     * @param shape       目标形状
     */
    public static void convertToMultiIndex(int linearIndex, int[] indices, ShapeCpu shape) {
        // 使用前向遍历与预计算的stride，将线性索引拆解为各维度索引
        // 注意：multipliers[i] 代表该维度的一步跨度，若维度长度为0则乘数为0
        int remaining = linearIndex;
        for (int i = 0; i < shape.dimension.length; i++) {
            int stride = shape.multipliers[i];
            indices[i] = stride == 0 ? 0 : remaining / stride;
            remaining = stride == 0 ? remaining : remaining % stride;
        }
    }

    /**
     * 将一维线性索引转换为指定Shape的多维索引
     *
     * @param linearIndex 一维索引
     * @param indices     多维索引输出
     * @param targetShape 目标形状
     */
    public static void flatToMultiIndex(int linearIndex, int[] indices, ShapeCpu targetShape) {
        int remaining = linearIndex;
        for (int i = 0; i < targetShape.dimension.length; i++) {
            int stride = targetShape.multipliers[i];
            indices[i] = stride == 0 ? 0 : remaining / stride;
            remaining = stride == 0 ? remaining : remaining % stride;
        }
    }
}

