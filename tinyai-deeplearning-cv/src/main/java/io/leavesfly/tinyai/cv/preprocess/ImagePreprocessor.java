package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 图像预处理器接口
 * 提供图像标准化、归一化等预处理功能
 */
public interface ImagePreprocessor {
    
    /**
     * 对图像进行标准化处理
     * @param image 图像数据 (形状: [height, width, channels] 或 [batch, height, width, channels])
     * @param mean 均值数组，每个通道一个均值
     * @param std 标准差数组，每个通道一个标准差
     * @return 标准化后的图像
     */
    NdArray normalize(NdArray image, float[] mean, float[] std);
    
    /**
     * 对图像进行归一化处理到指定范围
     * @param image 图像数据
     * @param min 目标范围最小值
     * @param max 目标范围最大值
     * @return 归一化后的图像
     */
    NdArray normalizeToRange(NdArray image, float min, float max);
    
    /**
     * 将图像数据缩放到指定尺寸
     * @param image 图像数据
     * @param targetHeight 目标高度
     * @param targetWidth 目标宽度
     * @return 缩放后的图像
     */
    NdArray resize(NdArray image, int targetHeight, int targetWidth);
    
    /**
     * 将图像数据转换为指定形状
     * @param image 图像数据
     * @param shape 目标形状
     * @return 转换后的图像
     */
    NdArray reshape(NdArray image, Shape shape);
}