package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 数据增强器接口
 * 提供常见的图像数据增强操作
 */
public interface DataAugmenter {
    
    /**
     * 随机水平翻转图像
     * @param image 图像数据
     * @param probability 翻转概率 (0.0-1.0)
     * @return 翻转后的图像
     */
    NdArray randomHorizontalFlip(NdArray image, float probability);
    
    /**
     * 随机垂直翻转图像
     * @param image 图像数据
     * @param probability 翻转概率 (0.0-1.0)
     * @return 翻转后的图像
     */
    NdArray randomVerticalFlip(NdArray image, float probability);
    
    /**
     * 随机旋转图像
     * @param image 图像数据
     * @param maxAngle 最大旋转角度（度）
     * @return 旋转后的图像
     */
    NdArray randomRotate(NdArray image, float maxAngle);
    
    /**
     * 随机裁剪图像
     * @param image 图像数据
     * @param cropHeight 裁剪高度
     * @param cropWidth 裁剪宽度
     * @return 裁剪后的图像
     */
    NdArray randomCrop(NdArray image, int cropHeight, int cropWidth);
    
    /**
     * 随机调整图像亮度
     * @param image 图像数据
     * @param maxDelta 最大亮度变化值
     * @return 调整后的图像
     */
    NdArray randomBrightness(NdArray image, float maxDelta);
    
    /**
     * 随机添加噪声到图像
     * @param image 图像数据
     * @param noiseLevel 噪声级别
     * @return 添加噪声后的图像
     */
    NdArray randomNoise(NdArray image, float noiseLevel);
    
    /**
     * 应用一系列随机增强操作
     * @param image 图像数据
     * @return 增强后的图像
     */
    NdArray augment(NdArray image);
}