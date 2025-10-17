package io.leavesfly.tinyai.agent.vla.utils;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 图像处理工具
 * 用于图像预处理和数据增强
 * 
 * @author TinyAI
 */
public class ImageProcessor {
    
    /**
     * 归一化图像到 [0, 1] 范围
     * 
     * @param image 原始图像，范围 [0, 255]
     * @return 归一化后的图像
     */
    public static NdArray normalize(NdArray image) {
        double[] data = image.toDoubleArray();
        double[] normalized = new double[data.length];
        
        for (int i = 0; i < data.length; i++) {
            normalized[i] = data[i] / 255.0;
        }
        
        return new NdArray(normalized).reshape(image.getShape());
    }
    
    /**
     * 标准化图像（减均值除标准差）
     * 
     * @param image 输入图像
     * @param mean RGB通道均值
     * @param std RGB通道标准差
     * @return 标准化后的图像
     */
    public static NdArray standardize(NdArray image, double[] mean, double[] std) {
        int[] shape = image.getShape();
        int height = shape[0];
        int width = shape[1];
        int channels = shape[2];
        
        double[][][] result = new double[height][width][channels];
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int idx = h * width * channels + w * channels + c;
                    result[h][w][c] = (image.get(idx) - mean[c]) / std[c];
                }
            }
        }
        
        return new NdArray(result);
    }
    
    /**
     * 调整图像大小（简单的最近邻插值）
     * 
     * @param image 输入图像
     * @param targetHeight 目标高度
     * @param targetWidth 目标宽度
     * @return 调整大小后的图像
     */
    public static NdArray resize(NdArray image, int targetHeight, int targetWidth) {
        int[] shape = image.getShape();
        int srcHeight = shape[0];
        int srcWidth = shape[1];
        int channels = shape[2];
        
        double[][][] result = new double[targetHeight][targetWidth][channels];
        
        double scaleH = (double) srcHeight / targetHeight;
        double scaleW = (double) srcWidth / targetWidth;
        
        for (int h = 0; h < targetHeight; h++) {
            for (int w = 0; w < targetWidth; w++) {
                int srcH = (int) (h * scaleH);
                int srcW = (int) (w * scaleW);
                
                for (int c = 0; c < channels; c++) {
                    int idx = srcH * srcWidth * channels + srcW * channels + c;
                    result[h][w][c] = image.get(idx);
                }
            }
        }
        
        return new NdArray(result);
    }
    
    /**
     * 随机水平翻转（数据增强）
     * 
     * @param image 输入图像
     * @param probability 翻转概率
     * @return 可能翻转后的图像
     */
    public static NdArray randomHorizontalFlip(NdArray image, double probability) {
        if (Math.random() > probability) {
            return image;
        }
        
        int[] shape = image.getShape();
        int height = shape[0];
        int width = shape[1];
        int channels = shape[2];
        
        double[][][] result = new double[height][width][channels];
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int srcIdx = h * width * channels + w * channels + c;
                    result[h][width - 1 - w][c] = image.get(srcIdx);
                }
            }
        }
        
        return new NdArray(result);
    }
    
    /**
     * 添加随机噪声（数据增强）
     * 
     * @param image 输入图像
     * @param noiseLevel 噪声级别
     * @return 添加噪声后的图像
     */
    public static NdArray addNoise(NdArray image, double noiseLevel) {
        double[] data = image.toDoubleArray();
        double[] noisy = new double[data.length];
        
        java.util.Random rand = new java.util.Random();
        
        for (int i = 0; i < data.length; i++) {
            double noise = (rand.nextDouble() - 0.5) * 2 * noiseLevel;
            noisy[i] = Math.max(0.0, Math.min(1.0, data[i] + noise));
        }
        
        return new NdArray(noisy).reshape(image.getShape());
    }
}
