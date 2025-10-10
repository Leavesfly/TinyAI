package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 图像预处理器实现类
 */
public class ImagePreprocessorImpl implements ImagePreprocessor {
    
    @Override
    public NdArray normalize(NdArray image, float[] mean, float[] std) {
        // 检查输入参数
        if (mean.length != std.length) {
            throw new IllegalArgumentException("Mean and std arrays must have the same length");
        }
        
        // 根据图像维度进行处理
        int dimNum = image.getShape().getDimNum();
        NdArray result = image;
        
        if (dimNum == 3) {
            // 单张图像 [height, width, channels]
            result = normalize3D(image, mean, std);
        } else if (dimNum == 4) {
            // 批量图像 [batch, height, width, channels]
            result = normalize4D(image, mean, std);
        } else {
            throw new IllegalArgumentException("Image must be 3D or 4D array");
        }
        
        return result;
    }
    
    /**
     * 对3D图像进行标准化处理
     */
    private NdArray normalize3D(NdArray image, float[] mean, float[] std) {
        Shape shape = image.getShape();
        int height = shape.getDimension(0);
        int width = shape.getDimension(1);
        int channels = shape.getDimension(2);
        
        if (channels != mean.length) {
            throw new IllegalArgumentException("Number of channels must match mean array length");
        }
        
        // 创建结果数组
        NdArray result = NdArray.of(shape);
        
        // 对每个通道进行标准化
        for (int c = 0; c < channels; c++) {
            // 提取通道数据
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float pixelValue = image.get(h, w, c);
                    float normalizedValue = (pixelValue - mean[c]) / std[c];
                    result.set(normalizedValue, h, w, c);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 对4D批量图像进行标准化处理
     */
    private NdArray normalize4D(NdArray image, float[] mean, float[] std) {
        Shape shape = image.getShape();
        int batch = shape.getDimension(0);
        int height = shape.getDimension(1);
        int width = shape.getDimension(2);
        int channels = shape.getDimension(3);
        
        if (channels != mean.length) {
            throw new IllegalArgumentException("Number of channels must match mean array length");
        }
        
        // 创建结果数组
        NdArray result = NdArray.of(shape);
        
        // 对每个样本的每个通道进行标准化
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                // 提取通道数据
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        float pixelValue = image.get(b, h, w, c);
                        float normalizedValue = (pixelValue - mean[c]) / std[c];
                        result.set(normalizedValue, b, h, w, c);
                    }
                }
            }
        }
        
        return result;
    }
    
    @Override
    public NdArray normalizeToRange(NdArray image, float min, float max) {
        // 找到图像的最小值和最大值
        float imageMin = findMin(image);
        float imageMax = findMax(image);
        
        // 避免除零错误
        if (imageMax == imageMin) {
            return NdArray.like(image.getShape(), min);
        }
        
        // 计算缩放因子
        float scale = (max - min) / (imageMax - imageMin);
        
        // 应用线性变换: new_value = (old_value - old_min) * scale + new_min
        NdArray normalized = image.sub(NdArray.like(image.getShape(), imageMin))
                                 .mulNum(scale)
                                 .add(NdArray.like(image.getShape(), min));
        
        return normalized;
    }
    
    /**
     * 查找图像中的最小值
     */
    private float findMin(NdArray image) {
        return image.flatten().min(0).getNumber().floatValue();
    }
    
    /**
     * 查找图像中的最大值
     */
    private float findMax(NdArray image) {
        return image.flatten().max(0).getNumber().floatValue();
    }
    
    @Override
    public NdArray resize(NdArray image, int targetHeight, int targetWidth) {
        // 简单实现：使用最近邻插值进行缩放
        // 实际应用中可以使用更复杂的插值算法
        
        Shape shape = image.getShape();
        int dimNum = shape.getDimNum();
        
        if (dimNum == 3) {
            // 单张图像 [height, width, channels]
            return resize3D(image, targetHeight, targetWidth);
        } else if (dimNum == 4) {
            // 批量图像 [batch, height, width, channels]
            return resize4D(image, targetHeight, targetWidth);
        } else {
            throw new IllegalArgumentException("Image must be 3D or 4D array");
        }
    }
    
    /**
     * 对3D图像进行缩放
     */
    private NdArray resize3D(NdArray image, int targetHeight, int targetWidth) {
        Shape originalShape = image.getShape();
        int originalHeight = originalShape.getDimension(0);
        int originalWidth = originalShape.getDimension(1);
        int channels = originalShape.getDimension(2);
        
        Shape targetShape = Shape.of(targetHeight, targetWidth, channels);
        NdArray result = NdArray.of(targetShape);
        
        // 计算缩放因子
        float heightScale = (float) originalHeight / targetHeight;
        float widthScale = (float) originalWidth / targetWidth;
        
        // 使用最近邻插值
        for (int h = 0; h < targetHeight; h++) {
            for (int w = 0; w < targetWidth; w++) {
                // 计算原始坐标
                int origH = Math.min((int) (h * heightScale), originalHeight - 1);
                int origW = Math.min((int) (w * widthScale), originalWidth - 1);
                
                // 复制所有通道的数据
                for (int c = 0; c < channels; c++) {
                    float value = image.get(origH, origW, c);
                    result.set(value, h, w, c);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 对4D批量图像进行缩放
     */
    private NdArray resize4D(NdArray image, int targetHeight, int targetWidth) {
        Shape originalShape = image.getShape();
        int batch = originalShape.getDimension(0);
        int originalHeight = originalShape.getDimension(1);
        int originalWidth = originalShape.getDimension(2);
        int channels = originalShape.getDimension(3);
        
        Shape targetShape = Shape.of(batch, targetHeight, targetWidth, channels);
        NdArray result = NdArray.of(targetShape);
        
        // 计算缩放因子
        float heightScale = (float) originalHeight / targetHeight;
        float widthScale = (float) originalWidth / targetWidth;
        
        // 对每个批次进行处理
        for (int b = 0; b < batch; b++) {
            // 使用最近邻插值
            for (int h = 0; h < targetHeight; h++) {
                for (int w = 0; w < targetWidth; w++) {
                    // 计算原始坐标
                    int origH = Math.min((int) (h * heightScale), originalHeight - 1);
                    int origW = Math.min((int) (w * widthScale), originalWidth - 1);
                    
                    // 复制所有通道的数据
                    for (int c = 0; c < channels; c++) {
                        float value = image.get(b, origH, origW, c);
                        result.set(value, b, h, w, c);
                    }
                }
            }
        }
        
        return result;
    }
    
    @Override
    public NdArray reshape(NdArray image, Shape shape) {
        return image.reshape(shape);
    }
}