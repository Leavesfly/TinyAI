package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Random;

/**
 * 数据增强器实现类
 */
public class DataAugmenterImpl implements DataAugmenter {
    
    private final Random random;
    private final float horizontalFlipProb;
    private final float verticalFlipProb;
    private final float maxRotationAngle;
    private final float maxBrightnessDelta;
    private final float noiseLevel;
    
    public DataAugmenterImpl() {
        this.random = new Random();
        this.horizontalFlipProb = 0.5f;
        this.verticalFlipProb = 0.5f;
        this.maxRotationAngle = 15.0f;
        this.maxBrightnessDelta = 0.2f;
        this.noiseLevel = 0.1f;
    }
    
    public DataAugmenterImpl(float horizontalFlipProb, float verticalFlipProb, 
                            float maxRotationAngle, float maxBrightnessDelta, float noiseLevel) {
        this.random = new Random();
        this.horizontalFlipProb = horizontalFlipProb;
        this.verticalFlipProb = verticalFlipProb;
        this.maxRotationAngle = maxRotationAngle;
        this.maxBrightnessDelta = maxBrightnessDelta;
        this.noiseLevel = noiseLevel;
    }
    
    @Override
    public NdArray randomHorizontalFlip(NdArray image, float probability) {
        if (random.nextFloat() < probability) {
            return flipHorizontally(image);
        }
        return image;
    }
    
    /**
     * 水平翻转图像
     */
    private NdArray flipHorizontally(NdArray image) {
        Shape shape = image.getShape();
        int dimNum = shape.getDimNum();
        
        if (dimNum == 3) {
            return flipHorizontally3D(image);
        } else if (dimNum == 4) {
            return flipHorizontally4D(image);
        } else {
            throw new IllegalArgumentException("Image must be 3D or 4D array");
        }
    }
    
    private NdArray flipHorizontally3D(NdArray image) {
        Shape shape = image.getShape();
        int height = shape.getDimension(0);
        int width = shape.getDimension(1);
        int channels = shape.getDimension(2);
        
        NdArray result = NdArray.of(shape);
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    float value = image.get(h, width - 1 - w, c);
                    result.set(value, h, w, c);
                }
            }
        }
        
        return result;
    }
    
    private NdArray flipHorizontally4D(NdArray image) {
        Shape shape = image.getShape();
        int batch = shape.getDimension(0);
        int height = shape.getDimension(1);
        int width = shape.getDimension(2);
        int channels = shape.getDimension(3);
        
        NdArray result = NdArray.of(shape);
        
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int c = 0; c < channels; c++) {
                        float value = image.get(b, h, width - 1 - w, c);
                        result.set(value, b, h, w, c);
                    }
                }
            }
        }
        
        return result;
    }
    
    @Override
    public NdArray randomVerticalFlip(NdArray image, float probability) {
        if (random.nextFloat() < probability) {
            return flipVertically(image);
        }
        return image;
    }
    
    /**
     * 垂直翻转图像
     */
    private NdArray flipVertically(NdArray image) {
        Shape shape = image.getShape();
        int dimNum = shape.getDimNum();
        
        if (dimNum == 3) {
            return flipVertically3D(image);
        } else if (dimNum == 4) {
            return flipVertically4D(image);
        } else {
            throw new IllegalArgumentException("Image must be 3D or 4D array");
        }
    }
    
    private NdArray flipVertically3D(NdArray image) {
        Shape shape = image.getShape();
        int height = shape.getDimension(0);
        int width = shape.getDimension(1);
        int channels = shape.getDimension(2);
        
        NdArray result = NdArray.of(shape);
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    float value = image.get(height - 1 - h, w, c);
                    result.set(value, h, w, c);
                }
            }
        }
        
        return result;
    }
    
    private NdArray flipVertically4D(NdArray image) {
        Shape shape = image.getShape();
        int batch = shape.getDimension(0);
        int height = shape.getDimension(1);
        int width = shape.getDimension(2);
        int channels = shape.getDimension(3);
        
        NdArray result = NdArray.of(shape);
        
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int c = 0; c < channels; c++) {
                        float value = image.get(b, height - 1 - h, w, c);
                        result.set(value, b, h, w, c);
                    }
                }
            }
        }
        
        return result;
    }
    
    @Override
    public NdArray randomRotate(NdArray image, float maxAngle) {
        float angle = (random.nextFloat() * 2 - 1) * maxAngle; // -maxAngle 到 +maxAngle
        return rotate(image, angle);
    }
    
    /**
     * 旋转图像
     */
    private NdArray rotate(NdArray image, float angle) {
        // 简化实现：这里只返回原图像
        // 实际应用中需要实现双线性插值旋转算法
        return image;
    }
    
    @Override
    public NdArray randomCrop(NdArray image, int cropHeight, int cropWidth) {
        Shape shape = image.getShape();
        int dimNum = shape.getDimNum();
        
        if (dimNum == 3) {
            return randomCrop3D(image, cropHeight, cropWidth);
        } else if (dimNum == 4) {
            return randomCrop4D(image, cropHeight, cropWidth);
        } else {
            throw new IllegalArgumentException("Image must be 3D or 4D array");
        }
    }
    
    private NdArray randomCrop3D(NdArray image, int cropHeight, int cropWidth) {
        Shape shape = image.getShape();
        int height = shape.getDimension(0);
        int width = shape.getDimension(1);
        
        if (cropHeight > height || cropWidth > width) {
            throw new IllegalArgumentException("Crop size must be smaller than image size");
        }
        
        // 随机选择裁剪起始位置
        int startY = random.nextInt(height - cropHeight + 1);
        int startX = random.nextInt(width - cropWidth + 1);
        
        // 提取裁剪区域
        return image.subNdArray(startY, startY + cropHeight, startX, startX + cropWidth);
    }
    
    private NdArray randomCrop4D(NdArray image, int cropHeight, int cropWidth) {
        Shape shape = image.getShape();
        int batch = shape.getDimension(0);
        int height = shape.getDimension(1);
        int width = shape.getDimension(2);
        int channels = shape.getDimension(3);
        
        if (cropHeight > height || cropWidth > width) {
            throw new IllegalArgumentException("Crop size must be smaller than image size");
        }
        
        // 创建结果数组
        Shape resultShape = Shape.of(batch, cropHeight, cropWidth, channels);
        NdArray result = NdArray.of(resultShape);
        
        // 随机选择裁剪起始位置
        int startY = random.nextInt(height - cropHeight + 1);
        int startX = random.nextInt(width - cropWidth + 1);
        
        // 对每个批次进行裁剪
        for (int b = 0; b < batch; b++) {
            NdArray cropped = image.subNdArray(b, b + 1, startY, startY + cropHeight, startX, startX + cropWidth);
            // 复制到结果数组
            for (int h = 0; h < cropHeight; h++) {
                for (int w = 0; w < cropWidth; w++) {
                    for (int c = 0; c < channels; c++) {
                        float value = cropped.get(0, h, w, c);
                        result.set(value, b, h, w, c);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 4D数组的子区域提取辅助方法
     */
    private NdArray subNdArray(NdArray image, int batchStart, int batchEnd, 
                              int heightStart, int heightEnd, int widthStart, int widthEnd) {
        Shape shape = image.getShape();
        int channels = shape.getDimension(3);
        
        Shape resultShape = Shape.of(batchEnd - batchStart, heightEnd - heightStart, 
                                    widthEnd - widthStart, channels);
        NdArray result = NdArray.of(resultShape);
        
        for (int b = batchStart; b < batchEnd; b++) {
            for (int h = heightStart; h < heightEnd; h++) {
                for (int w = widthStart; w < widthEnd; w++) {
                    for (int c = 0; c < channels; c++) {
                        float value = image.get(b, h, w, c);
                        result.set(value, b - batchStart, h - heightStart, w - widthStart, c);
                    }
                }
            }
        }
        
        return result;
    }
    
    @Override
    public NdArray randomBrightness(NdArray image, float maxDelta) {
        float delta = (random.nextFloat() * 2 - 1) * maxDelta; // -maxDelta 到 +maxDelta
        return image.add(NdArray.like(image.getShape(), delta));
    }
    
    @Override
    public NdArray randomNoise(NdArray image, float noiseLevel) {
        Shape shape = image.getShape();
        NdArray noise = NdArray.likeRandomN(shape).mulNum(noiseLevel);
        return image.add(noise);
    }
    
    @Override
    public NdArray augment(NdArray image) {
        NdArray augmented = image;
        
        // 应用各种增强操作
        augmented = randomHorizontalFlip(augmented, horizontalFlipProb);
        augmented = randomVerticalFlip(augmented, verticalFlipProb);
        augmented = randomRotate(augmented, maxRotationAngle);
        augmented = randomBrightness(augmented, maxBrightnessDelta);
        augmented = randomNoise(augmented, noiseLevel);
        
        return augmented;
    }
}