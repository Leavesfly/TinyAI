package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * 批处理管道类
 * 将单个图像处理结果组装成批次
 */
public class BatchProcessor {
    
    private final ImagePreprocessor preprocessor;
    private final DataAugmenter augmenter;
    
    public BatchProcessor() {
        this.preprocessor = new ImagePreprocessorImpl();
        this.augmenter = new DataAugmenterImpl();
    }
    
    public BatchProcessor(ImagePreprocessor preprocessor, DataAugmenter augmenter) {
        this.preprocessor = preprocessor;
        this.augmenter = augmenter;
    }
    
    /**
     * 将单个图像列表组装成批次
     * @param images 图像列表
     * @return 批次数据
     */
    public NdArray createBatch(List<NdArray> images) {
        if (images == null || images.isEmpty()) {
            throw new IllegalArgumentException("Images list cannot be null or empty");
        }
        
        // 检查所有图像是否具有相同的形状
        Shape firstShape = images.get(0).getShape();
        for (int i = 1; i < images.size(); i++) {
            if (!images.get(i).getShape().equals(firstShape)) {
                throw new IllegalArgumentException("All images must have the same shape");
            }
        }
        
        // 创建批次形状 [batch_size, height, width, channels]
        int batchSize = images.size();
        int[] dimensions = new int[firstShape.getDimNum() + 1];
        dimensions[0] = batchSize;
        for (int i = 0; i < firstShape.getDimNum(); i++) {
            dimensions[i + 1] = firstShape.getDimension(i);
        }
        
        Shape batchShape = Shape.of(dimensions);
        NdArray batch = NdArray.of(batchShape);
        
        // 填充批次数据
        for (int i = 0; i < batchSize; i++) {
            NdArray image = images.get(i);
            copyImageToBatch(batch, image, i);
        }
        
        return batch;
    }
    
    /**
     * 将单个图像复制到批次中的指定位置
     */
    private void copyImageToBatch(NdArray batch, NdArray image, int batchIndex) {
        Shape shape = image.getShape();
        int dimNum = shape.getDimNum();
        
        if (dimNum == 3) {
            copyImageToBatch3D(batch, image, batchIndex);
        } else {
            throw new IllegalArgumentException("Unsupported image dimension: " + dimNum);
        }
    }
    
    private void copyImageToBatch3D(NdArray batch, NdArray image, int batchIndex) {
        Shape shape = image.getShape();
        int height = shape.getDimension(0);
        int width = shape.getDimension(1);
        int channels = shape.getDimension(2);
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    float value = image.get(h, w, c);
                    batch.set(value, batchIndex, h, w, c);
                }
            }
        }
    }
    
    /**
     * 对图像列表进行预处理和增强，然后组装成批次
     * @param images 原始图像列表
     * @param mean 均值数组
     * @param std 标准差数组
     * @param augment 是否应用数据增强
     * @return 处理后的批次数据
     */
    public NdArray processBatch(List<NdArray> images, float[] mean, float[] std, boolean augment) {
        List<NdArray> processedImages = new ArrayList<>();
        
        for (NdArray image : images) {
            // 预处理
            NdArray processed = preprocessor.normalize(image, mean, std);
            
            // 数据增强（可选）
            if (augment) {
                processed = augmenter.augment(processed);
            }
            
            processedImages.add(processed);
        }
        
        // 组装批次
        return createBatch(processedImages);
    }
    
    /**
     * 对单个图像进行预处理和增强
     * @param image 原始图像
     * @param mean 均值数组
     * @param std 标准差数组
     * @param augment 是否应用数据增强
     * @return 处理后的图像
     */
    public NdArray processImage(NdArray image, float[] mean, float[] std, boolean augment) {
        // 预处理
        NdArray processed = preprocessor.normalize(image, mean, std);
        
        // 数据增强（可选）
        if (augment) {
            processed = augmenter.augment(processed);
        }
        
        return processed;
    }
}