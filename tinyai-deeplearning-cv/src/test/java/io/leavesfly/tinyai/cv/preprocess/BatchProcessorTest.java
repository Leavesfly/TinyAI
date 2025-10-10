package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class BatchProcessorTest {
    
    @Test
    public void testCreateBatch() {
        // 创建几个简单的2x2x1图像
        float[][][] imageData1 = {{{1, 2}}, {{3, 4}}};
        float[][][] imageData2 = {{{5, 6}}, {{7, 8}}};
        float[][][] imageData3 = {{{9, 10}}, {{11, 12}}};
        
        NdArray image1 = NdArray.of(imageData1);
        NdArray image2 = NdArray.of(imageData2);
        NdArray image3 = NdArray.of(imageData3);
        
        List<NdArray> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);
        
        BatchProcessor processor = new BatchProcessor();
        
        // 创建批次
        NdArray batch = processor.createBatch(images);
        
        // 验证输出形状
        Shape expectedShape = Shape.of(3, 2, 2, 1);
        assertEquals(expectedShape, batch.getShape());
        
        // 验证批次中的数据
        assertEquals(1.0f, batch.get(0, 0, 0, 0), 0.001f);
        assertEquals(5.0f, batch.get(1, 0, 0, 0), 0.001f);
        assertEquals(9.0f, batch.get(2, 0, 0, 0), 0.001f);
    }
    
    @Test
    public void testProcessBatch() {
        // 创建几个简单的2x2x3图像（模拟RGB图像）
        float[][][] imageData1 = {
            {{10, 20, 30}, {40, 50, 60}},
            {{70, 80, 90}, {100, 110, 120}}
        };
        float[][][] imageData2 = {
            {{130, 140, 150}, {160, 170, 180}},
            {{190, 200, 210}, {220, 230, 240}}
        };
        
        NdArray image1 = NdArray.of(imageData1);
        NdArray image2 = NdArray.of(imageData2);
        
        List<NdArray> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        
        BatchProcessor processor = new BatchProcessor();
        
        // RGB图像的均值和标准差（示例值）
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        
        // 处理批次
        NdArray processedBatch = processor.processBatch(images, mean, std, false);
        
        // 验证输出形状
        Shape expectedShape = Shape.of(2, 2, 2, 3);
        assertEquals(expectedShape, processedBatch.getShape());
    }
    
    @Test
    public void testProcessImage() {
        // 创建一个简单的2x2x3图像（模拟RGB图像）
        float[][][] imageData = {
            {{10, 20, 30}, {40, 50, 60}},
            {{70, 80, 90}, {100, 110, 120}}
        };
        
        NdArray image = NdArray.of(imageData);
        BatchProcessor processor = new BatchProcessor();
        
        // RGB图像的均值和标准差（示例值）
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        
        // 处理单个图像
        NdArray processed = processor.processImage(image, mean, std, false);
        
        // 验证输出形状
        assertEquals(image.getShape(), processed.getShape());
    }
    
    @Test
    public void testCreateBatchWithDifferentShapes() {
        // 创建不同形状的图像
        float[][][] imageData1 = {{{1, 2}}, {{3, 4}}}; // 2x2x1
        float[][][] imageData2 = {{{5, 6, 7}}, {{8, 9, 10}}}; // 2x3x1
        
        NdArray image1 = NdArray.of(imageData1);
        NdArray image2 = NdArray.of(imageData2);
        
        List<NdArray> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        
        BatchProcessor processor = new BatchProcessor();
        
        // 应该抛出异常，因为图像形状不同
        assertThrows(IllegalArgumentException.class, () -> {
            processor.createBatch(images);
        });
    }
    
    @Test
    public void testCreateBatchWithEmptyList() {
        List<NdArray> images = new ArrayList<>();
        BatchProcessor processor = new BatchProcessor();
        
        // 应该抛出异常，因为图像列表为空
        assertThrows(IllegalArgumentException.class, () -> {
            processor.createBatch(images);
        });
    }
}