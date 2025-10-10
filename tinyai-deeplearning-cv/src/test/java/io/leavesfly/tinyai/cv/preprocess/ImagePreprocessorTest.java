package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ImagePreprocessorTest {
    
    @Test
    public void testNormalize3D() {
        // 创建一个简单的3x3x3图像（模拟RGB图像）
        float[][][] imageData = {
            {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}},
            {{100, 110, 120}, {130, 140, 150}, {160, 170, 180}},
            {{190, 200, 210}, {220, 230, 240}, {250, 255, 255}}
        };
        
        NdArray image = NdArray.of(imageData);
        ImagePreprocessor preprocessor = new ImagePreprocessorImpl();
        
        // RGB图像的均值和标准差（示例值）
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        
        // 执行标准化
        NdArray normalized = preprocessor.normalize(image, mean, std);
        
        // 验证输出形状
        assertEquals(image.getShape(), normalized.getShape());
        
        // 验证部分值（第一个像素的R通道）
        float expected = (imageData[0][0][0] / 255.0f - mean[0]) / std[0];
        assertEquals(expected, normalized.get(0, 0, 0), 0.001f);
    }
    
    @Test
    public void testNormalize4D() {
        // 创建一个2x2x2x3的批量图像数据
        float[][][][] imageData = {
            {
                {{10, 20, 30}, {40, 50, 60}},
                {{70, 80, 90}, {100, 110, 120}}
            },
            {
                {{130, 140, 150}, {160, 170, 180}},
                {{190, 200, 210}, {220, 230, 240}}
            }
        };
        
        NdArray image = NdArray.of(imageData);
        ImagePreprocessor preprocessor = new ImagePreprocessorImpl();
        
        // RGB图像的均值和标准差（示例值）
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        
        // 执行标准化
        NdArray normalized = preprocessor.normalize(image, mean, std);
        
        // 验证输出形状
        assertEquals(image.getShape(), normalized.getShape());
        
        // 验证部分值（第一个样本第一个像素的R通道）
        float expected = (imageData[0][0][0][0] / 255.0f - mean[0]) / std[0];
        assertEquals(expected, normalized.get(0, 0, 0, 0), 0.001f);
    }
    
    @Test
    public void testNormalizeToRange() {
        // 创建一个简单的2x2x1图像
        float[][][] imageData = {
            {{10, 20}, {30, 40}},
            {{50, 60}, {70, 80}}
        };
        
        NdArray image = NdArray.of(imageData);
        ImagePreprocessor preprocessor = new ImagePreprocessorImpl();
        
        // 归一化到[0, 1]范围
        NdArray normalized = preprocessor.normalizeToRange(image, 0.0f, 1.0f);
        
        // 验证输出范围
        float min = normalized.flatten().min(0).getNumber().floatValue();
        float max = normalized.flatten().max(0).getNumber().floatValue();
        
        assertTrue(min >= 0.0f);
        assertTrue(max <= 1.0f);
        assertEquals(0.0f, min, 0.001f);
        assertEquals(1.0f, max, 0.001f);
    }
    
    @Test
    public void testResize3D() {
        // 创建一个2x2x1图像
        float[][][] imageData = {
            {{1, 2}, {3, 4}},
            {{5, 6}, {7, 8}}
        };
        
        NdArray image = NdArray.of(imageData);
        ImagePreprocessor preprocessor = new ImagePreprocessorImpl();
        
        // 缩放到4x4
        NdArray resized = preprocessor.resize(image, 4, 4);
        
        // 验证输出形状
        Shape expectedShape = Shape.of(4, 4, 1);
        assertEquals(expectedShape, resized.getShape());
    }
    
    @Test
    public void testResize4D() {
        // 创建一个1x2x2x1批量图像
        float[][][][] imageData = {
            {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}}
            }
        };
        
        NdArray image = NdArray.of(imageData);
        ImagePreprocessor preprocessor = new ImagePreprocessorImpl();
        
        // 缩放到1x4x4x1
        NdArray resized = preprocessor.resize(image, 4, 4);
        
        // 验证输出形状
        Shape expectedShape = Shape.of(1, 4, 4, 1);
        assertEquals(expectedShape, resized.getShape());
    }
}