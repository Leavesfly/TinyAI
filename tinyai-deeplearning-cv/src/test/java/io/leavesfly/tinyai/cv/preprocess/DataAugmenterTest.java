package io.leavesfly.tinyai.cv.preprocess;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DataAugmenterTest {
    
    @Test
    public void testRandomHorizontalFlip3D() {
        // 创建一个简单的3x3x1图像
        float[][][] imageData = {
            {{1, 2, 3}},
            {{4, 5, 6}},
            {{7, 8, 9}}
        };
        
        NdArray image = NdArray.of(imageData);
        DataAugmenter augmenter = new DataAugmenterImpl();
        
        // 执行水平翻转（100%概率）
        NdArray flipped = augmenter.randomHorizontalFlip(image, 1.0f);
        
        // 验证输出形状
        assertEquals(image.getShape(), flipped.getShape());
        
        // 验证翻转结果（第一行应该变成[3, 2, 1]）
        assertEquals(3.0f, flipped.get(0, 0, 0), 0.001f);
        assertEquals(2.0f, flipped.get(0, 1, 0), 0.001f);
        assertEquals(1.0f, flipped.get(0, 2, 0), 0.001f);
    }
    
    @Test
    public void testRandomVerticalFlip3D() {
        // 创建一个简单的3x3x1图像
        float[][][] imageData = {
            {{1}, {2}, {3}},
            {{4}, {5}, {6}},
            {{7}, {8}, {9}}
        };
        
        NdArray image = NdArray.of(imageData);
        DataAugmenter augmenter = new DataAugmenterImpl();
        
        // 执行垂直翻转（100%概率）
        NdArray flipped = augmenter.randomVerticalFlip(image, 1.0f);
        
        // 验证输出形状
        assertEquals(image.getShape(), flipped.getShape());
        
        // 验证翻转结果（第一列应该变成[7, 4, 1]）
        assertEquals(7.0f, flipped.get(0, 0, 0), 0.001f);
        assertEquals(4.0f, flipped.get(1, 0, 0), 0.001f);
        assertEquals(1.0f, flipped.get(2, 0, 0), 0.001f);
    }
    
    @Test
    public void testRandomCrop3D() {
        // 创建一个4x4x1图像
        float[][][] imageData = {
            {{1, 2, 3, 4}},
            {{5, 6, 7, 8}},
            {{9, 10, 11, 12}},
            {{13, 14, 15, 16}}
        };
        
        NdArray image = NdArray.of(imageData);
        DataAugmenter augmenter = new DataAugmenterImpl();
        
        // 执行裁剪到2x2
        NdArray cropped = augmenter.randomCrop(image, 2, 2);
        
        // 验证输出形状
        Shape expectedShape = Shape.of(2, 2, 1);
        assertEquals(expectedShape, cropped.getShape());
    }
    
    @Test
    public void testRandomBrightness() {
        // 创建一个简单的2x2x1图像
        float[][][] imageData = {
            {{10, 20}},
            {{30, 40}}
        };
        
        NdArray image = NdArray.of(imageData);
        DataAugmenter augmenter = new DataAugmenterImpl();
        
        // 调整亮度
        NdArray brightened = augmenter.randomBrightness(image, 0.1f);
        
        // 验证输出形状
        assertEquals(image.getShape(), brightened.getShape());
        
        // 验证亮度调整（这里只是简单验证形状，具体值因为随机性无法精确验证）
        assertNotNull(brightened);
    }
    
    @Test
    public void testRandomNoise() {
        // 创建一个简单的2x2x1图像
        float[][][] imageData = {
            {{10, 20}},
            {{30, 40}}
        };
        
        NdArray image = NdArray.of(imageData);
        DataAugmenter augmenter = new DataAugmenterImpl();
        
        // 添加噪声
        NdArray noisy = augmenter.randomNoise(image, 0.1f);
        
        // 验证输出形状
        assertEquals(image.getShape(), noisy.getShape());
        
        // 验证添加噪声（这里只是简单验证形状，具体值因为随机性无法精确验证）
        assertNotNull(noisy);
    }
    
    @Test
    public void testAugment() {
        // 创建一个简单的2x2x1图像
        float[][][] imageData = {
            {{10, 20}},
            {{30, 40}}
        };
        
        NdArray image = NdArray.of(imageData);
        DataAugmenter augmenter = new DataAugmenterImpl();
        
        // 应用所有增强
        NdArray augmented = augmenter.augment(image);
        
        // 验证输出形状
        assertEquals(image.getShape(), augmented.getShape());
        
        // 验证增强处理（这里只是简单验证形状）
        assertNotNull(augmented);
    }
}