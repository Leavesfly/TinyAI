package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class VGGTest {
    
    @Test
    public void testVGGConstruction() {
        // 测试VGG模型构造（默认VGG-11，1000分类）
        VGG vgg = new VGG();
        
        // 验证模型不为null
        assertNotNull(vgg);
        
        // 验证模型层数大于0
        assertTrue(vgg.getLayers().length > 0);
        
        // 验证参数数量大于0
        assertTrue(vgg.getParameterCount() > 0);
    }
    
    @Test
    public void testVGGWithCustomConfig() {
        // 测试VGG模型构造（自定义配置和分类数）
        int numClasses = 10;
        int[] blockConfig = {1, 1, 2, 2, 2}; // VGG-11
        VGG vgg = new VGG(numClasses, blockConfig);
        
        // 验证模型不为null
        assertNotNull(vgg);
    }
    
    @Test
    public void testVGGForward() {
        // 创建VGG模型（10分类以减少计算量）
        int numClasses = 10;
        VGG vgg = new VGG(numClasses);
        
        // 创建一个224x224的3通道图像批次
        // 形状: (batch_size=1, channels=3, height=224, width=224)
        float[][][][] imageData = new float[1][3][224][224];
        // 填充一些示例数据
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 224; h++) {
                for (int w = 0; w < 224; w++) {
                    imageData[0][c][h][w] = (float) (c * 224 * 224 + h * 224 + w) / (3 * 224 * 224);
                }
            }
        }
        
        NdArray image = NdArray.of(imageData);
        Variable input = new Variable(image);
        
        // 执行前向传播
        Variable output = vgg.forward(input);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(2, outputShape.getDimNum()); // 应该是2D (batch_size, num_classes)
        assertEquals(1, outputShape.getDimension(0)); // batch_size = 1
        assertEquals(numClasses, outputShape.getDimension(1)); // num_classes = 10
    }
    
    @Test
    public void testVGGParameterCount() {
        // 创建VGG模型（10分类）
        int numClasses = 10;
        VGG vgg = new VGG(numClasses);
        
        // 验证参数数量大于0
        assertTrue(vgg.getParameterCount() > 0);
    }
}