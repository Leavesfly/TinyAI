package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ResNetTest {
    
    @Test
    public void testResNetConstruction() {
        // 测试ResNet模型构造（默认ResNet-18，1000分类）
        ResNet resNet = new ResNet();
        
        // 验证模型不为null
        assertNotNull(resNet);
        
        // 验证模型层数大于0
        assertTrue(resNet.getLayers().length > 0);
        
        // 验证参数数量大于0
        assertTrue(resNet.getParameterCount() > 0);
    }
    
    @Test
    public void testResNetWithCustomConfig() {
        // 测试ResNet模型构造（自定义配置和分类数）
        int numClasses = 10;
        int[] layerConfig = {2, 2, 2, 2}; // ResNet-18
        ResNet resNet = new ResNet(numClasses, layerConfig);
        
        // 验证模型不为null
        assertNotNull(resNet);
    }
    
    @Test
    public void testResNetForward() {
        // 创建ResNet模型（10分类以减少计算量）
        int numClasses = 10;
        ResNet resNet = new ResNet(numClasses);
        
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
        Variable output = resNet.forward(input);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(2, outputShape.getDimNum()); // 应该是2D (batch_size, num_classes)
        assertEquals(1, outputShape.getDimension(0)); // batch_size = 1
        assertEquals(numClasses, outputShape.getDimension(1)); // num_classes = 10
    }
    
    @Test
    public void testResNetParameterCount() {
        // 创建ResNet模型（10分类）
        int numClasses = 10;
        ResNet resNet = new ResNet(numClasses);
        
        // 验证参数数量大于0
        assertTrue(resNet.getParameterCount() > 0);
    }
}