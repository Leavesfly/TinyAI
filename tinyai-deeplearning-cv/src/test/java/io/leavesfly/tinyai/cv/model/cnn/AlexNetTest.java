package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class AlexNetTest {
    
    @Test
    public void testAlexNetConstruction() {
        // 测试AlexNet模型构造（默认1000分类）
        AlexNet alexNet = new AlexNet();
        
        // 验证模型不为null
        assertNotNull(alexNet);
        
        // 验证模型层数
        assertEquals(27, alexNet.getLayers().length);
        
        // 验证参数数量大于0
        assertTrue(alexNet.getParameterCount() > 0);
    }
    
    @Test
    public void testAlexNetWithCustomClasses() {
        // 测试AlexNet模型构造（自定义分类数）
        int numClasses = 10;
        AlexNet alexNet = new AlexNet(numClasses);
        
        // 验证模型不为null
        assertNotNull(alexNet);
    }
    
    @Test
    public void testAlexNetForward() {
        // 创建AlexNet模型（10分类以减少计算量）
        int numClasses = 10;
        AlexNet alexNet = new AlexNet(numClasses);
        
        // 创建一个224x224的3通道图像批次
        // 形状: (batch_size=1, channels=3, height=224, width=224)
        // 注意：实际测试中使用较小的输入以减少计算量
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
        Variable output = alexNet.forward(input);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(2, outputShape.getDimNum()); // 应该是2D (batch_size, num_classes)
        assertEquals(1, outputShape.getDimension(0)); // batch_size = 1
        assertEquals(numClasses, outputShape.getDimension(1)); // num_classes = 10
    }
    
    @Test
    public void testAlexNetParameterCount() {
        // 创建AlexNet模型（10分类）
        int numClasses = 10;
        AlexNet alexNet = new AlexNet(numClasses);
        
        // 验证参数数量大于0
        assertTrue(alexNet.getParameterCount() > 0);
    }
}