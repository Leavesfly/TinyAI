package io.leavesfly.tinyai.cv.model.lenet;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LeNet5Test {
    
    @Test
    public void testLeNet5Construction() {
        // 测试LeNet-5模型构造
        LeNet5 lenet = new LeNet5();
        
        // 验证模型不为null
        assertNotNull(lenet);
        
        // 验证模型层数
        assertEquals(11, lenet.getLayers().length);
        
        // 验证参数数量大于0
        assertTrue(lenet.getParameterCount() > 0);
    }
    
    @Test
    public void testLeNet5Forward() {
        // 创建LeNet-5模型
        LeNet5 lenet = new LeNet5();
        
        // 创建一个32x32的单通道图像批次（模拟手写数字图像）
        // 形状: (batch_size=1, channels=1, height=32, width=32)
        float[][][][] imageData = new float[1][1][32][32];
        // 填充一些示例数据
        for (int h = 0; h < 32; h++) {
            for (int w = 0; w < 32; w++) {
                imageData[0][0][h][w] = (float) (h * 32 + w) / (32 * 32); // 归一化到[0,1]
            }
        }
        
        NdArray image = NdArray.of(imageData);
        Variable input = new Variable(image);
        
        // 执行前向传播
        Variable output = lenet.forward(input);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(2, outputShape.getDimNum()); // 应该是2D (batch_size, num_classes)
        assertEquals(1, outputShape.getDimension(0)); // batch_size = 1
        assertEquals(10, outputShape.getDimension(1)); // num_classes = 10
    }
    
    @Test
    public void testLeNet5ParameterCount() {
        // 创建LeNet-5模型
        LeNet5 lenet = new LeNet5();
        
        // 计算期望的参数数量
        // 卷积层1: (5*5*1*6) + 6 = 156
        // 卷积层2: (5*5*6*16) + 16 = 2416
        // 全连接层1: (400*120) + 120 = 48120
        // 全连接层2: (120*84) + 84 = 10164
        // 输出层: (84*10) + 10 = 850
        // 总计: 156 + 2416 + 48120 + 10164 + 850 = 61706
        int expectedParamCount = 61706;
        
        // 验证参数数量
        assertEquals(expectedParamCount, lenet.getParameterCount());
    }
}