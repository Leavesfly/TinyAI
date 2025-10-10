package io.leavesfly.tinyai.cv.model.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ResidualBlockTest {
    
    @Test
    public void testResidualBlockConstruction() {
        // 测试残差块构造（维度匹配）
        int inChannels = 64;
        int outChannels = 64;
        int stride = 1;
        ResidualBlock residualBlock = new ResidualBlock(inChannels, outChannels, stride);
        
        // 验证残差块不为null
        assertNotNull(residualBlock);
        
        // 验证层数
        assertEquals(7, residualBlock.getLayers().length); // 2 conv + 2 bn + 2 activation + 1 shortcut_bn = 7
        
        // 验证参数数量大于0
        assertTrue(residualBlock.getParameterCount() > 0);
    }
    
    @Test
    public void testResidualBlockWithDimensionMismatch() {
        // 测试残差块构造（维度不匹配，需要捷径连接）
        int inChannels = 64;
        int outChannels = 128;
        int stride = 2;
        ResidualBlock residualBlock = new ResidualBlock(inChannels, outChannels, stride);
        
        // 验证残差块不为null
        assertNotNull(residualBlock);
        
        // 验证层数（包含捷径连接的卷积层和批归一化层）
        assertEquals(9, residualBlock.getLayers().length); // 2 conv + 2 bn + 2 activation + 1 shortcut_conv + 1 shortcut_bn + 1 activation = 9
        
        // 验证参数数量大于0
        assertTrue(residualBlock.getParameterCount() > 0);
    }
    
    @Test
    public void testResidualBlockForward() {
        // 创建残差块（维度匹配）
        int inChannels = 64;
        int outChannels = 64;
        int stride = 1;
        ResidualBlock residualBlock = new ResidualBlock(inChannels, outChannels, stride);
        
        // 创建输入特征图
        // 形状: (batch_size=1, channels=64, height=56, width=56)
        float[][][][] inputData = new float[1][64][56][56];
        // 填充一些示例数据
        for (int c = 0; c < 64; c++) {
            for (int h = 0; h < 56; h++) {
                for (int w = 0; w < 56; w++) {
                    inputData[0][c][h][w] = (float) (c * 56 * 56 + h * 56 + w) / (64 * 56 * 56);
                }
            }
        }
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = residualBlock.forward(inputVar);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(4, outputShape.getDimNum()); // 应该是4D (batch_size, channels, height, width)
        assertEquals(1, outputShape.getDimension(0)); // batch_size = 1
        assertEquals(64, outputShape.getDimension(1)); // channels = 64
        assertEquals(56, outputShape.getDimension(2)); // height = 56
        assertEquals(56, outputShape.getDimension(3)); // width = 56
    }
    
    @Test
    public void testResidualBlockForwardWithDownsampling() {
        // 创建残差块（维度不匹配，需要下采样）
        int inChannels = 64;
        int outChannels = 128;
        int stride = 2;
        ResidualBlock residualBlock = new ResidualBlock(inChannels, outChannels, stride);
        
        // 创建输入特征图
        // 形状: (batch_size=1, channels=64, height=56, width=56)
        float[][][][] inputData = new float[1][64][56][56];
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = residualBlock.forward(inputVar);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(4, outputShape.getDimNum()); // 应该是4D (batch_size, channels, height, width)
        assertEquals(1, outputShape.getDimension(0)); // batch_size = 1
        assertEquals(128, outputShape.getDimension(1)); // channels = 128
        assertEquals(28, outputShape.getDimension(2)); // height = 28 (56/2 due to stride=2)
        assertEquals(28, outputShape.getDimension(3)); // width = 28 (56/2 due to stride=2)
    }
    
    @Test
    public void testResidualBlockParameterCount() {
        // 创建残差块（维度匹配）
        int inChannels = 64;
        int outChannels = 64;
        int stride = 1;
        ResidualBlock residualBlock = new ResidualBlock(inChannels, outChannels, stride);
        
        // 验证参数数量大于0
        assertTrue(residualBlock.getParameterCount() > 0);
    }
}