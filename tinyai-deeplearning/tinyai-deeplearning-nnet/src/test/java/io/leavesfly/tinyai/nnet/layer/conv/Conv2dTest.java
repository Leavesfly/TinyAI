package io.leavesfly.tinyai.nnet.layer.conv;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Conv2d层的单元测试
 */
public class Conv2dTest {

    @Test
    public void testConv2dCreation() {
        Conv2d conv = new Conv2d("conv", 3, 16, 3, 1, 1, true);

        assertEquals("conv", conv.getName());
        assertEquals(3, conv.getInChannels());
        assertEquals(16, conv.getOutChannels());
        assertEquals(3, conv.getKernelHeight());
        assertEquals(3, conv.getKernelWidth());
    }

    @Test
    public void testConv2dForward() {
        Conv2d conv = new Conv2d("conv", 3, 16, 3, 1, 1, true);

        // 创建输入 (batch=2, channels=3, height=32, width=32)
        NdArray inputData = NdArray.randn(Shape.of(2, 3, 32, 32));
        Variable input = new Variable(inputData);

        // 前向传播
        Variable output = conv.forward(input);

        // 验证输出形状 (batch=2, channels=16, height=32, width=32)
        assertEquals(Shape.of(2, 16, 32, 32), output.getShape());
    }

    @Test
    public void testConv2dGradientCheck() {
        Conv2d conv = new Conv2d("conv", 3, 16, 3, 1, 1, true);
        NdArray inputData = NdArray.randn(Shape.of(2, 3, 32, 32));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(conv, input);
    }

    @Test
    public void testConv2dWithoutBiasGradientCheck() {
        Conv2d conv = new Conv2d("conv", 3, 16, 3, 1, 1, false);
        NdArray inputData = NdArray.randn(Shape.of(2, 3, 32, 32));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(conv, input);
    }
}

