package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * LayerNorm层的单元测试
 */
public class LayerNormTest {

    @Test
    public void testLayerNormCreation() {
        LayerNorm layerNorm = new LayerNorm("ln", 64, 1e-5f);

        assertEquals("ln", layerNorm.getName());
        assertNotNull(layerNorm.getGamma());
        assertNotNull(layerNorm.getBeta());
    }

    @Test
    public void testLayerNormForward() {
        LayerNorm layerNorm = new LayerNorm("ln", 64, 1e-5f);

        // 创建输入 (batch=32, features=64)
        NdArray inputData = NdArray.randn(Shape.of(32, 64));
        Variable input = new Variable(inputData);

        // 前向传播
        Variable output = layerNorm.forward(input);

        // 验证输出形状
        assertEquals(Shape.of(32, 64), output.getShape());
    }

    @Test
    public void testLayerNormGradientCheck() {
        LayerNorm layerNorm = new LayerNorm("ln", 64, 1e-5f);
        NdArray inputData = NdArray.randn(Shape.of(32, 64));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(layerNorm, input);
    }
}

