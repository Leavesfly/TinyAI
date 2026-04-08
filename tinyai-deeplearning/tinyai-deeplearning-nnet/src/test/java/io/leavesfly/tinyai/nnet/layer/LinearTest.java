package io.leavesfly.tinyai.nnet.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Linear层的单元测试
 */
public class LinearTest {

    @Test
    public void testLinearCreation() {
        Linear layer = new Linear("fc", 128, 64, true);

        assertEquals("fc", layer.getName());
        assertEquals(128, layer.getInFeatures());
        assertEquals(64, layer.getOutFeatures());
        assertNotNull(layer.getWeight());
        assertNotNull(layer.getBias());
    }

    @Test
    public void testLinearWithoutBias() {
        Linear layer = new Linear("fc", 128, 64, false);

        assertNotNull(layer.getWeight());
        assertNull(layer.getBias());
    }

    @Test
    public void testLinearForward() {
        Linear layer = new Linear("fc", 128, 64, true);

        // 创建输入 (batch=32, features=128)
        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);

        // 前向传播
        Variable output = layer.forward(input);

        // 验证输出形状 (batch=32, features=64)
        assertEquals(Shape.of(32, 64), output.getShape());
    }

    @Test
    public void testLinearParameterShapes() {
        Linear layer = new Linear("fc", 100, 50, true);

        // 验证权重形状 (out_features, in_features)
        assertEquals(Shape.of(50, 100), layer.getWeight().data().getShape());

        // 验证偏置形状 (out_features)
        assertEquals(Shape.of(50), layer.getBias().data().getShape());
    }

    @Test
    public void testLinearInitialization() {
        Linear layer = new Linear("fc", 128, 64, true);

        // 验证参数已初始化（不全为0）
        float[] weightData = layer.getWeight().data().getArray();
        boolean hasNonZero = false;
        for (float v : weightData) {
            if (Math.abs(v) > 1e-6f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "权重应该被初始化为非零值");

        // 验证偏置初始化为0
        float[] biasData = layer.getBias().data().getArray();
        for (float v : biasData) {
            assertEquals(0.0f, v, 1e-6f);
        }
    }

    @Test
    public void testLinearBatchSizes() {
        Linear layer = new Linear("fc", 10, 5, true);

        // 测试不同的批次大小
        int[] batchSizes = {1, 16, 32, 64, 128};
        for (int batchSize : batchSizes) {
            NdArray inputData = NdArray.randn(Shape.of(batchSize, 10));
            Variable input = new Variable(inputData);
            Variable output = layer.forward(input);

            assertEquals(Shape.of(batchSize, 5), output.getShape());
        }
    }

    @Test
    public void testLinearToString() {
        Linear layer = new Linear("fc", 128, 64, true);
        String str = layer.toString();

        assertTrue(str.contains("Linear"));
        assertTrue(str.contains("fc"));
        assertTrue(str.contains("128"));
        assertTrue(str.contains("64"));
    }

    @Test
    public void testLinearGradientCheck() {
        Linear layer = new Linear("fc", 128, 64, true);
        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(layer, input);
    }

    @Test
    public void testLinearWithoutBiasGradientCheck() {
        Linear layer = new Linear("fc", 128, 64, false);
        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(layer, input);
    }
}
