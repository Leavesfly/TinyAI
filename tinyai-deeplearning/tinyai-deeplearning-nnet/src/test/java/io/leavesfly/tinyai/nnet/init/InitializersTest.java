package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 初始化器的单元测试
 */
public class InitializersTest {

    @Test
    public void testZerosInitializer() {
        NdArray tensor = NdArray.of(Shape.of(10, 10));
        Initializers.zeros(tensor);

        float[] data = tensor.getArray();
        for (float v : data) {
            assertEquals(0.0f, v, 1e-6f);
        }
    }

    @Test
    public void testOnesInitializer() {
        NdArray tensor = NdArray.of(Shape.of(10, 10));
        Initializers.ones(tensor);

        float[] data = tensor.getArray();
        for (float v : data) {
            assertEquals(1.0f, v, 1e-6f);
        }
    }

    @Test
    public void testConstantInitializer() {
        NdArray tensor = NdArray.of(Shape.of(10, 10));
        float constant = 3.14f;
        Initializers.constant(tensor, constant);

        float[] data = tensor.getArray();
        for (float v : data) {
            assertEquals(constant, v, 1e-6f);
        }
    }

    @Test
    public void testUniformInitializer() {
        NdArray tensor = NdArray.of(Shape.of(100, 100));
        float a = -1.0f;
        float b = 1.0f;
        Initializers.uniform(tensor, a, b);

        float[] data = tensor.getArray();
        
        // 验证所有值都在[a, b]范围内
        for (float v : data) {
            assertTrue(v >= a && v <= b);
        }

        // 验证不是所有值都相同
        boolean hasVariance = false;
        float first = data[0];
        for (float v : data) {
            if (Math.abs(v - first) > 1e-6f) {
                hasVariance = true;
                break;
            }
        }
        assertTrue(hasVariance);
    }

    @Test
    public void testNormalInitializer() {
        NdArray tensor = NdArray.of(Shape.of(1000, 100));
        float mean = 0.0f;
        float std = 1.0f;
        Initializers.normal(tensor, mean, std);

        float[] data = tensor.getArray();

        // 计算实际均值
        double sum = 0;
        for (float v : data) {
            sum += v;
        }
        double actualMean = sum / data.length;

        // 验证均值接近0（允许一定误差）
        assertEquals(mean, actualMean, 0.1);

        // 验证不是所有值都相同
        boolean hasVariance = false;
        float first = data[0];
        for (float v : data) {
            if (Math.abs(v - first) > 1e-6f) {
                hasVariance = true;
                break;
            }
        }
        assertTrue(hasVariance);
    }

    @Test
    public void testXavierUniformInitializer() {
        NdArray tensor = NdArray.of(Shape.of(64, 128));
        Initializers.xavierUniform(tensor, 1.0f);

        float[] data = tensor.getArray();

        // 验证不是所有值都相同
        boolean hasVariance = false;
        float first = data[0];
        for (float v : data) {
            if (Math.abs(v - first) > 1e-6f) {
                hasVariance = true;
                break;
            }
        }
        assertTrue(hasVariance);

        // 验证值的范围合理（Xavier初始化的值通常不会太大）
        for (float v : data) {
            assertTrue(Math.abs(v) < 1.0f, "Xavier初始化的值应该在合理范围内");
        }
    }

    @Test
    public void testKaimingUniformInitializer() {
        NdArray tensor = NdArray.of(Shape.of(64, 128));
        Initializers.kaimingUniform(tensor, 0, "fan_in", "relu");

        float[] data = tensor.getArray();

        // 验证不是所有值都相同
        boolean hasVariance = false;
        float first = data[0];
        for (float v : data) {
            if (Math.abs(v - first) > 1e-6f) {
                hasVariance = true;
                break;
            }
        }
        assertTrue(hasVariance);
    }

    @Test
    public void testCalculateFanInAndFanOut() {
        // 2D张量
        Shape shape2d = Shape.of(64, 128);
        int[] fan2d = Initializers.calculateFanInAndFanOut(shape2d);
        assertEquals(128, fan2d[0]); // fan_in
        assertEquals(64, fan2d[1]);  // fan_out

        // 4D张量（卷积核）
        Shape shape4d = Shape.of(32, 16, 3, 3);
        int[] fan4d = Initializers.calculateFanInAndFanOut(shape4d);
        assertEquals(16 * 3 * 3, fan4d[0]); // fan_in = in_channels * kernel_size^2
        assertEquals(32 * 3 * 3, fan4d[1]); // fan_out = out_channels * kernel_size^2
    }

    @Test
    public void testGetFan() {
        int fanIn = 128;
        int fanOut = 64;

        assertEquals(128, Initializers.getFan(fanIn, fanOut, "fan_in"));
        assertEquals(64, Initializers.getFan(fanIn, fanOut, "fan_out"));
        assertEquals(96, Initializers.getFan(fanIn, fanOut, "fan_avg"));
    }

    @Test
    public void testCalculateGain() {
        assertEquals(1.0f, Initializers.calculateGain(0, "linear"), 1e-6f);
        assertEquals(1.0f, Initializers.calculateGain(0, "sigmoid"), 1e-6f);
        assertEquals(5.0f / 3, Initializers.calculateGain(0, "tanh"), 1e-6f);
        assertTrue(Math.abs(Initializers.calculateGain(0, "relu") - Math.sqrt(2.0)) < 1e-6f);
    }

    @Test
    public void testInitializerInterface() {
        // 测试使用Initializer接口
        Initializer customInit = tensor -> {
            float[] data = tensor.getArray();
            for (int i = 0; i < data.length; i++) {
                data[i] = 0.5f;
            }
        };

        NdArray tensor = NdArray.of(Shape.of(10, 10));
        customInit.initialize(tensor);

        float[] data = tensor.getArray();
        for (float v : data) {
            assertEquals(0.5f, v, 1e-6f);
        }
    }
}
