package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 激活函数测试
 * 
 * 测试各种激活函数的前向和反向传播
 * 
 * @author TinyAI
 */
public class ActivationFunctionsTest {
    
    private static final float DELTA = 1e-5f;
    private boolean originalTrainMode;
    
    @Before
    public void setUp() {
        originalTrainMode = Config.train;
        Config.train = true;
    }
    
    @After
    public void tearDown() {
        Config.train = originalTrainMode;
    }
    
    // =============================================================================
    // ReLU测试
    // =============================================================================
    
    @Test
    public void testReLUForward() {
        ReLU relu = new ReLU();
        NdArray input = NdArray.of(new float[][]{{-2, -1, 0, 1, 2}});
        NdArray output = relu.forward(input);
        
        float[][] expected = {{0, 0, 0, 1, 2}};
        assertArrayEquals(expected, output.getMatrix());
    }
    
    @Test
    public void testReLUBackward() {
        ReLU relu = new ReLU();
        Variable x = new Variable(NdArray.of(new float[][]{{-2, -1, 0, 1, 2}}), "x");
        Variable y = relu.call(x);
        y.backward();
        
        // ReLU梯度：x>0时为1，否则为0
        float[][] expectedGrad = {{0, 0, 0, 1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }
    
    // =============================================================================
    // LeakyReLU测试
    // =============================================================================
    
    @Test
    public void testLeakyReLUForward() {
        LeakyReLU leakyRelu = new LeakyReLU();
        NdArray input = NdArray.of(new float[][]{{-2, -1, 0, 1, 2}});
        NdArray output = leakyRelu.forward(input);
        
        // LeakyReLU: max(0.01*x, x)
        float[] expected = {-0.02f, -0.01f, 0f, 1f, 2f};
        assertArrayEquals(expected, output.getArray(), DELTA);
    }
    
    // =============================================================================
    // ELU测试
    // =============================================================================
    
    @Test
    public void testELUForward() {
        ELU elu = new ELU();
        NdArray input = NdArray.of(new float[][]{{-1, 0, 1}});
        NdArray output = elu.forward(input);
        
        // ELU: x if x>0 else alpha*(exp(x)-1)
        assertNotNull(output);
        assertEquals(input.getShape(), output.getShape());
    }
    
    // =============================================================================
    // GELU测试
    // =============================================================================
    
    @Test
    public void testGELUForward() {
        GELU gelu = new GELU();
        NdArray input = NdArray.of(new float[][]{{-1, 0, 1}});
        NdArray output = gelu.forward(input);
        
        assertNotNull(output);
        assertEquals(input.getShape(), output.getShape());
        
        // GELU(0) ≈ 0
        assertTrue(Math.abs(output.get(0, 1)) < 0.1f);
    }
    
    // =============================================================================
    // SiLU (Swish)测试
    // =============================================================================
    
    @Test
    public void testSiLUForward() {
        SiLU silu = new SiLU();
        NdArray input = NdArray.of(new float[][]{{0f}});
        NdArray output = silu.forward(input);
        
        // SiLU(0) = 0
        assertEquals(0f, output.get(0, 0), DELTA);
    }
    
    // =============================================================================
    // Tanh测试
    // =============================================================================
    
    @Test
    public void testTanhForward() {
        Tanh tanh = new Tanh();
        NdArray input = NdArray.of(new float[][]{{0f}});
        NdArray output = tanh.forward(input);
        
        // tanh(0) = 0
        assertEquals(0f, output.get(0, 0), DELTA);
    }
    
    @Test
    public void testTanhSaturation() {
        Tanh tanh = new Tanh();
        NdArray input = NdArray.of(new float[][]{{10f, -10f}});
        NdArray output = tanh.forward(input);
        
        // tanh饱和到±1
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(-1f, output.get(0, 1), DELTA);
    }
    
    // =============================================================================
    // Sigmoid测试 (在Softmax中)
    // =============================================================================
    
    @Test
    public void testSigmoidLikeActivation() {
        // Sigmoid特性: sigmoid(0) = 0.5
        Tanh tanh = new Tanh();
        // 使用tanh构造sigmoid: sigmoid(x) = (tanh(x/2) + 1) / 2
        NdArray input = NdArray.of(new float[][]{{0f}});
        NdArray output = tanh.forward(input);
        
        assertEquals(0f, output.get(0, 0), DELTA);
    }
    
    // =============================================================================
    // 边界条件测试
    // =============================================================================
    
    @Test
    public void testActivationWithLargeValues() {
        // 测试大值的稳定性
        ReLU relu = new ReLU();
        NdArray input = NdArray.of(new float[][]{{1000f, -1000f}});
        NdArray output = relu.forward(input);
        
        assertEquals(1000f, output.get(0, 0), DELTA);
        assertEquals(0f, output.get(0, 1), DELTA);
    }
    
    @Test
    public void testActivationGradientFlow() {
        // 测试梯度流动
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "x");
        
        ReLU relu = new ReLU();
        Variable y = relu.call(x);
        y.backward();
        
        assertNotNull(x.getGrad());
        float[][] expectedGrad = {{1, 1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }
    
    @Test
    public void testActivationChaining() {
        // 测试激活函数链式调用
        Variable x = new Variable(NdArray.of(new float[][]{{1f}}), "x");
        
        ReLU relu = new ReLU();
        Tanh tanh = new Tanh();
        
        Variable y1 = relu.call(x);
        Variable y2 = tanh.call(y1);
        
        y2.backward();
        
        assertNotNull(x.getGrad());
    }
    
    // =============================================================================
    // GELU backward 测试
    // =============================================================================
    
    @Test
    public void testGELUBackward() {
        GELU gelu = new GELU();
        Variable x = new Variable(NdArray.of(new float[][]{{-1f, 0f, 1f, 2f}}), "x");
        Variable y = gelu.call(x);
        y.backward();
        
        assertNotNull("GELU backward 梯度不应为 null", x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // GELU'(0) ≈ 0.5
        assertEquals(0.5f, x.getGrad().get(0, 1), 0.01f);
        
        // 梯度不应包含 NaN 或 Infinity
        for (float val : x.getGrad().getArray()) {
            assertFalse("GELU backward 梯度包含 NaN", Float.isNaN(val));
            assertFalse("GELU backward 梯度包含 Infinity", Float.isInfinite(val));
        }
    }
    
    @Test
    public void testGELUForwardBackwardConsistency() {
        // 验证缓存优化后 forward+backward 的一致性
        GELU gelu = new GELU();
        Variable x = new Variable(NdArray.of(new float[][]{{0.5f}}), "x");
        Variable y = gelu.call(x);
        
        // GELU(0.5) ≈ 0.5 * 0.5 * (1 + tanh(...)) ≈ 0.3457
        assertTrue(y.getValue().get(0, 0) > 0.3f && y.getValue().get(0, 0) < 0.4f);
        
        y.backward();
        
        // 梯度应为正值且有限
        assertTrue("GELU'(0.5) 应为正值", x.getGrad().get(0, 0) > 0);
        assertTrue("GELU'(0.5) 应小于2", x.getGrad().get(0, 0) < 2f);
    }
    
    // =============================================================================
    // SiLU backward 测试
    // =============================================================================
    
    @Test
    public void testSiLUBackward() {
        SiLU silu = new SiLU();
        Variable x = new Variable(NdArray.of(new float[][]{{-2f, -1f, 0f, 1f, 2f}}), "x");
        Variable y = silu.call(x);
        y.backward();
        
        assertNotNull("SiLU backward 梯度不应为 null", x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // SiLU'(0) = sigmoid(0) * (1 + 0 * (1 - sigmoid(0))) = 0.5
        assertEquals(0.5f, x.getGrad().get(0, 2), 0.01f);
        
        // 梯度不应包含 NaN 或 Infinity
        for (float val : x.getGrad().getArray()) {
            assertFalse("SiLU backward 梯度包含 NaN", Float.isNaN(val));
            assertFalse("SiLU backward 梯度包含 Infinity", Float.isInfinite(val));
        }
    }
    
    @Test
    public void testSiLUForwardBackwardConsistency() {
        // 验证缓存优化后 forward+backward 的一致性
        SiLU silu = new SiLU();
        Variable x = new Variable(NdArray.of(new float[][]{{1f}}), "x");
        Variable y = silu.call(x);
        
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        float sigmoidOne = 1f / (1f + (float) Math.exp(-1f));
        assertEquals(sigmoidOne, y.getValue().get(0, 0), 0.01f);
        
        y.backward();
        
        // SiLU'(1) = sigmoid(1) * (1 + 1 * (1 - sigmoid(1)))
        float expectedGrad = sigmoidOne * (1f + 1f * (1f - sigmoidOne));
        assertEquals(expectedGrad, x.getGrad().get(0, 0), 0.01f);
    }
}
