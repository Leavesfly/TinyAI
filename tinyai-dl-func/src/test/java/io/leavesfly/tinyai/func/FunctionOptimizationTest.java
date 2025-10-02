package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Function类优化的单元测试
 *
 * @author TinyDL
 */
public class FunctionOptimizationTest {

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

    // 模拟一个简单的加法函数用于测试
    private static class TestAddFunction extends Function {
        @Override
        public NdArray forward(NdArray... inputs) {
            return inputs[0].add(inputs[1]);
        }

        @Override
        public List<NdArray> backward(NdArray yGrad) {
            return Arrays.asList(yGrad, yGrad);
        }

        @Override
        public int requireInputNum() {
            return 2;
        }
    }

    @Test
    public void testFunctionCall() {
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(NdArray.of(3.0f));
        Variable b = new Variable(NdArray.of(4.0f));

        Variable result = func.call(a, b);

        // 验证结果正确
        assertEquals(7.0f, result.getValue().getNumber().floatValue(), 1e-6);

        // 验证计算图构建
        assertNotNull(result.getCreator());
        assertSame(func, result.getCreator());
        assertSame(result, func.getOutput());
        assertSame(a, func.getInputs()[0]);
        assertSame(b, func.getInputs()[1]);
    }

    @Test
    public void testFunctionUnChain() {
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(NdArray.of(3.0f));
        Variable b = new Variable(NdArray.of(4.0f));

        Variable result = func.call(a, b);

        // 验证计算图已建立
        assertNotNull(func.getInputs());
        assertNotNull(func.getOutput());

        // 断开计算图
        func.unChain();

        // 验证计算图已断开
        assertNull(func.getInputs());
        assertNull(func.getOutput());
    }

    @Test(expected = RuntimeException.class)
    public void testInvalidInputNum() {
        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(NdArray.of(3.0f));

        // 应该抛出异常，因为加法函数需要2个输入，但只提供了1个
        func.call(a);
    }

    @Test
    public void testNonTrainMode() {
        Config.train = false;

        TestAddFunction func = new TestAddFunction();
        Variable a = new Variable(NdArray.of(3.0f));
        Variable b = new Variable(NdArray.of(4.0f));

        Variable result = func.call(a, b);

        // 验证结果正确
        assertEquals(7.0f, result.getValue().getNumber().floatValue(), 1e-6);

        // 在非训练模式下，不应该构建计算图
        assertNull(result.getCreator());
        assertNull(func.getInputs());
        assertNull(func.getOutput());
    }
}