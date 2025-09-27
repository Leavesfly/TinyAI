package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 数学函数的单元测试
 *
 * @author TinyDL
 */
public class MathFunctionsTest {

    private boolean originalTrainMode;

    @Before
    public void setUp() {
        originalTrainMode = Config.train;
        Config.train = true; // 启用训练模式以构建计算图
    }

    @After
    public void tearDown() {
        Config.train = originalTrainMode;
    }

    @Test
    public void testSin() {
        Sin sinFunc = new Sin();

        // 测试sin函数的前向传播
        NdArray input = NdArray.of(new float[]{0, (float) (Math.PI / 2), (float) Math.PI});
        NdArray result = sinFunc.forward(input);

        // 验证结果
        assertEquals(0f, result.getMatrix()[0][0], 1e-6);
        assertEquals(1f, result.getMatrix()[0][1], 1e-6);
        assertEquals(0f, result.getMatrix()[0][2], 1e-5); // π的精度可能有误差

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[]{0, (float) (Math.PI / 2)}), "x");
        Variable y = sinFunc.call(x);

        y.backward();

        // sin的导数是cos
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test
    public void testCos() {
        Cos cosFunc = new Cos();

        // 测试cos函数的前向传播
        NdArray input = NdArray.of(new float[]{0, (float) (Math.PI / 2), (float) Math.PI});
        NdArray result = cosFunc.forward(input);

        // 验证结果
        assertEquals(1f, result.getMatrix()[0][0], 1e-6);
        assertEquals(0f, result.getMatrix()[0][1], 1e-6);
        assertEquals(-1f, result.getMatrix()[0][2], 1e-5);

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[]{0, (float) (Math.PI / 2)}), "x");
        Variable y = cosFunc.call(x);

        y.backward();

        // cos的导数是-sin
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test
    public void testExp() {
        Exp expFunc = new Exp();

        // 测试exp函数的前向传播
        NdArray input = NdArray.of(new float[]{0, 1, 2});
        NdArray result = expFunc.forward(input);

        // 验证结果
        assertEquals(1f, result.getMatrix()[0][0], 1e-6); // e^0 = 1
        assertEquals((float) Math.E, result.getMatrix()[0][1], 1e-6); // e^1 = e
        assertEquals((float) (Math.E * Math.E), result.getMatrix()[0][2], 1e-5); // e^2 = e^2

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[]{1, 2}), "x");
        Variable y = expFunc.call(x);

        y.backward();

        // exp的导数是exp本身
        assertNotNull(x.getGrad());
        // 梯度应该等于exp(x)
        float[][] expectedGrad = {{(float) Math.E, (float) (Math.E * Math.E)}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testLog() {
        Log logFunc = new Log();

        // 测试log函数的前向传播
        NdArray input = NdArray.of(new float[]{1, (float) Math.E, (float) (Math.E * Math.E)});
        NdArray result = logFunc.forward(input);

        // 验证结果
        assertEquals(0f, result.getMatrix()[0][0], 1e-6); // ln(1) = 0
        assertEquals(1f, result.getMatrix()[0][1], 1e-6); // ln(e) = 1
        assertEquals(2f, result.getMatrix()[0][2], 1e-5); // ln(e^2) = 2

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[]{1, 2}), "x");
        Variable y = logFunc.call(x);

        y.backward();

        // log的导数是1/x
        assertNotNull(x.getGrad());
        float[][] expectedGrad = {{1f, 0.5f}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testSqu() {
        Squ squFunc = new Squ();

        // 测试平方函数的前向传播
        NdArray input = NdArray.of(new float[][]{{-2, -1}, {0, 1}, {2, 3}});
        NdArray result = squFunc.forward(input);

        // 验证结果
        float[][] expected = {{4, 1}, {0, 1}, {4, 9}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{2, 3}, {-1, 4}}), "x");
        Variable y = squFunc.call(x);

        y.backward();

        // 平方的导数是2x
        assertNotNull(x.getGrad());
        float[][] expectedGrad = {{4, 6}, {-2, 8}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testPow() {
        // 测试幂函数
        Pow powFunc = new Pow(3f);

        // 测试前向传播
        NdArray input = NdArray.of(new float[]{1, 2, 3});
        NdArray result = powFunc.forward(input);

        // 验证结果
        assertEquals(1f, result.getMatrix()[0][0], 1e-6); // 1^3 = 1
        assertEquals(8f, result.getMatrix()[0][1], 1e-6); // 2^3 = 8
        assertEquals(27f, result.getMatrix()[0][2], 1e-6); // 3^3 = 27

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[]{2, 3}), "x");
        Variable y = powFunc.call(x);

        y.backward();

        // x^3的导数是3x^2
        assertNotNull(x.getGrad());
        float[][] expectedGrad = {{12f, 27f}}; // 3*2^2=12, 3*3^2=27
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testReLU() {
        ReLu reluFunc = new ReLu();

        // 测试ReLU函数的前向传播
        NdArray input = NdArray.of(new float[][]{{-2, -1}, {0, 1}, {2, 3}});
        NdArray result = reluFunc.forward(input);

        // 验证结果：负数变0，正数保持
        float[][] expected = {{0, 0}, {0, 1}, {2, 3}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{-1, 0}, {1, 2}}), "x");
        Variable y = reluFunc.call(x);

        y.backward();

        // ReLU的导数：x>0时为1，x<=0时为0
        assertNotNull(x.getGrad());
        float[][] expectedGrad = {{0, 0}, {1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testSigmoid() {
        Sigmoid sigmoidFunc = new Sigmoid();

        // 测试sigmoid函数的前向传播
        NdArray input = NdArray.of(new float[]{-100, 0, 100});
        NdArray result = sigmoidFunc.forward(input);

        // 验证结果
        assertTrue(result.getMatrix()[0][0] < 0.01); // sigmoid(-100) ≈ 0
        assertEquals(0.5f, result.getMatrix()[0][1], 1e-6); // sigmoid(0) = 0.5
        assertTrue(result.getMatrix()[0][2] > 0.99); // sigmoid(100) ≈ 1

        // 测试所有值都在[0,1]范围内（sigmoid的值域）
        for (float val : result.getMatrix()[0]) {
            assertTrue(val >= 0 && val <= 1);
        }

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[]{0}), "x");
        Variable y = sigmoidFunc.call(x);

        y.backward();

        // sigmoid的导数在x=0处应该是0.25
        assertNotNull(x.getGrad());
        assertEquals(0.25f, x.getGrad().getNumber().floatValue(), 1e-6);
    }

    @Test
    public void testTanh() {
        Tanh tanhFunc = new Tanh();

        // 测试tanh函数的前向传播
        NdArray input = NdArray.of(new float[]{-100, 0, 100});
        NdArray result = tanhFunc.forward(input);

        // 验证结果
        assertTrue(result.getMatrix()[0][0] < -0.99); // tanh(-100) ≈ -1
        assertEquals(0f, result.getMatrix()[0][1], 1e-6); // tanh(0) = 0
        assertTrue(result.getMatrix()[0][2] > 0.99); // tanh(100) ≈ 1

        // 测试所有值都在[-1,1]范围内（tanh的值域）
        for (float val : result.getMatrix()[0]) {
            assertTrue(val >= -1 && val <= 1);
        }

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[]{0}), "x");
        Variable y = tanhFunc.call(x);

        y.backward();

        // tanh的导数在x=0处应该是1
        assertNotNull(x.getGrad());
        assertEquals(1f, x.getGrad().getNumber().floatValue(), 1e-6);
    }

    @Test
    public void testClip() {
        Clip clipFunc = new Clip(-1f, 1f);

        // 测试clip函数的前向传播
        NdArray input = NdArray.of(new float[][]{{-2, -0.5f}, {0, 0.5f}, {1.5f, 2}});
        NdArray result = clipFunc.forward(input);

        // 验证结果：值被限制在[-1, 1]范围内
        float[][] expected = {{-1, -0.5f}, {0, 0.5f}, {1, 1}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{-2, 0}, {2, 0.5f}}), "x");
        Variable y = clipFunc.call(x);

        y.backward();

        // clip的导数：在范围内为1，超出范围为0
        assertNotNull(x.getGrad());
        float[][] expectedGrad = {{0, 1}, {0, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testMax() {
        Max maxFunc = new Max(1, false); // 沿axis=1求最大值，不保持维度

        // 测试max函数的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 3, 2}, {6, 4, 5}});
        NdArray result = maxFunc.forward(input);

        // 验证结果：每行的最大值
        float[][] expected = {{3}, {6}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{1, 3, 2}, {6, 4, 5}}), "x");
        Variable y = maxFunc.call(x);

        y.backward();

        // max的梯度只在最大值位置为1，其他位置为0
        assertNotNull(x.getGrad());
        float[][] expectedGrad = {{0, 1, 0}, {1, 0, 0}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testMaxWithKeepDims() {
        Max maxFunc = new Max(1, true); // 沿axis=1求最大值，保持维度

        // 测试保持维度的max
        NdArray input = NdArray.of(new float[][]{{1, 3, 2}, {6, 4, 5}});
        NdArray result = maxFunc.forward(input);

        // 验证结果形状保持
        assertEquals(NdArray.of(new float[][]{{1, 3, 2}, {6, 4, 5}}).getShape(), result.getShape());

        // 验证结果值（应该是广播后的最大值）
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                assertTrue(matrix[i][j] >= input.getMatrix()[i][j]);
            }
        }
    }

    @Test
    public void testMin() {
        Min minFunc = new Min(0, false); // 沿axis=0求最小值

        // 测试min函数的前向传播
        NdArray input = NdArray.of(new float[][]{{3, 1, 4}, {2, 5, 1}});
        NdArray result = minFunc.forward(input);

        // 验证结果：每列的最小值
        float[][] expected = {{2, 1, 1}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testRequireInputNum() {
        // 测试各个函数需要的输入参数个数
        assertEquals(1, new Sin().requireInputNum());
        assertEquals(1, new Cos().requireInputNum());
        assertEquals(1, new Exp().requireInputNum());
        assertEquals(1, new Log().requireInputNum());
        assertEquals(1, new Squ().requireInputNum());
        assertEquals(1, new Pow(2f).requireInputNum());
        assertEquals(1, new ReLu().requireInputNum());
        assertEquals(1, new Sigmoid().requireInputNum());
        assertEquals(1, new Tanh().requireInputNum());
        assertEquals(1, new Clip(-1f, 1f).requireInputNum());
        assertEquals(1, new Max(0, false).requireInputNum());
        assertEquals(1, new Min(0, false).requireInputNum());
    }

    @Test(expected = RuntimeException.class)
    public void testInvalidInputNumber() {
        Sin sinFunc = new Sin();
        Variable x = new Variable(NdArray.of(1.0f));
        Variable y = new Variable(NdArray.of(2.0f));

        // Sin只需要一个输入，提供两个应该抛出异常
        sinFunc.call(x, y);
    }

    @Test
    public void testChainedMathOperations() {
        // 测试链式数学运算: y = sin(exp(x))
        Variable x = new Variable(NdArray.of(new float[]{0, 1}), "x");

        Exp expFunc = new Exp();
        Sin sinFunc = new Sin();

        Variable expX = expFunc.call(x);
        Variable y = sinFunc.call(expX);

        // 验证前向传播结果
        assertNotNull(y.getValue());
        assertEquals(x.getValue().getShape(), y.getValue().getShape());

        // 验证反向传播
        y.backward();

        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());

        // 复合函数的导数应该符合链式法则
        // d/dx[sin(exp(x))] = cos(exp(x)) * exp(x)
        float[] expected = new float[2];
        for (int i = 0; i < 2; i++) {
            float xi = x.getValue().getMatrix()[0][i];
            float exp_xi = (float) Math.exp(xi);
            expected[i] = (float) (Math.cos(exp_xi) * exp_xi);
        }

        float[][] actualGrad = x.getGrad().getMatrix();
        for (int i = 0; i < 2; i++) {
            assertEquals(expected[i], actualGrad[0][i], 1e-4);
        }
    }

    @Test
    public void testNonTrainMode() {
        // 在非训练模式下测试数学函数
        Config.train = false;

        Variable x = new Variable(NdArray.of(1.0f), "x");
        Sin sinFunc = new Sin();
        Variable y = sinFunc.call(x);

        // 结果应该正确
        assertEquals((float) Math.sin(1.0), y.getValue().getNumber().floatValue(), 1e-6);

        // 但不应该有创建者（计算图）
        assertNull(y.getCreator());
    }

    @Test
    public void testEdgeCases() {
        // 测试边界情况

        // 测试exp溢出保护
        Exp expFunc = new Exp();
        NdArray largeInput = NdArray.of(new float[]{700, 800}); // 很大的数
        NdArray expResult = expFunc.forward(largeInput);
        assertNotNull(expResult);
        // 应该不会抛出异常，即使结果可能是无穷大

        // 测试log的小数输入
        Log logFunc = new Log();
        NdArray smallInput = NdArray.of(new float[]{0.1f, 0.01f});
        NdArray logResult = logFunc.forward(smallInput);
        assertTrue(logResult.getMatrix()[0][0] < 0); // log(0.1) < 0
        assertTrue(logResult.getMatrix()[0][1] < logResult.getMatrix()[0][0]); // log(0.01) < log(0.1)

        // 测试ReLU的零输入
        ReLu reluFunc = new ReLu();
        NdArray zeroInput = NdArray.of(new float[]{0f});
        NdArray reluResult = reluFunc.forward(zeroInput);
        assertEquals(0f, reluResult.getNumber().floatValue(), 1e-6);
    }
}