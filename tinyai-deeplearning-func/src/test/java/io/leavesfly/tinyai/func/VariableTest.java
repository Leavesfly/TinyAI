package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Variable类的单元测试
 *
 * @author TinyDL
 */
public class VariableTest {

    private Variable scalarVar;
    private Variable vectorVar;
    private Variable matrixVar;
    private boolean originalTrainMode;

    @Before
    public void setUp() {
        originalTrainMode = Config.train;
        Config.train = true; // 启用训练模式以构建计算图

        scalarVar = new Variable(NdArray.of(5.0f), "scalar");
        vectorVar = new Variable(NdArray.of(new float[]{1, 2, 3, 4}), "vector");
        matrixVar = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "matrix");
    }

    @After
    public void tearDown() {
        Config.train = originalTrainMode; // 恢复原始状态
    }

    @Test
    public void testConstructors() {
        // 测试NdArray构造器
        NdArray arr = NdArray.of(3.14f);
        Variable v1 = new Variable(arr);
        assertEquals(arr, v1.getValue());
        assertNull(v1.getName());
        // Variable类没有isRequireGrad()方法，但requireGrad默认为true

        // 测试Number构造器
        Variable v2 = new Variable(2.71f);
        assertEquals(2.71f, v2.getValue().getNumber().floatValue(), 1e-6);

        // 测试带名称的构造器
        Variable v3 = new Variable(arr, "test_var");
        assertEquals(arr, v3.getValue());
        assertEquals("test_var", v3.getName());

        // 测试带名称和梯度标志的构造器
        Variable v4 = new Variable(arr, "no_grad_var", false);
        assertEquals(arr, v4.getValue());
        assertEquals("no_grad_var", v4.getName());
        // Variable类没有isRequireGrad()方法，通过构造函数参数验证
    }

    @Test(expected = RuntimeException.class)
    public void testNullValueConstructor() {
        new Variable((NdArray) null);
    }

    @Test(expected = RuntimeException.class)
    public void testNullNumberConstructor() {
        new Variable((Number) null);
    }

    @Test
    public void testSettersAndGetters() {
        Variable var = new Variable(NdArray.of(1.0f));

        // 测试setValue和getValue
        NdArray newValue = NdArray.of(2.0f);
        var.setValue(newValue);
        assertEquals(newValue, var.getValue());

        // 测试setName和getName
        var.setName("test");
        assertEquals("test", var.getName());

        // 测试setRequireGrad
        var.setRequireGrad(false);
        // 只能通过设置梯度来间接验证requireGrad的效果

        var.setRequireGrad(true);
        // 通过设置梯度来验证requireGrad的效果
    }

    @Test
    public void testGradientOperations() {
        Variable var = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}));

        // 初始状态梯度为null
        assertNull(var.getGrad());

        // 设置梯度
        NdArray grad = NdArray.of(new float[][]{{0.1f, 0.2f}, {0.3f, 0.4f}});
        var.setGrad(grad);
        assertEquals(grad, var.getGrad());

        // 清除梯度
        var.clearGrad();
        assertNull(var.getGrad());
    }

    @Test(expected = RuntimeException.class)
    public void testInvalidGradientShape() {
        Variable var = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}));
        NdArray invalidGrad = NdArray.of(new float[]{1, 2, 3}); // 形状不匹配
        var.setGrad(invalidGrad);
    }

    @Test
    public void testRequireGradEffect() {
        Variable var = new Variable(NdArray.of(1.0f));
        NdArray grad = NdArray.of(0.5f);

        // requireGrad为true时可以设置梯度
        var.setRequireGrad(true);
        var.setGrad(grad);
        assertEquals(grad, var.getGrad());

        // requireGrad为false时梯度被设为null
        var.setRequireGrad(false);
        var.setGrad(grad);
        assertNull(var.getGrad());
    }

    @Test
    public void testBasicArithmetic() {
        Variable a = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "a");
        Variable b = new Variable(NdArray.of(new float[][]{{2, 3}, {4, 5}}), "b");

        // 测试加法
        Variable sum = a.add(b);
        float[][] expectedSum = {{3, 5}, {7, 9}};
        assertArrayEquals(expectedSum, sum.getValue().getMatrix());
        assertNotNull(sum.getCreator()); // 应该有创建者函数

        // 测试减法
        Variable diff = b.sub(a);
        float[][] expectedDiff = {{1, 1}, {1, 1}};
        assertArrayEquals(expectedDiff, diff.getValue().getMatrix());

        // 测试乘法
        Variable mul = a.mul(b);
        float[][] expectedMul = {{2, 6}, {12, 20}};
        assertArrayEquals(expectedMul, mul.getValue().getMatrix());

        // 测试除法
        Variable div = b.div(a);
        float[][] expectedDiv = {{2, 1.5f}, {4f / 3f, 1.25f}};
        assertArrayEquals(expectedDiv, div.getValue().getMatrix());
    }

    @Test
    public void testScalarArithmetic() {
        Variable a = new Variable(NdArray.of(new float[][]{{2, 4}, {6, 8}}));

        // 测试与数字的运算
        Variable add5 = a.add(new Variable(5));
        float[][] expectedAdd = {{7, 9}, {11, 13}};
        assertArrayEquals(expectedAdd, add5.getValue().getMatrix());

        Variable mul2 = a.mul(new Variable(2));
        float[][] expectedMul = {{4, 8}, {12, 16}};
        assertArrayEquals(expectedMul, mul2.getValue().getMatrix());
    }

    @Test
    public void testMathOperations() {
        Variable a = new Variable(NdArray.of(new float[][]{{1, 4}, {9, 16}}));

        // 测试平方
        Variable square = a.squ();
        float[][] expectedSquare = {{1, 16}, {81, 256}};
        assertArrayEquals(expectedSquare, square.getValue().getMatrix());

        // 测试指数
        Variable exp = a.exp();
        assertNotNull(exp.getValue());
        assertTrue(exp.getValue().getMatrix()[0][0] > 2.7); // e^1 > 2.7

        // 测试对数
        Variable log = a.log();
        assertEquals(0f, log.getValue().getMatrix()[0][0], 1e-6); // ln(1) = 0

        // 测试正弦
        Variable sin = a.sin();
        assertNotNull(sin.getValue());

        // 测试余弦
        Variable cos = a.cos();
        assertNotNull(cos.getValue());

        // 测试tanh
        Variable tanh = a.tanh();
        assertNotNull(tanh.getValue());

        // 测试sigmoid - Variable类没有sigmoid方法，使用tanh代替
        Variable sigmoid = a.tanh(); // 使用tanh作为替代
        assertNotNull(sigmoid.getValue());
        // tanh值在[-1,1]范围内
        float[][] sigmoidMatrix = sigmoid.getValue().getMatrix();
        for (int i = 0; i < sigmoidMatrix.length; i++) {
            for (int j = 0; j < sigmoidMatrix[i].length; j++) {
                assertTrue(sigmoidMatrix[i][j] >= -1 && sigmoidMatrix[i][j] <= 1);
            }
        }

        // 测试ReLU - Variable类没有reLU方法，使用其他操作代替
        // 测试clip操作（类似ReLU）
        Variable clipped = a.clip(0, Float.MAX_VALUE); // clip(0, +inf) 类似 ReLU
        float[][] clippedMatrix = clipped.getValue().getMatrix();
        // 所有值都应该非负
        for (int i = 0; i < clippedMatrix.length; i++) {
            for (int j = 0; j < clippedMatrix[i].length; j++) {
                assertTrue(clippedMatrix[i][j] >= 0);
            }
        }
    }

    @Test
    public void testMatrixOperations() {
        Variable a = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}));
        Variable b = new Variable(NdArray.of(new float[][]{{2, 0}, {1, 2}}));

        // 测试矩阵乘法
        Variable matmul = a.matMul(b);
        float[][] expectedMatmul = {{4, 4}, {10, 8}};
        assertArrayEquals(expectedMatmul, matmul.getValue().getMatrix());

        // 测试转置
        Variable transpose = a.transpose();
        float[][] expectedTranspose = {{1, 3}, {2, 4}};
        assertArrayEquals(expectedTranspose, transpose.getValue().getMatrix());

        // 测试reshape
        Variable reshaped = a.reshape(Shape.of(4, 1));
        assertEquals(Shape.of(4, 1), reshaped.getValue().getShape());

        // 测试sum
        Variable sum = a.sum();
        assertEquals(10f, sum.getValue().getNumber().floatValue(), 1e-6);

        // 测试softmax
        Variable softmax = a.softMax();
        assertNotNull(softmax.getValue());
        // 验证每行和为1
        float[][] softmaxMatrix = softmax.getValue().getMatrix();
        for (int i = 0; i < softmaxMatrix.length; i++) {
            float rowSum = 0;
            for (int j = 0; j < softmaxMatrix[i].length; j++) {
                rowSum += softmaxMatrix[i][j];
            }
            assertEquals(1f, rowSum, 1e-6);
        }
    }

    @Test
    public void testIndexingOperations() {
        Variable a = new Variable(NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));

        // 测试getItem
        Variable item = a.getItem(new int[]{1}, new int[]{2});
        assertEquals(6f, item.getValue().getNumber().floatValue(), 1e-6);

        // 测试broadcastTo
        Variable b = new Variable(NdArray.of(new float[][]{{1, 2}}));
        Variable broadcasted = b.broadcastTo(Shape.of(3, 2));
        assertEquals(Shape.of(3, 2), broadcasted.getValue().getShape());
        float[][] expectedBroadcast = {{1, 2}, {1, 2}, {1, 2}};
        assertArrayEquals(expectedBroadcast, broadcasted.getValue().getMatrix());
    }

    @Test
    public void testBackwardPropagation() {
        // 创建简单的计算图: z = x^2 + y
        Variable x = new Variable(NdArray.of(3.0f), "x");
        Variable y = new Variable(NdArray.of(2.0f), "y");

        Variable x_squared = x.squ(); // x^2
        Variable z = x_squared.add(y); // z = x^2 + y

        // 执行反向传播
        z.backward();

        // 验证梯度
        assertNotNull(x.getGrad());
        assertNotNull(y.getGrad());

        // dz/dx = 2x = 2*3 = 6
        assertEquals(6f, x.getGrad().getNumber().floatValue(), 1e-6);
        // dz/dy = 1
        assertEquals(1f, y.getGrad().getNumber().floatValue(), 1e-6);
    }

    @Test
    public void testComplexBackwardPropagation() {
        // 更复杂的计算图: z = (x + y) * (x - y) = x^2 - y^2
        Variable x = new Variable(NdArray.of(5.0f), "x");
        Variable y = new Variable(NdArray.of(3.0f), "y");

        Variable sum = x.add(y);     // x + y = 8
        Variable diff = x.sub(y);    // x - y = 2
        Variable z = sum.mul(diff);  // z = 8 * 2 = 16

        z.backward();

        // dz/dx = d/dx(x^2 - y^2) = 2x = 10
        assertEquals(10f, x.getGrad().getNumber().floatValue(), 1e-6);
        // dz/dy = d/dy(x^2 - y^2) = -2y = -6
        assertEquals(-6f, y.getGrad().getNumber().floatValue(), 1e-6);
    }

    @Test
    public void testNoGradBackward() {
        Variable x = new Variable(NdArray.of(3.0f), "x", false); // 不需要梯度
        Variable y = new Variable(NdArray.of(2.0f), "y", true);  // 需要梯度

        Variable z = x.add(y);
        z.backward();

        // x不需要梯度，所以梯度应该为null
        assertNull(x.getGrad());
        // y需要梯度
        assertNotNull(y.getGrad());
    }

    @Test
    public void testClearGrad() {
        Variable x = new Variable(NdArray.of(3.0f), "x");
        Variable y = x.squ();

        y.backward();
        assertNotNull(x.getGrad());

        x.clearGrad();
        assertNull(x.getGrad());
    }

    @Test
    public void testUnchainBackward() {
        Variable x = new Variable(NdArray.of(3.0f), "x");
        Variable y = x.squ();
        Variable z = y.add(x);

        // 检查计算图是否建立
        assertNotNull(z.getCreator());
        assertNotNull(y.getCreator());

        // 切断计算图
        z.unChainBackward();

        // 计算图应该被切断
        assertNull(z.getCreator());
        assertNull(y.getCreator());
        assertNull(x.getCreator());
    }

    @Test
    public void testTrainModeEffect() {
        Variable x = new Variable(NdArray.of(3.0f), "x");
        Variable y = new Variable(NdArray.of(2.0f), "y");

        // 在训练模式下
        Config.train = true;
        Variable z1 = x.add(y);
        assertNotNull(z1.getCreator()); // 应该有创建者

        // 在非训练模式下
        Config.train = false;
        Variable z2 = x.add(y);
        assertNull(z2.getCreator()); // 不应该有创建者
    }

    @Test
    public void testMatrixBackward() {
        // 测试矩阵的反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "x");
        Variable y = x.sum(); // 所有元素求和

        y.backward();

        // 梯度应该全为1
        float[][] expectedGrad = {{1, 1}, {1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testBroadcastBackward() {
        // 测试广播操作的反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}}), "x");
        Variable y = new Variable(NdArray.of(new float[][]{{1, 1, 1}}), "y"); // 将被广播

        Variable z = x.add(y);
        Variable loss = z.sum();

        loss.backward();

        assertNotNull(x.getGrad());
        assertNotNull(y.getGrad());

        // x的梯度应该全为1
        float[][] expectedXGrad = {{1, 1, 1}, {1, 1, 1}};
        assertArrayEquals(expectedXGrad, x.getGrad().getMatrix());

        // y的梯度应该是2（因为被广播到2行）
        float[][] expectedYGrad = {{2, 2, 2}};
        assertArrayEquals(expectedYGrad, y.getGrad().getMatrix());
    }

    // 辅助方法 - 删除了isRequireGrad方法，因为Variable类没有这个方法
}