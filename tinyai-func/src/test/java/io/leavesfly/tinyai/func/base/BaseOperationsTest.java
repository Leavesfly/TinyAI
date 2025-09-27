package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 基础运算函数的单元测试
 * 
 * @author TinyDL
 */
public class BaseOperationsTest {
    
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
    public void testAdd() {
        Add addFunc = new Add();
        
        // 测试相同形状的加法
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        NdArray b = NdArray.of(new float[][]{{2, 3}, {4, 5}});
        
        NdArray result = addFunc.forward(a, b);
        float[][] expected = {{3, 5}, {7, 9}};
        assertArrayEquals(expected, result.getMatrix());
        
        // 测试反向传播
        Variable x = new Variable(a, "x");
        Variable y = new Variable(b, "y");
        Variable z = addFunc.call(x, y);
        
        z.backward();
        
        // 加法的梯度都是1
        float[][] expectedGrad = {{1, 1}, {1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
        assertArrayEquals(expectedGrad, y.getGrad().getMatrix());
    }
    
    @Test
    public void testAddWithBroadcast() {
        Add addFunc = new Add();
        
        // 测试广播加法
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        NdArray b = NdArray.of(new float[][]{{10}});  // 将被广播
        
        NdArray result = addFunc.forward(a, b);
        float[][] expected = {{11, 12, 13}, {14, 15, 16}};
        assertArrayEquals(expected, result.getMatrix());
        
        // 测试广播的反向传播
        Variable x = new Variable(a, "x");
        Variable y = new Variable(b, "y");
        Variable z = addFunc.call(x, y);
        
        z.backward();
        
        // x的梯度应该全为1
        float[][] expectedXGrad = {{1, 1, 1}, {1, 1, 1}};
        assertArrayEquals(expectedXGrad, x.getGrad().getMatrix());
        
        // y的梯度应该是广播后求和 = 6
        float[][] expectedYGrad = {{6}};
        assertArrayEquals(expectedYGrad, y.getGrad().getMatrix());
    }
    
    @Test
    public void testSub() {
        Sub subFunc = new Sub();
        
        // 测试减法
        NdArray a = NdArray.of(new float[][]{{5, 6}, {7, 8}});
        NdArray b = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        
        NdArray result = subFunc.forward(a, b);
        float[][] expected = {{4, 4}, {4, 4}};
        assertArrayEquals(expected, result.getMatrix());
        
        // 测试反向传播
        Variable x = new Variable(a, "x");
        Variable y = new Variable(b, "y");
        Variable z = subFunc.call(x, y);
        
        z.backward();
        
        // 减法的梯度：x为1，y为-1
        float[][] expectedXGrad = {{1, 1}, {1, 1}};
        float[][] expectedYGrad = {{-1, -1}, {-1, -1}};
        assertArrayEquals(expectedXGrad, x.getGrad().getMatrix());
        assertArrayEquals(expectedYGrad, y.getGrad().getMatrix());
    }
    
    @Test
    public void testMul() {
        Mul mulFunc = new Mul();
        
        // 测试乘法
        NdArray a = NdArray.of(new float[][]{{2, 3}, {4, 5}});
        NdArray b = NdArray.of(new float[][]{{3, 4}, {5, 6}});
        
        NdArray result = mulFunc.forward(a, b);
        float[][] expected = {{6, 12}, {20, 30}};
        assertArrayEquals(expected, result.getMatrix());
        
        // 测试反向传播
        Variable x = new Variable(a, "x");
        Variable y = new Variable(b, "y");
        Variable z = mulFunc.call(x, y);
        
        z.backward();
        
        // 乘法的梯度：dx = y, dy = x
        assertArrayEquals(b.getMatrix(), x.getGrad().getMatrix());
        assertArrayEquals(a.getMatrix(), y.getGrad().getMatrix());
    }
    
    @Test
    public void testDiv() {
        Div divFunc = new Div();
        
        // 测试除法
        NdArray a = NdArray.of(new float[][]{{6, 8}, {10, 12}});
        NdArray b = NdArray.of(new float[][]{{2, 4}, {5, 6}});
        
        NdArray result = divFunc.forward(a, b);
        float[][] expected = {{3, 2}, {2, 2}};
        assertArrayEquals(expected, result.getMatrix());
        
        // 测试反向传播
        Variable x = new Variable(a, "x");
        Variable y = new Variable(b, "y");
        Variable z = divFunc.call(x, y);
        
        z.backward();
        
        // 除法的梯度计算比较复杂，这里只验证梯度不为null
        assertNotNull(x.getGrad());
        assertNotNull(y.getGrad());
        
        // 验证形状正确
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        assertEquals(y.getValue().getShape(), y.getGrad().getShape());
    }
    
    @Test
    public void testNeg() {
        Neg negFunc = new Neg();
        
        // 测试取反
        NdArray a = NdArray.of(new float[][]{{1, -2}, {-3, 4}});
        
        NdArray result = negFunc.forward(a);
        float[][] expected = {{-1, 2}, {3, -4}};
        assertArrayEquals(expected, result.getMatrix());
        
        // 测试反向传播
        Variable x = new Variable(a, "x");
        Variable z = negFunc.call(x);
        
        z.backward();
        
        // 取反的梯度是-1
        float[][] expectedGrad = {{-1, -1}, {-1, -1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }
    
    @Test
    public void testRequireInputNum() {
        // 测试各个函数需要的输入参数个数
        assertEquals(2, new Add().requireInputNum());
        assertEquals(2, new Sub().requireInputNum());
        assertEquals(2, new Mul().requireInputNum());
        assertEquals(2, new Div().requireInputNum());
        assertEquals(1, new Neg().requireInputNum());
    }
    
    @Test(expected = RuntimeException.class)
    public void testAddInvalidInputs() {
        Add addFunc = new Add();
        Variable x = new Variable(NdArray.of(1.0f));
        
        // 只提供一个输入，应该抛出异常
        addFunc.call(x);
    }
    
    @Test(expected = RuntimeException.class)
    public void testSubInvalidInputs() {
        Sub subFunc = new Sub();
        Variable x = new Variable(NdArray.of(1.0f));
        Variable y = new Variable(NdArray.of(2.0f));
        Variable z = new Variable(NdArray.of(3.0f));
        
        // 提供三个输入，应该抛出异常
        subFunc.call(x, y, z);
    }
    
    @Test
    public void testChainedOperations() {
        // 测试链式运算: (a + b) * (a - b) = a^2 - b^2
        Variable a = new Variable(NdArray.of(new float[][]{{3, 4}, {5, 6}}), "a");
        Variable b = new Variable(NdArray.of(new float[][]{{1, 2}, {2, 3}}), "b");
        
        Add addFunc = new Add();
        Sub subFunc = new Sub();
        Mul mulFunc = new Mul();
        
        Variable sum = addFunc.call(a, b);      // a + b
        Variable diff = subFunc.call(a, b);     // a - b
        Variable result = mulFunc.call(sum, diff); // (a + b) * (a - b)
        
        // 验证结果 = a^2 - b^2
        float[][] expected = {{8, 12}, {21, 27}}; // {9-1, 16-4}, {25-4, 36-9}
        assertArrayEquals(expected, result.getValue().getMatrix());
        
        // 验证反向传播
        result.backward();
        
        // 梯度验证：d/da[(a+b)(a-b)] = (a-b) + (a+b) = 2a
        float[][] expectedAGrad = {{6, 8}, {10, 12}}; // 2 * a
        assertArrayEquals(expectedAGrad, a.getGrad().getMatrix());
        
        // 梯度验证：d/db[(a+b)(a-b)] = (a-b) - (a+b) = -2b
        float[][] expectedBGrad = {{-2, -4}, {-4, -6}}; // -2 * b
        assertArrayEquals(expectedBGrad, b.getGrad().getMatrix());
    }
    
    @Test
    public void testScalarOperations() {
        // 测试标量运算
        Variable a = new Variable(NdArray.of(5.0f), "a");
        Variable b = new Variable(NdArray.of(3.0f), "b");
        
        Add addFunc = new Add();
        Sub subFunc = new Sub();
        Mul mulFunc = new Mul();
        Div divFunc = new Div();
        
        // 测试各种运算
        Variable sum = addFunc.call(a, b);
        assertEquals(8.0f, sum.getValue().getNumber().floatValue(), 1e-6);
        
        Variable diff = subFunc.call(a, b);
        assertEquals(2.0f, diff.getValue().getNumber().floatValue(), 1e-6);
        
        Variable mul = mulFunc.call(a, b);
        assertEquals(15.0f, mul.getValue().getNumber().floatValue(), 1e-6);
        
        Variable div = divFunc.call(a, b);
        assertEquals(5.0f/3.0f, div.getValue().getNumber().floatValue(), 1e-6);
        
        // 测试取反
        Neg negFunc = new Neg();
        Variable neg = negFunc.call(a);
        assertEquals(-5.0f, neg.getValue().getNumber().floatValue(), 1e-6);
    }
    
    @Test
    public void testOperationsWithZero() {
        // 测试涉及零的运算
        Variable zero = new Variable(NdArray.of(0.0f), "zero");
        Variable a = new Variable(NdArray.of(5.0f), "a");
        
        Add addFunc = new Add();
        Sub subFunc = new Sub();
        Mul mulFunc = new Mul();
        
        // a + 0 = a
        Variable addResult = addFunc.call(a, zero);
        assertEquals(5.0f, addResult.getValue().getNumber().floatValue(), 1e-6);
        
        // a - 0 = a
        Variable subResult = subFunc.call(a, zero);
        assertEquals(5.0f, subResult.getValue().getNumber().floatValue(), 1e-6);
        
        // a * 0 = 0
        Variable mulResult = mulFunc.call(a, zero);
        assertEquals(0.0f, mulResult.getValue().getNumber().floatValue(), 1e-6);
        
        // 0的取反是0
        Neg negFunc = new Neg();
        Variable negResult = negFunc.call(zero);
        assertEquals(0.0f, negResult.getValue().getNumber().floatValue(), 1e-6);
    }
    
    @Test
    public void testBroadcastOperations() {
        // 测试各种广播情况
        Variable matrix = new Variable(NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}}), "matrix");
        Variable scalar = new Variable(NdArray.of(10.0f), "scalar");
        
        Add addFunc = new Add();
        
        // 矩阵 + 标量
        Variable result1 = addFunc.call(matrix, scalar);
        float[][] expected1 = {{11, 12, 13}, {14, 15, 16}};
        assertArrayEquals(expected1, result1.getValue().getMatrix());
        
        // 验证反向传播中的广播处理
        result1.backward();
        
        // matrix的梯度应该全为1
        float[][] expectedMatrixGrad = {{1, 1, 1}, {1, 1, 1}};
        assertArrayEquals(expectedMatrixGrad, matrix.getGrad().getMatrix());
        
        // 确保scalar的梯度不为null再进行验证
        assertNotNull("scalar的梯度不应该为null", scalar.getGrad());
        // scalar的梯度应该是6（2x3矩阵的元素个数）
        assertEquals(6.0f, scalar.getGrad().getNumber().floatValue(), 1e-6);
    }
    
    @Test
    public void testNonTrainMode() {
        // 在非训练模式下，不应该构建计算图
        Config.train = false;
        
        Variable a = new Variable(NdArray.of(1.0f), "a");
        Variable b = new Variable(NdArray.of(2.0f), "b");
        
        Add addFunc = new Add();
        Variable result = addFunc.call(a, b);
        
        // 结果应该正确
        assertEquals(3.0f, result.getValue().getNumber().floatValue(), 1e-6);
        
        // 但不应该有创建者（计算图）
        assertNull(result.getCreator());
    }
}