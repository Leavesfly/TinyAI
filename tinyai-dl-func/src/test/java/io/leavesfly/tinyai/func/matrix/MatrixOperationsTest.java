package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 矩阵操作函数的单元测试
 *
 * @author TinyDL
 */
public class MatrixOperationsTest {

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
    public void testMatMul() {
        MatMul matmulFunc = new MatMul();

        // 测试矩阵乘法的前向传播
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        NdArray b = NdArray.of(new float[][]{{2, 0}, {1, 2}});

        NdArray result = matmulFunc.forward(a, b);

        // 验证结果
        float[][] expected = {{4, 4}, {10, 8}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(a, "x");
        Variable w = new Variable(b, "w");
        Variable y = matmulFunc.call(x, w);

        y.backward();

        // 验证梯度：dx = dy * w^T, dw = x^T * dy
        assertNotNull(x.getGrad());
        assertNotNull(w.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        assertEquals(w.getValue().getShape(), w.getGrad().getShape());

        // 矩阵乘法梯度的具体验证
        float[][] expectedXGrad = {{2, 3}, {2, 3}}; // grad_y.dot(w.T)
        float[][] expectedWGrad = {{4, 4}, {6, 6}}; // x.T.dot(grad_y)
        assertArrayEquals(expectedXGrad, x.getGrad().getMatrix());
        assertArrayEquals(expectedWGrad, w.getGrad().getMatrix());
    }

    @Test
    public void testMatMulDifferentSizes() {
        MatMul matmulFunc = new MatMul();

        // 测试不同大小的矩阵乘法
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}});      // 1x3
        NdArray b = NdArray.of(new float[][]{{4}, {5}, {6}});  // 3x1

        NdArray result = matmulFunc.forward(a, b);

        // 验证结果：1x3 * 3x1 = 1x1
        assertEquals(Shape.of(1, 1), result.getShape());
        assertEquals(32f, result.getNumber().floatValue(), 1e-6); // 1*4 + 2*5 + 3*6 = 32

        // 测试反向传播
        Variable x = new Variable(a, "x");
        Variable w = new Variable(b, "w");
        Variable y = matmulFunc.call(x, w);

        y.backward();

        assertNotNull(x.getGrad());
        assertNotNull(w.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        assertEquals(w.getValue().getShape(), w.getGrad().getShape());
    }

    @Test
    public void testReshape() {
        Shape newShape = Shape.of(3, 2);
        Reshape reshapeFunc = new Reshape(newShape);

        // 测试reshape的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});  // 2x3
        NdArray result = reshapeFunc.forward(input);

        // 验证结果形状
        assertEquals(newShape, result.getShape());
        assertEquals(6, result.getShape().size()); // 元素总数不变

        // 测试反向传播
        Variable x = new Variable(input, "x");
        Variable y = reshapeFunc.call(x);

        y.backward();

        // 梯度形状应该与原始输入相同
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());

        // 梯度值应该全为1（因为reshape不改变值，只改变形状）
        float[][] expectedGrad = {{1, 1, 1}, {1, 1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test(expected = RuntimeException.class)
    public void testReshapeInvalidSize() {
        Shape invalidShape = Shape.of(2, 2); // 只有4个元素，但原始有6个
        Reshape reshapeFunc = new Reshape(invalidShape);

        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});  // 2x3 = 6个元素
        reshapeFunc.forward(input); // 应该抛出异常
    }

    @Test
    public void testBroadcastTo() {
        Shape targetShape = Shape.of(3, 2);
        BroadcastTo broadcastFunc = new BroadcastTo(targetShape);

        // 测试广播的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 2}});  // 1x2
        NdArray result = broadcastFunc.forward(input);

        // 验证结果形状和值
        assertEquals(targetShape, result.getShape());
        float[][] expected = {{1, 2}, {1, 2}, {1, 2}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(input, "x");
        Variable y = broadcastFunc.call(x);

        y.backward();

        // 广播的反向传播是求和到原始形状
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());

        // 梯度应该是3（因为广播到3行）
        float[][] expectedGrad = {{3, 3}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testSum() {
        Sum sumFunc = new Sum();

        // 测试求和的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        NdArray result = sumFunc.forward(input);

        // 验证结果：所有元素求和
        assertEquals(21f, result.getNumber().floatValue(), 1e-6); // 1+2+3+4+5+6=21

        // 测试反向传播
        Variable x = new Variable(input, "x");
        Variable y = sumFunc.call(x);

        y.backward();

        // 求和的梯度是将标量梯度广播到原始形状
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());

        // 所有位置的梯度都应该是1
        float[][] expectedGrad = {{1, 1, 1}, {1, 1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testSumTo() {
        Shape targetShape = Shape.of(2, 1);
        SumTo sumToFunc = new SumTo(targetShape);

        // 测试sumTo的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        NdArray result = sumToFunc.forward(input);

        // 验证结果：按列求和
        assertEquals(targetShape, result.getShape());
        float[][] expected = {{6}, {15}}; // 第一行和=6，第二行和=15
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(input, "x");
        Variable y = sumToFunc.call(x);

        y.backward();

        // sumTo的反向传播是广播
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test
    public void testTranspose() {
        Transpose transposeFunc = new Transpose();

        // 测试转置的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});  // 2x3
        NdArray result = transposeFunc.forward(input);

        // 验证结果形状和值
        assertEquals(Shape.of(3, 2), result.getShape());
        float[][] expected = {{1, 4}, {2, 5}, {3, 6}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(input, "x");
        Variable y = transposeFunc.call(x);

        y.backward();

        // 转置的梯度也是转置
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());

        // 梯度应该全为1
        float[][] expectedGrad = {{1, 1, 1}, {1, 1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testGetItem() {
        int[] rowSlices = {0, 2};
        int[] colSlices = {1, 2};
        GetItem getItemFunc = new GetItem(rowSlices, colSlices);

        // 测试getItem的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        NdArray result = getItemFunc.forward(input);

        // 验证结果：获取指定位置的元素（配对索引方式）
        // 行0列0列1=2, 行2列2=9
        float[][] expected = {{2, 9}};
        assertArrayEquals(expected, result.getMatrix());

        // 测试反向传播
        Variable x = new Variable(input, "x");
        Variable y = getItemFunc.call(x);

        y.backward();

        // getItem的梯度在选中的位置为1，其他位置为0
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test
    public void testSoftMax() {
        SoftMax softmaxFunc = new SoftMax();

        // 测试softmax的前向传播
        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {1, 2, 3}});
        NdArray result = softmaxFunc.forward(input);

        // 验证结果：每行和应该为1
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < matrix.length; i++) {
            float sum = 0;
            for (int j = 0; j < matrix[i].length; j++) {
                sum += matrix[i][j];
                assertTrue(matrix[i][j] > 0); // 所有值都应该为正
            }
            assertEquals(1f, sum, 1e-6);
        }

        // 测试反向传播
        Variable x = new Variable(input, "x");
        Variable y = softmaxFunc.call(x);

        y.backward();

        // softmax的梯度
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test
    public void testLinear() {
        Linear linearFunc = new Linear();

        // 测试线性变换（无偏置）
        NdArray x = NdArray.of(new float[][]{{1, 2}});      // 1x2
        NdArray w = NdArray.of(new float[][]{{1, 3}, {2, 4}}); // 2x2

        NdArray result = linearFunc.forward(x, w);

        // 验证结果：x * w
        float[][] expected = {{5, 11}}; // [1*1+2*2, 1*3+2*4] = [5, 11]
        assertArrayEquals(expected, result.getMatrix());

        // 测试带偏置的线性变换
        NdArray b = NdArray.of(new float[][]{{1, 1}}); // 1x2 偏置
        NdArray resultWithBias = linearFunc.forward(x, w, b);

        float[][] expectedWithBias = {{6, 12}}; // [5+1, 11+1] = [6, 12]
        assertArrayEquals(expectedWithBias, resultWithBias.getMatrix());

        // 测试反向传播（带偏置）
        Variable xVar = new Variable(x, "x");
        Variable wVar = new Variable(w, "w");
        Variable bVar = new Variable(b, "b");
        Variable y = linearFunc.call(xVar, wVar, bVar);

        y.backward();

        // 验证所有梯度都不为null
        assertNotNull(xVar.getGrad());
        assertNotNull(wVar.getGrad());
        assertNotNull(bVar.getGrad());

        // 验证梯度形状
        assertEquals(xVar.getValue().getShape(), xVar.getGrad().getShape());
        assertEquals(wVar.getValue().getShape(), wVar.getGrad().getShape());
        assertEquals(bVar.getValue().getShape(), bVar.getGrad().getShape());
    }

    @Test
    public void testRequireInputNum() {
        // 测试各个函数需要的输入参数个数
        assertEquals(2, new MatMul().requireInputNum());
        assertEquals(1, new Reshape(Shape.of(2, 2)).requireInputNum()); // Reshape需要一个输入
        assertEquals(1, new BroadcastTo(Shape.of(2, 2)).requireInputNum());
        assertEquals(1, new Sum().requireInputNum());
        assertEquals(1, new SumTo(Shape.of(2, 2)).requireInputNum());
        assertEquals(1, new Transpose().requireInputNum());
        assertEquals(1, new GetItem(new int[]{0}, new int[]{0}).requireInputNum());
        assertEquals(1, new SoftMax().requireInputNum());
        assertEquals(-1, new Linear().requireInputNum()); // -1表示可变参数
    }

    @Test
    public void testChainedMatrixOperations() {
        // 测试链式矩阵操作：transpose -> matmul -> sum
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "x");
        Variable w = new Variable(NdArray.of(new float[][]{{1, 0}, {0, 1}}), "w"); // 单位矩阵

        Transpose transposeFunc = new Transpose();
        MatMul matmulFunc = new MatMul();
        Sum sumFunc = new Sum();

        Variable xT = transposeFunc.call(x);        // 转置
        Variable y = matmulFunc.call(xT, w);        // 矩阵乘法
        Variable z = sumFunc.call(y);               // 求和

        // 验证前向传播结果
        assertEquals(10f, z.getValue().getNumber().floatValue(), 1e-6); // 1+3+2+4=10

        // 验证反向传播
        z.backward();

        assertNotNull(x.getGrad());
        assertNotNull(w.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        assertEquals(w.getValue().getShape(), w.getGrad().getShape());
    }

    @Test
    public void testBroadcastAndSum() {
        // 测试广播后求和的操作
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2}}), "x"); // 1x2

        BroadcastTo broadcastFunc = new BroadcastTo(Shape.of(3, 2));
        Sum sumFunc = new Sum();

        Variable broadcasted = broadcastFunc.call(x);
        Variable summed = sumFunc.call(broadcasted);

        // 验证结果：(1+2)*3 = 9
        assertEquals(9f, summed.getValue().getNumber().floatValue(), 1e-6);

        // 验证反向传播
        summed.backward();

        assertNotNull(x.getGrad());
        // 梯度应该是3（每个元素被复制到3次）
        float[][] expectedGrad = {{3, 3}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testReshapeAndTranspose() {
        // 测试reshape后转置
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3, 4, 5, 6}}), "x"); // 1x6

        Reshape reshapeFunc = new Reshape(Shape.of(2, 3));
        Transpose transposeFunc = new Transpose();

        Variable reshaped = reshapeFunc.call(x);    // 变为2x3
        Variable transposed = transposeFunc.call(reshaped); // 变为3x2

        // 验证最终形状
        assertEquals(Shape.of(3, 2), transposed.getValue().getShape());

        // 验证反向传播
        transposed.backward();

        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());

        // 梯度应该全为1
        float[][] expectedGrad = {{1, 1, 1, 1, 1, 1}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testNonTrainMode() {
        // 在非训练模式下测试矩阵操作
        Config.train = false;

        Variable x = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "x");
        Variable w = new Variable(NdArray.of(new float[][]{{1, 0}, {0, 1}}), "w");

        MatMul matmulFunc = new MatMul();
        Variable y = matmulFunc.call(x, w);

        // 结果应该正确
        float[][] expectedResult = {{1, 2}, {3, 4}};
        assertArrayEquals(expectedResult, y.getValue().getMatrix());

        // 但不应该有创建者（计算图）
        assertNull(y.getCreator());
    }

    @Test
    public void testEdgeCases() {
        // 测试边界情况

        // 1x1矩阵的转置
        Transpose transposeFunc = new Transpose();
        NdArray scalar = NdArray.of(new float[][]{{5}});
        NdArray transposed = transposeFunc.forward(scalar);
        assertEquals(scalar.getShape(), transposed.getShape());
        assertEquals(5f, transposed.getNumber().floatValue(), 1e-6);

        // 空矩阵的求和（如果支持）
        Sum sumFunc = new Sum();
        try {
            NdArray empty = NdArray.of(Shape.of(0, 0));
            NdArray sumResult = sumFunc.forward(empty);
            assertEquals(0f, sumResult.getNumber().floatValue(), 1e-6);
        } catch (Exception e) {
            // 如果不支持空矩阵，这是预期的
        }

        // 单元素的softmax
        SoftMax softmaxFunc = new SoftMax();
        NdArray singleElement = NdArray.of(new float[][]{{5}});
        NdArray softmaxResult = softmaxFunc.forward(singleElement);
        assertEquals(1f, softmaxResult.getNumber().floatValue(), 1e-6);
    }
}