package io.leavesfly.tinyai.ndarr;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray类的单元测试
 *
 * @author TinyDL
 */
public class NdArrayTest {

    private NdArray matrix2x3;
    private NdArray matrix3x2;
    private NdArray vector;
    private NdArray scalar;

    @Before
    public void setUp() {
        matrix2x3 = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        matrix3x2 = NdArray.of(new float[][]{{1, 2}, {3, 4}, {5, 6}});
        vector = NdArray.of(new float[]{1, 2, 3, 4});
        scalar = NdArray.of(5.0f);
    }

    @Test
    public void testConstructors() {
        // 测试标量构造器
        NdArray scalar = NdArray.of(3.14f);
        assertEquals(3.14f, scalar.getNumber().floatValue(), 1e-6);

        // 测试一维数组构造器
        float[] arr1d = {1, 2, 3};
        NdArray vector = NdArray.of(arr1d);
        assertEquals(Shape.of(3), vector.getShape());

        // 测试二维数组构造器
        float[][] arr2d = {{1, 2}, {3, 4}};
        NdArray matrix = NdArray.of(arr2d);
        assertEquals(Shape.of(2, 2), matrix.getShape());

        // 测试Shape构造器
        NdArray shaped = NdArray.of(Shape.of(2, 3));
        assertEquals(Shape.of(2, 3), shaped.getShape());
    }

    @Test
    public void testStaticCreationMethods() {
        // 测试zeros
        NdArray zeros = NdArray.zeros(Shape.of(2, 3));
        assertEquals(Shape.of(2, 3), zeros.getShape());
        float[][] zeroMatrix = zeros.getMatrix();
        for (int i = 0; i < zeroMatrix.length; i++) {
            for (int j = 0; j < zeroMatrix[i].length; j++) {
                assertEquals(0f, zeroMatrix[i][j], 1e-6);
            }
        }

        // 测试ones
        NdArray ones = NdArray.ones(Shape.of(2, 2));
        assertEquals(Shape.of(2, 2), ones.getShape());
        float[][] onesMatrix = ones.getMatrix();
        for (int i = 0; i < onesMatrix.length; i++) {
            for (int j = 0; j < onesMatrix[i].length; j++) {
                assertEquals(1f, onesMatrix[i][j], 1e-6);
            }
        }

        // 测试eye
        NdArray eye = NdArray.eye(Shape.of(3, 3));
        float[][] expected = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        assertArrayEquals(expected, eye.getMatrix());

        // 测试like
        NdArray like = NdArray.like(Shape.of(2, 2), 7);
        float[][] likeMatrix = like.getMatrix();
        for (int i = 0; i < likeMatrix.length; i++) {
            for (int j = 0; j < likeMatrix[i].length; j++) {
                assertEquals(7f, likeMatrix[i][j], 1e-6);
            }
        }

        // 测试likeRandomN
        NdArray random = NdArray.likeRandomN(Shape.of(3, 3));
        assertEquals(Shape.of(3, 3), random.getShape());
        // 随机数测试只验证形状，不验证具体值
    }

    @Test
    public void testBasicArithmetic() {
        // 测试加法
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        NdArray b = NdArray.of(new float[][]{{2, 3}, {4, 5}});
        NdArray sum = a.add(b);
        float[][] expectedSum = {{3, 5}, {7, 9}};
        assertArrayEquals(expectedSum, sum.getMatrix());

        // 测试减法
        NdArray diff = b.sub(a);
        float[][] expectedDiff = {{1, 1}, {1, 1}};
        assertArrayEquals(expectedDiff, diff.getMatrix());

        // 测试乘法
        NdArray mul = a.mul(b);
        float[][] expectedMul = {{2, 6}, {12, 20}};
        assertArrayEquals(expectedMul, mul.getMatrix());

        // 测试除法
        NdArray div = b.div(a);
        float[][] expectedDiv = {{2, 1.5f}, {4f / 3f, 1.25f}};
        assertArrayEquals(expectedDiv, div.getMatrix());
    }

    @Test
    public void testScalarArithmetic() {
        NdArray a = NdArray.of(new float[][]{{2, 4}, {6, 8}});

        // 测试标量加法
        NdArray addNum = a.add(NdArray.of(3));
        float[][] expectedAdd = {{5, 7}, {9, 11}};
        assertArrayEquals(expectedAdd, addNum.getMatrix());

        // 测试标量减法
        NdArray subNum = a.sub(NdArray.of(2));
        float[][] expectedSub = {{0, 2}, {4, 6}};
        assertArrayEquals(expectedSub, subNum.getMatrix());

        // 测试标量乘法
        NdArray mulNum = a.mulNum(2);
        float[][] expectedMul = {{4, 8}, {12, 16}};
        assertArrayEquals(expectedMul, mulNum.getMatrix());

        // 测试标量除法
        NdArray divNum = a.divNum(2);
        float[][] expectedDiv = {{1, 2}, {3, 4}};
        assertArrayEquals(expectedDiv, divNum.getMatrix());
    }

    @Test
    public void testMathFunctions() {
        NdArray a = NdArray.of(new float[][]{{1, 4}, {9, 16}});

        // 测试平方
        NdArray square = a.square();
        float[][] expectedSquare = {{1, 16}, {81, 256}};
        assertArrayEquals(expectedSquare, square.getMatrix());

        // 测试开方
        NdArray sqrt = a.sqrt();
        float[][] expectedSqrt = {{1, 2}, {3, 4}};
        assertArrayEquals(expectedSqrt, sqrt.getMatrix());

        // 测试指数
        NdArray exp = a.exp();
        float[][] expMatrix = exp.getMatrix();
        assertTrue(expMatrix[0][0] > 2.7 && expMatrix[0][0] < 2.8); // e^1 ≈ 2.718

        // 测试对数
        NdArray log = a.log();
        float[][] logMatrix = log.getMatrix();
        assertEquals(0f, logMatrix[0][0], 1e-6); // ln(1) = 0

        // 测试绝对值
        NdArray negative = NdArray.of(new float[][]{{-1, 2}, {-3, 4}});
        NdArray abs = negative.abs();
        float[][] expectedAbs = {{1, 2}, {3, 4}};
        assertArrayEquals(expectedAbs, abs.getMatrix());

        // 测试取反
        NdArray neg = a.neg();
        float[][] expectedNeg = {{-1, -4}, {-9, -16}};
        assertArrayEquals(expectedNeg, neg.getMatrix());
    }

    @Test
    public void testMatrixOperations() {
        // 测试矩阵乘法
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        NdArray b = NdArray.of(new float[][]{{2, 0}, {1, 2}});
        NdArray dot = a.dot(b);
        float[][] expectedDot = {{4, 4}, {10, 8}};
        assertArrayEquals(expectedDot, dot.getMatrix());

        // 测试转置
        NdArray transpose = matrix2x3.transpose();
        assertEquals(Shape.of(3, 2), transpose.getShape());
        float[][] expectedTranspose = {{1, 4}, {2, 5}, {3, 6}};
        assertArrayEquals(expectedTranspose, transpose.getMatrix());

        // 测试reshape
        NdArray reshaped = matrix2x3.reshape(Shape.of(3, 2));
        assertEquals(Shape.of(3, 2), reshaped.getShape());

        // 测试flatten
        NdArray flattened = matrix2x3.flatten();
        assertEquals(Shape.of(1, 6), flattened.getShape());
    }

    @Test
    public void testAggregationOperations() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});

        // 测试sum
        NdArray sum = a.sum();
        assertEquals(21f, sum.getNumber().floatValue(), 1e-6);

        // 测试按轴求和
        NdArray sumAxis0 = a.sum(0);
        float[][] expectedSumAxis0 = {{5, 7, 9}};
        assertArrayEquals(expectedSumAxis0, sumAxis0.getMatrix());

        NdArray sumAxis1 = a.sum(1);
        float[][] expectedSumAxis1 = {{6}, {15}};
        assertArrayEquals(expectedSumAxis1, sumAxis1.getMatrix());

        // 测试mean
        NdArray meanAxis0 = a.mean(0);
        float[][] expectedMeanAxis0 = {{2.5f, 3.5f, 4.5f}};
        assertArrayEquals(expectedMeanAxis0, meanAxis0.getMatrix());

        NdArray meanAxis1 = a.mean(1);
        float[][] expectedMeanAxis1 = {{2f}, {5f}};
        assertArrayEquals(expectedMeanAxis1, meanAxis1.getMatrix());

        // 测试max
        NdArray maxAxis1 = a.max(1);
        float[][] expectedMaxAxis1 = {{3}, {6}};
        assertArrayEquals(expectedMaxAxis1, maxAxis1.getMatrix());

        // 测试argMax
        NdArray argMaxAxis0 = a.argMax(0);
        float[][] expectedArgMaxAxis0 = {{1, 1, 1}};
        assertArrayEquals(expectedArgMaxAxis0, argMaxAxis0.getMatrix());

        NdArray argMaxAxis1 = a.argMax(1);
        float[][] expectedArgMaxAxis1 = {{2}, {2}};
        assertArrayEquals(expectedArgMaxAxis1, argMaxAxis1.getMatrix());
    }

    @Test
    public void testBroadcasting() {
        NdArray a = NdArray.of(new float[][]{{1, 2}});
        NdArray broadcasted = a.broadcastTo(Shape.of(3, 2));

        assertEquals(Shape.of(3, 2), broadcasted.getShape());
        float[][] expected = {{1, 2}, {1, 2}, {1, 2}};
        assertArrayEquals(expected, broadcasted.getMatrix());
    }

    @Test
    public void testIndexingAndSlicing() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

        // 测试单元素获取
        NdArray item = a.getItem(new int[]{1}, new int[]{2});
        assertEquals(6f, item.getNumber().floatValue(), 1e-6);

        // 测试行切片
        NdArray rowSlice = a.getItem(new int[]{0, 2}, null);
        float[][] expectedRowSlice = {{1, 2, 3}, {7, 8, 9}};
        assertArrayEquals(expectedRowSlice, rowSlice.getMatrix());

        // 测试列切片
        NdArray colSlice = a.getItem(null, new int[]{0, 2});
        float[][] expectedColSlice = {{1, 3}, {4, 6}, {7, 9}};
        assertArrayEquals(expectedColSlice, colSlice.getMatrix());
    }

    @Test
    public void testSoftMax() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {1, 2, 3}});
        NdArray softmax = a.softMax();

        // 检查每行和是否为1
        for (int i = 0; i < softmax.getShape().getRow(); i++) {
            float sum = 0;
            for (int j = 0; j < softmax.getShape().getColumn(); j++) {
                sum += softmax.getMatrix()[i][j];
            }
            assertEquals(1f, sum, 1e-6);
        }

        // 检查所有元素是否为正数
        float[][] softmaxMatrix = softmax.getMatrix();
        for (int i = 0; i < softmaxMatrix.length; i++) {
            for (int j = 0; j < softmaxMatrix[i].length; j++) {
                assertTrue(softmaxMatrix[i][j] > 0);
            }
        }
    }

    @Test
    public void testMask() {
        NdArray a = NdArray.of(new float[][]{{-1, 0, 1}, {2, -3, 4}});
        NdArray mask = a.mask(0);

        float[][] expectedMask = {{0, 0, 1}, {1, 0, 1}};
        assertArrayEquals(expectedMask, mask.getMatrix());
    }

    @Test
    public void testMaximum() {
        NdArray a = NdArray.of(new float[][]{{-1, 0, 1}, {2, -3, 4}});
        NdArray maximum = a.maximum(0);

        float[][] expectedMaximum = {{0, 0, 1}, {2, 0, 4}};
        assertArrayEquals(expectedMaximum, maximum.getMatrix());
    }

    @Test
    public void testComparison() {
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        NdArray b = NdArray.of(new float[][]{{1, 1}, {4, 4}});

        // 测试相等
        NdArray eq = a.eq(b);
        float[][] expectedEq = {{1, 0}, {0, 1}};
        assertArrayEquals(expectedEq, eq.getMatrix());

        // 测试大于
        assertTrue(a.isLar(NdArray.of(new float[][]{{0, 1}, {2, 3}})));
        assertFalse(a.isLar(NdArray.of(new float[][]{{2, 3}, {4, 5}})));
    }

    @Test
    public void testAddAt() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        NdArray b = NdArray.of(new float[][]{{10}, {20}});

        NdArray result = a.addAt(new int[]{0, 2}, new int[]{1, 1}, b);

        // 验证指定位置的值被正确添加
        float[][] matrix = result.getMatrix();
        assertEquals(12f, matrix[0][1], 1e-6); // 2 + 10
        assertEquals(28f, matrix[2][1], 1e-6); // 8 + 20
    }

    @Test
    public void testAddTo() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        NdArray b = NdArray.of(new float[][]{{10, 20}});

        a.addTo(1, 1, b);

        // 验证从指定位置开始的值被正确添加
        float[][] matrix = a.getMatrix();
        assertEquals(15f, matrix[1][1], 1e-6); // 5 + 10
        assertEquals(26f, matrix[1][2], 1e-6); // 6 + 20
    }

    @Test
    public void testSubNdArray() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        NdArray sub = a.subNdArray(1, 3, 1, 3);

        float[][] expectedSub = {{6, 7}, {10, 11}};
        assertArrayEquals(expectedSub, sub.getMatrix());
    }

    @Test(expected = RuntimeException.class)
    public void testInvalidMatrixDot() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}});
        NdArray b = NdArray.of(new float[][]{{1, 2}});
        a.dot(b); // 应该抛出异常，因为形状不匹配
    }

    @Test(expected = RuntimeException.class)
    public void testInvalidReshape() {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        a.reshape(Shape.of(2, 2)); // 应该抛出异常，因为大小不匹配
    }

    @Test
    public void testGettersAndSetters() {
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});

        // 测试getShape
        assertEquals(Shape.of(2, 2), a.getShape());

        // 测试通过getMatrix获取数据
        float[][] resultMatrix = a.getMatrix();
        float[][] expected = {{1, 2}, {3, 4}};
        assertArrayEquals(expected, resultMatrix);

        // 测试getMatrix
        float[][] matrix = a.getMatrix();
        float[][] expectedMatrix = {{1, 2}, {3, 4}};
        assertArrayEquals(expectedMatrix, matrix);

        // 测试getNumber（标量）
        NdArray scalar = NdArray.of(3.14f);
        assertEquals(3.14f, scalar.getNumber().floatValue(), 1e-6);
    }

    @Test
    public void testToString() {
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        String str = a.toString();
        assertNotNull(str);
        assertFalse(str.isEmpty());
    }
}