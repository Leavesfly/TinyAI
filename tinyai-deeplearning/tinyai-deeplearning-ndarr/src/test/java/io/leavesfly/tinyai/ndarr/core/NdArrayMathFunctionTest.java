package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray数学函数测试
 * 
 * 测试各种数学函数，包括：
 * - 基本数学函数（exp, log, sqrt, pow等）
 * - 三角函数（sin, cos, tan等）
 * - 激活函数（sigmoid, tanh, softmax等）
 * - 逻辑运算（abs, neg, mask等）
 *
 * @author TinyAI
 */
public class NdArrayMathFunctionTest {

    private static final float DELTA = 1e-5f;

    // =============================================================================
    // 基本数学函数测试
    // =============================================================================

    @Test
    public void testExp() {
        // 测试指数函数
        NdArray input = NdArray.of(new float[][]{{0f, 1f, 2f}, {-1f, -2f, 3f}});
        NdArray result = input.exp();
        
        assertEquals((float)Math.exp(0), result.get(0, 0), DELTA);
        assertEquals((float)Math.exp(1), result.get(0, 1), DELTA);
        assertEquals((float)Math.exp(2), result.get(0, 2), DELTA);
        assertEquals((float)Math.exp(-1), result.get(1, 0), DELTA);
        assertEquals((float)Math.exp(-2), result.get(1, 1), DELTA);
        assertEquals((float)Math.exp(3), result.get(1, 2), DELTA);
    }

    @Test
    public void testExpOfZero() {
        // e^0 = 1
        NdArray zeros = NdArray.zeros(Shape.of(2, 3));
        NdArray result = zeros.exp();
        
        float[][] expected = {{1f, 1f, 1f}, {1f, 1f, 1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testLog() {
        // 测试自然对数
        NdArray input = NdArray.of(new float[][]{{1f, (float)Math.E, 10f}, {100f, 1000f, 0.1f}});
        NdArray result = input.log();
        
        assertEquals(0f, result.get(0, 0), DELTA);
        assertEquals(1f, result.get(0, 1), DELTA);
        assertEquals((float)Math.log(10), result.get(0, 2), DELTA);
        assertEquals((float)Math.log(100), result.get(1, 0), DELTA);
        assertEquals((float)Math.log(1000), result.get(1, 1), DELTA);
        assertEquals((float)Math.log(0.1), result.get(1, 2), DELTA);
    }

    @Test(expected = ArithmeticException.class)
    public void testLogOfZero() {
        // log(0) 应该抛出异常
        NdArray zeros = NdArray.zeros(Shape.of(2, 2));
        zeros.log();
    }

    @Test(expected = ArithmeticException.class)
    public void testLogOfNegative() {
        // log(负数) 应该抛出异常
        NdArray negative = NdArray.of(new float[][]{{-1f, -2f}, {-3f, -4f}});
        negative.log();
    }

    @Test
    public void testSqrt() {
        // 测试平方根
        NdArray input = NdArray.of(new float[][]{{0f, 1f, 4f}, {9f, 16f, 25f}});
        NdArray result = input.sqrt();
        
        float[][] expected = {{0f, 1f, 2f}, {3f, 4f, 5f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSqrtOfLargeNumbers() {
        // 测试大数的平方根
        NdArray input = NdArray.of(new float[][]{{100f, 10000f}, {1000000f, 4f}});
        NdArray result = input.sqrt();
        
        assertEquals(10f, result.get(0, 0), DELTA);
        assertEquals(100f, result.get(0, 1), DELTA);
        assertEquals(1000f, result.get(1, 0), DELTA);
        assertEquals(2f, result.get(1, 1), DELTA);
    }

    @Test
    public void testPow() {
        // 测试幂运算
        NdArray input = NdArray.of(new float[][]{{2f, 3f, 4f}, {5f, 10f, 0f}});
        NdArray result = input.pow(2);
        
        float[][] expected = {{4f, 9f, 16f}, {25f, 100f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testPowWithFractionalExponent() {
        // 测试小数次幂（开方）
        NdArray input = NdArray.of(new float[][]{{4f, 9f, 16f}, {25f, 36f, 49f}});
        NdArray result = input.pow(0.5f);
        
        float[][] expected = {{2f, 3f, 4f}, {5f, 6f, 7f}};
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expected[i][j], result.getMatrix()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testPowWithZeroExponent() {
        // 任何数的0次方都是1
        NdArray input = NdArray.of(new float[][]{{2f, 3f, 100f}, {-5f, 0f, 1f}});
        NdArray result = input.pow(0);
        
        float[][] expected = {{1f, 1f, 1f}, {1f, 1f, 1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSquare() {
        // 测试平方
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = input.square();
        
        float[][] expected = {{1f, 4f, 9f}, {16f, 25f, 36f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    // =============================================================================
    // 三角函数测试
    // =============================================================================

    @Test
    public void testSin() {
        // 测试正弦函数
        float[] angles = {0f, (float)Math.PI/6, (float)Math.PI/4, (float)Math.PI/2};
        NdArray input = NdArray.of(angles);
        NdArray result = input.sin();
        
        assertEquals(0f, result.get(0, 0), DELTA);
        assertEquals(0.5f, result.get(0, 1), DELTA);
        assertEquals((float)Math.sqrt(2)/2, result.get(0, 2), DELTA);
        assertEquals(1f, result.get(0, 3), DELTA);
    }

    @Test
    public void testCos() {
        // 测试余弦函数
        float[] angles = {0f, (float)Math.PI/3, (float)Math.PI/2, (float)Math.PI};
        NdArray input = NdArray.of(angles);
        NdArray result = input.cos();
        
        assertEquals(1f, result.get(0, 0), DELTA);
        assertEquals(0.5f, result.get(0, 1), DELTA);
        assertEquals(0f, result.get(0, 2), DELTA);
        assertEquals(-1f, result.get(0, 3), DELTA);
    }

    @Test
    public void testSinCosRelation() {
        // 测试 sin^2 + cos^2 = 1
        NdArray angles = NdArray.of(new float[]{0f, 0.5f, 1f, 1.5f, 2f});
        NdArray sin = angles.sin();
        NdArray cos = angles.cos();
        
        NdArray sinSquared = sin.square();
        NdArray cosSquared = cos.square();
        NdArray sum = sinSquared.add(cosSquared);
        
        for (int i = 0; i < 5; i++) {
            assertEquals(1f, sum.get(0, i), DELTA);
        }
    }

    // =============================================================================
    // 激活函数测试
    // =============================================================================

    @Test
    public void testTanh() {
        // 测试双曲正切函数
        NdArray input = NdArray.of(new float[][]{{-2f, -1f, 0f}, {1f, 2f, 3f}});
        NdArray result = input.tanh();
        
        // tanh的值域是(-1, 1)
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertTrue(matrix[i][j] > -1f && matrix[i][j] < 1f);
            }
        }
        
        // tanh(0) = 0
        assertEquals(0f, result.get(0, 2), DELTA);
        
        // tanh是奇函数：tanh(-x) = -tanh(x)
        assertEquals(-result.get(1, 0), result.get(0, 1), DELTA);
        assertEquals(-result.get(1, 1), result.get(0, 0), DELTA);
    }

    @Test
    public void testSigmoid() {
        // 测试Sigmoid函数
        NdArray input = NdArray.of(new float[][]{{-10f, 0f, 10f}, {-1f, 1f, 2f}});
        NdArray result = input.sigmoid();
        
        // sigmoid(0) = 0.5
        assertEquals(0.5f, result.get(0, 1), DELTA);
        
        // sigmoid(-x) = 1 - sigmoid(x)
        assertEquals(1f - result.get(1, 1), result.get(1, 0), DELTA);
        
        // sigmoid的值域是(0, 1)
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertTrue(matrix[i][j] > 0f && matrix[i][j] < 1f);
            }
        }
        
        // 极端值测试
        assertTrue(result.get(0, 0) < 0.01f);  // sigmoid(-10) ≈ 0
        assertTrue(result.get(0, 2) > 0.99f);  // sigmoid(10) ≈ 1
    }

    @Test
    public void testSoftMax2D() {
        // 测试Softmax函数（2D）
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {1f, 2f, 3f}});
        NdArray result = input.softMax();
        
        // 每行的和应该为1
        for (int i = 0; i < 2; i++) {
            float rowSum = 0f;
            for (int j = 0; j < 3; j++) {
                float val = result.get(i, j);
                assertTrue(val > 0f && val < 1f);
                rowSum += val;
            }
            assertEquals(1f, rowSum, DELTA);
        }
    }

    @Test
    public void testSoftMaxNumericalStability() {
        // 测试数值稳定性（大数值输入）
        NdArray input = NdArray.of(new float[][]{{1000f, 1001f, 999f}});
        NdArray result = input.softMax();
        
        float sum = 0f;
        for (int i = 0; i < 3; i++) {
            float val = result.get(0, i);
            assertFalse(Float.isNaN(val));
            assertFalse(Float.isInfinite(val));
            sum += val;
        }
        assertEquals(1f, sum, DELTA);
    }

    // =============================================================================
    // 逻辑运算测试
    // =============================================================================

    @Test
    public void testAbs() {
        // 测试绝对值
        NdArray input = NdArray.of(new float[][]{{-1f, 2f, -3f}, {4f, -5f, 6f}});
        NdArray result = input.abs();
        
        float[][] expected = {{1f, 2f, 3f}, {4f, 5f, 6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testAbsOfPositive() {
        // 正数的绝对值是自身
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = input.abs();
        
        assertArrayEquals(input.getMatrix(), result.getMatrix());
    }

    @Test
    public void testNeg() {
        // 测试取反
        NdArray input = NdArray.of(new float[][]{{1f, -2f, 3f}, {-4f, 5f, -6f}});
        NdArray result = input.neg();
        
        float[][] expected = {{-1f, 2f, -3f}, {4f, -5f, 6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testDoubleNeg() {
        // 两次取反应该回到原值
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = input.neg().neg();
        
        assertArrayEquals(input.getArray(), result.getArray(), DELTA);
    }

    @Test
    public void testMaximum() {
        // 测试最大值截断
        NdArray input = NdArray.of(new float[][]{{-1f, 0f, 1f}, {2f, -3f, 4f}});
        NdArray result = input.maximum(0);
        
        float[][] expected = {{0f, 0f, 1f}, {2f, 0f, 4f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMaximumWithNegativeThreshold() {
        // 测试负数阈值
        NdArray input = NdArray.of(new float[][]{{-5f, -2f, 0f}, {1f, 3f, -1f}});
        NdArray result = input.maximum(-3);
        
        float[][] expected = {{-3f, -2f, 0f}, {1f, 3f, -1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMask() {
        // 测试掩码运算
        NdArray input = NdArray.of(new float[][]{{-1f, 0f, 1f}, {2f, -3f, 4f}});
        NdArray result = input.mask(0);
        
        float[][] expected = {{0f, 0f, 1f}, {1f, 0f, 1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMaskWithDifferentThreshold() {
        // 测试不同阈值的掩码
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = input.mask(3);
        
        float[][] expected = {{0f, 0f, 0f}, {1f, 1f, 1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testClip() {
        // 测试裁剪
        NdArray input = NdArray.of(new float[][]{{-5f, 0f, 5f}, {-10f, 3f, 10f}});
        NdArray result = input.clip(-3f, 6f);
        
        float[][] expected = {{-3f, 0f, 5f}, {-3f, 3f, 6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testClipWithSameMinMax() {
        // 测试相同的最小最大值
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = input.clip(2.5f, 2.5f);
        
        float[][] expected = {{2.5f, 2.5f}, {2.5f, 2.5f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testClipWithInvalidRange() {
        // 最小值大于最大值应该抛出异常
        NdArray input = NdArray.of(new float[][]{{1f, 2f}});
        input.clip(5f, 2f);
    }

    // =============================================================================
    // 比较运算测试
    // =============================================================================

    @Test
    public void testEq() {
        // 测试相等比较
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray b = NdArray.of(new float[][]{{1f, 0f, 3f}, {0f, 5f, 0f}});
        NdArray result = a.eq(b);
        
        float[][] expected = {{1f, 0f, 1f}, {0f, 1f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testGt() {
        // 测试大于比较
        NdArray a = NdArray.of(new float[][]{{1f, 5f, 3f}, {4f, 2f, 6f}});
        NdArray b = NdArray.of(new float[][]{{2f, 3f, 3f}, {3f, 5f, 6f}});
        NdArray result = a.gt(b);
        
        float[][] expected = {{0f, 1f, 0f}, {1f, 0f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testLt() {
        // 测试小于比较
        NdArray a = NdArray.of(new float[][]{{1f, 5f, 3f}, {4f, 2f, 6f}});
        NdArray b = NdArray.of(new float[][]{{2f, 3f, 3f}, {3f, 5f, 6f}});
        NdArray result = a.lt(b);
        
        float[][] expected = {{1f, 0f, 0f}, {0f, 1f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testIsLar() {
        // 测试所有元素都大于
        NdArray a = NdArray.of(new float[][]{{2f, 3f}, {4f, 5f}});
        NdArray b = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        assertTrue(a.isAllGreaterThan(b));
        assertFalse(b.isAllGreaterThan(a));
    }

    @Test
    public void testIsLarWithEqual() {
        // 有相等元素时应该返回false
        NdArray a = NdArray.of(new float[][]{{2f, 3f}, {4f, 5f}});
        NdArray b = NdArray.of(new float[][]{{1f, 3f}, {3f, 4f}});
        
        assertFalse(a.isAllGreaterThan(b));
    }
}
