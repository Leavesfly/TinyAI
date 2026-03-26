package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArrayLogicalTest - 测试逻辑运算操作
 * 
 * 测试通过 NdArray 门面调用的 LogicalOperations 方法
 *
 * @author TinyAI
 */
public class NdArrayLogicalTest {

    @Test
    public void testNeg() {
        // neg：[1,-2,3]→[-1,2,-3]
        NdArray array = NdArray.of(new float[]{1, -2, 3});
        NdArray result = array.neg();
        float[] buf = result.getArray();

        assertEquals(-1.0f, buf[0], 1e-6);
        assertEquals(2.0f, buf[1], 1e-6);
        assertEquals(-3.0f, buf[2], 1e-6);
    }

    @Test
    public void testNegAllZeros() {
        // neg：全零数组neg
        NdArray array = NdArray.of(new float[]{0, 0, 0});
        NdArray result = array.neg();
        float[] buf = result.getArray();

        assertEquals(0.0f, buf[0], 1e-6);
        assertEquals(0.0f, buf[1], 1e-6);
        assertEquals(0.0f, buf[2], 1e-6);
    }

    @Test
    public void testAbs() {
        // abs：[-1,2,-3]→[1,2,3]
        NdArray array = NdArray.of(new float[]{-1, 2, -3});
        NdArray result = array.abs();
        float[] buf = result.getArray();

        assertEquals(1.0f, buf[0], 1e-6);
        assertEquals(2.0f, buf[1], 1e-6);
        assertEquals(3.0f, buf[2], 1e-6);
    }

    @Test
    public void testAbsAllPositive() {
        // abs：已全正数组abs
        NdArray array = NdArray.of(new float[]{1, 2, 3});
        NdArray result = array.abs();
        float[] buf = result.getArray();

        assertEquals(1.0f, buf[0], 1e-6);
        assertEquals(2.0f, buf[1], 1e-6);
        assertEquals(3.0f, buf[2], 1e-6);
    }

    @Test
    public void testEqEqual() {
        // eq：相等返回1.0
        NdArray array1 = NdArray.of(new float[]{1, 2, 3});
        NdArray array2 = NdArray.of(new float[]{1, 2, 3});
        NdArray result = array1.eq(array2);
        float[] buf = result.getArray();

        assertEquals(1.0f, buf[0], 1e-6);
        assertEquals(1.0f, buf[1], 1e-6);
        assertEquals(1.0f, buf[2], 1e-6);
    }

    @Test
    public void testEqNotEqual() {
        // eq：不等返回0.0
        NdArray array1 = NdArray.of(new float[]{1, 2, 3});
        NdArray array2 = NdArray.of(new float[]{4, 5, 6});
        NdArray result = array1.eq(array2);
        float[] buf = result.getArray();

        assertEquals(0.0f, buf[0], 1e-6);
        assertEquals(0.0f, buf[1], 1e-6);
        assertEquals(0.0f, buf[2], 1e-6);
    }

    @Test
    public void testEqMixed() {
        // eq：混合情况
        NdArray array1 = NdArray.of(new float[]{1, 2, 3});
        NdArray array2 = NdArray.of(new float[]{1, 5, 3});
        NdArray result = array1.eq(array2);
        float[] buf = result.getArray();

        assertEquals(1.0f, buf[0], 1e-6); // 1 == 1
        assertEquals(0.0f, buf[1], 1e-6); // 2 != 5
        assertEquals(1.0f, buf[2], 1e-6); // 3 == 3
    }

    @Test
    public void testGt() {
        // gt：大于返回1.0，不大于返回0.0
        NdArray array1 = NdArray.of(new float[]{5, 2, 7});
        NdArray array2 = NdArray.of(new float[]{3, 5, 7});
        NdArray result = array1.gt(array2);
        float[] buf = result.getArray();

        assertEquals(1.0f, buf[0], 1e-6); // 5 > 3
        assertEquals(0.0f, buf[1], 1e-6); // 2 <= 5
        assertEquals(0.0f, buf[2], 1e-6); // 7 <= 7
    }

    @Test
    public void testLt() {
        // lt：小于返回1.0，不小于返回0.0
        NdArray array1 = NdArray.of(new float[]{3, 5, 7});
        NdArray array2 = NdArray.of(new float[]{5, 2, 7});
        NdArray result = array1.lt(array2);
        float[] buf = result.getArray();

        assertEquals(1.0f, buf[0], 1e-6); // 3 < 5
        assertEquals(0.0f, buf[1], 1e-6); // 5 >= 2
        assertEquals(0.0f, buf[2], 1e-6); // 7 >= 7
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLogicalOperationShapeMismatch() {
        // 形状不匹配时抛 IllegalArgumentException
        NdArray array1 = NdArray.of(new float[]{1, 2, 3});
        NdArray array2 = NdArray.of(new float[]{1, 2, 3, 4});

        array1.eq(array2);
    }

    @Test
    public void testNegMatrix() {
        // neg：矩阵取反
        NdArray array = NdArray.of(new float[][]{
            {1, -2},
            {-3, 4}
        });
        NdArray result = array.neg();
        
        assertEquals(-1.0f, result.get(0, 0), 1e-6);
        assertEquals(2.0f, result.get(0, 1), 1e-6);
        assertEquals(3.0f, result.get(1, 0), 1e-6);
        assertEquals(-4.0f, result.get(1, 1), 1e-6);
    }

    @Test
    public void testAbsMatrix() {
        // abs：矩阵绝对值
        NdArray array = NdArray.of(new float[][]{
            {-1, 2},
            {-3, -4}
        });
        NdArray result = array.abs();
        
        assertEquals(1.0f, result.get(0, 0), 1e-6);
        assertEquals(2.0f, result.get(0, 1), 1e-6);
        assertEquals(3.0f, result.get(1, 0), 1e-6);
        assertEquals(4.0f, result.get(1, 1), 1e-6);
    }

    @Test
    public void testEqMatrix() {
        // eq：矩阵相等比较
        NdArray array1 = NdArray.of(new float[][]{
            {1, 2},
            {3, 4}
        });
        NdArray array2 = NdArray.of(new float[][]{
            {1, 5},
            {3, 4}
        });
        NdArray result = array1.eq(array2);
        
        assertEquals(1.0f, result.get(0, 0), 1e-6);
        assertEquals(0.0f, result.get(0, 1), 1e-6);
        assertEquals(1.0f, result.get(1, 0), 1e-6);
        assertEquals(1.0f, result.get(1, 1), 1e-6);
    }

    @Test
    public void testGtMatrix() {
        // gt：矩阵大于比较
        NdArray array1 = NdArray.of(new float[][]{
            {5, 2},
            {7, 1}
        });
        NdArray array2 = NdArray.of(new float[][]{
            {3, 5},
            {7, 0}
        });
        NdArray result = array1.gt(array2);
        
        assertEquals(1.0f, result.get(0, 0), 1e-6);
        assertEquals(0.0f, result.get(0, 1), 1e-6);
        assertEquals(0.0f, result.get(1, 0), 1e-6);
        assertEquals(1.0f, result.get(1, 1), 1e-6);
    }

    @Test
    public void testLtMatrix() {
        // lt：矩阵小于比较
        NdArray array1 = NdArray.of(new float[][]{
            {3, 5},
            {7, 0}
        });
        NdArray array2 = NdArray.of(new float[][]{
            {5, 2},
            {7, 1}
        });
        NdArray result = array1.lt(array2);
        
        assertEquals(1.0f, result.get(0, 0), 1e-6);
        assertEquals(0.0f, result.get(0, 1), 1e-6);
        assertEquals(0.0f, result.get(1, 0), 1e-6);
        assertEquals(1.0f, result.get(1, 1), 1e-6);
    }
}
