package io.leavesfly.tinyai.ndarr;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArrayUtil类的单元测试
 *
 * @author TinyDL
 */
public class NdArrayUtilTest {

    private NdArray array2x3;
    private NdArray array2x3_2;
    private NdArray array1x3;
    private NdArray array3x2;

    @Before
    public void setUp() {
        array2x3 = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        array2x3_2 = NdArray.of(new float[][]{{7, 8, 9}, {10, 11, 12}});
        array1x3 = NdArray.of(new float[][]{{13, 14, 15}});
        array3x2 = NdArray.of(new float[][]{{1, 2}, {3, 4}, {5, 6}});
    }

    @Test
    public void testMergeAxis0_TwoMatrices() {
        // 测试按行合并两个2x3矩阵
        NdArray result = NdArrayUtil.merge(0, array2x3, array2x3_2);

        assertEquals(Shape.of(4, 3), result.getShape());

        float[][] expected = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMergeAxis0_MultipleMatrices() {
        // 测试按行合并多个矩阵
        NdArray result = NdArrayUtil.merge(0, array2x3, array1x3, array2x3_2);

        assertEquals(Shape.of(5, 3), result.getShape());

        float[][] expected = {
                {1, 2, 3},
                {4, 5, 6},
                {13, 14, 15},
                {7, 8, 9},
                {10, 11, 12}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMergeAxis1_TwoMatrices() {
        // 测试按列合并两个2x3矩阵
        NdArray result = NdArrayUtil.merge(1, array2x3, array2x3_2);

        assertEquals(Shape.of(2, 6), result.getShape());

        float[][] expected = {
                {1, 2, 3, 7, 8, 9},
                {4, 5, 6, 10, 11, 12}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMergeAxis1_DifferentShapes() {
        // 测试按列合并不同形状的矩阵
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        NdArray b = NdArray.of(new float[][]{{5, 6, 7}, {8, 9, 10}});

        NdArray result = NdArrayUtil.merge(1, a, b);

        assertEquals(Shape.of(2, 5), result.getShape());

        float[][] expected = {
                {1, 2, 5, 6, 7},
                {3, 4, 8, 9, 10}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMergeAxis1_MultipleMatrices() {
        // 测试按列合并多个矩阵
        NdArray a = NdArray.of(new float[][]{{1}, {2}});
        NdArray b = NdArray.of(new float[][]{{3, 4}, {5, 6}});
        NdArray c = NdArray.of(new float[][]{{7}, {8}});

        NdArray result = NdArrayUtil.merge(1, a, b, c);

        assertEquals(Shape.of(2, 4), result.getShape());

        float[][] expected = {
                {1, 3, 4, 7},
                {2, 5, 6, 8}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMergeAxis0_SingleMatrix() {
        // 测试合并单个矩阵
        NdArray result = NdArrayUtil.merge(0, array2x3);

        assertEquals(array2x3.getShape(), result.getShape());
        assertArrayEquals(array2x3.getMatrix(), result.getMatrix());
    }

    @Test
    public void testMergeAxis1_SingleMatrix() {
        // 测试合并单个矩阵
        NdArray result = NdArrayUtil.merge(1, array2x3);

        assertEquals(array2x3.getShape(), result.getShape());
        assertArrayEquals(array2x3.getMatrix(), result.getMatrix());
    }

    @Test
    public void testMergeAxis0_VectorMatrices() {
        // 测试合并向量矩阵
        NdArray v1 = NdArray.of(new float[][]{{1, 2, 3}});
        NdArray v2 = NdArray.of(new float[][]{{4, 5, 6}});
        NdArray v3 = NdArray.of(new float[][]{{7, 8, 9}});

        NdArray result = NdArrayUtil.merge(0, v1, v2, v3);

        assertEquals(Shape.of(3, 3), result.getShape());

        float[][] expected = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMergeAxis1_VectorMatrices() {
        // 测试按列合并向量矩阵
        NdArray v1 = NdArray.of(new float[][]{{1, 2}});
        NdArray v2 = NdArray.of(new float[][]{{3, 4}});
        NdArray v3 = NdArray.of(new float[][]{{5, 6}});

        NdArray result = NdArrayUtil.merge(1, v1, v2, v3);

        assertEquals(Shape.of(1, 6), result.getShape());

        float[][] expected = {{1, 2, 3, 4, 5, 6}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMerge3DArraysAxis0() {
        // 测试三维数组按axis=0合并
        NdArray a1 = NdArray.of(Shape.of(2, 2, 2));
        NdArray a2 = NdArray.of(Shape.of(1, 2, 2));

        // 手动设置一些值进行测试
        a1.set(1f, 0, 0, 0);
        a1.set(2f, 0, 0, 1);
        a1.set(3f, 0, 1, 0);
        a1.set(4f, 0, 1, 1);
        a1.set(5f, 1, 0, 0);
        a1.set(6f, 1, 0, 1);
        a1.set(7f, 1, 1, 0);
        a1.set(8f, 1, 1, 1);

        a2.set(9f, 0, 0, 0);
        a2.set(10f, 0, 0, 1);
        a2.set(11f, 0, 1, 0);
        a2.set(12f, 0, 1, 1);

        NdArray result = NdArrayUtil.merge(0, a1, a2);

        assertEquals(Shape.of(3, 2, 2), result.getShape());

        // 验证合并后的值
        assertEquals(1f, result.get(0, 0, 0), 1e-6);
        assertEquals(2f, result.get(0, 0, 1), 1e-6);
        assertEquals(5f, result.get(1, 0, 0), 1e-6);
        assertEquals(9f, result.get(2, 0, 0), 1e-6);
        assertEquals(12f, result.get(2, 1, 1), 1e-6);
    }

    @Test
    public void testMerge3DArraysAxis1() {
        // 测试三维数组按axis=1合并
        NdArray a1 = NdArray.of(Shape.of(2, 2, 3));
        NdArray a2 = NdArray.of(Shape.of(2, 1, 3));

        // 设置一些测试值
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    a1.set(i * 6 + j * 3 + k + 1, i, j, k);
                }
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int k = 0; k < 3; k++) {
                a2.set(100 + i * 3 + k, i, 0, k);
            }
        }

        NdArray result = NdArrayUtil.merge(1, a1, a2);

        assertEquals(Shape.of(2, 3, 3), result.getShape());

        // 验证一些关键值
        assertEquals(1f, result.get(0, 0, 0), 1e-6);
        assertEquals(100f, result.get(0, 2, 0), 1e-6);
        assertEquals(103f, result.get(1, 2, 0), 1e-6);
    }

    @Test(expected = RuntimeException.class)
    public void testMergeInvalidAxis() {
        // 测试无效的axis值
        NdArrayUtil.merge(2, array2x3, array2x3_2);
    }

    @Test
    public void testMergeEmptyArrays() {
        // 测试空数组情况（虽然实际使用中可能不会遇到）
        NdArray[] emptyArrays = {};
        try {
            NdArray result = NdArrayUtil.merge(0, emptyArrays);
            // 如果没有抛出异常，验证结果
            assertNotNull(result);
        } catch (IllegalArgumentException e) {
            // 这是预期的行为，因为我们在merge方法中验证参数
            // 这个测试主要是确保我们了解当前的行为
            assertEquals("至少需要一个NdArray进行合并", e.getMessage());
        }
    }

    @Test
    public void testMergeIdenticalShapes() {
        // 测试形状完全相同的数组合并
        NdArray a = NdArray.ones(Shape.of(3, 3));
        NdArray b = NdArray.like(Shape.of(3, 3), 2);
        NdArray c = NdArray.like(Shape.of(3, 3), 3);

        // 按行合并
        NdArray resultAxis0 = NdArrayUtil.merge(0, a, b, c);
        assertEquals(Shape.of(9, 3), resultAxis0.getShape());

        // 按列合并
        NdArray resultAxis1 = NdArrayUtil.merge(1, a, b, c);
        assertEquals(Shape.of(3, 9), resultAxis1.getShape());

        // 验证axis0合并的值
        float[][] matrix0 = resultAxis0.getMatrix();
        // 前3行应该都是1
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(1f, matrix0[i][j], 1e-6);
            }
        }
        // 中间3行应该都是2
        for (int i = 3; i < 6; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(2f, matrix0[i][j], 1e-6);
            }
        }
        // 最后3行应该都是3
        for (int i = 6; i < 9; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(3f, matrix0[i][j], 1e-6);
            }
        }

        // 验证axis1合并的值
        float[][] matrix1 = resultAxis1.getMatrix();
        for (int i = 0; i < 3; i++) {
            // 前3列应该是1
            for (int j = 0; j < 3; j++) {
                assertEquals(1f, matrix1[i][j], 1e-6);
            }
            // 中间3列应该是2
            for (int j = 3; j < 6; j++) {
                assertEquals(2f, matrix1[i][j], 1e-6);
            }
            // 最后3列应该是3
            for (int j = 6; j < 9; j++) {
                assertEquals(3f, matrix1[i][j], 1e-6);
            }
        }
    }

    @Test
    public void testMergeLargeArrays() {
        // 测试较大数组的合并性能和正确性
        NdArray large1 = NdArray.like(Shape.of(100, 50), 1);
        NdArray large2 = NdArray.like(Shape.of(100, 50), 2);

        NdArray resultAxis0 = NdArrayUtil.merge(0, large1, large2);
        assertEquals(Shape.of(200, 50), resultAxis0.getShape());

        NdArray resultAxis1 = NdArrayUtil.merge(1, large1, large2);
        assertEquals(Shape.of(100, 100), resultAxis1.getShape());

        // 验证一些关键位置的值
        assertEquals(1f, resultAxis0.get(0, 0), 1e-6);
        assertEquals(1f, resultAxis0.get(99, 49), 1e-6);
        assertEquals(2f, resultAxis0.get(100, 0), 1e-6);
        assertEquals(2f, resultAxis0.get(199, 49), 1e-6);

        assertEquals(1f, resultAxis1.get(0, 0), 1e-6);
        assertEquals(1f, resultAxis1.get(99, 49), 1e-6);
        assertEquals(2f, resultAxis1.get(0, 50), 1e-6);
        assertEquals(2f, resultAxis1.get(99, 99), 1e-6);
    }
}