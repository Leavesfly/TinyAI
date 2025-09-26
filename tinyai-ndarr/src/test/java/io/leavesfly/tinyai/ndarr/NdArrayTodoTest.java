package io.leavesfly.tinyai.ndarr;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 专门测试NdArray类中新完善的TODO部分功能
 * 
 * @author TinyDL
 */
public class NdArrayTodoTest {
    
    private NdArray testMatrix;
    
    @Before
    public void setUp() {
        // 创建一个测试矩阵
        testMatrix = new NdArray(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
    }
    
    /**
     * 测试max(int axis)方法当axis=0时的实现
     */
    @Test
    public void testMaxAxis0() {
        // 测试按列查找最大值（axis=0表示按行方向查找每列的最大值）
        NdArray maxAxis0 = testMatrix.max(0);
        
        // 验证结果形状
        assertEquals(new Shape(1, 3), maxAxis0.getShape());
        
        // 验证结果值：每列的最大值应该是 [7, 8, 9]
        float[][] expected = {{7, 8, 9}};
        assertArrayEquals(expected, maxAxis0.getMatrix());
        
        // 打印结果以便调试
        System.out.println("max(axis=0) 结果: " + java.util.Arrays.deepToString(maxAxis0.getMatrix()));
    }
    
    /**
     * 测试max(int axis)方法当axis=1时的实现（确保原有功能仍然正常）
     */
    @Test
    public void testMaxAxis1() {
        // 测试按行查找最大值（axis=1表示按列方向查找每行的最大值）
        NdArray maxAxis1 = testMatrix.max(1);
        
        // 验证结果形状
        assertEquals(new Shape(3, 1), maxAxis1.getShape());
        
        // 验证结果值：每行的最大值应该是 [[3], [6], [9]]
        float[][] expected = {{3}, {6}, {9}};
        assertArrayEquals(expected, maxAxis1.getMatrix());
        
        // 打印结果以便调试
        System.out.println("max(axis=1) 结果: " + java.util.Arrays.deepToString(maxAxis1.getMatrix()));
    }
    
    /**
     * 测试min(int axis)方法当axis=0时的实现
     */
    @Test
    public void testMinAxis0() {
        // 测试按列查找最小值（axis=0表示按行方向查找每列的最小值）
        NdArray minAxis0 = testMatrix.min(0);
        
        // 验证结果形状
        assertEquals(new Shape(1, 3), minAxis0.getShape());
        
        // 验证结果值：每列的最小值应该是 [1, 2, 3]
        float[][] expected = {{1, 2, 3}};
        assertArrayEquals(expected, minAxis0.getMatrix());
        
        // 打印结果以便调试
        System.out.println("min(axis=0) 结果: " + java.util.Arrays.deepToString(minAxis0.getMatrix()));
    }
    
    /**
     * 测试min(int axis)方法当axis=1时的实现（确保原有功能仍然正常）
     */
    @Test
    public void testMinAxis1() {
        // 测试按行查找最小值（axis=1表示按列方向查找每行的最小值）
        NdArray minAxis1 = testMatrix.min(1);
        
        // 验证结果形状
        assertEquals(new Shape(3, 1), minAxis1.getShape());
        
        // 验证结果值：每行的最小值应该是 [[1], [4], [7]]
        float[][] expected = {{1}, {4}, {7}};
        assertArrayEquals(expected, minAxis1.getMatrix());
        
        // 打印结果以便调试
        System.out.println("min(axis=1) 结果: " + java.util.Arrays.deepToString(minAxis1.getMatrix()));
    }
    
    /**
     * 测试max方法在不同数据分布下的正确性
     */
    @Test
    public void testMaxWithDifferentData() {
        // 创建一个混合数据的矩阵
        NdArray mixedMatrix = new NdArray(new float[][]{
            {-1, 0, 5},
            {2, -3, 1},
            {10, 8, -2}
        });
        
        // 测试axis=0（按列最大值）
        NdArray maxAxis0 = mixedMatrix.max(0);
        float[][] expectedAxis0 = {{10, 8, 5}};
        assertArrayEquals(expectedAxis0, maxAxis0.getMatrix());
        
        // 测试axis=1（按行最大值）
        NdArray maxAxis1 = mixedMatrix.max(1);
        float[][] expectedAxis1 = {{5}, {2}, {10}};
        assertArrayEquals(expectedAxis1, maxAxis1.getMatrix());
    }
    
    /**
     * 测试min方法在不同数据分布下的正确性
     */
    @Test
    public void testMinWithDifferentData() {
        // 创建一个混合数据的矩阵
        NdArray mixedMatrix = new NdArray(new float[][]{
            {-1, 0, 5},
            {2, -3, 1},
            {10, 8, -2}
        });
        
        // 测试axis=0（按列最小值）
        NdArray minAxis0 = mixedMatrix.min(0);
        float[][] expectedAxis0 = {{-1, -3, -2}};
        assertArrayEquals(expectedAxis0, minAxis0.getMatrix());
        
        // 测试axis=1（按行最小值）
        NdArray minAxis1 = mixedMatrix.min(1);
        float[][] expectedAxis1 = {{-1}, {-3}, {-2}};
        assertArrayEquals(expectedAxis1, minAxis1.getMatrix());
    }
    
    /**
     * 测试异常处理改进 - 无效轴参数
     */
    @Test
    public void testInvalidAxisException() {
        try {
            testMatrix.max(2); // 无效的轴参数
            fail("应该抛出IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("不支持的轴参数"));
            System.out.println("正确捕获异常: " + e.getMessage());
        }
        
        try {
            testMatrix.min(-1); // 无效的轴参数
            fail("应该抛出IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("不支持的轴参数"));
            System.out.println("正确捕获异常: " + e.getMessage());
        }
    }
    
    /**
     * 测试改进的异常处理 - 矩阵乘法维度不匹配
     */
    @Test
    public void testImprovedDotException() {
        NdArray a = new NdArray(new float[][]{{1, 2, 3}});
        NdArray b = new NdArray(new float[][]{{1, 2}});
        
        try {
            a.dot(b);
            fail("应该抛出IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("矩阵乘法维度不匹配"));
            System.out.println("正确捕获异常: " + e.getMessage());
        }
    }
    
    /**
     * 测试边界情况 - 单行矩阵
     */
    @Test
    public void testSingleRowMatrix() {
        NdArray singleRow = new NdArray(new float[][]{{1, 2, 3, 4}});
        
        // axis=0应该返回原行
        NdArray maxAxis0 = singleRow.max(0);
        float[][] expectedAxis0 = {{1, 2, 3, 4}};
        assertArrayEquals(expectedAxis0, maxAxis0.getMatrix());
        
        // axis=1应该返回行最大值
        NdArray maxAxis1 = singleRow.max(1);
        float[][] expectedAxis1 = {{4}};
        assertArrayEquals(expectedAxis1, maxAxis1.getMatrix());
    }
    
    /**
     * 测试边界情况 - 单列矩阵
     */
    @Test
    public void testSingleColumnMatrix() {
        NdArray singleCol = new NdArray(new float[][]{{1}, {2}, {3}, {4}});
        
        // axis=0应该返回列最大值
        NdArray maxAxis0 = singleCol.max(0);
        float[][] expectedAxis0 = {{4}};
        assertArrayEquals(expectedAxis0, maxAxis0.getMatrix());
        
        // axis=1应该返回每行值
        NdArray maxAxis1 = singleCol.max(1);
        float[][] expectedAxis1 = {{1}, {2}, {3}, {4}};
        assertArrayEquals(expectedAxis1, maxAxis1.getMatrix());
    }
    
    /**
     * 综合测试 - 验证所有完善的功能
     */
    @Test
    public void testOverallImprovement() {
        System.out.println("=== NdArray TODO功能完善测试 ===");
        
        // 创建测试数据
        NdArray testData = new NdArray(new float[][]{
            {1.5f, 2.5f, 3.5f},
            {4.5f, 5.5f, 6.5f}
        });
        
        System.out.println("测试数据:");
        System.out.println(java.util.Arrays.deepToString(testData.getMatrix()));
        
        // 测试各种操作
        System.out.println("axis=0 最大值: " + java.util.Arrays.deepToString(testData.max(0).getMatrix()));
        System.out.println("axis=0 最小值: " + java.util.Arrays.deepToString(testData.min(0).getMatrix()));
        System.out.println("axis=1 最大值: " + java.util.Arrays.deepToString(testData.max(1).getMatrix()));
        System.out.println("axis=1 最小值: " + java.util.Arrays.deepToString(testData.min(1).getMatrix()));
        
        // 验证结果正确性
        assertArrayEquals(new float[][]{{4.5f, 5.5f, 6.5f}}, testData.max(0).getMatrix());
        assertArrayEquals(new float[][]{{1.5f, 2.5f, 3.5f}}, testData.min(0).getMatrix());
        assertArrayEquals(new float[][]{{3.5f}, {6.5f}}, testData.max(1).getMatrix());
        assertArrayEquals(new float[][]{{1.5f}, {4.5f}}, testData.min(1).getMatrix());
        
        System.out.println("=== 所有测试通过! ===");
    }
}