package io.leavesfly.tinyai.ndarr;

import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * ShapeCpu类的单元测试
 *
 * @author TinyDL
 */
public class ShapeCpuTest {

    private ShapeCpu shape1D;
    private ShapeCpu shape2D;
    private ShapeCpu shape3D;
    private ShapeCpu shape4D;

    @Before
    public void setUp() {
        shape1D = ShapeCpu.of(5);
        shape2D = ShapeCpu.of(3, 4);
        shape3D = ShapeCpu.of(2, 3, 4);
        shape4D = ShapeCpu.of(2, 3, 4, 5);
    }

    @Test
    public void testConstructors() {
        // 测试一维构造器
        ShapeCpu s1 = ShapeCpu.of(10);
        assertEquals(1, s1.dimension.length);
        assertEquals(10, s1.dimension[0]);

        // 测试二维构造器
        ShapeCpu s2 = ShapeCpu.of(5, 6);
        assertEquals(2, s2.dimension.length);
        assertEquals(5, s2.dimension[0]);
        assertEquals(6, s2.dimension[1]);

        // 测试多维构造器
        ShapeCpu s3 = ShapeCpu.of(2, 3, 4, 5);
        assertEquals(4, s3.dimension.length);
        assertEquals(2, s3.dimension[0]);
        assertEquals(3, s3.dimension[1]);
        assertEquals(4, s3.dimension[2]);
        assertEquals(5, s3.dimension[3]);

        // 测试从数组构造器
        int[] dims = {3, 4, 5};
        ShapeCpu s4 = ShapeCpu.of(dims);
        assertEquals(3, s4.dimension.length);
        assertArrayEquals(dims, s4.dimension);
    }

    @Test
    public void testSize() {
        // 测试一维
        assertEquals(5, shape1D.size());

        // 测试二维
        assertEquals(12, shape2D.size()); // 3 * 4

        // 测试三维
        assertEquals(24, shape3D.size()); // 2 * 3 * 4

        // 测试四维
        assertEquals(120, shape4D.size()); // 2 * 3 * 4 * 5
    }

    @Test
    public void testIsMatrix() {
        // 测试一维不是矩阵
        assertFalse(shape1D.isMatrix());

        // 测试二维是矩阵
        assertTrue(shape2D.isMatrix());

        // 测试三维不是矩阵
        assertFalse(shape3D.isMatrix());

        // 测试四维不是矩阵
        assertFalse(shape4D.isMatrix());
    }

    @Test
    public void testMatrixDimensions() {
        // 只有二维shape才有row和column
        assertEquals(3, shape2D.getRow());
        assertEquals(4, shape2D.getColumn());
    }

    @Test(expected = RuntimeException.class)
    public void testGetRowOnNonMatrix() {
        // 非矩阵形状调用getRow应该抛出异常
        shape1D.getRow();
    }

    @Test(expected = RuntimeException.class)
    public void testGetColumnOnNonMatrix() {
        // 非矩阵形状调用getColumn应该抛出异常
        shape1D.getColumn();
    }

    @Test
    public void testGetIndex() {
        // 测试一维索引
        assertEquals(0, shape1D.getIndex(0));
        assertEquals(2, shape1D.getIndex(2));
        assertEquals(4, shape1D.getIndex(4));

        // 测试二维索引
        assertEquals(0, shape2D.getIndex(0, 0)); // [0,0]
        assertEquals(1, shape2D.getIndex(0, 1)); // [0,1]
        assertEquals(4, shape2D.getIndex(1, 0)); // [1,0]
        assertEquals(5, shape2D.getIndex(1, 1)); // [1,1]
        assertEquals(11, shape2D.getIndex(2, 3)); // [2,3]

        // 测试三维索引
        assertEquals(0, shape3D.getIndex(0, 0, 0)); // [0,0,0]
        assertEquals(1, shape3D.getIndex(0, 0, 1)); // [0,0,1]
        assertEquals(4, shape3D.getIndex(0, 1, 0)); // [0,1,0]
        assertEquals(12, shape3D.getIndex(1, 0, 0)); // [1,0,0]
        assertEquals(23, shape3D.getIndex(1, 2, 3)); // [1,2,3]
    }

    @Test
    public void testEquals() {
        // 测试相等
        ShapeCpu s1 = ShapeCpu.of(3, 4);
        ShapeCpu s2 = ShapeCpu.of(3, 4);
        assertTrue(s1.equals(s2));
        assertTrue(s2.equals(s1));
        assertEquals(s1.hashCode(), s2.hashCode());

        // 测试不相等
        ShapeCpu s3 = ShapeCpu.of(3, 5);
        assertFalse(s1.equals(s3));
        assertFalse(s3.equals(s1));

        ShapeCpu s4 = ShapeCpu.of(4, 3);
        assertFalse(s1.equals(s4));

        ShapeCpu s5 = ShapeCpu.of(3, 4, 5);
        assertFalse(s1.equals(s5));

        // 测试与null和其他类型比较
        assertFalse(s1.equals(null));
        assertFalse(s1.equals("not a shape"));

        // 测试自身相等
        assertTrue(s1.equals(s1));
    }

    @Test
    public void testHashCode() {
        // 相等的ShapeCpu应该有相同的hashCode
        ShapeCpu s1 = ShapeCpu.of(3, 4);
        ShapeCpu s2 = ShapeCpu.of(3, 4);
        assertEquals(s1.hashCode(), s2.hashCode());

        // 不同的ShapeCpu通常应该有不同的hashCode（虽然不保证）
        ShapeCpu s3 = ShapeCpu.of(3, 5);
        // 注意：这里我们不能保证hashCode一定不同，只是通常情况下
        // assertNotEquals(s1.hashCode(), s3.hashCode());
    }

    @Test
    public void testToString() {
        // 测试toString方法
        assertEquals("[5]", shape1D.toString());
        assertEquals("[3,4]", shape2D.toString());
        assertEquals("[2,3,4]", shape3D.toString());
        assertEquals("[2,3,4,5]", shape4D.toString());

        // 测试空字符串不会出现
        assertFalse(shape1D.toString().isEmpty());
        assertFalse(shape2D.toString().isEmpty());

        // 测试格式正确
        assertTrue(shape2D.toString().startsWith("["));
        assertTrue(shape2D.toString().endsWith("]"));
        assertTrue(shape2D.toString().contains(","));
    }

    @Test
    public void testMultipliers() {
        // 验证multipliers是否正确计算（通过getIndex方法间接验证）

        // 对于shape (2,3,4)，multipliers应该是[12, 4, 1]
        // 验证几个关键位置的索引计算
        assertEquals(0, shape3D.getIndex(0, 0, 0));   // 0*12 + 0*4 + 0*1 = 0
        assertEquals(12, shape3D.getIndex(1, 0, 0));  // 1*12 + 0*4 + 0*1 = 12
        assertEquals(4, shape3D.getIndex(0, 1, 0));   // 0*12 + 1*4 + 0*1 = 4
        assertEquals(1, shape3D.getIndex(0, 0, 1));   // 0*12 + 0*4 + 1*1 = 1
        assertEquals(17, shape3D.getIndex(1, 1, 1));  // 1*12 + 1*4 + 1*1 = 17
    }

    @Test
    public void testBoundaryConditions() {
        // 测试边界条件
        ShapeCpu tiny = ShapeCpu.of(1);
        assertEquals(1, tiny.size());
        assertEquals("[1]", tiny.toString());

        ShapeCpu single = ShapeCpu.of(1, 1);
        assertTrue(single.isMatrix());
        assertEquals(1, single.getRow());
        assertEquals(1, single.getColumn());
        assertEquals(1, single.size());

        // 测试大尺寸
        ShapeCpu large = ShapeCpu.of(100, 100);
        assertTrue(large.isMatrix());
        assertEquals(100, large.getRow());
        assertEquals(100, large.getColumn());
        assertEquals(10000, large.size());
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testInvalidIndexAccess() {
        // 测试越界访问
        shape2D.getIndex(0, 1, 2); // 超出维度
    }

    @Test
    public void testZeroDimension() {
        // 测试包含0的维度
        ShapeCpu zeroShapeCpu = ShapeCpu.of(0, 5);
        assertEquals(0, zeroShapeCpu.size());
        assertTrue(zeroShapeCpu.isMatrix());
        assertEquals(0, zeroShapeCpu.getRow());
        assertEquals(5, zeroShapeCpu.getColumn());
    }

    @Test
    public void testCopyDimensions() {
        // 验证dimension数组是独立的副本
        int[] originalDims = {2, 3, 4};
        ShapeCpu shape = ShapeCpu.of(originalDims);

        // 修改原始数组不应该影响ShapeCpu
        originalDims[0] = 999;
        assertEquals(2, shape.dimension[0]); // 应该仍然是2，不是999

        // 修改ShapeCpu的dimension也不应该影响外部
        shape.dimension[1] = 888;
        assertEquals(888, shape.dimension[1]); // ShapeCpu内部已修改
        // 但这是直接修改的，实际使用中应该避免
    }
}