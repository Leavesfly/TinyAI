package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;
import static org.junit.Assert.*;

public class NdArrayAxisOperationsTest {

    @Test
    public void testArgMaxAxis1() {
        // 2x3矩阵 [[1,3,2],[4,1,5]]
        NdArray array = NdArray.of(new float[]{1, 3, 2, 4, 1, 5}, Shape.of(2, 3));

        // axis=1（最后一轴），每行最大值索引→[1,2]
        NdArray result = array.argMax(1);
        float[] buf = result.getArray();
        assertEquals(1.0f, buf[0], 0.001f);
        assertEquals(2.0f, buf[1], 0.001f);
    }

    @Test
    public void testArgMaxAxis0() {
        // 2x3矩阵 [[1,3,2],[4,1,5]]
        NdArray array = NdArray.of(new float[]{1, 3, 2, 4, 1, 5}, Shape.of(2, 3));

        // axis=0（倒数第二轴），每列最大值索引→[1,0,1]
        NdArray result = array.argMax(0);
        float[] buf = result.getArray();
        assertEquals(1.0f, buf[0], 0.001f);
        assertEquals(0.0f, buf[1], 0.001f);
        assertEquals(1.0f, buf[2], 0.001f);
    }

    @Test
    public void testMaxAxis1() {
        // 2x3矩阵 [[1,3,2],[4,1,5]]
        NdArray array = NdArray.of(new float[]{1, 3, 2, 4, 1, 5}, Shape.of(2, 3));

        // axis=1，每行最大值→[3,5]
        NdArray result = array.max(1);
        float[] buf = result.getArray();
        assertEquals(3.0f, buf[0], 0.001f);
        assertEquals(5.0f, buf[1], 0.001f);
    }

    @Test
    public void testMaxAxis0() {
        // 2x3矩阵 [[1,3,2],[4,1,5]]
        NdArray array = NdArray.of(new float[]{1, 3, 2, 4, 1, 5}, Shape.of(2, 3));

        // axis=0，每列最大值→[4,3,5]
        NdArray result = array.max(0);
        float[] buf = result.getArray();
        assertEquals(4.0f, buf[0], 0.001f);
        assertEquals(3.0f, buf[1], 0.001f);
        assertEquals(5.0f, buf[2], 0.001f);
    }

    @Test
    public void testMinAxis1() {
        // 2x3矩阵 [[1,3,2],[4,1,5]]
        NdArray array = NdArray.of(new float[]{1, 3, 2, 4, 1, 5}, Shape.of(2, 3));

        // axis=1，每行最小值→[1,1]
        NdArray result = array.min(1);
        float[] buf = result.getArray();
        assertEquals(1.0f, buf[0], 0.001f);
        assertEquals(1.0f, buf[1], 0.001f);
    }

    @Test
    public void testMinAxis0() {
        // 2x3矩阵 [[1,3,2],[4,1,5]]
        NdArray array = NdArray.of(new float[]{1, 3, 2, 4, 1, 5}, Shape.of(2, 3));

        // axis=0，每列最小值→[1,1,2]
        NdArray result = array.min(0);
        float[] buf = result.getArray();
        assertEquals(1.0f, buf[0], 0.001f);
        assertEquals(1.0f, buf[1], 0.001f);
        assertEquals(2.0f, buf[2], 0.001f);
    }

    @Test
    public void test3DTensorAxisOperations() {
        // 2x2x3张量
        float[] data = new float[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        };
        NdArray array = NdArray.of(data, Shape.of(2, 2, 3));

        // axis=2（最后一轴），每行最大值
        NdArray resultMaxAxis2 = array.max(2);
        float[] buf2 = resultMaxAxis2.getArray();
        assertEquals(3.0f, buf2[0], 0.001f);
        assertEquals(6.0f, buf2[1], 0.001f);
        assertEquals(9.0f, buf2[2], 0.001f);
        assertEquals(12.0f, buf2[3], 0.001f);

        // axis=1（倒数第二轴）
        NdArray resultMaxAxis1 = array.max(1);
        float[] buf1 = resultMaxAxis1.getArray();
        assertEquals(4.0f, buf1[0], 0.001f);
        assertEquals(5.0f, buf1[1], 0.001f);
        assertEquals(6.0f, buf1[2], 0.001f);
        assertEquals(10.0f, buf1[3], 0.001f);
        assertEquals(11.0f, buf1[4], 0.001f);
        assertEquals(12.0f, buf1[5], 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testUnsupportedAxis() {
        // 2x3矩阵
        NdArray array = NdArray.of(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3));

        // 尝试使用不支持的轴（axis=2，超出了最后两个轴的范围）
        array.max(2);
    }
}
