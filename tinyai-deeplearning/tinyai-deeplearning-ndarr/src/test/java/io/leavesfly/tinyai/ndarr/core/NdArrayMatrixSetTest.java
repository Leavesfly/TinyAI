package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;
import static org.junit.Assert.*;

public class NdArrayMatrixSetTest {

    @Test
    public void testSetBlock() {
        // 3x4全零矩阵
        NdArray array = NdArray.zeros(Shape.of(3, 4));

        // 设置[1:3, 1:3]区域为[1,2,3,4]
        float[] blockData = new float[]{1, 2, 3, 4};
        array.setBlock(1, 3, 1, 3, blockData);

        // 验证结果
        float[] result = array.getArray();
        // row0: [0,0,0,0]
        assertEquals(0.0f, result[0], 0.001f);
        assertEquals(0.0f, result[1], 0.001f);
        assertEquals(0.0f, result[2], 0.001f);
        assertEquals(0.0f, result[3], 0.001f);
        // row1: [0,1,2,0]
        assertEquals(0.0f, result[4], 0.001f);
        assertEquals(1.0f, result[5], 0.001f);
        assertEquals(2.0f, result[6], 0.001f);
        assertEquals(0.0f, result[7], 0.001f);
        // row2: [0,3,4,0]
        assertEquals(0.0f, result[8], 0.001f);
        assertEquals(3.0f, result[9], 0.001f);
        assertEquals(4.0f, result[10], 0.001f);
        assertEquals(0.0f, result[11], 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetBlockDataLengthMismatch() {
        NdArray array = NdArray.zeros(Shape.of(3, 4));
        // 数据长度不匹配：区域大小为2x2=4，但提供3个数据
        array.setBlock(1, 3, 1, 3, new float[]{1, 2, 3});
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetBlockOutOfBounds() {
        NdArray array = NdArray.zeros(Shape.of(3, 4));
        // 边界超出：endRow=4超出行数3
        array.setBlock(1, 4, 1, 3, new float[]{1, 2, 3, 4});
    }

    @Test
    public void testSetRows() {
        // 3x4全零矩阵
        NdArray array = NdArray.zeros(Shape.of(3, 4));

        // 设置第0行和第2行
        int[] rowIndices = new int[]{0, 2};
        float[] rowData = new float[]{1, 2, 3, 4, 5, 6, 7, 8};
        array.setRows(rowIndices, rowData);

        // 验证结果
        float[] result = array.getArray();
        // row0: [1,2,3,4]
        assertEquals(1.0f, result[0], 0.001f);
        assertEquals(2.0f, result[1], 0.001f);
        assertEquals(3.0f, result[2], 0.001f);
        assertEquals(4.0f, result[3], 0.001f);
        // row1: [0,0,0,0] (未修改)
        assertEquals(0.0f, result[4], 0.001f);
        assertEquals(0.0f, result[5], 0.001f);
        assertEquals(0.0f, result[6], 0.001f);
        assertEquals(0.0f, result[7], 0.001f);
        // row2: [5,6,7,8]
        assertEquals(5.0f, result[8], 0.001f);
        assertEquals(6.0f, result[9], 0.001f);
        assertEquals(7.0f, result[10], 0.001f);
        assertEquals(8.0f, result[11], 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetRowsIndexOutOfBounds() {
        NdArray array = NdArray.zeros(Shape.of(3, 4));
        // 行索引超出范围：行索引3超出了0-2的范围
        array.setRows(new int[]{0, 3}, new float[]{1, 2, 3, 4, 5, 6, 7, 8});
    }

    @Test
    public void testSetCols() {
        // 3x4全零矩阵
        NdArray array = NdArray.zeros(Shape.of(3, 4));

        // 设置第1列和第3列
        int[] colIndices = new int[]{1, 3};
        float[] colData = new float[]{1, 2, 3, 4, 5, 6};
        array.setCols(colIndices, colData);

        // 验证结果
        float[] result = array.getArray();
        // row0: [0,1,0,2]
        assertEquals(0.0f, result[0], 0.001f);
        assertEquals(1.0f, result[1], 0.001f);
        assertEquals(0.0f, result[2], 0.001f);
        assertEquals(2.0f, result[3], 0.001f);
        // row1: [0,3,0,4]
        assertEquals(0.0f, result[4], 0.001f);
        assertEquals(3.0f, result[5], 0.001f);
        assertEquals(0.0f, result[6], 0.001f);
        assertEquals(4.0f, result[7], 0.001f);
        // row2: [0,5,0,6]
        assertEquals(0.0f, result[8], 0.001f);
        assertEquals(5.0f, result[9], 0.001f);
        assertEquals(0.0f, result[10], 0.001f);
        assertEquals(6.0f, result[11], 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetColsIndexOutOfBounds() {
        NdArray array = NdArray.zeros(Shape.of(3, 4));
        // 列索引超出范围：列索引4超出了0-3的范围
        array.setCols(new int[]{1, 4}, new float[]{1, 2, 3, 4, 5, 6});
    }
}
