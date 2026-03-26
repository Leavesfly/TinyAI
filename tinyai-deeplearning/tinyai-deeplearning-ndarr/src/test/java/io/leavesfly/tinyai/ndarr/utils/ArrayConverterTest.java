package io.leavesfly.tinyai.ndarr.utils;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayConverter;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * ArrayConverter 类的单元测试
 */
public class ArrayConverterTest {

    @Test
    public void testToMatrix_2x3Matrix() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        NdArrayCpu ndArray = new NdArrayCpu(data, new ShapeCpu(2, 3));
        
        float[][] matrix = ArrayConverter.toMatrix(ndArray);
        
        assertEquals(2, matrix.length);
        assertEquals(3, matrix[0].length);
        assertEquals(1.0f, matrix[0][0], 0.001f);
        assertEquals(2.0f, matrix[0][1], 0.001f);
        assertEquals(3.0f, matrix[0][2], 0.001f);
        assertEquals(4.0f, matrix[1][0], 0.001f);
        assertEquals(5.0f, matrix[1][1], 0.001f);
        assertEquals(6.0f, matrix[1][2], 0.001f);
    }

    @Test
    public void testToMatrix_1DVector() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        NdArrayCpu ndArray = new NdArrayCpu(data, new ShapeCpu(5));
        
        float[][] matrix = ArrayConverter.toMatrix(ndArray);
        
        assertEquals(1, matrix.length);
        assertEquals(5, matrix[0].length);
        assertEquals(1.0f, matrix[0][0], 0.001f);
        assertEquals(2.0f, matrix[0][1], 0.001f);
        assertEquals(3.0f, matrix[0][2], 0.001f);
        assertEquals(4.0f, matrix[0][3], 0.001f);
        assertEquals(5.0f, matrix[0][4], 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testToMatrix_3DArrayThrowsException() {
        float[] data = new float[24];
        NdArrayCpu ndArray = new NdArrayCpu(data, new ShapeCpu(2, 3, 4));
        
        ArrayConverter.toMatrix(ndArray);
    }

    @Test
    public void testTo3dArray_2x3x4Tensor() {
        float[] data = new float[24];
        for (int i = 0; i < 24; i++) {
            data[i] = i;
        }
        NdArrayCpu ndArray = new NdArrayCpu(data, new ShapeCpu(2, 3, 4));
        
        float[][][] array3d = ArrayConverter.to3dArray(ndArray);
        
        assertEquals(2, array3d.length);
        assertEquals(3, array3d[0].length);
        assertEquals(4, array3d[0][0].length);
        
        // 验证部分元素
        assertEquals(0.0f, array3d[0][0][0], 0.001f);
        assertEquals(1.0f, array3d[0][0][1], 0.001f);
        assertEquals(4.0f, array3d[0][1][0], 0.001f);
        assertEquals(12.0f, array3d[1][0][0], 0.001f);
        assertEquals(23.0f, array3d[1][2][3], 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTo3dArray_2DArrayThrowsException() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        NdArrayCpu ndArray = new NdArrayCpu(data, new ShapeCpu(2, 3));
        
        ArrayConverter.to3dArray(ndArray);
    }

    @Test
    public void testTo4dArray_2x3x2x2Tensor() {
        float[] data = new float[24];
        for (int i = 0; i < 24; i++) {
            data[i] = i;
        }
        NdArrayCpu ndArray = new NdArrayCpu(data, new ShapeCpu(2, 3, 2, 2));
        
        float[][][][] array4d = ArrayConverter.to4dArray(ndArray);
        
        assertEquals(2, array4d.length);
        assertEquals(3, array4d[0].length);
        assertEquals(2, array4d[0][0].length);
        assertEquals(2, array4d[0][0][0].length);
        
        // 验证部分元素
        assertEquals(0.0f, array4d[0][0][0][0], 0.001f);
        assertEquals(1.0f, array4d[0][0][0][1], 0.001f);
        assertEquals(2.0f, array4d[0][0][1][0], 0.001f);
        assertEquals(4.0f, array4d[0][1][0][0], 0.001f);
        assertEquals(12.0f, array4d[1][0][0][0], 0.001f);
        assertEquals(23.0f, array4d[1][2][1][1], 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTo4dArray_3DArrayThrowsException() {
        float[] data = new float[24];
        NdArrayCpu ndArray = new NdArrayCpu(data, new ShapeCpu(2, 3, 4));
        
        ArrayConverter.to4dArray(ndArray);
    }

    @Test
    public void testFlattenArray_2DArray() {
        float[][] array2d = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        float[] buffer = new float[6];
        
        int resultIndex = ArrayConverter.flattenArray(array2d, buffer, 0);
        
        assertEquals(6, resultIndex);
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, buffer, 0.001f);
    }

    @Test
    public void testFlattenArray_3DArray() {
        float[][][] array3d = {
            {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
            {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}
        };
        float[] buffer = new float[12];
        
        int resultIndex = ArrayConverter.flattenArray(array3d, buffer, 0);
        
        assertEquals(12, resultIndex);
        float[] expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        assertArrayEquals(expected, buffer, 0.001f);
    }

    @Test
    public void testFlattenArray_WithStartingIndex() {
        float[][] array2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] buffer = new float[6];
        buffer[0] = 0.0f;
        buffer[1] = 0.0f;
        
        int resultIndex = ArrayConverter.flattenArray(array2d, buffer, 2);
        
        assertEquals(6, resultIndex);
        assertArrayEquals(new float[]{0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, buffer, 0.001f);
    }
}
