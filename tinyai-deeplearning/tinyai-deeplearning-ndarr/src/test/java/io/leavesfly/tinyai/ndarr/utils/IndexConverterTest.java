package io.leavesfly.tinyai.ndarr.utils;

import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.IndexConverter;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * IndexConverter 类的单元测试
 */
public class IndexConverterTest {

    @Test
    public void testConvertToMultiIndex_2DMatrix() {
        ShapeCpu shape = new ShapeCpu(3, 4);
        
        // 测试 linearIndex=0 → [0,0]
        int[] indices = new int[2];
        IndexConverter.convertToMultiIndex(0, indices, shape);
        assertArrayEquals(new int[]{0, 0}, indices);
        
        // 测试 linearIndex=5 → [1,1]
        indices = new int[2];
        IndexConverter.convertToMultiIndex(5, indices, shape);
        assertArrayEquals(new int[]{1, 1}, indices);
        
        // 测试 linearIndex=11 → [2,3]
        indices = new int[2];
        IndexConverter.convertToMultiIndex(11, indices, shape);
        assertArrayEquals(new int[]{2, 3}, indices);
    }

    @Test
    public void testConvertToMultiIndex_3DTensor() {
        ShapeCpu shape = new ShapeCpu(2, 3, 4);
        
        // 测试 linearIndex=0 → [0,0,0]
        int[] indices = new int[3];
        IndexConverter.convertToMultiIndex(0, indices, shape);
        assertArrayEquals(new int[]{0, 0, 0}, indices);
        
        // 测试 linearIndex=13 → [1,0,1]
        indices = new int[3];
        IndexConverter.convertToMultiIndex(13, indices, shape);
        assertArrayEquals(new int[]{1, 0, 1}, indices);
        
        // 测试 linearIndex=23 → [1,2,3]
        indices = new int[3];
        IndexConverter.convertToMultiIndex(23, indices, shape);
        assertArrayEquals(new int[]{1, 2, 3}, indices);
    }

    @Test
    public void testConvertToMultiIndex_1DVector() {
        ShapeCpu shape = new ShapeCpu(5);
        
        // 测试 linearIndex=3 → [3]
        int[] indices = new int[1];
        IndexConverter.convertToMultiIndex(3, indices, shape);
        assertArrayEquals(new int[]{3}, indices);
    }

    @Test
    public void testFlatToMultiIndex_Consistency() {
        ShapeCpu shape = new ShapeCpu(3, 4);
        int linearIndex = 7;
        
        // 测试 flatToMultiIndex 与 convertToMultiIndex 结果一致性
        int[] indices1 = new int[2];
        int[] indices2 = new int[2];
        
        IndexConverter.convertToMultiIndex(linearIndex, indices1, shape);
        IndexConverter.flatToMultiIndex(linearIndex, indices2, shape);
        
        assertArrayEquals(indices1, indices2);
    }

    @Test
    public void testRoundTripWithShapeCpuGetIndex_2D() {
        ShapeCpu shape = new ShapeCpu(3, 4);
        int[] testIndices = {0, 5, 11};
        
        for (int linearIndex : testIndices) {
            int[] indices = new int[2];
            IndexConverter.convertToMultiIndex(linearIndex, indices, shape);
            
            // 使用 ShapeCpu.getIndex 转回 linearIndex
            int resultIndex = shape.getIndex(indices);
            assertEquals(linearIndex, resultIndex);
        }
    }

    @Test
    public void testRoundTripWithShapeCpuGetIndex_3D() {
        ShapeCpu shape = new ShapeCpu(2, 3, 4);
        int[] testIndices = {0, 13, 23};
        
        for (int linearIndex : testIndices) {
            int[] indices = new int[3];
            IndexConverter.convertToMultiIndex(linearIndex, indices, shape);
            
            // 使用 ShapeCpu.getIndex 转回 linearIndex
            int resultIndex = shape.getIndex(indices);
            assertEquals(linearIndex, resultIndex);
        }
    }

    @Test
    public void testRoundTripWithShapeCpuGetIndex_1D() {
        ShapeCpu shape = new ShapeCpu(5);
        int[] testIndices = {0, 2, 4};
        
        for (int linearIndex : testIndices) {
            int[] indices = new int[1];
            IndexConverter.convertToMultiIndex(linearIndex, indices, shape);
            
            // 使用 ShapeCpu.getIndex 转回 linearIndex
            int resultIndex = shape.getIndex(indices);
            assertEquals(linearIndex, resultIndex);
        }
    }
}
