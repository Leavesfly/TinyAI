package io.leavesfly.tinyai.func.util;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * BroadcastUtils 工具类单元测试
 */
public class BroadcastUtilsTest {

    @Test
    public void testIsBroadcastableSameShape() {
        Shape shape = Shape.of(2, 3);
        assertTrue(BroadcastUtils.isBroadcastable(shape, shape));
    }

    @Test
    public void testIsBroadcastableScalarToMatrix() {
        Shape scalar = Shape.of(1);
        Shape matrix = Shape.of(2, 3);
        assertTrue(BroadcastUtils.isBroadcastable(scalar, matrix));
    }

    @Test
    public void testIsBroadcastableRowToMatrix() {
        Shape row = Shape.of(1, 3);
        Shape matrix = Shape.of(2, 3);
        assertTrue(BroadcastUtils.isBroadcastable(row, matrix));
    }

    @Test
    public void testIsBroadcastableColumnToMatrix() {
        Shape col = Shape.of(2, 1);
        Shape matrix = Shape.of(2, 3);
        assertTrue(BroadcastUtils.isBroadcastable(col, matrix));
    }

    @Test
    public void testIsBroadcastableIncompatible() {
        Shape shape1 = Shape.of(2, 3);
        Shape shape2 = Shape.of(3, 2);
        assertFalse(BroadcastUtils.isBroadcastable(shape1, shape2));
    }

    @Test
    public void testIsBroadcastableLargerToSmaller() {
        Shape larger = Shape.of(2, 3);
        Shape smaller = Shape.of(1, 3);
        // 大形状不能广播到小形状
        assertFalse(BroadcastUtils.isBroadcastable(larger, smaller));
    }

    @Test
    public void testSumToShapeSameDims() {
        NdArray grad = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        Shape targetShape = Shape.of(1, 3);
        NdArray result = BroadcastUtils.sumToShape(grad, targetShape);
        
        assertEquals(targetShape, result.getShape());
        // 按行求和：[5, 7, 9]
        assertEquals(5f, result.get(0, 0), 1e-5f);
        assertEquals(7f, result.get(0, 1), 1e-5f);
        assertEquals(9f, result.get(0, 2), 1e-5f);
    }

    @Test
    public void testSumToShapeDifferentNdim() {
        // 3D 梯度 sumTo 2D 目标
        NdArray grad = NdArray.of(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 1, 3));
        Shape targetShape = Shape.of(1, 3);
        NdArray result = BroadcastUtils.sumToShape(grad, targetShape);
        
        assertEquals(targetShape, result.getShape());
    }

    @Test
    public void testSumToShapeScalar() {
        NdArray grad = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        Shape scalarShape = Shape.of(1);
        NdArray result = BroadcastUtils.sumToShape(grad, scalarShape);
        
        // 所有元素求和 = 10
        assertEquals(10f, result.getNumber().floatValue(), 1e-5f);
    }

    @Test
    public void testSumToShapeTwoArgVersion() {
        NdArray grad = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        Shape targetShape = Shape.of(1, 3);
        NdArray result = BroadcastUtils.sumToShape(grad, targetShape);
        
        assertEquals(targetShape, result.getShape());
    }

    @Test
    public void testSumToShape1DTarget() {
        // 2D 梯度 sumTo 1D 目标 shape(1,)：验证返回的 shape 是 (1,) 而非标量 ()
        NdArray grad = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        Shape targetShape = Shape.of(1);
        NdArray result = BroadcastUtils.sumToShape(grad, targetShape);

        assertEquals(targetShape, result.getShape());
        assertEquals(1, result.getShape().getDimNum());
        assertEquals(10f, result.getArray()[0], 1e-5f);
    }

    @Test
    public void testSumToShape3DTo1DTarget() {
        // 高维梯度 sumTo 到 1D 单元素目标
        NdArray grad = NdArray.of(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 1, 3));
        Shape targetShape = Shape.of(1);
        NdArray result = BroadcastUtils.sumToShape(grad, targetShape);

        assertEquals(targetShape, result.getShape());
        assertEquals(21f, result.getArray()[0], 1e-5f);
    }

    @Test
    public void testSumToShapeSameShape() {
        // 梯度和目标形状完全相同，应直接返回
        NdArray grad = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        Shape targetShape = Shape.of(2, 2);
        NdArray result = BroadcastUtils.sumToShape(grad, targetShape);

        assertEquals(targetShape, result.getShape());
        assertArrayEquals(grad.getArray(), result.getArray(), 1e-5f);
    }

    @Test
    public void testSumToShapeColumnVector() {
        // 2D 梯度 sumTo 列向量目标 shape(2,1)
        NdArray grad = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        Shape targetShape = Shape.of(2, 1);
        NdArray result = BroadcastUtils.sumToShape(grad, targetShape);

        assertEquals(targetShape, result.getShape());
        assertEquals(6f, result.getArray()[0], 1e-5f);   // 1+2+3
        assertEquals(15f, result.getArray()[1], 1e-5f);  // 4+5+6
    }
}
