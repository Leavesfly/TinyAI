package io.leavesfly.tinyai.ndarr.utils;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * ArrayValidator数组验证工具测试
 * 
 * 测试各种验证功能
 *
 * @author TinyAI
 */
public class ArrayValidatorTest {

    @Test
    public void testValidateDataShapeMatch() {
        // 测试数据长度与形状匹配
        ArrayValidator.validateDataShape(6, 6);
        assertTrue(true); // 不抛异常即通过
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateDataShapeMismatch() {
        // 测试数据长度与形状不匹配
        ArrayValidator.validateDataShape(5, 6);
    }

    @Test
    public void testValidateShapeCompatibility() {
        Shape shape1 = Shape.of(2, 3);
        Shape shape2 = Shape.of(2, 3);
        
        ArrayValidator.validateShapeCompatibility(shape1, shape2, "测试");
        assertTrue(true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateShapeIncompatibility() {
        Shape shape1 = Shape.of(2, 3);
        Shape shape2 = Shape.of(3, 2);
        
        ArrayValidator.validateShapeCompatibility(shape1, shape2, "测试");
    }

    @Test
    public void testValidateAxisValid() {
        ArrayValidator.validateAxis(0, 3);
        ArrayValidator.validateAxis(1, 3);
        ArrayValidator.validateAxis(2, 3);
        assertTrue(true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateAxisNegative() {
        ArrayValidator.validateAxis(-4, 3);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateAxisOutOfBounds() {
        ArrayValidator.validateAxis(3, 3);
    }

    @Test
    public void testValidateTransposeOrderValid() {
        int[] order = {0, 1, 2};
        ArrayValidator.validateTransposeOrder(order, 3);
        assertTrue(true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateTransposeOrderInvalidLength() {
        int[] order = {0, 1};
        ArrayValidator.validateTransposeOrder(order, 3);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateTransposeOrderDuplicate() {
        int[] order = {0, 1, 1};
        ArrayValidator.validateTransposeOrder(order, 3);
    }

    @Test
    public void testValidateArrayDimensionsValid() {
        float[][] array = {{1f, 2f}, {3f, 4f}};
        ArrayValidator.validateArrayDimensions(array);
        assertTrue(true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateArrayDimensionsNull() {
        ArrayValidator.validateArrayDimensions(null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testValidateArrayDimensionsInconsistent() {
        float[][] array = {{1f, 2f}, {3f, 4f, 5f}};
        ArrayValidator.validateArrayDimensions(array);
    }
}
