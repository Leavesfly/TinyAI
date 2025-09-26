package io.leavesfly.tinyai.ndarr;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * NDArray类的单元测试
 * 
 * @author TinyAI Team
 * @version 1.0
 */
public class NDArrayTest {
    
    @Test
    public void testCreateArray() {
        // 测试创建2x3数组
        NDArray array = new NDArray(2, 3);
        
        assertEquals(2, array.getDimensions());
        assertArrayEquals(new int[]{2, 3}, array.getShape());
        assertEquals(6, array.getSize());
    }
    
    @Test
    public void testSetAndGet() {
        NDArray array = new NDArray(2, 3);
        
        // 设置值
        array.set(5.5, 0, 1);
        array.set(-2.3, 1, 2);
        
        // 获取值
        assertEquals(5.5, array.get(0, 1), 1e-10);
        assertEquals(-2.3, array.get(1, 2), 1e-10);
        assertEquals(0.0, array.get(0, 0), 1e-10); // 默认值
    }
    
    @Test
    public void testFill() {
        NDArray array = new NDArray(2, 2);
        array.fill(3.14);
        
        // 检查所有元素都被填充
        assertEquals(3.14, array.get(0, 0), 1e-10);
        assertEquals(3.14, array.get(0, 1), 1e-10);
        assertEquals(3.14, array.get(1, 0), 1e-10);
        assertEquals(3.14, array.get(1, 1), 1e-10);
    }
    
    @Test
    public void testAdd() {
        NDArray array1 = new NDArray(2, 2);
        NDArray array2 = new NDArray(2, 2);
        
        array1.set(1.0, 0, 0);
        array1.set(2.0, 0, 1);
        array1.set(3.0, 1, 0);
        array1.set(4.0, 1, 1);
        
        array2.set(0.5, 0, 0);
        array2.set(1.5, 0, 1);
        array2.set(2.5, 1, 0);
        array2.set(3.5, 1, 1);
        
        NDArray result = array1.add(array2);
        
        assertEquals(1.5, result.get(0, 0), 1e-10);
        assertEquals(3.5, result.get(0, 1), 1e-10);
        assertEquals(5.5, result.get(1, 0), 1e-10);
        assertEquals(7.5, result.get(1, 1), 1e-10);
    }
    
    @Test
    public void testMultiply() {
        NDArray array = new NDArray(2, 2);
        array.set(2.0, 0, 0);
        array.set(3.0, 0, 1);
        array.set(4.0, 1, 0);
        array.set(5.0, 1, 1);
        
        NDArray result = array.multiply(2.5);
        
        assertEquals(5.0, result.get(0, 0), 1e-10);
        assertEquals(7.5, result.get(0, 1), 1e-10);
        assertEquals(10.0, result.get(1, 0), 1e-10);
        assertEquals(12.5, result.get(1, 1), 1e-10);
    }
    
    @Test
    public void test3DArray() {
        // 测试三维数组
        NDArray array = new NDArray(2, 2, 2);
        
        assertEquals(3, array.getDimensions());
        assertEquals(8, array.getSize());
        assertArrayEquals(new int[]{2, 2, 2}, array.getShape());
        
        // 设置和获取三维索引
        array.set(1.23, 1, 0, 1);
        assertEquals(1.23, array.get(1, 0, 1), 1e-10);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidShape() {
        // 测试无效形状 - 包含0或负数
        new NDArray(2, 0, 3);
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testIndexOutOfBounds() {
        NDArray array = new NDArray(2, 3);
        array.get(2, 1); // 超出边界
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testDimensionMismatch() {
        NDArray array = new NDArray(2, 3);
        array.get(1); // 维度不匹配，应该提供2个索引
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAddShapeMismatch() {
        NDArray array1 = new NDArray(2, 3);
        NDArray array2 = new NDArray(3, 2);
        array1.add(array2); // 形状不匹配
    }
    
    @Test
    public void testConstructorWithData() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        NDArray array = new NDArray(data, 2, 3);
        
        assertEquals(1.0, array.get(0, 0), 1e-10);
        assertEquals(2.0, array.get(0, 1), 1e-10);
        assertEquals(3.0, array.get(0, 2), 1e-10);
        assertEquals(4.0, array.get(1, 0), 1e-10);
        assertEquals(5.0, array.get(1, 1), 1e-10);
        assertEquals(6.0, array.get(1, 2), 1e-10);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorDataSizeMismatch() {
        double[] data = {1.0, 2.0, 3.0}; // 3个元素
        new NDArray(data, 2, 3); // 但需要6个元素
    }
}