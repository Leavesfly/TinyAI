package io.leavesfly.tinyai.agent.embodied.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 三维向量测试类
 *
 * @author TinyAI Team
 */
public class Vector3DTest {

    @Test
    public void testDefaultConstructor() {
        Vector3D vector = new Vector3D();
        assertEquals(0.0, vector.getX(), 1e-6);
        assertEquals(0.0, vector.getY(), 1e-6);
        assertEquals(0.0, vector.getZ(), 1e-6);
    }

    @Test
    public void testParameterizedConstructor() {
        Vector3D vector = new Vector3D(1.0, 2.0, 3.0);
        assertEquals(1.0, vector.getX(), 1e-6);
        assertEquals(2.0, vector.getY(), 1e-6);
        assertEquals(3.0, vector.getZ(), 1e-6);
    }

    @Test
    public void testMagnitude() {
        Vector3D vector = new Vector3D(3.0, 4.0, 0.0);
        assertEquals(5.0, vector.magnitude(), 1e-6);
        
        Vector3D vector3D = new Vector3D(1.0, 2.0, 2.0);
        assertEquals(3.0, vector3D.magnitude(), 1e-6);
    }

    @Test
    public void testAdd() {
        Vector3D v1 = new Vector3D(1.0, 2.0, 3.0);
        Vector3D v2 = new Vector3D(4.0, 5.0, 6.0);
        Vector3D result = v1.add(v2);
        
        assertEquals(5.0, result.getX(), 1e-6);
        assertEquals(7.0, result.getY(), 1e-6);
        assertEquals(9.0, result.getZ(), 1e-6);
    }

    @Test
    public void testDistanceTo() {
        Vector3D v1 = new Vector3D(0.0, 0.0, 0.0);
        Vector3D v2 = new Vector3D(3.0, 4.0, 0.0);
        assertEquals(5.0, v1.distanceTo(v2), 1e-6);
        
        Vector3D v3 = new Vector3D(1.0, 1.0, 1.0);
        Vector3D v4 = new Vector3D(4.0, 5.0, 1.0);
        assertEquals(5.0, v3.distanceTo(v4), 1e-6);
    }

    @Test
    public void testSettersAndGetters() {
        Vector3D vector = new Vector3D();
        
        vector.setX(5.0);
        assertEquals(5.0, vector.getX(), 1e-6);
        
        vector.setY(6.0);
        assertEquals(6.0, vector.getY(), 1e-6);
        
        vector.setZ(7.0);
        assertEquals(7.0, vector.getZ(), 1e-6);
    }

    @Test
    public void testToString() {
        Vector3D vector = new Vector3D(1.5, 2.5, 3.5);
        String str = vector.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("1.5"));
        assertTrue(str.contains("2.5"));
        assertTrue(str.contains("3.5"));
    }

    @Test
    public void testZeroVectorMagnitude() {
        Vector3D zero = new Vector3D(0.0, 0.0, 0.0);
        assertEquals(0.0, zero.magnitude(), 1e-6);
    }

    @Test
    public void testSubtract() {
        Vector3D v1 = new Vector3D(4.0, 5.0, 6.0);
        Vector3D v2 = new Vector3D(1.0, 2.0, 3.0);
        Vector3D result = v1.subtract(v2);
        
        assertEquals(3.0, result.getX(), 1e-6);
        assertEquals(3.0, result.getY(), 1e-6);
        assertEquals(3.0, result.getZ(), 1e-6);
    }

    @Test
    public void testMultiply() {
        Vector3D v = new Vector3D(2.0, 3.0, 4.0);
        Vector3D result = v.multiply(2.0);
        
        assertEquals(4.0, result.getX(), 1e-6);
        assertEquals(6.0, result.getY(), 1e-6);
        assertEquals(8.0, result.getZ(), 1e-6);
    }

    @Test
    public void testDistanceSymmetry() {
        Vector3D v1 = new Vector3D(1.0, 2.0, 3.0);
        Vector3D v2 = new Vector3D(4.0, 5.0, 6.0);
        
        double d1 = v1.distanceTo(v2);
        double d2 = v2.distanceTo(v1);
        
        assertEquals(d1, d2, 1e-6);
    }
}
