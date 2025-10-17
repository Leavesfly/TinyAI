package io.leavesfly.tinyai.agent.robot.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Vector2D测试类
 * 
 * @author TinyAI Team
 */
public class Vector2DTest {
    
    @Test
    public void testConstructor() {
        Vector2D v = new Vector2D(3.0, 4.0);
        assertEquals(3.0, v.getX(), 1e-6);
        assertEquals(4.0, v.getY(), 1e-6);
    }
    
    @Test
    public void testDefaultConstructor() {
        Vector2D v = new Vector2D();
        assertEquals(0.0, v.getX(), 1e-6);
        assertEquals(0.0, v.getY(), 1e-6);
    }
    
    @Test
    public void testCopyConstructor() {
        Vector2D v1 = new Vector2D(3.0, 4.0);
        Vector2D v2 = new Vector2D(v1);
        assertEquals(v1.getX(), v2.getX(), 1e-6);
        assertEquals(v1.getY(), v2.getY(), 1e-6);
    }
    
    @Test
    public void testMagnitude() {
        Vector2D v = new Vector2D(3.0, 4.0);
        assertEquals(5.0, v.magnitude(), 1e-6);
    }
    
    @Test
    public void testDistanceTo() {
        Vector2D v1 = new Vector2D(0.0, 0.0);
        Vector2D v2 = new Vector2D(3.0, 4.0);
        assertEquals(5.0, v1.distanceTo(v2), 1e-6);
    }
    
    @Test
    public void testAdd() {
        Vector2D v1 = new Vector2D(1.0, 2.0);
        Vector2D v2 = new Vector2D(3.0, 4.0);
        Vector2D result = v1.add(v2);
        assertEquals(4.0, result.getX(), 1e-6);
        assertEquals(6.0, result.getY(), 1e-6);
    }
    
    @Test
    public void testSubtract() {
        Vector2D v1 = new Vector2D(5.0, 7.0);
        Vector2D v2 = new Vector2D(2.0, 3.0);
        Vector2D result = v1.subtract(v2);
        assertEquals(3.0, result.getX(), 1e-6);
        assertEquals(4.0, result.getY(), 1e-6);
    }
    
    @Test
    public void testMultiply() {
        Vector2D v = new Vector2D(2.0, 3.0);
        Vector2D result = v.multiply(2.0);
        assertEquals(4.0, result.getX(), 1e-6);
        assertEquals(6.0, result.getY(), 1e-6);
    }
    
    @Test
    public void testNormalize() {
        Vector2D v = new Vector2D(3.0, 4.0);
        Vector2D normalized = v.normalize();
        assertEquals(1.0, normalized.magnitude(), 1e-6);
        assertEquals(0.6, normalized.getX(), 1e-6);
        assertEquals(0.8, normalized.getY(), 1e-6);
    }
    
    @Test
    public void testAngle() {
        Vector2D v = new Vector2D(1.0, 0.0);
        assertEquals(0.0, v.angle(), 1e-6);
        
        Vector2D v2 = new Vector2D(0.0, 1.0);
        assertEquals(Math.PI / 2, v2.angle(), 1e-6);
    }
    
    @Test
    public void testDot() {
        Vector2D v1 = new Vector2D(1.0, 2.0);
        Vector2D v2 = new Vector2D(3.0, 4.0);
        assertEquals(11.0, v1.dot(v2), 1e-6);
    }
    
    @Test
    public void testEquals() {
        Vector2D v1 = new Vector2D(3.0, 4.0);
        Vector2D v2 = new Vector2D(3.0, 4.0);
        Vector2D v3 = new Vector2D(3.0, 5.0);
        
        assertTrue(v1.equals(v2));
        assertFalse(v1.equals(v3));
    }
}
