package io.leavesfly.tinyai.agent.robot.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * RobotState测试类
 * 
 * @author TinyAI Team
 */
public class RobotStateTest {
    
    @Test
    public void testConstructor() {
        Vector2D position = new Vector2D(1.0, 2.0);
        RobotState state = new RobotState(position, Math.PI / 4);
        
        assertEquals(1.0, state.getPosition().getX(), 1e-6);
        assertEquals(2.0, state.getPosition().getY(), 1e-6);
        assertEquals(Math.PI / 4, state.getHeading(), 1e-6);
        assertEquals(100.0, state.getBatteryLevel(), 1e-6);
    }
    
    @Test
    public void testDefaultConstructor() {
        RobotState state = new RobotState();
        assertEquals(0.0, state.getPosition().getX(), 1e-6);
        assertEquals(0.0, state.getPosition().getY(), 1e-6);
        assertEquals(100.0, state.getBatteryLevel(), 1e-6);
    }
    
    @Test
    public void testCopyConstructor() {
        RobotState state1 = new RobotState(new Vector2D(1.0, 2.0), Math.PI / 4);
        state1.setBatteryLevel(80.0);
        
        RobotState state2 = new RobotState(state1);
        assertEquals(state1.getPosition().getX(), state2.getPosition().getX(), 1e-6);
        assertEquals(state1.getBatteryLevel(), state2.getBatteryLevel(), 1e-6);
    }
    
    @Test
    public void testSetLinearSpeed() {
        RobotState state = new RobotState();
        state.setLinearSpeed(0.3);
        assertEquals(0.3, state.getLinearSpeed(), 1e-6);
        
        // 测试自动限幅
        state.setLinearSpeed(1.0);
        assertEquals(0.5, state.getLinearSpeed(), 1e-6);
        
        state.setLinearSpeed(-0.1);
        assertEquals(0.0, state.getLinearSpeed(), 1e-6);
    }
    
    @Test
    public void testSetAngularSpeed() {
        RobotState state = new RobotState();
        state.setAngularSpeed(1.0);
        assertEquals(1.0, state.getAngularSpeed(), 1e-6);
        
        // 测试自动限幅
        state.setAngularSpeed(Math.PI);
        assertEquals(Math.PI / 2, state.getAngularSpeed(), 1e-6);
    }
    
    @Test
    public void testSetBatteryLevel() {
        RobotState state = new RobotState();
        state.setBatteryLevel(50.0);
        assertEquals(50.0, state.getBatteryLevel(), 1e-6);
        
        // 测试自动限幅
        state.setBatteryLevel(150.0);
        assertEquals(100.0, state.getBatteryLevel(), 1e-6);
        
        state.setBatteryLevel(-10.0);
        assertEquals(0.0, state.getBatteryLevel(), 1e-6);
    }
    
    @Test
    public void testNeedsCharging() {
        RobotState state = new RobotState();
        state.setBatteryLevel(30.0);
        assertFalse(state.needsCharging());
        
        state.setBatteryLevel(15.0);
        assertTrue(state.needsCharging());
    }
    
    @Test
    public void testNeedsEmptying() {
        RobotState state = new RobotState();
        state.setDustCapacity(85.0);
        assertFalse(state.needsEmptying());
        
        state.setDustCapacity(95.0);
        assertTrue(state.needsEmptying());
    }
    
    @Test
    public void testIsOperational() {
        RobotState state = new RobotState();
        assertTrue(state.isOperational());
        
        state.setBatteryLevel(3.0);
        assertFalse(state.isOperational());
        
        state.setBatteryLevel(50.0);
        state.setDustCapacity(100.0);
        assertFalse(state.isOperational());
    }
    
    @Test
    public void testSetHeading() {
        RobotState state = new RobotState();
        
        // 测试归一化
        state.setHeading(3 * Math.PI);
        assertEquals(Math.PI, state.getHeading(), 1e-6);
        
        state.setHeading(-Math.PI / 2);
        assertEquals(3 * Math.PI / 2, state.getHeading(), 1e-6);
    }
}
