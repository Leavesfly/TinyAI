package io.leavesfly.tinyai.agent.embodied.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 车辆状态测试类
 *
 * @author TinyAI Team
 */
public class VehicleStateTest {

    @Test
    public void testDefaultConstructor() {
        VehicleState state = new VehicleState();
        
        assertNotNull(state.getPosition());
        assertEquals(0.0, state.getPosition().getX(), 1e-6);
        assertEquals(0.0, state.getHeading(), 1e-6);
        assertEquals(0.0, state.getSpeed(), 1e-6);
        assertEquals(0.0, state.getAcceleration(), 1e-6);
        assertEquals(0.0, state.getSteeringAngle(), 1e-6);
        assertEquals(0.0, state.getAngularVelocity(), 1e-6);
    }

    @Test
    public void testCopyConstructor() {
        VehicleState original = new VehicleState();
        original.setPosition(new Vector3D(10.0, 20.0, 0.0));
        original.setHeading(Math.PI / 4);
        original.setSpeed(25.0);
        original.setAcceleration(2.0);
        original.setSteeringAngle(0.1);
        original.setAngularVelocity(0.05);
        
        VehicleState copy = new VehicleState(original);
        
        assertEquals(original.getPosition().getX(), copy.getPosition().getX(), 1e-6);
        assertEquals(original.getPosition().getY(), copy.getPosition().getY(), 1e-6);
        assertEquals(original.getHeading(), copy.getHeading(), 1e-6);
        assertEquals(original.getSpeed(), copy.getSpeed(), 1e-6);
        assertEquals(original.getAcceleration(), copy.getAcceleration(), 1e-6);
        assertEquals(original.getSteeringAngle(), copy.getSteeringAngle(), 1e-6);
        assertEquals(original.getAngularVelocity(), copy.getAngularVelocity(), 1e-6);
    }

    @Test
    public void testCopyConstructorIndependence() {
        VehicleState original = new VehicleState();
        original.setPosition(new Vector3D(10.0, 20.0, 0.0));
        
        VehicleState copy = new VehicleState(original);
        copy.getPosition().setX(30.0);
        
        // 原始对象不应该被修改
        assertEquals(10.0, original.getPosition().getX(), 1e-6);
        assertEquals(30.0, copy.getPosition().getX(), 1e-6);
    }

    @Test
    public void testGetVelocityVector() {
        VehicleState state = new VehicleState();
        state.setSpeed(10.0);
        state.setHeading(0.0); // 朝向正东
        
        Vector3D velocity = state.getVelocityVector();
        
        assertEquals(10.0, velocity.getX(), 1e-6);
        assertEquals(0.0, velocity.getY(), 1e-6);
        assertEquals(0.0, velocity.getZ(), 1e-6);
    }

    @Test
    public void testGetVelocityVectorWithHeading() {
        VehicleState state = new VehicleState();
        state.setSpeed(10.0);
        state.setHeading(Math.PI / 2); // 朝向正北
        
        Vector3D velocity = state.getVelocityVector();
        
        assertEquals(0.0, velocity.getX(), 1e-5);
        assertEquals(10.0, velocity.getY(), 1e-5);
        assertEquals(0.0, velocity.getZ(), 1e-6);
    }

    @Test
    public void testSettersAndGetters() {
        VehicleState state = new VehicleState();
        
        Vector3D position = new Vector3D(5.0, 10.0, 0.0);
        state.setPosition(position);
        assertEquals(position, state.getPosition());
        
        state.setHeading(1.57);
        assertEquals(1.57, state.getHeading(), 1e-6);
        
        state.setSpeed(30.0);
        assertEquals(30.0, state.getSpeed(), 1e-6);
        
        state.setAcceleration(2.5);
        assertEquals(2.5, state.getAcceleration(), 1e-6);
        
        state.setSteeringAngle(0.2);
        assertEquals(0.2, state.getSteeringAngle(), 1e-6);
        
        state.setAngularVelocity(0.1);
        assertEquals(0.1, state.getAngularVelocity(), 1e-6);
    }

    @Test
    public void testToString() {
        VehicleState state = new VehicleState();
        state.setSpeed(25.0);
        state.setAcceleration(2.0);
        
        String str = state.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("VehicleState"));
        assertTrue(str.contains("25.00"));
        assertTrue(str.contains("2.00"));
    }

    @Test
    public void testNegativeSpeed() {
        VehicleState state = new VehicleState();
        state.setSpeed(-10.0); // 倒车
        
        assertEquals(-10.0, state.getSpeed(), 1e-6);
    }

    @Test
    public void testNegativeAcceleration() {
        VehicleState state = new VehicleState();
        state.setAcceleration(-3.0); // 减速
        
        assertEquals(-3.0, state.getAcceleration(), 1e-6);
    }
}
