package io.leavesfly.tinyai.agent.embodied.model;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 驾驶动作测试类
 *
 * @author TinyAI Team
 */
public class DrivingActionTest {

    @Test
    public void testDefaultConstructor() {
        DrivingAction action = new DrivingAction();
        assertEquals(0.0, action.getSteering(), 1e-6);
        assertEquals(0.0, action.getThrottle(), 1e-6);
        assertEquals(0.0, action.getBrake(), 1e-6);
    }

    @Test
    public void testParameterizedConstructor() {
        DrivingAction action = new DrivingAction(0.5, 0.8, 0.2);
        assertEquals(0.5, action.getSteering(), 1e-6);
        assertEquals(0.8, action.getThrottle(), 1e-6);
        assertEquals(0.2, action.getBrake(), 1e-6);
    }

    @Test
    public void testToArray() {
        DrivingAction action = new DrivingAction(0.3, 0.7, 0.1);
        NdArray array = action.toArray();
        
        assertNotNull(array);
        assertEquals(3, array.getShape().size());
        assertEquals(0.3f, array.get(0), 1e-6);
        assertEquals(0.7f, array.get(1), 1e-6);
        assertEquals(0.1f, array.get(2), 1e-6);
    }

    @Test
    public void testFromArray() {
        NdArray array = NdArray.of(new float[]{0.4f, 0.6f, 0.3f}, Shape.of(3));
        DrivingAction action = DrivingAction.fromArray(array);
        
        assertNotNull(action);
        assertEquals(0.4, action.getSteering(), 1e-6);
        assertEquals(0.6, action.getThrottle(), 1e-6);
        assertEquals(0.3, action.getBrake(), 1e-6);
    }

    @Test
    public void testFromArrayInvalidSize() {
        NdArray array = NdArray.of(new float[]{0.4f, 0.6f}, Shape.of(2));
        assertThrows(IllegalArgumentException.class, () -> {
            DrivingAction.fromArray(array);
        });
    }

    @Test
    public void testClip() {
        DrivingAction action = new DrivingAction(1.5, -0.2, 1.2);
        action.clip();
        
        assertEquals(1.0, action.getSteering(), 1e-6);
        assertEquals(0.0, action.getThrottle(), 1e-6);
        assertEquals(1.0, action.getBrake(), 1e-6);
    }

    @Test
    public void testClipNegativeSteering() {
        DrivingAction action = new DrivingAction(-1.8, 0.5, 0.3);
        action.clip();
        
        assertEquals(-1.0, action.getSteering(), 1e-6);
        assertEquals(0.5, action.getThrottle(), 1e-6);
        assertEquals(0.3, action.getBrake(), 1e-6);
    }

    @Test
    public void testIsNullAction() {
        DrivingAction nullAction = new DrivingAction(0.0, 0.0, 0.0);
        assertTrue(nullAction.isNullAction());
        
        DrivingAction nonNullAction = new DrivingAction(0.1, 0.0, 0.0);
        assertFalse(nonNullAction.isNullAction());
    }

    @Test
    public void testIsEmergencyBrake() {
        DrivingAction emergencyAction = new DrivingAction(0.0, 0.0, 0.9);
        assertTrue(emergencyAction.isEmergencyBrake());
        
        DrivingAction normalAction = new DrivingAction(0.0, 0.5, 0.3);
        assertFalse(normalAction.isEmergencyBrake());
    }

    @Test
    public void testSettersAndGetters() {
        DrivingAction action = new DrivingAction();
        
        action.setSteering(0.6);
        assertEquals(0.6, action.getSteering(), 1e-6);
        
        action.setThrottle(0.8);
        assertEquals(0.8, action.getThrottle(), 1e-6);
        
        action.setBrake(0.4);
        assertEquals(0.4, action.getBrake(), 1e-6);
    }

    @Test
    public void testToString() {
        DrivingAction action = new DrivingAction(0.5, 0.7, 0.2);
        String str = action.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("0.500"));
        assertTrue(str.contains("0.700"));
        assertTrue(str.contains("0.200"));
    }

    @Test
    public void testRoundTripConversion() {
        DrivingAction original = new DrivingAction(0.3, 0.6, 0.1);
        NdArray array = original.toArray();
        DrivingAction converted = DrivingAction.fromArray(array);
        
        assertEquals(original.getSteering(), converted.getSteering(), 1e-6);
        assertEquals(original.getThrottle(), converted.getThrottle(), 1e-6);
        assertEquals(original.getBrake(), converted.getBrake(), 1e-6);
    }
}
