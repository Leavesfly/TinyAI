package io.leavesfly.tinyai.agent.embodied.decision;

import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;
import io.leavesfly.tinyai.agent.embodied.model.VehicleState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 决策模块测试类
 *
 * @author TinyAI Team
 */
public class DecisionModuleTest {

    private DecisionModule decisionModule;

    @BeforeEach
    public void setUp() {
        decisionModule = new DecisionModule();
    }

    @Test
    public void testConstructor() {
        assertNotNull(decisionModule);
    }

    @Test
    public void testDecideWithNullState() {
        assertThrows(NullPointerException.class, () -> {
            decisionModule.decide(null);
        });
    }

    @Test
    public void testDecideWithValidState() {
        PerceptionState state = new PerceptionState();
        VehicleState vehicleState = new VehicleState();
        vehicleState.setSpeed(20.0);
        state.setVehicleState(vehicleState);
        
        DrivingAction action = decisionModule.decide(state);
        
        assertNotNull(action);
        assertNotNull(action.getSteering());
        assertNotNull(action.getThrottle());
        assertNotNull(action.getBrake());
    }

    @Test
    public void testDecideActionInValidRange() {
        PerceptionState state = new PerceptionState();
        VehicleState vehicleState = new VehicleState();
        vehicleState.setSpeed(25.0);
        state.setVehicleState(vehicleState);
        
        DrivingAction action = decisionModule.decide(state);
        
        // 验证动作在有效范围内
        assertTrue(action.getSteering() >= -1.0 && action.getSteering() <= 1.0);
        assertTrue(action.getThrottle() >= 0.0 && action.getThrottle() <= 1.0);
        assertTrue(action.getBrake() >= 0.0 && action.getBrake() <= 1.0);
    }

    @Test
    public void testSetPolicyNetwork() {
        PolicyNetwork customPolicy = new SimplePolicy();
        
        assertDoesNotThrow(() -> {
            decisionModule.setPolicyNetwork(customPolicy);
        });
    }

    @Test
    public void testMultipleDecisions() {
        PerceptionState state = new PerceptionState();
        VehicleState vehicleState = new VehicleState();
        vehicleState.setSpeed(30.0);
        state.setVehicleState(vehicleState);
        
        // 多次决策应该都能成功
        for (int i = 0; i < 10; i++) {
            DrivingAction action = decisionModule.decide(state);
            assertNotNull(action);
        }
    }

    @Test
    public void testDecideWithDifferentSpeeds() {
        PerceptionState state = new PerceptionState();
        VehicleState vehicleState = new VehicleState();
        
        // 测试不同速度下的决策
        double[] speeds = {0.0, 10.0, 30.0, 60.0, 100.0};
        for (double speed : speeds) {
            vehicleState.setSpeed(speed);
            state.setVehicleState(vehicleState);
            
            DrivingAction action = decisionModule.decide(state);
            assertNotNull(action);
            
            // 验证动作有效性
            assertTrue(action.getSteering() >= -1.0 && action.getSteering() <= 1.0);
            assertTrue(action.getThrottle() >= 0.0 && action.getThrottle() <= 1.0);
            assertTrue(action.getBrake() >= 0.0 && action.getBrake() <= 1.0);
        }
    }
}
