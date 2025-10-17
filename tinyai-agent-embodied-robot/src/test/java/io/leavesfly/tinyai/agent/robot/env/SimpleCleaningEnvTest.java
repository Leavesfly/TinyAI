package io.leavesfly.tinyai.agent.robot.env;

import io.leavesfly.tinyai.agent.robot.model.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * SimpleCleaningEnv测试类
 * 
 * @author TinyAI Team
 */
public class SimpleCleaningEnvTest {
    
    private SimpleCleaningEnv env;
    private EnvironmentConfig config;
    
    @BeforeEach
    public void setUp() {
        config = EnvironmentConfig.createSimpleRoomConfig();
        env = new SimpleCleaningEnv(config);
    }
    
    @Test
    public void testReset() {
        CleaningState state = env.reset();
        
        assertNotNull(state);
        assertNotNull(state.getRobotState());
        assertNotNull(state.getFloorMap());
        assertNotNull(state.getObstacleMap());
        
        // 验证初始状态
        assertEquals(100.0, state.getRobotState().getBatteryLevel(), 1e-6);
        assertEquals(0.0, state.getRobotState().getDustCapacity(), 1e-6);
        assertEquals(0.0, state.getFloorMap().getCoverageRate(), 1e-6);
    }
    
    @Test
    public void testStep() {
        env.reset();
        
        CleaningAction action = CleaningAction.moveForward(0.5);
        StepResult result = env.step(action);
        
        assertNotNull(result);
        assertNotNull(result.getObservation());
        assertFalse(result.isDone());
        
        // 验证位置变化
        Vector2D newPosition = result.getObservation().getRobotState().getPosition();
        assertTrue(newPosition.getX() > 0.5 || newPosition.getY() > 0.5);
    }
    
    @Test
    public void testMultipleSteps() {
        env.reset();
        
        for (int i = 0; i < 10; i++) {
            CleaningAction action = CleaningAction.moveForward(0.5);
            StepResult result = env.step(action);
            assertNotNull(result);
        }
    }
    
    @Test
    public void testCollision() {
        env.reset();
        
        // 尝试移出边界
        for (int i = 0; i < 100; i++) {
            CleaningAction action = CleaningAction.moveForward(1.0);
            StepResult result = env.step(action);
            
            // 机器人应该被边界限制
            Vector2D position = result.getObservation().getRobotState().getPosition();
            assertTrue(position.getX() >= 0 && position.getX() <= config.getRoomWidth());
            assertTrue(position.getY() >= 0 && position.getY() <= config.getRoomHeight());
        }
    }
    
    @Test
    public void testBatteryConsumption() {
        env.reset();
        double initialBattery = env.getObservation().getRobotState().getBatteryLevel();
        
        // 执行多步动作
        for (int i = 0; i < 50; i++) {
            CleaningAction action = CleaningAction.moveForward(0.5);
            env.step(action);
        }
        
        double finalBattery = env.getObservation().getRobotState().getBatteryLevel();
        assertTrue(finalBattery < initialBattery);
    }
    
    @Test
    public void testTerminationByBattery() {
        env.reset();
        
        // 持续执行直到电量耗尽
        boolean terminated = false;
        for (int i = 0; i < 10000 && !terminated; i++) {
            CleaningAction action = CleaningAction.moveForward(1.0);
            StepResult result = env.step(action);
            terminated = result.isDone();
        }
        
        assertTrue(terminated);
    }
    
    @Test
    public void testGetSensorData() {
        env.reset();
        
        assertNotNull(env.getSensorData(SensorType.CAMERA));
        assertNotNull(env.getSensorData(SensorType.LIDAR));
        assertNotNull(env.getSensorData(SensorType.DIRT_SENSOR));
        assertNotNull(env.getSensorData(SensorType.ODOMETER));
    }
    
    @Test
    public void testScenarioType() {
        assertEquals(ScenarioType.SIMPLE_ROOM, env.getScenarioType());
    }
    
    @Test
    public void testClose() {
        env.reset();
        env.close();
        // 确保关闭后不抛出异常
        assertTrue(true);
    }
}
