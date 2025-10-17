package io.leavesfly.tinyai.agent.embodied.env;

import io.leavesfly.tinyai.agent.embodied.model.ScenarioType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 环境配置测试类
 *
 * @author TinyAI Team
 */
public class EnvironmentConfigTest {

    @Test
    public void testDefaultConstructor() {
        EnvironmentConfig config = new EnvironmentConfig();
        
        assertEquals(3, config.getLaneCount());
        assertEquals(3.5, config.getLaneWidth(), 1e-6);
        assertEquals(1000.0, config.getRoadLength(), 1e-6);
        assertEquals(20, config.getVehicleDensity());
        assertEquals(120.0, config.getSpeedLimit(), 1e-6);
        assertEquals(ScenarioType.HIGHWAY, config.getScenarioType());
    }

    @Test
    public void testCreateTestConfig() {
        EnvironmentConfig config = EnvironmentConfig.createTestConfig();
        
        assertEquals(2, config.getLaneCount());
        assertEquals(5, config.getVehicleDensity());
        assertEquals(500.0, config.getRoadLength(), 1e-6);
        assertEquals(1000, config.getMaxSteps());
        assertEquals(ScenarioType.TEST, config.getScenarioType());
    }

    @Test
    public void testCreateHighwayConfig() {
        EnvironmentConfig config = EnvironmentConfig.createHighwayConfig();
        
        assertEquals(3, config.getLaneCount());
        assertEquals(20, config.getVehicleDensity());
        assertEquals(120.0, config.getSpeedLimit(), 1e-6);
        assertEquals(ScenarioType.HIGHWAY, config.getScenarioType());
    }

    @Test
    public void testCreateUrbanConfig() {
        EnvironmentConfig config = EnvironmentConfig.createUrbanConfig();
        
        assertEquals(2, config.getLaneCount());
        assertEquals(40, config.getVehicleDensity());
        assertEquals(60.0, config.getSpeedLimit(), 1e-6);
        assertEquals(100.0, config.getCurvatureRadius(), 1e-6);
        assertEquals(ScenarioType.URBAN, config.getScenarioType());
    }

    @Test
    public void testSettersAndGetters() {
        EnvironmentConfig config = new EnvironmentConfig();
        
        config.setLaneCount(4);
        assertEquals(4, config.getLaneCount());
        
        config.setLaneWidth(4.0);
        assertEquals(4.0, config.getLaneWidth(), 1e-6);
        
        config.setRoadLength(2000.0);
        assertEquals(2000.0, config.getRoadLength(), 1e-6);
        
        config.setCurvatureRadius(200.0);
        assertEquals(200.0, config.getCurvatureRadius(), 1e-6);
        
        config.setVehicleDensity(30);
        assertEquals(30, config.getVehicleDensity());
        
        config.setSpeedLimit(100.0);
        assertEquals(100.0, config.getSpeedLimit(), 1e-6);
        
        config.setTargetSpeed(90.0);
        assertEquals(90.0, config.getTargetSpeed(), 1e-6);
        
        config.setVisibility(500.0);
        assertEquals(500.0, config.getVisibility(), 1e-6);
        
        config.setFrictionCoeff(0.7);
        assertEquals(0.7, config.getFrictionCoeff(), 1e-6);
        
        config.setTimeStep(0.1);
        assertEquals(0.1, config.getTimeStep(), 1e-6);
        
        config.setMaxSteps(5000);
        assertEquals(5000, config.getMaxSteps());
        
        config.setScenarioType(ScenarioType.URBAN);
        assertEquals(ScenarioType.URBAN, config.getScenarioType());
    }

    @Test
    public void testRewardWeights() {
        EnvironmentConfig config = new EnvironmentConfig();
        
        assertEquals(0.3, config.getRewardSpeedWeight(), 1e-6);
        assertEquals(0.4, config.getRewardLaneWeight(), 1e-6);
        assertEquals(1.0, config.getRewardCollisionWeight(), 1e-6);
        assertEquals(0.1, config.getRewardComfortWeight(), 1e-6);
        
        config.setRewardSpeedWeight(0.5);
        assertEquals(0.5, config.getRewardSpeedWeight(), 1e-6);
        
        config.setRewardLaneWeight(0.3);
        assertEquals(0.3, config.getRewardLaneWeight(), 1e-6);
        
        config.setRewardCollisionWeight(2.0);
        assertEquals(2.0, config.getRewardCollisionWeight(), 1e-6);
        
        config.setRewardComfortWeight(0.2);
        assertEquals(0.2, config.getRewardComfortWeight(), 1e-6);
    }

    @Test
    public void testToString() {
        EnvironmentConfig config = EnvironmentConfig.createHighwayConfig();
        String str = config.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("EnvConfig"));
        assertTrue(str.contains("lanes=3"));
        assertTrue(str.contains("120"));
    }

    @Test
    public void testCustomConfiguration() {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setLaneCount(5);
        config.setRoadLength(3000.0);
        config.setVehicleDensity(50);
        config.setSpeedLimit(80.0);
        config.setScenarioType(ScenarioType.URBAN);
        
        assertEquals(5, config.getLaneCount());
        assertEquals(3000.0, config.getRoadLength(), 1e-6);
        assertEquals(50, config.getVehicleDensity());
        assertEquals(80.0, config.getSpeedLimit(), 1e-6);
        assertEquals(ScenarioType.URBAN, config.getScenarioType());
    }
}
