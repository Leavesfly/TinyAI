package io.leavesfly.tinyai.agent.embodied.perception;

import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.env.impl.SimpleDrivingEnv;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;
import io.leavesfly.tinyai.agent.embodied.sensor.SensorSuite;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 感知模块测试类
 *
 * @author TinyAI Team
 */
public class PerceptionModuleTest {

    private PerceptionModule perceptionModule;
    private DrivingEnvironment environment;
    private SensorSuite sensorSuite;

    @BeforeEach
    public void setUp() {
        EnvironmentConfig config = EnvironmentConfig.createTestConfig();
        environment = new SimpleDrivingEnv(config);
        sensorSuite = new SensorSuite(environment);
        perceptionModule = new PerceptionModule(sensorSuite);
    }

    @AfterEach
    public void tearDown() {
        if (environment != null) {
            environment.close();
        }
    }

    @Test
    public void testConstructor() {
        assertNotNull(perceptionModule);
        assertFalse(perceptionModule.isInitialized());
    }

    @Test
    public void testInitialize() {
        perceptionModule.initialize();
        
        assertTrue(perceptionModule.isInitialized());
        assertTrue(sensorSuite.isInitialized());
    }

    @Test
    public void testProcessWithoutInitialize() {
        // 未初始化时处理，应该自动初始化
        PerceptionState rawState = environment.reset();
        
        PerceptionState processedState = perceptionModule.process(rawState);
        
        assertNotNull(processedState);
        assertTrue(perceptionModule.isInitialized());
    }

    @Test
    public void testProcessWithInitialize() {
        perceptionModule.initialize();
        PerceptionState rawState = environment.reset();
        
        PerceptionState processedState = perceptionModule.process(rawState);
        
        assertNotNull(processedState);
        assertNotNull(processedState.getVisualFeatures());
        assertNotNull(processedState.getLidarFeatures());
    }

    @Test
    public void testProcessMultipleTimes() {
        perceptionModule.initialize();
        PerceptionState rawState = environment.reset();
        
        for (int i = 0; i < 5; i++) {
            PerceptionState processedState = perceptionModule.process(rawState);
            assertNotNull(processedState);
            assertNotNull(processedState.getVisualFeatures());
            assertNotNull(processedState.getLidarFeatures());
        }
    }

    @Test
    public void testReset() {
        perceptionModule.initialize();
        
        assertDoesNotThrow(() -> {
            perceptionModule.reset();
        });
    }

    @Test
    public void testGetSensorSuite() {
        SensorSuite suite = perceptionModule.getSensorSuite();
        
        assertNotNull(suite);
        assertEquals(sensorSuite, suite);
    }

    @Test
    public void testFeatureExtraction() {
        perceptionModule.initialize();
        PerceptionState rawState = environment.reset();
        
        PerceptionState processedState = perceptionModule.process(rawState);
        
        // 验证特征已经被提取
        assertNotNull(processedState.getVisualFeatures());
        assertNotNull(processedState.getLidarFeatures());
        
        // 验证特征维度
        assertTrue(processedState.getVisualFeatures().getShape().size() > 0);
        assertTrue(processedState.getLidarFeatures().getShape().size() > 0);
    }
}
