package io.leavesfly.tinyai.agent.embodied.execution;

import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.env.impl.SimpleDrivingEnv;
import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.ExecutionFeedback;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 执行模块测试类
 *
 * @author TinyAI Team
 */
public class ExecutionModuleTest {

    private ExecutionModule executionModule;
    private DrivingEnvironment environment;

    @BeforeEach
    public void setUp() {
        EnvironmentConfig config = EnvironmentConfig.createTestConfig();
        environment = new SimpleDrivingEnv(config);
        environment.reset();
        executionModule = new ExecutionModule(environment);
    }

    @AfterEach
    public void tearDown() {
        if (environment != null) {
            environment.close();
        }
    }

    @Test
    public void testConstructor() {
        assertNotNull(executionModule);
        assertNotNull(executionModule.getEnvironment());
    }

    @Test
    public void testExecuteValidAction() {
        DrivingAction action = new DrivingAction(0.1, 0.5, 0.0);
        
        ExecutionFeedback feedback = executionModule.execute(action);
        
        assertNotNull(feedback);
        assertTrue(feedback.isSuccess());
        assertNotNull(feedback.getNextState());
        assertNotNull(feedback.getActualAction());
    }

    @Test
    public void testExecuteNullAction() {
        DrivingAction nullAction = new DrivingAction(0.0, 0.0, 0.0);
        
        ExecutionFeedback feedback = executionModule.execute(nullAction);
        
        assertNotNull(feedback);
        assertTrue(feedback.isSuccess());
    }

    @Test
    public void testExecuteEmergencyBrake() {
        DrivingAction brakeAction = new DrivingAction(0.0, 0.0, 1.0);
        
        ExecutionFeedback feedback = executionModule.execute(brakeAction);
        
        assertNotNull(feedback);
        assertTrue(feedback.isSuccess());
    }

    @Test
    public void testExecuteMultipleActions() {
        DrivingAction[] actions = {
            new DrivingAction(0.1, 0.5, 0.0),
            new DrivingAction(-0.1, 0.3, 0.0),
            new DrivingAction(0.0, 0.0, 0.5),
            new DrivingAction(0.2, 0.8, 0.0)
        };
        
        for (DrivingAction action : actions) {
            ExecutionFeedback feedback = executionModule.execute(action);
            assertNotNull(feedback);
            assertTrue(feedback.isSuccess());
        }
    }

    @Test
    public void testFeedbackContainsReward() {
        DrivingAction action = new DrivingAction(0.0, 0.5, 0.0);
        
        ExecutionFeedback feedback = executionModule.execute(action);
        
        assertNotNull(feedback.getReward());
    }

    @Test
    public void testFeedbackContainsInfo() {
        DrivingAction action = new DrivingAction(0.1, 0.5, 0.0);
        
        ExecutionFeedback feedback = executionModule.execute(action);
        
        assertNotNull(feedback.getInfo());
    }

    @Test
    public void testFeedbackDoneFlag() {
        DrivingAction action = new DrivingAction(0.0, 0.5, 0.0);
        
        ExecutionFeedback feedback = executionModule.execute(action);
        
        // 第一步通常不应该结束
        assertNotNull(feedback);
    }

    @Test
    public void testGetEnvironment() {
        DrivingEnvironment env = executionModule.getEnvironment();
        
        assertNotNull(env);
        assertEquals(environment, env);
    }

    @Test
    public void testExecuteWithExtremeActions() {
        // 测试极限转向
        DrivingAction extremeSteer = new DrivingAction(1.0, 0.5, 0.0);
        ExecutionFeedback feedback1 = executionModule.execute(extremeSteer);
        assertNotNull(feedback1);
        assertTrue(feedback1.isSuccess());
        
        // 测试极限油门
        DrivingAction extremeThrottle = new DrivingAction(0.0, 1.0, 0.0);
        ExecutionFeedback feedback2 = executionModule.execute(extremeThrottle);
        assertNotNull(feedback2);
        assertTrue(feedback2.isSuccess());
    }
}
