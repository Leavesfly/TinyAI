package io.leavesfly.tinyai.agent.vla;

import io.leavesfly.tinyai.agent.vla.env.RobotEnvironment;
import io.leavesfly.tinyai.agent.vla.env.SimpleRobotEnv;
import io.leavesfly.tinyai.agent.vla.model.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * VLA智能体集成测试
 * 
 * @author TinyAI
 */
public class VLAAgentTest {
    
    @Test
    public void testAgentCreation() {
        VLAAgent agent = new VLAAgent(768, 8, 6, 7);
        assertNotNull(agent);
        assertTrue(agent.getParameterCount() > 0);
    }
    
    @Test
    public void testAgentPredict() {
        VLAAgent agent = new VLAAgent(768, 8, 6, 7);
        
        // 创建测试状态
        double[][][] imageData = new double[64][64][3];
        VisionInput visionInput = new VisionInput(new NdArray(imageData));
        LanguageInput languageInput = new LanguageInput("Pick up the red cube");
        
        double[] jointPos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] jointVel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        ProprioceptionInput proprioInput = new ProprioceptionInput(
            new NdArray(jointPos),
            new NdArray(jointVel)
        );
        
        VLAState state = new VLAState(visionInput, languageInput, proprioInput);
        
        // 预测动作
        VLAAction action = agent.predict(state);
        
        assertNotNull(action);
        assertNotNull(action.getContinuousAction());
        assertNotNull(action.getActionType());
        assertTrue(action.getConfidence() >= 0.0 && action.getConfidence() <= 1.0);
        assertNotNull(action.getLanguageFeedback());
    }
    
    @Test
    public void testEnvironmentInteraction() {
        TaskConfig config = new TaskConfig();
        config.setTaskName("PickAndPlace");
        config.setMaxSteps(10);
        
        RobotEnvironment env = new SimpleRobotEnv(config);
        VLAAgent agent = new VLAAgent(768, 8, 6, 7);
        
        VLAState state = env.reset();
        assertNotNull(state);
        
        VLAAction action = agent.predict(state);
        RobotEnvironment.EnvironmentStep step = env.step(action);
        
        assertNotNull(step);
        assertNotNull(step.getNextState());
        assertNotNull(step.getInfo());
    }
    
    @Test
    public void testActionSpaceSpec() {
        TaskConfig config = new TaskConfig();
        RobotEnvironment env = new SimpleRobotEnv(config);
        
        RobotEnvironment.ActionSpaceSpec actionSpace = env.getActionSpace();
        
        assertNotNull(actionSpace);
        assertEquals(7, actionSpace.getContinuousDim());
        assertEquals(7, actionSpace.getDiscreteNum());
        assertNotNull(actionSpace.getContinuousLow());
        assertNotNull(actionSpace.getContinuousHigh());
    }
    
    @Test
    public void testObservationSpaceSpec() {
        TaskConfig config = new TaskConfig();
        RobotEnvironment env = new SimpleRobotEnv(config);
        
        RobotEnvironment.ObservationSpaceSpec obsSpace = env.getObservationSpace();
        
        assertNotNull(obsSpace);
        assertTrue(obsSpace.getImageHeight() > 0);
        assertTrue(obsSpace.getImageWidth() > 0);
        assertEquals(3, obsSpace.getImageChannels());
        assertTrue(obsSpace.getProprietaryDim() > 0);
    }
}
