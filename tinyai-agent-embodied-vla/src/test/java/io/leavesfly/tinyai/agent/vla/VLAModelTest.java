package io.leavesfly.tinyai.agent.vla;

import io.leavesfly.tinyai.agent.vla.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * VLA数据模型测试
 * 
 * @author TinyAI
 */
public class VLAModelTest {
    
    @Test
    public void testVisionInput() {
        double[][][] imageData = new double[64][64][3];
        NdArray rgbImage = new NdArray(imageData);
        
        VisionInput visionInput = new VisionInput(rgbImage);
        
        assertNotNull(visionInput);
        assertNotNull(visionInput.getRgbImage());
        assertEquals(64, visionInput.getRgbImage().getShape()[0]);
        assertEquals(64, visionInput.getRgbImage().getShape()[1]);
        assertEquals(3, visionInput.getRgbImage().getShape()[2]);
        assertTrue(visionInput.getTimestamp() > 0);
    }
    
    @Test
    public void testLanguageInput() {
        String instruction = "Pick up the red cube";
        LanguageInput languageInput = new LanguageInput(instruction);
        
        assertNotNull(languageInput);
        assertEquals(instruction, languageInput.getInstruction());
    }
    
    @Test
    public void testProprioceptionInput() {
        double[] jointPositions = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
        double[] jointVelocities = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        NdArray positions = new NdArray(jointPositions);
        NdArray velocities = new NdArray(jointVelocities);
        
        ProprioceptionInput proprioInput = new ProprioceptionInput(positions, velocities);
        
        assertNotNull(proprioInput);
        assertNotNull(proprioInput.getJointPositions());
        assertNotNull(proprioInput.getJointVelocities());
        assertEquals(7, proprioInput.getJointPositions().getShape()[0]);
        assertEquals(0.0, proprioInput.getGripperState());
    }
    
    @Test
    public void testVLAState() {
        double[][][] imageData = new double[64][64][3];
        VisionInput visionInput = new VisionInput(new NdArray(imageData));
        LanguageInput languageInput = new LanguageInput("Test instruction");
        
        VLAState state = new VLAState(visionInput, languageInput);
        
        assertNotNull(state);
        assertNotNull(state.getVisionInput());
        assertNotNull(state.getLanguageInput());
        assertNotNull(state.getAttentionWeights());
        assertTrue(state.getTimestamp() > 0);
    }
    
    @Test
    public void testVLAAction() {
        double[] actionValues = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
        NdArray continuousAction = new NdArray(actionValues);
        
        VLAAction action = new VLAAction(continuousAction, 0, ActionType.MOVE_END_EFFECTOR);
        
        assertNotNull(action);
        assertNotNull(action.getContinuousAction());
        assertEquals(0, action.getDiscreteAction());
        assertEquals(ActionType.MOVE_END_EFFECTOR, action.getActionType());
        assertEquals(1.0, action.getConfidence());
    }
    
    @Test
    public void testActionType() {
        assertEquals(7, ActionType.values().length);
        assertEquals("移动末端执行器", ActionType.MOVE_END_EFFECTOR.getDescription());
        assertEquals("抓取物体", ActionType.GRASP_OBJECT.getDescription());
    }
    
    @Test
    public void testTaskConfig() {
        TaskConfig config = new TaskConfig();
        config.setTaskName("PickAndPlace");
        config.setMaxSteps(100);
        config.setSuccessReward(100.0);
        
        assertEquals("PickAndPlace", config.getTaskName());
        assertEquals(100, config.getMaxSteps());
        assertEquals(100.0, config.getSuccessReward());
        assertFalse(config.isRender());
    }
}
