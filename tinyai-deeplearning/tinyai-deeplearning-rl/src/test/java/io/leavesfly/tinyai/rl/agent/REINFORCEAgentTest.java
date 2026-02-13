package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.rl.Experience;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * REINFORCEAgent类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class REINFORCEAgentTest {

    private REINFORCEAgent agent;
    private final int STATE_DIM = 4;
    private final int ACTION_DIM = 2;
    private final int[] HIDDEN_SIZES = {32, 32};
    private final float LEARNING_RATE = 0.001f;
    private final float GAMMA = 0.99f;
    private final String AGENT_NAME = "TestREINFORCEAgent";

    @Before
    public void setUp() {
        agent = new REINFORCEAgent(
            AGENT_NAME, STATE_DIM, ACTION_DIM, HIDDEN_SIZES,
            LEARNING_RATE, GAMMA, true  // 使用基线
        );
    }

    /**
     * 创建测试Experience
     */
    private Experience createExperience(float[] stateValues, int action, float reward, 
                                       float[] nextStateValues, boolean done) {
        Variable state = new Variable(NdArray.of(stateValues, Shape.of(1, stateValues.length)));
        Variable actionVar = new Variable(NdArray.of(new float[]{action}, Shape.of(1, 1)));
        Variable nextState = new Variable(NdArray.of(nextStateValues, Shape.of(1, nextStateValues.length)));
        return new Experience(state, actionVar, reward, nextState, done);
    }

    /**
     * 测试初始状态
     */
    @Test
    public void testInitialState() {
        assertEquals(AGENT_NAME, agent.getName());
        assertEquals(STATE_DIM, agent.getStateDim());
        assertEquals(ACTION_DIM, agent.getActionDim());
        assertEquals(LEARNING_RATE, agent.getLearningRate(), 0.001f);
        assertEquals(GAMMA, agent.getGamma(), 0.001f);
        assertEquals(0, agent.getTrainingStep());
        assertTrue(agent.isUsingBaseline());
    }

    /**
     * 测试动作选择
     */
    @Test
    public void testSelectAction() {
        Variable state = new Variable(NdArray.of(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, Shape.of(1, 4)));
        
        // 测试多次动作选择
        for (int i = 0; i < 20; i++) {
            Variable action = agent.selectAction(state);
            assertNotNull(action);
            assertNotNull(action.getValue());
            
            // 动作应该在有效范围内
            float actionValue = action.getValue().getNumber().floatValue();
            assertTrue("Action should be 0 or 1", actionValue == 0 || actionValue == 1);
        }
    }

    /**
     * 测试评估模式
     */
    @Test
    public void testEvalMode() {
        Variable state = new Variable(NdArray.of(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, Shape.of(1, 4)));
        
        // 训练模式下动作选择会存储log概率
        agent.train();
        assertTrue(agent.isTraining());
        
        // 评估模式下动作选择不存储log概率
        agent.eval();
        assertFalse(agent.isTraining());
        
        // 评估模式下仍然可以选动作
        Variable action = agent.selectAction(state);
        assertNotNull(action);
    }

    /**
     * 测试经验存储
     */
    @Test
    public void testStoreExperience() {
        Experience exp = createExperience(
            new float[]{0.1f, 0.2f, 0.3f, 0.4f},
            0,
            1.0f,
            new float[]{0.2f, 0.3f, 0.4f, 0.5f},
            false
        );
        
        agent.storeExperience(exp);
        // 经验已存储，后续learnFromEpisode会使用
    }

    /**
     * 测试回合学习（简化版）
     */
    @Test
    public void testLearnFromEpisode() {
        // 回合结束时调用learnFromEpisode（空回合）
        int stepsBefore = agent.getTrainingStep();
        agent.learnFromEpisode();
        
        // 空回合不应该增加训练步数
        assertEquals(stepsBefore, agent.getTrainingStep());
    }

    /**
     * 测试学习率设置
     */
    @Test
    public void testLearningRate() {
        float newLearningRate = 0.0001f;
        agent.setLearningRate(newLearningRate);
        assertEquals(newLearningRate, agent.getLearningRate(), 0.0001f);
    }

    /**
     * 测试折扣因子
     */
    @Test
    public void testGamma() {
        assertEquals(GAMMA, agent.getGamma(), 0.001f);
    }

    /**
     * 测试参数获取
     */
    @Test
    public void testGetAllParams() {
        var params = agent.getAllParams();
        assertNotNull(params);
        assertFalse(params.isEmpty());
    }

    /**
     * 测试模型重置
     */
    @Test
    public void testReset() {
        agent.reset();
        // 重置后应该可以正常选择动作
        Variable state = new Variable(NdArray.of(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, Shape.of(1, 4)));
        Variable action = agent.selectAction(state);
        assertNotNull(action);
    }

    /**
     * 测试训练统计信息
     */
    @Test
    public void testGetTrainingStats() {
        java.util.Map<String, Object> stats = agent.getTrainingStats();
        assertNotNull(stats);
        assertTrue(stats.containsKey("episode_count"));
        assertTrue(stats.containsKey("average_return"));
        assertTrue(stats.containsKey("average_policy_loss"));
        assertTrue(stats.containsKey("use_baseline"));
    }

    /**
     * 测试统计信息重置
     */
    @Test
    public void testResetTrainingStats() {
        // 先进行一些学习
        for (int i = 0; i < 5; i++) {
            Experience exp = createExperience(
                new float[]{0.1f * i, 0.2f * i, 0.3f * i, 0.4f * i},
                i % ACTION_DIM,
                1.0f,
                new float[]{0.1f * (i + 1), 0.2f * (i + 1), 0.3f * (i + 1), 0.4f * (i + 1)},
                i == 4
            );
            agent.learn(exp);
        }
        agent.learnFromEpisode();
        
        // 重置统计
        agent.resetTrainingStats();
        
        java.util.Map<String, Object> stats = agent.getTrainingStats();
        assertEquals(0, stats.get("episode_count"));
    }

    /**
     * 测试不使用基线的智能体
     */
    @Test
    public void testWithoutBaseline() {
        REINFORCEAgent agentNoBaseline = new REINFORCEAgent(
            "NoBaselineAgent", STATE_DIM, ACTION_DIM, HIDDEN_SIZES,
            LEARNING_RATE, GAMMA, false  // 不使用基线
        );
        
        assertFalse(agentNoBaseline.isUsingBaseline());
        
        // 测试动作选择
        Variable state = new Variable(NdArray.of(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, Shape.of(1, 4)));
        Variable action = agentNoBaseline.selectAction(state);
        assertNotNull(action);
    }
}
