package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.rl.Experience;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * DQNAgent类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class DQNAgentTest {

    private DQNAgent agent;
    private final int STATE_DIM = 4;
    private final int ACTION_DIM = 2;
    private final int[] HIDDEN_SIZES = {32, 32};
    private final float LEARNING_RATE = 0.001f;
    private final float EPSILON = 0.1f;
    private final float GAMMA = 0.99f;
    private final int BATCH_SIZE = 8;
    private final int BUFFER_SIZE = 100;
    private final int TARGET_UPDATE_FREQ = 10;
    private final String AGENT_NAME = "TestDQNAgent";

    @Before
    public void setUp() {
        agent = new DQNAgent(
            AGENT_NAME, STATE_DIM, ACTION_DIM, HIDDEN_SIZES,
            LEARNING_RATE, EPSILON, GAMMA, BATCH_SIZE, BUFFER_SIZE, TARGET_UPDATE_FREQ
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
        assertEquals(EPSILON, agent.getEpsilon(), 0.001f);
        assertEquals(GAMMA, agent.getGamma(), 0.001f);
        assertEquals(0, agent.getTrainingStep());
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
     * 测试探索率衰减
     */
    @Test
    public void testEpsilonDecay() {
        float initialEpsilon = agent.getEpsilon();
        agent.decayEpsilon(0.95f);
        float decayedEpsilon = agent.getEpsilon();
        
        assertTrue("Epsilon should decay", decayedEpsilon < initialEpsilon);
        assertEquals(initialEpsilon * 0.95f, decayedEpsilon, 0.001f);
    }

    /**
     * 测试探索率下限
     */
    @Test
    public void testEpsilonLowerBound() {
        // 多次衰减，测试下限
        for (int i = 0; i < 100; i++) {
            agent.decayEpsilon(0.9f);
        }
        
        float epsilon = agent.getEpsilon();
        assertTrue("Epsilon should not go below 0.01", epsilon >= 0.01f);
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
        // 验证经验已存储（通过检查缓冲区大小）
        // 注意：DQNAgent内部使用ReplayBuffer，但canSample方法不直接暴露
    }

    /**
     * 测试学习功能（简化版，验证接口调用）
     */
    @Test
    public void testLearn() {
        // 添加经验（不触发实际学习，因为batch size不够）
        for (int i = 0; i < 3; i++) {
            Experience exp = createExperience(
                new float[]{0.1f * i, 0.2f * i, 0.3f * i, 0.4f * i},
                i % ACTION_DIM,
                1.0f,
                new float[]{0.1f * (i + 1), 0.2f * (i + 1), 0.3f * (i + 1), 0.4f * (i + 1)},
                false
            );
            agent.learn(exp);
        }
        
        // 训练步数应该为0，因为经验不足batch size
        assertEquals(0, agent.getTrainingStep());
    }

    /**
     * 测试批量学习接口
     */
    @Test
    public void testLearnBatch() {
        // 批量学习接口测试（空批次）
        int stepsBefore = agent.getTrainingStep();
        agent.learnBatch(new Experience[0]);
        
        // 空批次不应该增加训练步数
        assertEquals(stepsBefore, agent.getTrainingStep());
    }

    /**
     * 测试训练模式切换
     */
    @Test
    public void testTrainingMode() {
        assertTrue(agent.isTraining());
        
        agent.eval();
        assertFalse(agent.isTraining());
        
        agent.train();
        assertTrue(agent.isTraining());
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
     * 测试获取训练统计信息
     */
    @Test
    public void testGetTrainingStats() {
        java.util.Map<String, Object> stats = agent.getTrainingStats();
        assertNotNull(stats);
        assertTrue(stats.containsKey("training_step"));
        assertTrue(stats.containsKey("epsilon"));
        assertTrue(stats.containsKey("average_loss"));
    }
}
