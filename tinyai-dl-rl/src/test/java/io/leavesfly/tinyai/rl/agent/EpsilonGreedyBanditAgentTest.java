package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.rl.Experience;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * EpsilonGreedyBanditAgent类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class EpsilonGreedyBanditAgentTest {

    private EpsilonGreedyBanditAgent agent;
    private final int NUM_ARMS = 3;
    private final float INITIAL_EPSILON = 0.1f;
    private final String AGENT_NAME = "TestEpsilonGreedyAgent";

    @Before
    public void setUp() {
        agent = new EpsilonGreedyBanditAgent(AGENT_NAME, NUM_ARMS, INITIAL_EPSILON);
    }

    /**
     * 创建测试Experience
     */
    private Experience createExperience(int action, float reward) {
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        Variable actionVar = new Variable(NdArray.of(new float[]{action}, Shape.of(1)));
        Variable nextState = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        return new Experience(state, actionVar, reward, nextState, false);
    }

    /**
     * 测试初始状态
     */
    @Test
    public void testInitialState() {
        assertEquals(AGENT_NAME, agent.getName());
        assertEquals(NUM_ARMS, agent.getActionDim());
        assertEquals(1, agent.getStateDim());
        assertEquals(INITIAL_EPSILON, agent.getCurrentEpsilon(), 0.001f);
        assertEquals(0, agent.getTotalActions());
        assertEquals(0, agent.getTrainingStep());
        
        // 验证所有臂的初始统计
        for (int i = 0; i < NUM_ARMS; i++) {
            assertEquals(0, agent.getActionCount(i));
            assertEquals(0.0f, agent.getEstimatedReward(i), 0.001f);
            assertEquals(0.0f, agent.getTotalReward(i), 0.001f);
        }
    }

    /**
     * 测试动作选择范围
     */
    @Test
    public void testActionSelectionRange() {
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        
        // 多次选择动作，验证都在有效范围内
        for (int i = 0; i < 100; i++) {
            Variable action = agent.selectAction(state);
            int armIndex = (int) action.getValue().get(0);
            assertTrue("动作应该在有效范围内", armIndex >= 0 && armIndex < NUM_ARMS);
        }
    }

    /**
     * 测试selectArm方法
     */
    @Test
    public void testSelectArm() {
        // 多次选择，验证范围
        for (int i = 0; i < 50; i++) {
            int arm = agent.selectArm();
            assertTrue("臂索引应该在有效范围内", arm >= 0 && arm < NUM_ARMS);
        }
    }

    /**
     * 测试学习和统计更新
     */
    @Test
    public void testLearningAndStatistics() {
        // 模拟学习过程：臂0获得高奖励，臂1获得低奖励
        for (int i = 0; i < 10; i++) {
            Experience exp0 = createExperience(0, 1.0f);
            Experience exp1 = createExperience(1, 0.1f);
            
            agent.learn(exp0);
            agent.learn(exp1);
        }
        
        // 验证统计信息
        assertEquals(10, agent.getActionCount(0));
        assertEquals(10, agent.getActionCount(1));
        assertEquals(0, agent.getActionCount(2));
        assertEquals(20, agent.getTotalActions());
        assertEquals(20, agent.getTrainingStep());
        
        assertEquals(1.0f, agent.getEstimatedReward(0), 0.001f);
        assertEquals(0.1f, agent.getEstimatedReward(1), 0.001f);
        assertEquals(0.0f, agent.getEstimatedReward(2), 0.001f);
        
        assertEquals(10.0f, agent.getTotalReward(0), 0.001f);
        assertEquals(1.0f, agent.getTotalReward(1), 0.001f);
        assertEquals(0.0f, agent.getTotalReward(2), 0.001f);
    }

    /**
     * 测试最优臂识别
     */
    @Test
    public void testBestArmIdentification() {
        // 让臂1获得最高奖励
        for (int i = 0; i < 5; i++) {
            agent.learn(createExperience(0, 0.5f));
            agent.learn(createExperience(1, 1.0f));
            agent.learn(createExperience(2, 0.2f));
        }
        
        assertEquals(1, agent.getBestArmIndex());
        assertEquals(1.0f, agent.getBestEstimatedReward(), 0.001f);
    }

    /**
     * 测试epsilon衰减
     */
    @Test
    public void testEpsilonDecay() {
        float initialEpsilon = agent.getCurrentEpsilon();
        
        // 进行一些学习，epsilon应该衰减
        for (int i = 0; i < 10; i++) {
            agent.learn(createExperience(0, 1.0f));
        }
        
        assertTrue("Epsilon应该衰减", agent.getCurrentEpsilon() < initialEpsilon);
        assertTrue("Epsilon不应该低于最小值", agent.getCurrentEpsilon() >= agent.getMinEpsilon());
    }

    /**
     * 测试epsilon参数设置
     */
    @Test
    public void testEpsilonParameterSetting() {
        // 测试设置epsilon
        agent.setCurrentEpsilon(0.5f);
        assertEquals(0.5f, agent.getCurrentEpsilon(), 0.001f);
        
        // 测试边界值
        agent.setCurrentEpsilon(-0.1f);
        assertEquals(0.0f, agent.getCurrentEpsilon(), 0.001f);
        
        agent.setCurrentEpsilon(1.1f);
        assertEquals(1.0f, agent.getCurrentEpsilon(), 0.001f);
        
        // 测试衰减率设置
        agent.setEpsilonDecay(0.9f);
        assertEquals(0.9f, agent.getEpsilonDecay(), 0.001f);
        
        // 测试最小epsilon设置
        agent.setMinEpsilon(0.05f);
        assertEquals(0.05f, agent.getMinEpsilon(), 0.001f);
    }

    /**
     * 测试贪婪行为（epsilon=0）
     */
    @Test
    public void testGreedyBehavior() {
        // 创建一个新的代理实例确保干净的测试环境
        EpsilonGreedyBanditAgent testAgent = new EpsilonGreedyBanditAgent("TestGreedyAgent", NUM_ARMS, 0.0f);
        testAgent.setEpsilonDecay(1.0f); // 不衰减
        testAgent.setSeed(12345L); // 设置固定种子确保可重现性
        
        // 训练使臂1成为最优，且显著优于其他臂
        for (int i = 0; i < 20; i++) {
            testAgent.learn(createExperience(0, 0.1f)); // 臂0低奖励
            testAgent.learn(createExperience(1, 2.0f)); // 臂1非常高奖励
            testAgent.learn(createExperience(2, 0.2f)); // 臂2低奖励
        }
        
        // 验证臂1确实成为最优
        assertEquals("Best arm should be 1", 1, testAgent.getBestArmIndex());
        assertTrue("Arm 1 should have highest reward", testAgent.getEstimatedReward(1) > 1.5f);
        
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        
        // 多次选择，应该总是选择臂1
        for (int i = 0; i < 20; i++) {
            Variable action = testAgent.selectAction(state);
            int selectedArm = (int) action.getValue().get(0);
            assertEquals("Should always select best arm (1)", 1, selectedArm);
        }
    }

    /**
     * 测试探索行为（epsilon=1）
     */
    @Test
    public void testExplorationBehavior() {
        // 设置epsilon为1，确保纯探索行为
        agent.setCurrentEpsilon(1.0f);
        agent.setEpsilonDecay(1.0f); // 不衰减
        
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        
        // 收集选择的动作
        boolean[] actionsSeen = new boolean[NUM_ARMS];
        for (int i = 0; i < 100; i++) {
            Variable action = agent.selectAction(state);
            int armIndex = (int) action.getValue().get(0);
            actionsSeen[armIndex] = true;
        }
        
        // 应该看到所有动作都被选择过
        for (int i = 0; i < NUM_ARMS; i++) {
            assertTrue("所有动作都应该被探索过", actionsSeen[i]);
        }
    }

    /**
     * 测试批量学习
     */
    @Test
    public void testBatchLearning() {
        Experience[] experiences = {
            createExperience(0, 1.0f),
            createExperience(1, 0.5f),
            createExperience(0, 1.2f)
        };
        
        agent.learnBatch(experiences);
        
        assertEquals(2, agent.getActionCount(0));
        assertEquals(1, agent.getActionCount(1));
        assertEquals(0, agent.getActionCount(2));
        assertEquals(1.1f, agent.getEstimatedReward(0), 0.001f);
        assertEquals(0.5f, agent.getEstimatedReward(1), 0.001f);
    }

    /**
     * 测试经验存储
     */
    @Test
    public void testExperienceStorage() {
        Experience experience = createExperience(0, 1.5f);
        agent.storeExperience(experience);
        
        // storeExperience应该直接调用learn
        assertEquals(1, agent.getActionCount(0));
        assertEquals(1.5f, agent.getEstimatedReward(0), 0.001f);
    }

    /**
     * 测试重置功能
     */
    @Test
    public void testReset() {
        // 先进行一些学习
        for (int i = 0; i < 5; i++) {
            agent.learn(createExperience(0, 1.0f));
        }
        
        float epsilonBeforeReset = agent.getCurrentEpsilon();
        
        agent.reset();
        
        // 验证统计信息被重置
        assertEquals(0, agent.getTotalActions());
        assertEquals(0, agent.getTrainingStep());
        for (int i = 0; i < NUM_ARMS; i++) {
            assertEquals(0, agent.getActionCount(i));
            assertEquals(0.0f, agent.getEstimatedReward(i), 0.001f);
            assertEquals(0.0f, agent.getTotalReward(i), 0.001f);
        }
        
        // epsilon通常不重置
        assertEquals(epsilonBeforeReset, agent.getCurrentEpsilon(), 0.001f);
    }

    /**
     * 测试随机种子设置
     */
    @Test
    public void testRandomSeed() {
        agent.setCurrentEpsilon(1.0f); // 纯探索以测试随机性
        agent.setSeed(12345L);
        
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        
        // 记录第一次的选择序列
        int[] sequence1 = new int[10];
        for (int i = 0; i < 10; i++) {
            sequence1[i] = (int) agent.selectAction(state).getValue().get(0);
        }
        
        // 重置并使用相同种子
        agent.setSeed(12345L);
        
        // 记录第二次的选择序列
        int[] sequence2 = new int[10];
        for (int i = 0; i < 10; i++) {
            sequence2[i] = (int) agent.selectAction(state).getValue().get(0);
        }
        
        // 序列应该相同
        assertArrayEquals("相同种子应该产生相同的随机序列", sequence1, sequence2);
    }

    /**
     * 测试最优选择率计算
     */
    @Test
    public void testOptimalActionRate() {
        // 初始应该为0
        assertEquals(0.0f, agent.getOptimalActionRate(), 0.001f);
        
        // 让臂0成为最优，但选择其他臂
        agent.learn(createExperience(0, 1.0f));
        agent.learn(createExperience(1, 0.5f));
        agent.learn(createExperience(1, 0.5f));
        
        // 最优臂是0，但只选择了1次，总共3次
        assertEquals(1.0f / 3.0f, agent.getOptimalActionRate(), 0.001f);
    }

    /**
     * 测试算法描述信息
     */
    @Test
    public void testAlgorithmDescription() {
        String description = agent.getAlgorithmDescription();
        assertNotNull(description);
        assertTrue(description.contains("ε-贪心"));
        assertTrue(description.contains(String.valueOf(agent.getCurrentEpsilon())));
        assertTrue(description.contains(String.valueOf(agent.getEpsilonDecay())));
        assertTrue(description.contains(String.valueOf(agent.getMinEpsilon())));
    }

    /**
     * 测试getAllEstimatedRewards方法返回副本
     */
    @Test
    public void testGetAllEstimatedRewardsReturnsCopy() {
        agent.learn(createExperience(0, 1.0f));
        
        float[] rewards1 = agent.getAllEstimatedRewards();
        float[] rewards2 = agent.getAllEstimatedRewards();
        
        // 应该是不同的对象但内容相同
        assertNotSame(rewards1, rewards2);
        assertArrayEquals(rewards1, rewards2, 0.001f);
        
        // 修改返回的数组不应影响内部状态
        rewards1[0] = 999.0f;
        assertEquals(1.0f, agent.getEstimatedReward(0), 0.001f);
    }

    /**
     * 测试getAllActionCounts方法返回副本
     */
    @Test
    public void testGetAllActionCountsReturnsCopy() {
        agent.learn(createExperience(0, 1.0f));
        
        int[] counts1 = agent.getAllActionCounts();
        int[] counts2 = agent.getAllActionCounts();
        
        // 应该是不同的对象但内容相同
        assertNotSame(counts1, counts2);
        assertArrayEquals(counts1, counts2);
        
        // 修改返回的数组不应影响内部状态
        counts1[0] = 999;
        assertEquals(1, agent.getActionCount(0));
    }
}