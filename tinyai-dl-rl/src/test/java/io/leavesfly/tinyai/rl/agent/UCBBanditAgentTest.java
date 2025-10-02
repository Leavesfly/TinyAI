package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.rl.Experience;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * UCBBanditAgent类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class UCBBanditAgentTest {

    private UCBBanditAgent agent;
    private final int NUM_ARMS = 4;
    private final String AGENT_NAME = "TestUCBAgent";
    private final float DEFAULT_CONFIDENCE = (float) Math.sqrt(2.0);

    @Before
    public void setUp() {
        agent = new UCBBanditAgent(AGENT_NAME, NUM_ARMS);
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
        assertEquals(DEFAULT_CONFIDENCE, agent.getConfidenceParam(), 0.001f);
        assertEquals(0, agent.getTotalActions());
        
        // 验证所有臂的初始统计
        for (int i = 0; i < NUM_ARMS; i++) {
            assertEquals(0, agent.getActionCount(i));
            assertEquals(0.0f, agent.getEstimatedReward(i), 0.001f);
            assertEquals(Float.POSITIVE_INFINITY, agent.getUCBValue(i), 0.001f);
            assertEquals(Float.POSITIVE_INFINITY, agent.getConfidenceInterval(i), 0.001f);
        }
    }

    /**
     * 测试自定义置信度参数的构造函数
     */
    @Test
    public void testCustomConfidenceConstructor() {
        float customConfidence = 2.0f;
        UCBBanditAgent customAgent = new UCBBanditAgent(AGENT_NAME, NUM_ARMS, customConfidence);
        assertEquals(customConfidence, customAgent.getConfidenceParam(), 0.001f);
    }

    /**
     * 测试初始选择行为（优先选择未尝试的臂）
     */
    @Test
    public void testInitialArmSelection() {
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        
        // 前NUM_ARMS次选择应该覆盖所有臂
        boolean[] armSelected = new boolean[NUM_ARMS];
        for (int i = 0; i < NUM_ARMS; i++) {
            Variable action = agent.selectAction(state);
            int armIndex = (int) action.getValue().get(0);
            
            assertTrue("选择的臂应该在有效范围内", armIndex >= 0 && armIndex < NUM_ARMS);
            assertFalse("每个臂应该只被选择一次", armSelected[armIndex]);
            armSelected[armIndex] = true;
            
            // 模拟学习该选择
            agent.learn(createExperience(armIndex, 1.0f));
        }
        
        // 验证所有臂都被选择过
        for (boolean selected : armSelected) {
            assertTrue("所有臂都应该被选择过", selected);
        }
    }

    /**
     * 测试UCB值计算
     */
    @Test
    public void testUCBCalculation() {
        // 初始状态，所有臂的UCB应该是无穷大
        for (int i = 0; i < NUM_ARMS; i++) {
            assertEquals(Float.POSITIVE_INFINITY, agent.getUCBValue(i), 0.001f);
        }
        
        // 选择一些臂后计算UCB
        agent.learn(createExperience(0, 1.0f));
        agent.learn(createExperience(1, 0.5f));
        agent.learn(createExperience(0, 0.8f)); // 再次选择臂0
        
        // 臂0被选择2次，平均奖励0.9
        float ucb0 = agent.getUCBValue(0);
        assertFalse("UCB值应该是有限的", Float.isInfinite(ucb0));
        assertTrue("UCB应该大于平均奖励", ucb0 > 0.9f);
        
        // 臂1被选择1次，平均奖励0.5
        float ucb1 = agent.getUCBValue(1);
        assertFalse("UCB值应该是有限的", Float.isInfinite(ucb1));
        assertTrue("UCB应该大于平均奖励", ucb1 > 0.5f);
        
        // 未选择的臂仍应该是无穷大
        assertEquals(Float.POSITIVE_INFINITY, agent.getUCBValue(2), 0.001f);
    }

    /**
     * 测试置信区间计算
     */
    @Test
    public void testConfidenceIntervalCalculation() {
        // 选择一些臂
        agent.learn(createExperience(0, 1.0f));
        agent.learn(createExperience(1, 0.5f));
        agent.learn(createExperience(0, 0.8f));
        agent.learn(createExperience(1, 0.6f));
        
        float interval0 = agent.getConfidenceInterval(0);
        float interval1 = agent.getConfidenceInterval(1);
        
        assertFalse("置信区间应该是有限的", Float.isInfinite(interval0));
        assertFalse("置信区间应该是有限的", Float.isInfinite(interval1));
        assertTrue("置信区间应该为正", interval0 > 0);
        assertTrue("置信区间应该为正", interval1 > 0);
        
        // 两个臂被选择次数相同，置信区间应该相等
        assertEquals(interval0, interval1, 0.001f);
    }

    /**
     * 测试UCB选择策略
     */
    @Test
    public void testUCBSelectionStrategy() {
        // 让臂0获得高奖励但被选择多次，臂1获得中等奖励但选择较少
        for (int i = 0; i < 10; i++) {
            agent.learn(createExperience(0, 0.8f));
        }
        agent.learn(createExperience(1, 0.6f));
        agent.learn(createExperience(2, 0.4f));
        
        // 获取UCB值
        float ucb0 = agent.getUCBValue(0);
        float ucb1 = agent.getUCBValue(1);
        float ucb2 = agent.getUCBValue(2);
        
        // 由于臂1和臂2被选择次数较少，它们的UCB值应该较高
        assertTrue("臂1的UCB应该很高由于不确定性", ucb1 > ucb0);
        assertTrue("臂2的UCB应该很高由于不确定性", ucb2 > ucb0);
        
        // 未选择的臂3应该有最高的UCB（无穷大）
        assertEquals(Float.POSITIVE_INFINITY, agent.getUCBValue(3), 0.001f);
    }

    /**
     * 测试getAllUCBValues方法
     */
    @Test
    public void testGetAllUCBValues() {
        // 训练一些臂
        agent.learn(createExperience(0, 1.0f));
        agent.learn(createExperience(1, 0.5f));
        
        float[] ucbValues = agent.getAllUCBValues();
        assertEquals(NUM_ARMS, ucbValues.length);
        
        // 验证与单独获取的值一致
        for (int i = 0; i < NUM_ARMS; i++) {
            assertEquals(agent.getUCBValue(i), ucbValues[i], 0.001f);
        }
    }

    /**
     * 测试getAllConfidenceIntervals方法
     */
    @Test
    public void testGetAllConfidenceIntervals() {
        // 训练一些臂
        agent.learn(createExperience(0, 1.0f));
        agent.learn(createExperience(1, 0.5f));
        
        float[] intervals = agent.getAllConfidenceIntervals();
        assertEquals(NUM_ARMS, intervals.length);
        
        // 验证与单独获取的值一致
        for (int i = 0; i < NUM_ARMS; i++) {
            assertEquals(agent.getConfidenceInterval(i), intervals[i], 0.001f);
        }
    }

    /**
     * 测试置信度参数设置
     */
    @Test
    public void testConfidenceParameterSetting() {
        float newConfidence = 3.0f;
        agent.setConfidenceParam(newConfidence);
        assertEquals(newConfidence, agent.getConfidenceParam(), 0.001f);
        
        // 测试负值（应该被设为0）
        agent.setConfidenceParam(-1.0f);
        assertEquals(0.0f, agent.getConfidenceParam(), 0.001f);
    }

    /**
     * 测试算法收敛行为
     */
    @Test
    public void testConvergenceBehavior() {
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        
        // 模拟一个明显最优的环境：臂1远优于其他臂
        float[] trueRewards = {0.2f, 0.9f, 0.3f, 0.1f};
        
        // 进行大量学习
        for (int round = 0; round < 50; round++) {
            Variable action = agent.selectAction(state);
            int armIndex = (int) action.getValue().get(0);
            
            // 根据真实奖励给予反馈（加入一些噪声）
            float reward = trueRewards[armIndex] + (float) (Math.random() - 0.5) * 0.1f;
            agent.learn(createExperience(armIndex, reward));
        }
        
        // 经过充分学习后，最优臂应该被识别
        int bestArm = agent.getBestArmIndex();
        assertTrue("应该倾向于识别出最优臂", bestArm == 1 || agent.getEstimatedReward(1) > 0.7f);
    }

    /**
     * 测试选择信息获取
     */
    @Test
    public void testSelectionInfo() {
        agent.learn(createExperience(0, 1.0f));
        agent.learn(createExperience(1, 0.5f));
        
        String info = agent.getSelectionInfo();
        assertNotNull(info);
        assertTrue(info.contains("UCB选择策略"));
        assertTrue(info.contains("推荐选择臂"));
        assertTrue(info.contains("各臂详情"));
    }

    /**
     * 测试算法描述
     */
    @Test
    public void testAlgorithmDescription() {
        String description = agent.getAlgorithmDescription();
        assertNotNull(description);
        assertTrue("Description should contain UCB", description.contains("UCB"));
        // 检查是否包含置信度参数值
        String confidenceStr = String.format("%.4f", agent.getConfidenceParam());
        assertTrue("Description should contain confidence parameter: " + description, 
                  description.contains(confidenceStr));
    }

    /**
     * 测试批量学习
     */
    @Test
    public void testBatchLearning() {
        Experience[] experiences = {
            createExperience(0, 1.0f),
            createExperience(1, 0.5f),
            createExperience(0, 1.2f),
            createExperience(2, 0.8f)
        };
        
        agent.learnBatch(experiences);
        
        assertEquals(2, agent.getActionCount(0));
        assertEquals(1, agent.getActionCount(1));
        assertEquals(1, agent.getActionCount(2));
        assertEquals(0, agent.getActionCount(3));
        
        assertEquals(1.1f, agent.getEstimatedReward(0), 0.001f);
        assertEquals(0.5f, agent.getEstimatedReward(1), 0.001f);
        assertEquals(0.8f, agent.getEstimatedReward(2), 0.001f);
    }

    /**
     * 测试重置功能
     */
    @Test
    public void testReset() {
        // 先进行一些学习
        for (int i = 0; i < 3; i++) {
            agent.learn(createExperience(i, 1.0f));
        }
        
        agent.reset();
        
        // 验证统计信息被重置
        assertEquals(0, agent.getTotalActions());
        for (int i = 0; i < NUM_ARMS; i++) {
            assertEquals(0, agent.getActionCount(i));
            assertEquals(0.0f, agent.getEstimatedReward(i), 0.001f);
            assertEquals(Float.POSITIVE_INFINITY, agent.getUCBValue(i), 0.001f);
            assertEquals(Float.POSITIVE_INFINITY, agent.getConfidenceInterval(i), 0.001f);
        }
    }

    /**
     * 测试置信度参数对行为的影响
     */
    @Test
    public void testConfidenceParameterEffect() {
        // 创建两个相同配置但不同置信度的智能体
        UCBBanditAgent conservativeAgent = new UCBBanditAgent("Conservative", NUM_ARMS, 0.5f);
        UCBBanditAgent aggressiveAgent = new UCBBanditAgent("Aggressive", NUM_ARMS, 3.0f);
        
        // 让两个智能体都学习相同的经验
        for (int i = 0; i < 2; i++) {
            Experience exp = createExperience(0, 1.0f);
            conservativeAgent.learn(exp);
            aggressiveAgent.learn(exp);
        }
        
        // 高置信度参数应该导致更大的置信区间
        float conservativeInterval = conservativeAgent.getConfidenceInterval(0);
        float aggressiveInterval = aggressiveAgent.getConfidenceInterval(0);
        
        assertTrue("高置信度参数应该产生更大的置信区间", 
                  aggressiveInterval > conservativeInterval);
    }

    /**
     * 测试边界情况：单臂情况
     */
    @Test
    public void testSingleArm() {
        UCBBanditAgent singleArmAgent = new UCBBanditAgent("SingleArm", 1);
        Variable state = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        
        // 单臂情况下，总是应该选择唯一的臂
        for (int i = 0; i < 10; i++) {
            Variable action = singleArmAgent.selectAction(state);
            assertEquals(0.0f, action.getValue().get(0), 0.001f);
            singleArmAgent.learn(createExperience(0, (float) Math.random()));
        }
        
        assertEquals(10, singleArmAgent.getTotalActions());
        assertEquals(10, singleArmAgent.getActionCount(0));
    }

    /**
     * 测试UCB值的单调性（更多选择导致更小的置信区间）
     */
    @Test
    public void testUCBMonotonicity() {
        // 只测试置信区间的单调性，因为UCB值受总选择次数影响
        // 让臂0获得一些经验以获得有意义的置信区间
        for (int i = 0; i < 5; i++) {
            agent.learn(createExperience(0, 1.0f));
        }
        
        float interval1 = agent.getConfidenceInterval(0);
        
        // 再增加一些经验
        for (int i = 0; i < 5; i++) {
            agent.learn(createExperience(0, 1.0f));
        }
        
        float interval2 = agent.getConfidenceInterval(0);
        
        // 更多的选择应该导致更小的置信区间
        assertTrue(String.format("Interval should decrease: %.6f -> %.6f", interval1, interval2),
                  interval2 < interval1);
    }
}