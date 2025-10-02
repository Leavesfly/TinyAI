package io.leavesfly.tinyai.rl.policy;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * EpsilonGreedyPolicy类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class EpsilonGreedyPolicyTest {

    private EpsilonGreedyPolicy policy;
    private MockQFunction qFunction;
    private final int STATE_DIM = 2;
    private final int ACTION_DIM = 3;
    private final float EPSILON = 0.1f;

    /**
     * 模拟Q函数用于测试
     */
    private static class MockQFunction implements EpsilonGreedyPolicy.QFunction {
        private float[] qValues;

        public MockQFunction(float[] qValues) {
            this.qValues = qValues;
        }

        @Override
        public Variable getQValues(Variable state) {
            return new Variable(NdArray.of(qValues, Shape.of(1, qValues.length)));
        }

        public void setQValues(float[] qValues) {
            this.qValues = qValues;
        }
    }

    @Before
    public void setUp() {
        // 设置Q值：动作0=1.0, 动作1=2.0, 动作2=0.5
        // 因此动作1应该是贪婪选择
        qFunction = new MockQFunction(new float[]{1.0f, 2.0f, 0.5f});
        policy = new EpsilonGreedyPolicy(STATE_DIM, ACTION_DIM, EPSILON, qFunction);
    }

    /**
     * 测试策略的基本属性
     */
    @Test
    public void testBasicProperties() {
        assertEquals("EpsilonGreedy", policy.getName());
        assertEquals(STATE_DIM, policy.getStateDim());
        assertEquals(ACTION_DIM, policy.getActionDim());
        assertEquals(EPSILON, policy.getEpsilon(), 0.001f);
    }

    /**
     * 测试贪婪动作选择（epsilon=0时）
     */
    @Test
    public void testGreedyActionSelection() {
        // 设置epsilon为0，强制贪婪选择
        policy.setEpsilon(0.0f);
        
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        
        // 多次选择，应该总是选择动作1（Q值最高）
        for (int i = 0; i < 10; i++) {
            Variable action = policy.selectAction(state);
            assertEquals(1.0f, action.getValue().getNumber().floatValue(), 0.001f);
        }
    }

    /**
     * 测试探索行为（epsilon=1时）
     */
    @Test
    public void testExplorationBehavior() {
        // 设置epsilon为1，强制探索
        policy.setEpsilon(1.0f);
        
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        
        // 收集多次选择的结果
        boolean[] actionsSeen = new boolean[ACTION_DIM];
        for (int i = 0; i < 100; i++) {
            Variable action = policy.selectAction(state);
            int actionIndex = (int) action.getValue().getNumber().floatValue();
            assertTrue("动作索引应该在有效范围内", actionIndex >= 0 && actionIndex < ACTION_DIM);
            actionsSeen[actionIndex] = true;
        }
        
        // 由于是随机探索，应该看到多个不同的动作
        int distinctActions = 0;
        for (boolean seen : actionsSeen) {
            if (seen) distinctActions++;
        }
        assertTrue("探索应该产生多个不同的动作", distinctActions > 1);
    }

    /**
     * 测试动作概率分布计算
     */
    @Test
    public void testActionProbabilities() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        Variable probs = policy.getActionProbabilities(state);
        
        assertEquals(Shape.of(1, ACTION_DIM), probs.getValue().getShape());
        
        // 验证概率总和为1
        float sum = 0.0f;
        for (int i = 0; i < ACTION_DIM; i++) {
            float prob = probs.getValue().get(0, i);
            assertTrue("概率应该非负", prob >= 0.0f);
            sum += prob;
        }
        assertEquals(1.0f, sum, 0.001f);
        
        // 最优动作（动作1）应该有最高概率
        float bestActionProb = probs.getValue().get(0, 1);
        float expectedBestProb = (1.0f - EPSILON) + (EPSILON / ACTION_DIM);
        assertEquals(expectedBestProb, bestActionProb, 0.001f);
        
        // 其他动作应该有相等的较低概率
        float otherActionProb = EPSILON / ACTION_DIM;
        assertEquals(otherActionProb, probs.getValue().get(0, 0), 0.001f);
        assertEquals(otherActionProb, probs.getValue().get(0, 2), 0.001f);
    }

    /**
     * 测试特定状态-动作对的概率
     */
    @Test
    public void testActionProbability() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        
        // 测试最优动作的概率
        Variable bestAction = new Variable(NdArray.of(1.0f));
        float bestProb = policy.getActionProbability(state, bestAction);
        float expectedBestProb = (1.0f - EPSILON) + (EPSILON / ACTION_DIM);
        assertEquals(expectedBestProb, bestProb, 0.001f);
        
        // 测试次优动作的概率
        Variable suboptimalAction = new Variable(NdArray.of(0.0f));
        float suboptimalProb = policy.getActionProbability(state, suboptimalAction);
        float expectedSuboptimalProb = EPSILON / ACTION_DIM;
        assertEquals(expectedSuboptimalProb, suboptimalProb, 0.001f);
    }

    /**
     * 测试对数概率计算
     */
    @Test
    public void testLogProbability() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        Variable action = new Variable(NdArray.of(1.0f)); // 最优动作
        
        Variable logProb = policy.getLogProbability(state, action);
        float expectedProb = (1.0f - EPSILON) + (EPSILON / ACTION_DIM);
        float expectedLogProb = (float) Math.log(expectedProb);
        
        assertEquals(expectedLogProb, logProb.getValue().getNumber().floatValue(), 0.001f);
    }

    /**
     * 测试epsilon设置和边界值
     */
    @Test
    public void testEpsilonSetting() {
        // 测试正常范围内的值
        policy.setEpsilon(0.5f);
        assertEquals(0.5f, policy.getEpsilon(), 0.001f);
        
        // 测试边界值
        policy.setEpsilon(0.0f);
        assertEquals(0.0f, policy.getEpsilon(), 0.001f);
        
        policy.setEpsilon(1.0f);
        assertEquals(1.0f, policy.getEpsilon(), 0.001f);
        
        // 测试超出范围的值（应该被限制）
        policy.setEpsilon(-0.1f);
        assertEquals(0.0f, policy.getEpsilon(), 0.001f);
        
        policy.setEpsilon(1.1f);
        assertEquals(1.0f, policy.getEpsilon(), 0.001f);
    }

    /**
     * 测试epsilon衰减
     */
    @Test
    public void testEpsilonDecay() {
        float initialEpsilon = 0.5f;
        float decayRate = 0.9f;
        float minEpsilon = 0.01f;
        
        policy.setEpsilon(initialEpsilon);
        policy.decayEpsilon(decayRate, minEpsilon);
        
        float expectedEpsilon = initialEpsilon * decayRate;
        assertEquals(expectedEpsilon, policy.getEpsilon(), 0.001f);
        
        // 测试多次衰减
        for (int i = 0; i < 10; i++) {
            policy.decayEpsilon(decayRate, minEpsilon);
        }
        
        // epsilon应该不会低于最小值
        assertTrue(policy.getEpsilon() >= minEpsilon);
    }

    /**
     * 测试不同Q值配置下的行为
     */
    @Test
    public void testDifferentQValueConfigurations() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        
        // 测试Q值相等的情况
        qFunction.setQValues(new float[]{1.0f, 1.0f, 1.0f});
        policy.setEpsilon(0.0f); // 纯贪婪
        
        // 当Q值相等时，应该选择第一个动作（索引0）
        Variable action = policy.selectAction(state);
        assertEquals(0.0f, action.getValue().getNumber().floatValue(), 0.001f);
        
        // 测试负Q值
        qFunction.setQValues(new float[]{-1.0f, -0.5f, -2.0f});
        action = policy.selectAction(state);
        assertEquals(1.0f, action.getValue().getNumber().floatValue(), 0.001f); // 应该选择-0.5（最大值）
    }

    /**
     * 测试不同状态的处理
     */
    @Test
    public void testDifferentStates() {
        // 测试不同的状态输入，Q函数返回相同值
        Variable state1 = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        Variable state2 = new Variable(NdArray.of(new float[]{3.0f, 4.0f}, Shape.of(1, 2)));
        
        policy.setEpsilon(0.0f); // 纯贪婪确保一致性
        
        Variable action1 = policy.selectAction(state1);
        Variable action2 = policy.selectAction(state2);
        
        // 由于Q函数对所有状态返回相同的Q值，行为应该一致
        assertEquals(action1.getValue().getNumber().floatValue(), 
                    action2.getValue().getNumber().floatValue(), 0.001f);
    }

    /**
     * 测试概率分布的一致性
     */
    @Test
    public void testProbabilityConsistency() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        
        // 多次计算概率分布，结果应该一致
        Variable probs1 = policy.getActionProbabilities(state);
        Variable probs2 = policy.getActionProbabilities(state);
        
        for (int i = 0; i < ACTION_DIM; i++) {
            assertEquals(probs1.getValue().get(0, i), probs2.getValue().get(0, i), 0.001f);
        }
    }

    /**
     * 测试极小概率的对数概率计算（避免log(0)）
     */
    @Test
    public void testLogProbabilityNumericalStability() {
        // 创建一个返回极端Q值的Q函数
        MockQFunction extremeQFunction = new MockQFunction(new float[]{1000.0f, -1000.0f, -1000.0f});
        EpsilonGreedyPolicy extremePolicy = new EpsilonGreedyPolicy(STATE_DIM, ACTION_DIM, 0.001f, extremeQFunction);
        
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        Variable suboptimalAction = new Variable(NdArray.of(1.0f)); // 次优动作
        
        // 即使概率很小，对数概率也应该是有限值
        Variable logProb = extremePolicy.getLogProbability(state, suboptimalAction);
        assertFalse("对数概率不应该是无穷大", Float.isInfinite(logProb.getValue().getNumber().floatValue()));
        assertFalse("对数概率不应该是NaN", Float.isNaN(logProb.getValue().getNumber().floatValue()));
    }
}