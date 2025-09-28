package io.leavesfly.tinyai.rl.environment;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.rl.Environment;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * MultiArmedBanditEnvironment类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class MultiArmedBanditEnvironmentTest {

    private MultiArmedBanditEnvironment environment;
    private final float[] TRUE_REWARDS = {0.1f, 0.8f, 0.3f, 0.6f};
    private final float[] VARIANCES = {1.0f, 1.0f, 1.0f, 1.0f};
    private final int MAX_STEPS = 100;

    @Before
    public void setUp() {
        environment = new MultiArmedBanditEnvironment(TRUE_REWARDS, VARIANCES, MAX_STEPS);
    }

    /**
     * 创建动作变量
     */
    private Variable createAction(int armIndex) {
        return new Variable(NdArray.of(new float[]{armIndex}, Shape.of(1)));
    }

    /**
     * 测试初始状态
     */
    @Test
    public void testInitialState() {
        assertEquals(1, environment.getStateDim());
        assertEquals(TRUE_REWARDS.length, environment.getActionDim());
        assertEquals(MAX_STEPS, environment.getInfo().get("maxSteps"));
        assertFalse(environment.isDone());
        assertEquals(0, environment.getCurrentStep());
        
        // 验证最优臂识别（臂1有最高奖励0.8）
        assertEquals(1, environment.getOptimalArm());
        assertEquals(0.8f, environment.getOptimalReward(), 0.001f);
        
        // 验证真实奖励
        assertArrayEquals(TRUE_REWARDS, environment.getTrueRewards(), 0.001f);
    }

    /**
     * 测试简化构造函数（单位方差）
     */
    @Test
    public void testSimpleConstructor() {
        MultiArmedBanditEnvironment simpleEnv = new MultiArmedBanditEnvironment(TRUE_REWARDS, MAX_STEPS);
        assertEquals(TRUE_REWARDS.length, simpleEnv.getActionDim());
        assertEquals(1, simpleEnv.getOptimalArm());
        assertArrayEquals(TRUE_REWARDS, simpleEnv.getTrueRewards(), 0.001f);
    }

    /**
     * 测试重置功能
     */
    @Test
    public void testReset() {
        // 先执行一些步骤
        environment.step(createAction(0));
        environment.step(createAction(1));
        
        Variable state = environment.reset();
        
        assertNotNull(state);
        assertEquals(Shape.of(1), state.getValue().getShape());
        assertEquals(0.0f, state.getValue().get(0), 0.001f);
        assertFalse(environment.isDone());
        assertEquals(0, environment.getCurrentStep());
        assertEquals(0.0f, environment.getTotalReward(), 0.001f);
        assertEquals(0.0f, environment.getTotalRegret(), 0.001f);
    }

    /**
     * 测试有效动作执行
     */
    @Test
    public void testValidActionExecution() {
        environment.reset();
        
        for (int arm = 0; arm < TRUE_REWARDS.length; arm++) {
            Variable action = createAction(arm);
            Environment.StepResult result = environment.step(action);
            
            assertNotNull(result);
            assertNotNull(result.getNextState());
            assertNotNull(result.getInfo());
            assertFalse(Float.isNaN(result.getReward()));
            
            // 验证信息
            Map<String, Object> info = result.getInfo();
            assertEquals(arm, info.get("selectedArm"));
            assertEquals(TRUE_REWARDS[arm], (Float) info.get("trueReward"), 0.001f);
            assertEquals(environment.getOptimalArm(), info.get("optimalArm"));
            assertEquals(arm == environment.getOptimalArm(), info.get("isOptimal"));
            
            environment.reset(); // 为下次测试重置
        }
    }

    /**
     * 测试无效动作
     */
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidActionNegative() {
        environment.reset();
        environment.step(createAction(-1));
    }

    /**
     * 测试无效动作（超出范围）
     */
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidActionTooLarge() {
        environment.reset();
        environment.step(createAction(TRUE_REWARDS.length));
    }

    /**
     * 测试环境结束后执行动作
     */
    @Test(expected = IllegalStateException.class)
    public void testActionAfterDone() {
        environment.reset();
        
        // 执行最大步数的动作
        for (int i = 0; i < MAX_STEPS; i++) {
            environment.step(createAction(0));
        }
        
        assertTrue(environment.isDone());
        
        // 尝试在环境结束后执行动作，应该抛出异常
        environment.step(createAction(0));
    }

    /**
     * 测试累积奖励和悔恨值计算
     */
    @Test
    public void testRewardAndRegretAccumulation() {
        environment.reset();
        
        float expectedTotalReward = 0.0f;
        float expectedTotalRegret = 0.0f;
        int steps = 5;
        
        for (int i = 0; i < steps; i++) {
            int armIndex = i % TRUE_REWARDS.length;
            Variable action = createAction(armIndex);
            Environment.StepResult result = environment.step(action);
            
            expectedTotalReward += result.getReward();
            expectedTotalRegret += (environment.getOptimalReward() - TRUE_REWARDS[armIndex]);
            
            // 验证累积值（考虑浮点精度）
            assertEquals(expectedTotalReward, environment.getTotalReward(), 0.01f);
            assertEquals(expectedTotalRegret, environment.getTotalRegret(), 0.001f);
            
            // 验证平均值
            assertEquals(expectedTotalReward / (i + 1), environment.getAverageReward(), 0.01f);
            assertEquals(expectedTotalRegret / (i + 1), environment.getAverageRegret(), 0.001f);
        }
    }

    /**
     * 测试最优动作选择
     */
    @Test
    public void testOptimalActionSelection() {
        environment.reset();
        
        // 选择最优臂
        int optimalArm = environment.getOptimalArm();
        Variable action = createAction(optimalArm);
        Environment.StepResult result = environment.step(action);
        
        // 验证这是最优选择
        assertTrue((Boolean) result.getInfo().get("isOptimal"));
        assertEquals(0.0f, (Float) result.getInfo().get("instantRegret"), 0.001f);
    }

    /**
     * 测试次优动作选择
     */
    @Test
    public void testSuboptimalActionSelection() {
        environment.reset();
        
        // 选择次优臂（非最优臂）
        int suboptimalArm = (environment.getOptimalArm() + 1) % TRUE_REWARDS.length;
        Variable action = createAction(suboptimalArm);
        Environment.StepResult result = environment.step(action);
        
        // 验证这不是最优选择
        assertFalse((Boolean) result.getInfo().get("isOptimal"));
        
        float expectedRegret = environment.getOptimalReward() - TRUE_REWARDS[suboptimalArm];
        assertEquals(expectedRegret, (Float) result.getInfo().get("instantRegret"), 0.001f);
    }

    /**
     * 测试随机动作采样
     */
    @Test
    public void testRandomActionSampling() {
        boolean[] actionsSeen = new boolean[TRUE_REWARDS.length];
        
        // 多次采样随机动作
        for (int i = 0; i < 100; i++) {
            Variable randomAction = environment.sampleAction();
            int armIndex = (int) randomAction.getValue().get(0);
            
            assertTrue("随机动作应该在有效范围内", armIndex >= 0 && armIndex < TRUE_REWARDS.length);
            actionsSeen[armIndex] = true;
        }
        
        // 验证所有动作都被采样过（概率很高）
        int distinctActions = 0;
        for (boolean seen : actionsSeen) {
            if (seen) distinctActions++;
        }
        assertTrue("随机采样应该产生多样的动作", distinctActions >= TRUE_REWARDS.length / 2);
    }

    /**
     * 测试动作有效性检查
     */
    @Test
    public void testActionValidation() {
        // 测试有效动作
        for (int i = 0; i < TRUE_REWARDS.length; i++) {
            assertTrue(environment.isValidAction(createAction(i)));
        }
        
        // 测试无效动作
        assertFalse(environment.isValidAction(createAction(-1)));
        assertFalse(environment.isValidAction(createAction(TRUE_REWARDS.length)));
        assertFalse(environment.isValidAction(null));
        
        // 测试空值动作
        try {
            Variable nullAction = new Variable((NdArray) null);
            assertFalse(environment.isValidAction(nullAction));
        } catch (RuntimeException e) {
            // Variable构造器不允许null，这是预期的
            assertTrue(e.getMessage().contains("null"));
        }
    }

    /**
     * 测试环境信息获取
     */
    @Test
    public void testEnvironmentInfo() {
        environment.reset();
        Map<String, Object> info = environment.getInfo();
        
        assertNotNull(info);
        assertEquals(0, info.get("currentStep"));
        assertEquals(MAX_STEPS, info.get("maxSteps"));
        assertEquals(false, info.get("done"));
        assertEquals(1, info.get("stateDim"));
        assertEquals(TRUE_REWARDS.length, info.get("actionDim"));
        
        // 执行一步后检查信息
        environment.step(createAction(0));
        info = environment.getInfo();
        assertEquals(1, info.get("currentStep"));
    }

    /**
     * 测试随机种子设置
     */
    @Test
    public void testRandomSeed() {
        environment.setSeed(12345L);
        environment.reset();
        
        // 记录第一次的奖励序列
        float[] rewards1 = new float[10];
        for (int i = 0; i < 10; i++) {
            Environment.StepResult result = environment.step(createAction(0));
            rewards1[i] = result.getReward();
        }
        
        // 重置并使用相同种子
        environment.setSeed(12345L);
        environment.reset();
        
        // 记录第二次的奖励序列
        float[] rewards2 = new float[10];
        for (int i = 0; i < 10; i++) {
            Environment.StepResult result = environment.step(createAction(0));
            rewards2[i] = result.getReward();
        }
        
        // 序列应该相同
        assertArrayEquals("相同种子应该产生相同的奖励序列", rewards1, rewards2, 0.001f);
    }

    /**
     * 测试奖励分布的统计特性
     */
    @Test
    public void testRewardDistribution() {
        environment.setSeed(42L); // 固定种子以获得可重现的结果
        environment.reset();
        
        int numSamples = 1000;
        float[] rewards = new float[numSamples];
        
        // 采样大量的奖励值
        for (int i = 0; i < numSamples; i++) {
            Environment.StepResult result = environment.step(createAction(0)); // 总是选择臂0
            rewards[i] = result.getReward();
            environment.reset(); // 重置以独立采样
        }
        
        // 计算样本均值
        float sampleMean = 0.0f;
        for (float reward : rewards) {
            sampleMean += reward;
        }
        sampleMean /= numSamples;
        
        // 样本均值应该接近真实均值（在统计误差范围内）
        assertEquals("样本均值应该接近真实均值", TRUE_REWARDS[0], sampleMean, 0.1f);
    }

    /**
     * 测试完整回合执行
     */
    @Test
    public void testFullEpisode() {
        environment.reset();
        
        int stepCount = 0;
        while (!environment.isDone()) {
            Variable action = createAction(stepCount % TRUE_REWARDS.length);
            Environment.StepResult result = environment.step(action);
            
            assertNotNull(result);
            assertEquals(stepCount + 1, environment.getCurrentStep());
            stepCount++;
        }
        
        assertEquals(MAX_STEPS, stepCount);
        assertTrue(environment.isDone());
        assertEquals(MAX_STEPS, environment.getCurrentStep());
    }

    /**
     * 测试不同奖励配置
     */
    @Test
    public void testDifferentRewardConfigurations() {
        // 测试所有奖励相等的情况
        float[] equalRewards = {0.5f, 0.5f, 0.5f};
        MultiArmedBanditEnvironment equalEnv = new MultiArmedBanditEnvironment(equalRewards, 50);
        
        assertEquals(0, equalEnv.getOptimalArm()); // 应该选择第一个作为最优
        assertEquals(0.5f, equalEnv.getOptimalReward(), 0.001f);
        
        // 测试负奖励的情况
        float[] negativeRewards = {-0.8f, -0.2f, -0.9f};
        MultiArmedBanditEnvironment negativeEnv = new MultiArmedBanditEnvironment(negativeRewards, 50);
        
        assertEquals(1, negativeEnv.getOptimalArm()); // -0.2是最大值
        assertEquals(-0.2f, negativeEnv.getOptimalReward(), 0.001f);
    }

    /**
     * 测试渲染功能（不会抛出异常）
     */
    @Test
    public void testRender() {
        environment.reset();
        environment.step(createAction(0));
        
        // 渲染不应该抛出异常
        try {
            environment.render();
        } catch (Exception e) {
            fail("渲染不应该抛出异常: " + e.getMessage());
        }
    }

    /**
     * 测试边界情况：单臂环境
     */
    @Test
    public void testSingleArmEnvironment() {
        float[] singleReward = {0.7f};
        MultiArmedBanditEnvironment singleEnv = new MultiArmedBanditEnvironment(singleReward, 10);
        
        assertEquals(1, singleEnv.getActionDim());
        assertEquals(0, singleEnv.getOptimalArm());
        assertEquals(0.7f, singleEnv.getOptimalReward(), 0.001f);
        
        singleEnv.reset();
        Environment.StepResult result = singleEnv.step(createAction(0));
        
        assertTrue((Boolean) result.getInfo().get("isOptimal"));
        assertEquals(0.0f, (Float) result.getInfo().get("instantRegret"), 0.001f);
    }

    /**
     * 测试零步数限制
     */
    @Test
    public void testZeroMaxSteps() {
        MultiArmedBanditEnvironment zeroStepEnv = new MultiArmedBanditEnvironment(TRUE_REWARDS, 0);
        zeroStepEnv.reset();
        assertTrue("零步数环境应该立即结束", zeroStepEnv.isDone());
    }
}