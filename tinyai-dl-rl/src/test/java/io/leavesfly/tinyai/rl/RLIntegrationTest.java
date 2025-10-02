package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.rl.agent.EpsilonGreedyBanditAgent;
import io.leavesfly.tinyai.rl.agent.UCBBanditAgent;
import io.leavesfly.tinyai.rl.environment.MultiArmedBanditEnvironment;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 强化学习模块集成测试
 * 测试智能体与环境的完整交互流程
 * 
 * @author leavesfly
 * @version 0.01
 */
public class RLIntegrationTest {

    private MultiArmedBanditEnvironment environment;
    private EpsilonGreedyBanditAgent epsilonAgent;
    private UCBBanditAgent ucbAgent;
    
    private final float[] TRUE_REWARDS = {0.1f, 0.8f, 0.3f, 0.6f};
    private final int MAX_STEPS = 100;
    private final int NUM_ARMS = TRUE_REWARDS.length;

    @Before
    public void setUp() {
        environment = new MultiArmedBanditEnvironment(TRUE_REWARDS, MAX_STEPS);
        epsilonAgent = new EpsilonGreedyBanditAgent("EpsilonGreedy", NUM_ARMS, 0.1f);
        ucbAgent = new UCBBanditAgent("UCB", NUM_ARMS);
        
        // 设置随机种子以获得可重现的结果
        environment.setSeed(42L);
        epsilonAgent.setSeed(42L);
    }

    /**
     * 测试完整的训练循环
     */
    @Test
    public void testCompleteTrainingLoop() {
        Variable state = environment.reset();
        assertNotNull(state);
        
        int totalSteps = 0;
        float totalReward = 0.0f;
        
        while (!environment.isDone()) {
            // 智能体选择动作
            Variable action = epsilonAgent.selectAction(state);
            assertNotNull(action);
            
            // 环境执行动作
            Environment.StepResult result = environment.step(action);
            assertNotNull(result);
            
            // 创建经验并让智能体学习
            Experience experience = new Experience(
                state,
                action,
                result.getReward(),
                result.getNextState(),
                result.isDone()
            );
            epsilonAgent.learn(experience);
            
            // 更新状态
            state = result.getNextState();
            totalReward += result.getReward();
            totalSteps++;
        }
        
        assertEquals(MAX_STEPS, totalSteps);
        assertEquals(MAX_STEPS, epsilonAgent.getTotalActions());
        assertTrue("总奖励应该为正", totalReward > 0);
        
        // 验证智能体学到了一些东西
        assertTrue("智能体应该识别出一些最优选择", epsilonAgent.getOptimalActionRate() > 0);
    }

    /**
     * 测试不同智能体的性能比较
     */
    @Test
    public void testAgentPerformanceComparison() {
        int numEpisodes = 5;
        float epsilonTotalReward = 0.0f;
        float ucbTotalReward = 0.0f;
        
        for (int episode = 0; episode < numEpisodes; episode++) {
            // 测试 Epsilon-Greedy 智能体
            epsilonTotalReward += runEpisode(epsilonAgent);
            
            // 测试 UCB 智能体
            ucbTotalReward += runEpisode(ucbAgent);
        }
        
        float epsilonAverage = epsilonTotalReward / numEpisodes;
        float ucbAverage = ucbTotalReward / numEpisodes;
        
        assertTrue("Epsilon-Greedy平均奖励应该为正", epsilonAverage > 0);
        assertTrue("UCB平均奖励应该为正", ucbAverage > 0);
        
        // 验证两个智能体都学会了一些策略
        assertTrue("智能体应该学会避免最差的臂", epsilonAverage > TRUE_REWARDS[0] * 0.8f);
        assertTrue("智能体应该学会避免最差的臂", ucbAverage > TRUE_REWARDS[0] * 0.8f);
    }

    /**
     * 运行单个回合
     */
    private float runEpisode(Agent agent) {
        agent.reset();
        Variable state = environment.reset();
        float episodeReward = 0.0f;
        
        while (!environment.isDone()) {
            Variable action = agent.selectAction(state);
            Environment.StepResult result = environment.step(action);
            
            Experience experience = new Experience(
                state,
                action,
                result.getReward(),
                result.getNextState(),
                result.isDone()
            );
            agent.learn(experience);
            
            state = result.getNextState();
            episodeReward += result.getReward();
        }
        
        return episodeReward;
    }

    /**
     * 测试经验回放缓冲区与智能体的集成
     */
    @Test
    public void testReplayBufferIntegration() {
        ReplayBuffer buffer = new ReplayBuffer(50);
        Variable state = environment.reset();
        
        // 收集一些经验
        for (int i = 0; i < 20 && !environment.isDone(); i++) {
            Variable action = epsilonAgent.selectAction(state);
            Environment.StepResult result = environment.step(action);
            
            Experience experience = new Experience(
                state,
                action,
                result.getReward(),
                result.getNextState(),
                result.isDone()
            );
            
            buffer.push(experience);
            state = result.getNextState();
        }
        
        assertTrue("缓冲区应该有经验", buffer.size() > 0);
        assertTrue("缓冲区应该可以采样", buffer.canSample(5));
        
        // 从缓冲区采样并进行批量学习
        Experience[] batch = buffer.sample(5);
        epsilonAgent.learnBatch(batch);
        
        assertEquals(5, batch.length);
        assertTrue("智能体应该从批量学习中获得训练步数", epsilonAgent.getTrainingStep() > 0);
    }

    /**
     * 测试多回合学习的收敛性
     */
    @Test
    public void testMultiEpisodeLearning() {
        int numEpisodes = 10;
        float[] episodeRewards = new float[numEpisodes];
        
        for (int episode = 0; episode < numEpisodes; episode++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            
            while (!environment.isDone()) {
                Variable action = epsilonAgent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state,
                    action,
                    result.getReward(),
                    result.getNextState(),
                    result.isDone()
                );
                epsilonAgent.learn(experience);
                
                state = result.getNextState();
                episodeReward += result.getReward();
            }
            
            episodeRewards[episode] = episodeReward;
        }
        
        // 验证学习有所改进（后半段应该比前半段好）
        float firstHalfAverage = 0.0f;
        float secondHalfAverage = 0.0f;
        
        for (int i = 0; i < numEpisodes / 2; i++) {
            firstHalfAverage += episodeRewards[i];
            secondHalfAverage += episodeRewards[i + numEpisodes / 2];
        }
        
        firstHalfAverage /= (numEpisodes / 2);
        secondHalfAverage /= (numEpisodes / 2);
        
        // 后半段的性能应该不比前半段差（允许一些波动）
        assertTrue("学习应该有所改进", secondHalfAverage >= firstHalfAverage * 0.9f);
    }

    /**
     * 测试智能体与环境的状态一致性
     */
    @Test
    public void testStateConsistency() {
        Variable state = environment.reset();
        
        for (int i = 0; i < 10 && !environment.isDone(); i++) {
            // 记录环境状态
            Variable envState = environment.getCurrentState();
            
            // 智能体选择动作
            Variable action = epsilonAgent.selectAction(state);
            
            // 执行动作
            Environment.StepResult result = environment.step(action);
            
            // 验证状态转移的一致性
            assertNotNull(result.getNextState());
            state = result.getNextState();
            
            // 多臂老虎机的状态应该保持不变（虚拟状态）
            assertEquals(envState.getValue().get(0), state.getValue().get(0), 0.001f);
        }
    }

    /**
     * 测试异常情况处理
     */
    @Test
    public void testExceptionHandling() {
        Variable state = environment.reset();
        
        // 尝试无效动作
        Variable invalidAction = new Variable(NdArray.of(new float[]{-1}, Shape.of(1)));
        
        try {
            environment.step(invalidAction);
            fail("应该抛出异常");
        } catch (IllegalArgumentException e) {
            // 预期的异常
            assertTrue(e.getMessage().contains("无效的动作"));
        }
        
        // 环境应该仍然可用
        Variable validAction = new Variable(NdArray.of(new float[]{0}, Shape.of(1)));
        Environment.StepResult result = environment.step(validAction);
        assertNotNull(result);
    }

    /**
     * 测试智能体参数动态调整
     */
    @Test
    public void testDynamicParameterAdjustment() {
        Variable state = environment.reset();
        
        float initialEpsilon = epsilonAgent.getCurrentEpsilon();
        
        // 执行一些步骤
        for (int i = 0; i < 10 && !environment.isDone(); i++) {
            Variable action = epsilonAgent.selectAction(state);
            Environment.StepResult result = environment.step(action);
            
            Experience experience = new Experience(
                state,
                action,
                result.getReward(),
                result.getNextState(),
                result.isDone()
            );
            epsilonAgent.learn(experience);
            
            state = result.getNextState();
        }
        
        // epsilon应该有所衰减
        assertTrue("Epsilon应该衰减", epsilonAgent.getCurrentEpsilon() <= initialEpsilon);
        
        // 动态调整epsilon
        epsilonAgent.setCurrentEpsilon(0.5f);
        assertEquals(0.5f, epsilonAgent.getCurrentEpsilon(), 0.001f);
    }

    /**
     * 测试不同环境配置下的智能体适应性
     */
    @Test
    public void testAgentAdaptability() {
        // 创建一个更有挑战性的环境（奖励差异更小）
        float[] challengingRewards = {0.45f, 0.55f, 0.50f, 0.48f};
        MultiArmedBanditEnvironment challengingEnv = new MultiArmedBanditEnvironment(challengingRewards, 50);
        challengingEnv.setSeed(123L);
        
        EpsilonGreedyBanditAgent adaptiveAgent = new EpsilonGreedyBanditAgent("Adaptive", 4, 0.2f);
        adaptiveAgent.setSeed(123L);
        
        Variable state = challengingEnv.reset();
        
        while (!challengingEnv.isDone()) {
            Variable action = adaptiveAgent.selectAction(state);
            Environment.StepResult result = challengingEnv.step(action);
            
            Experience experience = new Experience(
                state,
                action,
                result.getReward(),
                result.getNextState(),
                result.isDone()
            );
            adaptiveAgent.learn(experience);
            
            state = result.getNextState();
        }
        
        // 即使在困难环境中，智能体也应该有一些学习效果
        assertTrue("智能体应该在困难环境中也有学习效果", adaptiveAgent.getTotalActions() > 0);
        assertTrue("智能体应该识别出某种模式", adaptiveAgent.getBestEstimatedReward() > 0.4f);
    }

    /**
     * 测试经验数据的完整性
     */
    @Test
    public void testExperienceDataIntegrity() {
        Variable state = environment.reset();
        
        for (int i = 0; i < 5 && !environment.isDone(); i++) {
            Variable action = epsilonAgent.selectAction(state);
            Environment.StepResult result = environment.step(action);
            
            Experience experience = new Experience(
                state,
                action,
                result.getReward(),
                result.getNextState(),
                result.isDone(),
                i
            );
            
            // 验证经验数据的完整性
            assertNotNull(experience.getState());
            assertNotNull(experience.getAction());
            assertNotNull(experience.getNextState());
            assertEquals(result.getReward(), experience.getReward(), 0.001f);
            assertEquals(result.isDone(), experience.isDone());
            assertEquals(i, experience.getTimeStep());
            
            epsilonAgent.learn(experience);
            state = result.getNextState();
        }
    }

    /**
     * 测试性能统计信息的准确性
     */
    @Test
    public void testPerformanceStatistics() {
        Variable state = environment.reset();
        int optimalActions = 0;
        int totalActions = 0;
        
        while (!environment.isDone()) {
            Variable action = epsilonAgent.selectAction(state);
            Environment.StepResult result = environment.step(action);
            
            // 检查是否选择了最优动作
            boolean isOptimal = (Boolean) result.getInfo().get("isOptimal");
            if (isOptimal) {
                optimalActions++;
            }
            totalActions++;
            
            Experience experience = new Experience(
                state,
                action,
                result.getReward(),
                result.getNextState(),
                result.isDone()
            );
            epsilonAgent.learn(experience);
            
            state = result.getNextState();
        }
        
        // 验证统计信息的准确性
        assertEquals(totalActions, epsilonAgent.getTotalActions());
        
        float expectedOptimalRate = (float) optimalActions / totalActions;
        assertEquals(expectedOptimalRate, epsilonAgent.getOptimalActionRate(), 0.001f);
    }
}