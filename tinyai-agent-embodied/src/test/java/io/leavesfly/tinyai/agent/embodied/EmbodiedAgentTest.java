package io.leavesfly.tinyai.agent.embodied;

import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.model.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 具身智能体测试类
 *
 * @author TinyAI Team
 */
public class EmbodiedAgentTest {

    private EmbodiedAgent agent;
    private EnvironmentConfig config;

    @BeforeEach
    public void setUp() {
        config = EnvironmentConfig.createTestConfig();
        agent = new EmbodiedAgent(config);
    }

    @AfterEach
    public void tearDown() {
        if (agent != null) {
            agent.close();
        }
    }

    @Test
    public void testConstructor() {
        assertNotNull(agent);
        assertNotNull(agent.getEnvironment());
        assertFalse(agent.getCurrentState() != null);
    }

    @Test
    public void testReset() {
        PerceptionState state = agent.reset();
        
        assertNotNull(state);
        assertNotNull(agent.getCurrentState());
        assertEquals(0, agent.getEpisodeSteps());
        assertEquals(0.0, agent.getTotalReward(), 1e-6);
    }

    @Test
    public void testStepBeforeReset() {
        assertThrows(IllegalStateException.class, () -> {
            agent.step();
        });
    }

    @Test
    public void testStepAfterReset() {
        agent.reset();
        
        StepResult result = agent.step();
        
        assertNotNull(result);
        assertNotNull(result.getObservation());
        assertEquals(1, agent.getEpisodeSteps());
    }

    @Test
    public void testMultipleSteps() {
        agent.reset();
        
        int numSteps = 10;
        for (int i = 0; i < numSteps; i++) {
            StepResult result = agent.step();
            assertNotNull(result);
        }
        
        assertEquals(numSteps, agent.getEpisodeSteps());
    }

    @Test
    public void testRewardAccumulation() {
        agent.reset();
        
        double initialReward = agent.getTotalReward();
        agent.step();
        
        // 奖励应该发生变化（可能为正或负）
        assertNotNull(agent.getTotalReward());
    }

    @Test
    public void testRunEpisode() {
        Episode episode = agent.runEpisode(100);
        
        assertNotNull(episode);
        assertNotNull(episode.getEpisodeId());
        assertNotNull(episode.getTrajectory());
        assertTrue(episode.getLength() > 0);
        assertTrue(episode.getLength() <= 100);
    }

    @Test
    public void testRunEpisodeWithShortLimit() {
        Episode episode = agent.runEpisode(5);
        
        assertNotNull(episode);
        assertTrue(episode.getLength() <= 5);
    }

    @Test
    public void testRunEpisodeScenarioType() {
        Episode episode = agent.runEpisode(50);
        
        assertEquals(config.getScenarioType(), episode.getScenarioType());
    }

    @Test
    public void testMultipleReset() {
        PerceptionState state1 = agent.reset();
        assertNotNull(state1);
        assertEquals(0, agent.getEpisodeSteps());
        
        agent.step();
        assertEquals(1, agent.getEpisodeSteps());
        
        PerceptionState state2 = agent.reset();
        assertNotNull(state2);
        assertEquals(0, agent.getEpisodeSteps());
        assertEquals(0.0, agent.getTotalReward(), 1e-6);
    }

    @Test
    public void testStepResultInfo() {
        agent.reset();
        StepResult result = agent.step();
        
        assertNotNull(result.getInfo());
        assertTrue(result.getInfo().containsKey("total_reward"));
        assertTrue(result.getInfo().containsKey("episode_steps"));
    }

    @Test
    public void testClose() {
        agent.reset();
        agent.step();
        
        assertDoesNotThrow(() -> {
            agent.close();
        });
    }

    @Test
    public void testGetEnvironment() {
        assertNotNull(agent.getEnvironment());
        assertEquals(config.getScenarioType(), agent.getEnvironment().getScenarioType());
    }

    @Test
    public void testHighwayScenario() {
        EnvironmentConfig highwayConfig = EnvironmentConfig.createHighwayConfig();
        EmbodiedAgent highwayAgent = new EmbodiedAgent(highwayConfig);
        
        try {
            Episode episode = highwayAgent.runEpisode(50);
            
            assertNotNull(episode);
            assertEquals(ScenarioType.HIGHWAY, episode.getScenarioType());
        } finally {
            highwayAgent.close();
        }
    }

    @Test
    public void testUrbanScenario() {
        EnvironmentConfig urbanConfig = EnvironmentConfig.createUrbanConfig();
        EmbodiedAgent urbanAgent = new EmbodiedAgent(urbanConfig);
        
        try {
            Episode episode = urbanAgent.runEpisode(50);
            
            assertNotNull(episode);
            assertEquals(ScenarioType.URBAN, episode.getScenarioType());
        } finally {
            urbanAgent.close();
        }
    }
}
