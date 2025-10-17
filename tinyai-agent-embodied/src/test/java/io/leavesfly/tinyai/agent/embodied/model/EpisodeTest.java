package io.leavesfly.tinyai.agent.embodied.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 情景记录测试类
 *
 * @author TinyAI Team
 */
public class EpisodeTest {

    @Test
    public void testConstructor() {
        Episode episode = new Episode("test_episode", ScenarioType.HIGHWAY);
        
        assertEquals("test_episode", episode.getEpisodeId());
        assertEquals(ScenarioType.HIGHWAY, episode.getScenarioType());
        assertNotNull(episode.getTrajectory());
        assertTrue(episode.getTrajectory().isEmpty());
        assertEquals(0.0, episode.getTotalReward(), 1e-6);
        assertTrue(episode.getStartTime() > 0);
    }

    @Test
    public void testAddTransition() {
        Episode episode = new Episode("test", ScenarioType.TEST);
        
        PerceptionState state1 = new PerceptionState();
        DrivingAction action = new DrivingAction(0.1, 0.5, 0.0);
        PerceptionState state2 = new PerceptionState();
        
        Transition transition = new Transition(state1, action, 1.5, state2, false);
        episode.addTransition(transition);
        
        assertEquals(1, episode.getLength());
        assertEquals(1.5, episode.getTotalReward(), 1e-6);
    }

    @Test
    public void testAddMultipleTransitions() {
        Episode episode = new Episode("test", ScenarioType.HIGHWAY);
        
        for (int i = 0; i < 10; i++) {
            PerceptionState state1 = new PerceptionState();
            DrivingAction action = new DrivingAction();
            PerceptionState state2 = new PerceptionState();
            Transition transition = new Transition(state1, action, 0.5, state2, false);
            episode.addTransition(transition);
        }
        
        assertEquals(10, episode.getLength());
        assertEquals(5.0, episode.getTotalReward(), 1e-6);
    }

    @Test
    public void testGetAverageReward() {
        Episode episode = new Episode("test", ScenarioType.URBAN);
        
        for (int i = 0; i < 5; i++) {
            PerceptionState state1 = new PerceptionState();
            DrivingAction action = new DrivingAction();
            PerceptionState state2 = new PerceptionState();
            Transition transition = new Transition(state1, action, 2.0, state2, false);
            episode.addTransition(transition);
        }
        
        assertEquals(2.0, episode.getAverageReward(), 1e-6);
    }

    @Test
    public void testGetAverageRewardEmpty() {
        Episode episode = new Episode("test", ScenarioType.TEST);
        assertEquals(0.0, episode.getAverageReward(), 1e-6);
    }

    @Test
    public void testAddCriticalEvent() {
        Episode episode = new Episode("test", ScenarioType.HIGHWAY);
        
        episode.addCriticalEvent("Near collision");
        episode.addCriticalEvent("Emergency brake");
        
        assertEquals(2, episode.getCriticalEvents().size());
        assertTrue(episode.getCriticalEvents().contains("Near collision"));
        assertTrue(episode.getCriticalEvents().contains("Emergency brake"));
    }

    @Test
    public void testAddLearnedLesson() {
        Episode episode = new Episode("test", ScenarioType.URBAN);
        
        episode.addLearnedLesson("Keep safe distance");
        episode.addLearnedLesson("Slow down in turns");
        
        assertEquals(2, episode.getLearnedLessons().size());
        assertTrue(episode.getLearnedLessons().contains("Keep safe distance"));
    }

    @Test
    public void testFinish() throws InterruptedException {
        Episode episode = new Episode("test", ScenarioType.HIGHWAY);
        long startTime = episode.getStartTime();
        
        Thread.sleep(10); // 确保有时间差
        episode.finish();
        
        assertTrue(episode.getEndTime() > 0);
        assertTrue(episode.getEndTime() >= startTime);
        assertTrue(episode.getDuration() > 0);
    }

    @Test
    public void testGetDurationBeforeFinish() throws InterruptedException {
        Episode episode = new Episode("test", ScenarioType.TEST);
        
        Thread.sleep(10);
        long duration = episode.getDuration();
        
        assertTrue(duration > 0);
    }

    @Test
    public void testGetDurationAfterFinish() throws InterruptedException {
        Episode episode = new Episode("test", ScenarioType.HIGHWAY);
        
        Thread.sleep(10);
        episode.finish();
        long duration = episode.getDuration();
        
        assertTrue(duration > 0);
        assertEquals(duration, episode.getDuration()); // 应该保持不变
    }

    @Test
    public void testMetadata() {
        Episode episode = new Episode("test", ScenarioType.URBAN);
        
        episode.getMetadata().put("weather", "sunny");
        episode.getMetadata().put("difficulty", 5);
        
        assertEquals("sunny", episode.getMetadata().get("weather"));
        assertEquals(5, episode.getMetadata().get("difficulty"));
    }

    @Test
    public void testToString() {
        Episode episode = new Episode("episode_123", ScenarioType.HIGHWAY);
        
        PerceptionState state1 = new PerceptionState();
        DrivingAction action = new DrivingAction();
        PerceptionState state2 = new PerceptionState();
        Transition transition = new Transition(state1, action, 3.5, state2, false);
        episode.addTransition(transition);
        
        String str = episode.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("Episode"));
        assertTrue(str.contains("episode_123"));
        assertTrue(str.contains("length=1"));
        assertTrue(str.contains("3.50"));
    }

    @Test
    public void testNegativeRewards() {
        Episode episode = new Episode("test", ScenarioType.TEST);
        
        PerceptionState state1 = new PerceptionState();
        DrivingAction action = new DrivingAction();
        PerceptionState state2 = new PerceptionState();
        Transition transition = new Transition(state1, action, -10.0, state2, false);
        episode.addTransition(transition);
        
        assertEquals(-10.0, episode.getTotalReward(), 1e-6);
        assertEquals(-10.0, episode.getAverageReward(), 1e-6);
    }

    @Test
    public void testMixedRewards() {
        Episode episode = new Episode("test", ScenarioType.HIGHWAY);
        
        double[] rewards = {1.0, -2.0, 3.0, -1.0, 2.0};
        for (double reward : rewards) {
            PerceptionState state1 = new PerceptionState();
            DrivingAction action = new DrivingAction();
            PerceptionState state2 = new PerceptionState();
            Transition transition = new Transition(state1, action, reward, state2, false);
            episode.addTransition(transition);
        }
        
        assertEquals(3.0, episode.getTotalReward(), 1e-6);
        assertEquals(0.6, episode.getAverageReward(), 1e-6);
    }
}
