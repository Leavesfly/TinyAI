package io.leavesfly.tinyai.agent.embodied.memory;

import io.leavesfly.tinyai.agent.embodied.model.Episode;
import io.leavesfly.tinyai.agent.embodied.model.ScenarioType;
import io.leavesfly.tinyai.agent.embodied.model.Transition;

import java.util.*;

/**
 * 情景记忆管理
 * 存储和管理驾驶情景数据
 *
 * @author TinyAI Team
 */
public class EpisodicMemory {
    private Map<String, Episode> episodes;
    private List<Transition> replayBuffer;
    private int maxBufferSize;
    private int maxEpisodes;

    public EpisodicMemory() {
        this(10000, 100);
    }

    public EpisodicMemory(int maxBufferSize, int maxEpisodes) {
        this.episodes = new LinkedHashMap<>();
        this.replayBuffer = new ArrayList<>();
        this.maxBufferSize = maxBufferSize;
        this.maxEpisodes = maxEpisodes;
    }

    /**
     * 存储完整情景
     */
    public void storeEpisode(Episode episode) {
        episodes.put(episode.getEpisodeId(), episode);
        
        // 将情景中的转移添加到回放缓冲区
        for (Transition transition : episode.getTrajectory()) {
            storeTransition(transition);
        }
        
        // 限制情景数量
        while (episodes.size() > maxEpisodes) {
            String firstKey = episodes.keySet().iterator().next();
            episodes.remove(firstKey);
        }
    }

    /**
     * 存储单个转移
     */
    public void storeTransition(Transition transition) {
        replayBuffer.add(transition);
        
        // 限制缓冲区大小
        while (replayBuffer.size() > maxBufferSize) {
            replayBuffer.remove(0);
        }
    }

    /**
     * 随机采样批次
     */
    public List<Transition> sampleBatch(int batchSize) {
        if (replayBuffer.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<Transition> batch = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < Math.min(batchSize, replayBuffer.size()); i++) {
            int idx = random.nextInt(replayBuffer.size());
            batch.add(replayBuffer.get(idx));
        }
        
        return batch;
    }

    /**
     * 获取指定场景的所有情景
     */
    public List<Episode> getEpisodesByScenario(ScenarioType type) {
        List<Episode> result = new ArrayList<>();
        for (Episode episode : episodes.values()) {
            if (episode.getScenarioType() == type) {
                result.add(episode);
            }
        }
        return result;
    }

    /**
     * 获取最佳情景（按总奖励排序）
     */
    public List<Episode> getTopEpisodes(int n) {
        List<Episode> sorted = new ArrayList<>(episodes.values());
        sorted.sort((e1, e2) -> Double.compare(e2.getTotalReward(), e1.getTotalReward()));
        return sorted.subList(0, Math.min(n, sorted.size()));
    }

    /**
     * 清空记忆
     */
    public void clear() {
        episodes.clear();
        replayBuffer.clear();
    }

    public int getEpisodeCount() {
        return episodes.size();
    }

    public int getBufferSize() {
        return replayBuffer.size();
    }

    public Episode getEpisode(String episodeId) {
        return episodes.get(episodeId);
    }
}
