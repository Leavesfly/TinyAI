package io.leavesfly.tinyai.agent.robot.learning;

import io.leavesfly.tinyai.agent.robot.model.*;

import java.util.*;

/**
 * 情景记忆
 * 
 * <p>存储和管理学习经验。</p>
 * 
 * @author TinyAI Team
 */
public class EpisodicMemory {
    private List<Episode> buffer;
    private int maxSize;
    private int position;
    
    public EpisodicMemory(int maxSize) {
        this.maxSize = maxSize;
        this.buffer = new ArrayList<>(maxSize);
        this.position = 0;
    }
    
    public void storeEpisode(Episode episode) {
        if (buffer.size() < maxSize) {
            buffer.add(episode);
        } else {
            buffer.set(position, episode);
        }
        position = (position + 1) % maxSize;
    }
    
    public List<Episode> sampleBatch(int batchSize) {
        if (buffer.isEmpty()) {
            return Collections.emptyList();
        }
        
        int actualSize = Math.min(batchSize, buffer.size());
        List<Episode> batch = new ArrayList<>(actualSize);
        Random random = new Random();
        
        for (int i = 0; i < actualSize; i++) {
            int idx = random.nextInt(buffer.size());
            batch.add(buffer.get(idx));
        }
        
        return batch;
    }
    
    public List<Episode> filterByScenario(ScenarioType type) {
        List<Episode> filtered = new ArrayList<>();
        for (Episode episode : buffer) {
            if (episode.getScenarioType() == type) {
                filtered.add(episode);
            }
        }
        return filtered;
    }
    
    public void clear() {
        buffer.clear();
        position = 0;
    }
    
    public int size() {
        return buffer.size();
    }
    
    public boolean isEmpty() {
        return buffer.isEmpty();
    }
}
