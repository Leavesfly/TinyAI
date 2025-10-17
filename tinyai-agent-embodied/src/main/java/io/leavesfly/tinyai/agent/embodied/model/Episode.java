package io.leavesfly.tinyai.agent.embodied.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 情景类
 * 记录完整的驾驶情景（一个episode）
 *
 * @author TinyAI Team
 */
public class Episode {
    private String episodeId;                // 情景唯一标识
    private List<Transition> trajectory;     // 完整轨迹序列
    private double totalReward;              // 累积奖励
    private ScenarioType scenarioType;       // 场景类型
    private List<String> criticalEvents;     // 关键事件列表
    private List<String> learnedLessons;     // 经验教训总结
    private long startTime;                  // 开始时间
    private long endTime;                    // 结束时间
    private Map<String, Object> metadata;    // 元数据

    public Episode(String episodeId, ScenarioType scenarioType) {
        this.episodeId = episodeId;
        this.scenarioType = scenarioType;
        this.trajectory = new ArrayList<>();
        this.criticalEvents = new ArrayList<>();
        this.learnedLessons = new ArrayList<>();
        this.metadata = new HashMap<>();
        this.totalReward = 0.0;
        this.startTime = System.currentTimeMillis();
    }

    /**
     * 添加转移
     */
    public void addTransition(Transition transition) {
        trajectory.add(transition);
        totalReward += transition.getReward();
    }

    /**
     * 添加关键事件
     */
    public void addCriticalEvent(String event) {
        criticalEvents.add(event);
    }

    /**
     * 添加经验教训
     */
    public void addLearnedLesson(String lesson) {
        learnedLessons.add(lesson);
    }

    /**
     * 结束情景
     */
    public void finish() {
        this.endTime = System.currentTimeMillis();
    }

    /**
     * 获取情景长度
     */
    public int getLength() {
        return trajectory.size();
    }

    /**
     * 获取平均奖励
     */
    public double getAverageReward() {
        if (trajectory.isEmpty()) {
            return 0.0;
        }
        return totalReward / trajectory.size();
    }

    /**
     * 获取持续时间（毫秒）
     */
    public long getDuration() {
        if (endTime == 0) {
            return System.currentTimeMillis() - startTime;
        }
        return endTime - startTime;
    }

    // Getters and Setters
    public String getEpisodeId() {
        return episodeId;
    }

    public void setEpisodeId(String episodeId) {
        this.episodeId = episodeId;
    }

    public List<Transition> getTrajectory() {
        return trajectory;
    }

    public void setTrajectory(List<Transition> trajectory) {
        this.trajectory = trajectory;
    }

    public double getTotalReward() {
        return totalReward;
    }

    public void setTotalReward(double totalReward) {
        this.totalReward = totalReward;
    }

    public ScenarioType getScenarioType() {
        return scenarioType;
    }

    public void setScenarioType(ScenarioType scenarioType) {
        this.scenarioType = scenarioType;
    }

    public List<String> getCriticalEvents() {
        return criticalEvents;
    }

    public void setCriticalEvents(List<String> criticalEvents) {
        this.criticalEvents = criticalEvents;
    }

    public List<String> getLearnedLessons() {
        return learnedLessons;
    }

    public void setLearnedLessons(List<String> learnedLessons) {
        this.learnedLessons = learnedLessons;
    }

    public long getStartTime() {
        return startTime;
    }

    public void setStartTime(long startTime) {
        this.startTime = startTime;
    }

    public long getEndTime() {
        return endTime;
    }

    public void setEndTime(long endTime) {
        this.endTime = endTime;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    @Override
    public String toString() {
        return String.format("Episode[id=%s, type=%s, length=%d, totalReward=%.2f, avgReward=%.3f]",
                episodeId, scenarioType.getName(), getLength(), totalReward, getAverageReward());
    }
}
