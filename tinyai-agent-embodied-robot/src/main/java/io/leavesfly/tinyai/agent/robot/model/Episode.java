package io.leavesfly.tinyai.agent.robot.model;

import java.util.ArrayList;
import java.util.List;

/**
 * 情景类
 * 
 * <p>表示一个完整的清扫情景，包含所有状态转移序列。</p>
 * 
 * @author TinyAI Team
 */
public class Episode {
    /**
     * 情景ID
     */
    private String episodeId;
    
    /**
     * 场景类型
     */
    private ScenarioType scenarioType;
    
    /**
     * 状态转移序列
     */
    private List<Transition> transitions;
    
    /**
     * 总奖励
     */
    private double totalReward;
    
    /**
     * 情景长度（步数）
     */
    private int length;
    
    /**
     * 清扫效率
     */
    private double cleaningEfficiency;
    
    /**
     * 能量效率
     */
    private double energyEfficiency;
    
    /**
     * 构造函数
     * 
     * @param episodeId 情景ID
     * @param scenarioType 场景类型
     */
    public Episode(String episodeId, ScenarioType scenarioType) {
        this.episodeId = episodeId;
        this.scenarioType = scenarioType;
        this.transitions = new ArrayList<>();
        this.totalReward = 0.0;
        this.length = 0;
        this.cleaningEfficiency = 0.0;
        this.energyEfficiency = 0.0;
    }
    
    /**
     * 添加状态转移
     * 
     * @param transition 状态转移
     */
    public void addTransition(Transition transition) {
        transitions.add(transition);
        totalReward += transition.getReward();
        length++;
    }
    
    /**
     * 获取平均奖励
     * 
     * @return 平均奖励
     */
    public double getAverageReward() {
        return length > 0 ? totalReward / length : 0.0;
    }
    
    /**
     * 获取覆盖率
     * 
     * @return 覆盖率（0-1）
     */
    public double getCoverageRate() {
        if (transitions.isEmpty()) {
            return 0.0;
        }
        Transition lastTransition = transitions.get(transitions.size() - 1);
        CleaningState finalState = lastTransition.getNextState();
        if (finalState.getFloorMap() != null) {
            return finalState.getFloorMap().getCoverageRate();
        }
        return 0.0;
    }
    
    /**
     * 判断是否成功（达到目标覆盖率）
     * 
     * @param targetCoverage 目标覆盖率
     * @return 是否成功
     */
    public boolean isSuccessful(double targetCoverage) {
        return getCoverageRate() >= targetCoverage;
    }
    
    // Getters and Setters
    public String getEpisodeId() {
        return episodeId;
    }
    
    public void setEpisodeId(String episodeId) {
        this.episodeId = episodeId;
    }
    
    public ScenarioType getScenarioType() {
        return scenarioType;
    }
    
    public void setScenarioType(ScenarioType scenarioType) {
        this.scenarioType = scenarioType;
    }
    
    public List<Transition> getTransitions() {
        return transitions;
    }
    
    public void setTransitions(List<Transition> transitions) {
        this.transitions = transitions;
    }
    
    public double getTotalReward() {
        return totalReward;
    }
    
    public void setTotalReward(double totalReward) {
        this.totalReward = totalReward;
    }
    
    public int getLength() {
        return length;
    }
    
    public void setLength(int length) {
        this.length = length;
    }
    
    public double getCleaningEfficiency() {
        return cleaningEfficiency;
    }
    
    public void setCleaningEfficiency(double cleaningEfficiency) {
        this.cleaningEfficiency = cleaningEfficiency;
    }
    
    public double getEnergyEfficiency() {
        return energyEfficiency;
    }
    
    public void setEnergyEfficiency(double energyEfficiency) {
        this.energyEfficiency = energyEfficiency;
    }
    
    @Override
    public String toString() {
        return String.format("Episode(id=%s, type=%s, length=%d, reward=%.2f, coverage=%.1f%%)",
                             episodeId, scenarioType, length, totalReward, getCoverageRate() * 100);
    }
}
