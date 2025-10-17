package io.leavesfly.tinyai.agent.robot.model;

import java.util.HashMap;
import java.util.Map;

/**
 * 步进结果类
 * 
 * <p>表示环境执行一步动作后的结果。</p>
 * 
 * @author TinyAI Team
 */
public class StepResult {
    /**
     * 新的观测状态
     */
    private CleaningState observation;
    
    /**
     * 即时奖励
     */
    private double reward;
    
    /**
     * 是否结束
     */
    private boolean done;
    
    /**
     * 附加信息
     */
    private Map<String, Object> info;
    
    /**
     * 构造函数
     * 
     * @param observation 观测状态
     * @param reward 奖励
     * @param done 是否结束
     */
    public StepResult(CleaningState observation, double reward, boolean done) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = new HashMap<>();
    }
    
    /**
     * 添加附加信息
     * 
     * @param key 键
     * @param value 值
     */
    public void addInfo(String key, Object value) {
        info.put(key, value);
    }
    
    /**
     * 获取附加信息
     * 
     * @param key 键
     * @return 值
     */
    public Object getInfo(String key) {
        return info.get(key);
    }
    
    // Getters and Setters
    public CleaningState getObservation() {
        return observation;
    }
    
    public void setObservation(CleaningState observation) {
        this.observation = observation;
    }
    
    public double getReward() {
        return reward;
    }
    
    public void setReward(double reward) {
        this.reward = reward;
    }
    
    public boolean isDone() {
        return done;
    }
    
    public void setDone(boolean done) {
        this.done = done;
    }
    
    public Map<String, Object> getInfo() {
        return info;
    }
    
    public void setInfo(Map<String, Object> info) {
        this.info = info;
    }
    
    @Override
    public String toString() {
        return String.format("StepResult(reward=%.2f, done=%b, info=%s)",
                             reward, done, info);
    }
}
