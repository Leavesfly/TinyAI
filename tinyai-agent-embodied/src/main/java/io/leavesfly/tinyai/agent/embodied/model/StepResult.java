package io.leavesfly.tinyai.agent.embodied.model;

import java.util.HashMap;
import java.util.Map;

/**
 * 环境步进结果类
 * 封装环境执行动作后的返回信息
 *
 * @author TinyAI Team
 */
public class StepResult {
    private PerceptionState observation;  // 执行后的观测
    private double reward;                // 即时奖励
    private boolean done;                 // 是否终止
    private Map<String, Object> info;     // 附加信息

    public StepResult() {
        this.info = new HashMap<>();
        this.done = false;
        this.reward = 0.0;
    }

    public StepResult(PerceptionState observation, double reward, boolean done) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = new HashMap<>();
    }

    /**
     * 添加附加信息
     */
    public void addInfo(String key, Object value) {
        info.put(key, value);
    }

    /**
     * 获取附加信息
     */
    public Object getInfo(String key) {
        return info.get(key);
    }

    // Getters and Setters
    public PerceptionState getObservation() {
        return observation;
    }

    public void setObservation(PerceptionState observation) {
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
        return String.format("StepResult[reward=%.3f, done=%b, info_count=%d]",
                reward, done, info.size());
    }
}
