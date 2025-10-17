package io.leavesfly.tinyai.agent.embodied.model;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.HashMap;
import java.util.Map;

/**
 * 执行反馈类
 * 封装动作执行后的反馈信息
 *
 * @author TinyAI Team
 */
public class ExecutionFeedback {
    private boolean success;              // 动作是否成功执行
    private NdArray actualAction;         // 实际执行的动作参数
    private PerceptionState nextState;    // 执行后的环境状态
    private double reward;                // 即时奖励值
    private boolean done;                 // 是否到达终止状态
    private Map<String, Object> info;     // 附加调试信息

    public ExecutionFeedback() {
        this.info = new HashMap<>();
        this.success = true;
        this.done = false;
    }

    public ExecutionFeedback(boolean success, NdArray actualAction, PerceptionState nextState,
                            double reward, boolean done) {
        this.success = success;
        this.actualAction = actualAction;
        this.nextState = nextState;
        this.reward = reward;
        this.done = done;
        this.info = new HashMap<>();
    }

    /**
     * 添加调试信息
     */
    public void addInfo(String key, Object value) {
        info.put(key, value);
    }

    // Getters and Setters
    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public NdArray getActualAction() {
        return actualAction;
    }

    public void setActualAction(NdArray actualAction) {
        this.actualAction = actualAction;
    }

    public PerceptionState getNextState() {
        return nextState;
    }

    public void setNextState(PerceptionState nextState) {
        this.nextState = nextState;
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
        return String.format("ExecutionFeedback[success=%b, reward=%.3f, done=%b]",
                success, reward, done);
    }
}
