package io.leavesfly.tinyai.agent.evol;

import java.util.HashMap;
import java.util.Map;

/**
 * 经验记录数据结构
 * 用于存储Agent在执行任务过程中的经验数据
 * 
 * @author 山泽
 */
public class Experience {
    /** 任务描述 */
    private String task;
    
    /** 上下文信息 */
    private Map<String, Object> context;
    
    /** 执行的动作 */
    private String action;
    
    /** 执行结果 */
    private Object result;
    
    /** 是否成功 */
    private boolean success;
    
    /** 奖励值 */
    private double reward;
    
    /** 时间戳 */
    private long timestamp;
    
    /** 反思内容 */
    private String reflection;
    
    /**
     * 默认构造函数
     */
    public Experience() {
        this.context = new HashMap<>();
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 完整构造函数
     * 
     * @param task 任务描述
     * @param context 上下文信息
     * @param action 执行的动作
     * @param result 执行结果
     * @param success 是否成功
     * @param reward 奖励值
     */
    public Experience(String task, Map<String, Object> context, String action, 
                     Object result, boolean success, double reward) {
        this.task = task;
        this.context = context != null ? new HashMap<>(context) : new HashMap<>();
        this.action = action;
        this.result = result;
        this.success = success;
        this.reward = reward;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 转换为Map格式，便于序列化
     * 
     * @return Map格式的经验数据
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("task", task);
        map.put("context", context);
        map.put("action", action);
        map.put("result", result);
        map.put("success", success);
        map.put("reward", reward);
        map.put("timestamp", timestamp);
        map.put("reflection", reflection);
        return map;
    }
    
    /**
     * 从Map格式创建Experience对象
     * 
     * @param map Map格式的数据
     * @return Experience对象
     */
    @SuppressWarnings("unchecked")
    public static Experience fromMap(Map<String, Object> map) {
        Experience exp = new Experience();
        exp.task = (String) map.get("task");
        exp.context = (Map<String, Object>) map.getOrDefault("context", new HashMap<>());
        exp.action = (String) map.get("action");
        exp.result = map.get("result");
        exp.success = (Boolean) map.getOrDefault("success", false);
        exp.reward = ((Number) map.getOrDefault("reward", 0.0)).doubleValue();
        exp.timestamp = ((Number) map.getOrDefault("timestamp", System.currentTimeMillis())).longValue();
        exp.reflection = (String) map.get("reflection");
        return exp;
    }
    
    // Getters and Setters
    
    public String getTask() {
        return task;
    }
    
    public void setTask(String task) {
        this.task = task;
    }
    
    public Map<String, Object> getContext() {
        return context;
    }
    
    public void setContext(Map<String, Object> context) {
        this.context = context != null ? new HashMap<>(context) : new HashMap<>();
    }
    
    public String getAction() {
        return action;
    }
    
    public void setAction(String action) {
        this.action = action;
    }
    
    public Object getResult() {
        return result;
    }
    
    public void setResult(Object result) {
        this.result = result;
    }
    
    public boolean isSuccess() {
        return success;
    }
    
    public void setSuccess(boolean success) {
        this.success = success;
    }
    
    public double getReward() {
        return reward;
    }
    
    public void setReward(double reward) {
        this.reward = reward;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
    public String getReflection() {
        return reflection;
    }
    
    public void setReflection(String reflection) {
        this.reflection = reflection;
    }
    
    @Override
    public String toString() {
        return String.format("Experience{task='%s', action='%s', success=%s, reward=%.2f, timestamp=%d}", 
                           task, action, success, reward, timestamp);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Experience that = (Experience) obj;
        return timestamp == that.timestamp && 
               Double.compare(that.reward, reward) == 0 &&
               success == that.success &&
               task.equals(that.task) &&
               action.equals(that.action);
    }
    
    @Override
    public int hashCode() {
        int result = task != null ? task.hashCode() : 0;
        result = 31 * result + (action != null ? action.hashCode() : 0);
        result = 31 * result + (success ? 1 : 0);
        result = 31 * result + Long.hashCode(timestamp);
        return result;
    }
}