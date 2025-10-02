package io.leavesfly.tinyai.agent.evol;

import java.util.*;

/**
 * 策略记录数据结构
 * 用于存储Agent学习到的策略信息
 * 
 * @author 山泽
 */
public class Strategy {
    /** 策略名称 */
    private String name;
    
    /** 策略描述 */
    private String description;
    
    /** 适用条件 */
    private Map<String, Object> conditions;
    
    /** 动作序列 */
    private List<String> actions;
    
    /** 成功率 */
    private double successRate;
    
    /** 使用次数 */
    private int usageCount;
    
    /** 最后更新时间 */
    private long lastUpdated;
    
    /**
     * 默认构造函数
     */
    public Strategy() {
        this.conditions = new HashMap<>();
        this.actions = new ArrayList<>();
        this.lastUpdated = System.currentTimeMillis();
        this.successRate = 0.0;
        this.usageCount = 0;
    }
    
    /**
     * 完整构造函数
     * 
     * @param name 策略名称
     * @param description 策略描述
     * @param conditions 适用条件
     * @param actions 动作序列
     * @param successRate 成功率
     * @param usageCount 使用次数
     */
    public Strategy(String name, String description, Map<String, Object> conditions,
                   List<String> actions, double successRate, int usageCount) {
        this.name = name;
        this.description = description;
        this.conditions = conditions != null ? new HashMap<>(conditions) : new HashMap<>();
        this.actions = actions != null ? new ArrayList<>(actions) : new ArrayList<>();
        this.successRate = successRate;
        this.usageCount = usageCount;
        this.lastUpdated = System.currentTimeMillis();
    }
    
    /**
     * 更新策略的成功率
     * 使用指数移动平均法更新成功率
     * 
     * @param success 本次是否成功
     * @param learningRate 学习率
     */
    public void updateSuccessRate(boolean success, double learningRate) {
        double newValue = success ? 1.0 : 0.0;
        this.successRate = (1 - learningRate) * this.successRate + learningRate * newValue;
        this.usageCount++;
        this.lastUpdated = System.currentTimeMillis();
    }
    
    /**
     * 检查策略是否匹配给定的上下文
     * 
     * @param context 当前上下文
     * @return 是否匹配
     */
    public boolean matchesContext(Map<String, Object> context) {
        if (conditions.isEmpty()) {
            return true;
        }
        
        for (Map.Entry<String, Object> condition : conditions.entrySet()) {
            String key = condition.getKey();
            Object expectedValue = condition.getValue();
            
            if (!context.containsKey(key)) {
                // 如果上下文中没有这个条件，根据条件类型进行推断
                if (!inferConditionMatch(key, expectedValue, context)) {
                    return false;
                }
            } else {
                Object actualValue = context.get(key);
                if (!Objects.equals(expectedValue, actualValue)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * 推断条件匹配
     * 当上下文中没有直接的条件值时，尝试推断是否匹配
     * 
     * @param conditionKey 条件键
     * @param expectedValue 期望值
     * @param context 上下文
     * @return 是否匹配
     */
    private boolean inferConditionMatch(String conditionKey, Object expectedValue, Map<String, Object> context) {
        switch (conditionKey) {
            case "uncertainty":
                double uncertainty = assessUncertainty(context);
                if ("high".equals(expectedValue) && uncertainty < 0.5) return false;
                if ("low".equals(expectedValue) && uncertainty > 0.5) return false;
                break;
            case "task_type":
                String task = (String) context.get("task");
                if (task != null) {
                    String taskType = task.contains(":") ? task.split(":")[0] : task;
                    return Objects.equals(expectedValue, taskType);
                }
                return false;
            default:
                // 对于未知条件，默认不匹配
                return false;
        }
        return true;
    }
    
    /**
     * 评估上下文的不确定性
     * 
     * @param context 上下文
     * @return 不确定性值 (0-1)
     */
    private double assessUncertainty(Map<String, Object> context) {
        // 简单的不确定性评估
        // 基于任务复杂度和可用信息量
        double complexity = 0.0;
        
        // 任务复杂度
        String task = (String) context.get("task");
        if (task != null) {
            complexity = Math.min(task.split("\\s+").length / 10.0, 1.0);
        }
        
        // 上下文信息量
        double infoAmount = Math.min(context.size() / 5.0, 1.0);
        double uncertainty = complexity * (1.0 - infoAmount * 0.5);
        
        return Math.max(0.0, Math.min(1.0, uncertainty));
    }
    
    /**
     * 转换为Map格式，便于序列化
     * 
     * @return Map格式的策略数据
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("name", name);
        map.put("description", description);
        map.put("conditions", conditions);
        map.put("actions", actions);
        map.put("successRate", successRate);
        map.put("usageCount", usageCount);
        map.put("lastUpdated", lastUpdated);
        return map;
    }
    
    /**
     * 从Map格式创建Strategy对象
     * 
     * @param map Map格式的数据
     * @return Strategy对象
     */
    @SuppressWarnings("unchecked")
    public static Strategy fromMap(Map<String, Object> map) {
        Strategy strategy = new Strategy();
        strategy.name = (String) map.get("name");
        strategy.description = (String) map.get("description");
        strategy.conditions = (Map<String, Object>) map.getOrDefault("conditions", new HashMap<>());
        strategy.actions = (List<String>) map.getOrDefault("actions", new ArrayList<>());
        strategy.successRate = ((Number) map.getOrDefault("successRate", 0.0)).doubleValue();
        strategy.usageCount = ((Number) map.getOrDefault("usageCount", 0)).intValue();
        strategy.lastUpdated = ((Number) map.getOrDefault("lastUpdated", System.currentTimeMillis())).longValue();
        return strategy;
    }
    
    /**
     * 创建策略的副本
     * 
     * @return 策略副本
     */
    public Strategy copy() {
        return new Strategy(name, description, conditions, actions, successRate, usageCount);
    }
    
    // Getters and Setters
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public Map<String, Object> getConditions() {
        return conditions;
    }
    
    public void setConditions(Map<String, Object> conditions) {
        this.conditions = conditions != null ? new HashMap<>(conditions) : new HashMap<>();
    }
    
    public List<String> getActions() {
        return actions;
    }
    
    public void setActions(List<String> actions) {
        this.actions = actions != null ? new ArrayList<>(actions) : new ArrayList<>();
    }
    
    public double getSuccessRate() {
        return successRate;
    }
    
    public void setSuccessRate(double successRate) {
        this.successRate = successRate;
    }
    
    public int getUsageCount() {
        return usageCount;
    }
    
    public void setUsageCount(int usageCount) {
        this.usageCount = usageCount;
    }
    
    public long getLastUpdated() {
        return lastUpdated;
    }
    
    public void setLastUpdated(long lastUpdated) {
        this.lastUpdated = lastUpdated;
    }
    
    @Override
    public String toString() {
        return String.format("Strategy{name='%s', successRate=%.2f, usageCount=%d, actions=%s}", 
                           name, successRate, usageCount, actions);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Strategy strategy = (Strategy) obj;
        return Objects.equals(name, strategy.name) && 
               Objects.equals(description, strategy.description);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(name, description);
    }
}