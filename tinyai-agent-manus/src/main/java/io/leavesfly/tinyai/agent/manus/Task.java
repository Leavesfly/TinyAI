package io.leavesfly.tinyai.agent.manus;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 任务类
 * 计划驱动模式中的基本执行单元
 * 
 * @author 山泽
 */
public class Task {
    
    private String id;                          // 任务ID
    private String description;                 // 任务描述
    private String type;                        // 任务类型：thinking, action, observation
    private Map<String, Object> parameters;     // 任务参数
    private String status;                      // 任务状态：pending, running, completed, failed
    private Object result;                      // 任务结果
    private String errorMessage;                // 错误信息
    private LocalDateTime createdAt;            // 创建时间
    private LocalDateTime startedAt;            // 开始时间
    private LocalDateTime completedAt;          // 完成时间
    private int priority;                       // 优先级
    
    /**
     * 构造函数
     */
    public Task() {
        this.id = generateTaskId();
        this.status = "pending";
        this.parameters = new HashMap<>();
        this.createdAt = LocalDateTime.now();
        this.priority = 0;
    }
    
    public Task(String description, String type) {
        this();
        this.description = description;
        this.type = type;
    }
    
    public Task(String description, String type, Map<String, Object> parameters) {
        this(description, type);
        this.parameters = parameters != null ? new HashMap<>(parameters) : new HashMap<>();
    }
    
    /**
     * 生成任务ID
     */
    private String generateTaskId() {
        return "task_" + System.currentTimeMillis() + "_" + (int)(Math.random() * 1000);
    }
    
    /**
     * 开始执行任务
     */
    public void start() {
        this.status = "running";
        this.startedAt = LocalDateTime.now();
    }
    
    /**
     * 完成任务
     */
    public void complete(Object result) {
        this.status = "completed";
        this.result = result;
        this.completedAt = LocalDateTime.now();
    }
    
    /**
     * 任务失败
     */
    public void fail(String errorMessage) {
        this.status = "failed";
        this.errorMessage = errorMessage;
        this.completedAt = LocalDateTime.now();
    }
    
    /**
     * 判断任务是否完成
     */
    public boolean isCompleted() {
        return "completed".equals(status);
    }
    
    /**
     * 判断任务是否失败
     */
    public boolean isFailed() {
        return "failed".equals(status);
    }
    
    /**
     * 判断任务是否运行中
     */
    public boolean isRunning() {
        return "running".equals(status);
    }
    
    /**
     * 获取执行时长（毫秒）
     */
    public long getExecutionTime() {
        if (startedAt == null) {
            return 0;
        }
        LocalDateTime endTime = completedAt != null ? completedAt : LocalDateTime.now();
        return java.time.Duration.between(startedAt, endTime).toMillis();
    }
    
    // Getter 和 Setter 方法
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public String getType() {
        return type;
    }
    
    public void setType(String type) {
        this.type = type;
    }
    
    public Map<String, Object> getParameters() {
        return new HashMap<>(parameters);
    }
    
    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters != null ? new HashMap<>(parameters) : new HashMap<>();
    }
    
    public void addParameter(String key, Object value) {
        this.parameters.put(key, value);
    }
    
    public Object getParameter(String key) {
        return this.parameters.get(key);
    }
    
    public String getStatus() {
        return status;
    }
    
    public void setStatus(String status) {
        this.status = status;
    }
    
    public Object getResult() {
        return result;
    }
    
    public void setResult(Object result) {
        this.result = result;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public LocalDateTime getStartedAt() {
        return startedAt;
    }
    
    public LocalDateTime getCompletedAt() {
        return completedAt;
    }
    
    public int getPriority() {
        return priority;
    }
    
    public void setPriority(int priority) {
        this.priority = priority;
    }
    
    @Override
    public String toString() {
        return String.format("Task{id='%s', type='%s', status='%s', description='%s'}", 
                           id, type, status, description);
    }
}