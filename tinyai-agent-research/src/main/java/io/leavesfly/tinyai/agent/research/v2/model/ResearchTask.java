package io.leavesfly.tinyai.agent.research.v2.model;

import java.time.LocalDateTime;
import java.util.*;

/**
 * 研究任务
 * 表示一个完整的研究任务实例
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ResearchTask {
    
    /**
     * 任务ID（唯一标识）
     */
    private String taskId;
    
    /**
     * 研究主题/问题
     */
    private String topic;
    
    /**
     * 任务状态
     */
    private TaskStatus status;
    
    /**
     * 研究计划
     */
    private ResearchPlan plan;
    
    /**
     * 智能体任务列表
     */
    private List<AgentTask> agentTasks;
    
    /**
     * 研究上下文（存储中间结果和状态）
     */
    private Map<String, Object> context;
    
    /**
     * 最终研究报告
     */
    private ResearchReport report;
    
    /**
     * 任务创建时间
     */
    private LocalDateTime createdAt;
    
    /**
     * 任务更新时间
     */
    private LocalDateTime updatedAt;
    
    /**
     * 任务完成时间
     */
    private LocalDateTime completedAt;
    
    /**
     * 错误信息（如果失败）
     */
    private String errorMessage;
    
    /**
     * 用户ID
     */
    private String userId;
    
    /**
     * 项目ID
     */
    private String projectId;
    
    /**
     * 配置参数
     */
    private Map<String, Object> config;
    
    public ResearchTask() {
        this.taskId = UUID.randomUUID().toString();
        this.status = TaskStatus.SUBMITTED;
        this.agentTasks = new ArrayList<>();
        this.context = new HashMap<>();
        this.config = new HashMap<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public ResearchTask(String topic) {
        this();
        this.topic = topic;
    }
    
    /**
     * 更新任务状态
     */
    public void updateStatus(TaskStatus newStatus) {
        if (this.status.canTransitionTo(newStatus)) {
            this.status = newStatus;
            this.updatedAt = LocalDateTime.now();
            
            if (newStatus.isTerminal()) {
                this.completedAt = LocalDateTime.now();
            }
        } else {
            throw new IllegalStateException(
                String.format("无法从状态 %s 转换到 %s", this.status, newStatus)
            );
        }
    }
    
    /**
     * 添加智能体任务
     */
    public void addAgentTask(AgentTask task) {
        this.agentTasks.add(task);
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 设置研究计划
     */
    public void setPlan(ResearchPlan plan) {
        this.plan = plan;
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 设置研究报告
     */
    public void setReport(ResearchReport report) {
        this.report = report;
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 添加上下文信息
     */
    public void putContext(String key, Object value) {
        this.context.put(key, value);
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 获取上下文信息
     */
    @SuppressWarnings("unchecked")
    public <T> T getContext(String key) {
        return (T) this.context.get(key);
    }
    
    /**
     * 设置配置参数
     */
    public void setConfig(String key, Object value) {
        this.config.put(key, value);
    }
    
    /**
     * 获取配置参数
     */
    @SuppressWarnings("unchecked")
    public <T> T getConfig(String key) {
        return (T) this.config.get(key);
    }
    
    /**
     * 计算任务执行时长（秒）
     */
    public long getDurationSeconds() {
        if (completedAt == null) {
            return java.time.Duration.between(createdAt, LocalDateTime.now()).getSeconds();
        }
        return java.time.Duration.between(createdAt, completedAt).getSeconds();
    }
    
    // Getters and Setters
    
    public String getTaskId() {
        return taskId;
    }
    
    public void setTaskId(String taskId) {
        this.taskId = taskId;
    }
    
    public String getTopic() {
        return topic;
    }
    
    public void setTopic(String topic) {
        this.topic = topic;
    }
    
    public TaskStatus getStatus() {
        return status;
    }
    
    public void setStatus(TaskStatus status) {
        this.status = status;
    }
    
    public ResearchPlan getPlan() {
        return plan;
    }
    
    public List<AgentTask> getAgentTasks() {
        return agentTasks;
    }
    
    public void setAgentTasks(List<AgentTask> agentTasks) {
        this.agentTasks = agentTasks;
    }
    
    public Map<String, Object> getContext() {
        return context;
    }
    
    public void setContext(Map<String, Object> context) {
        this.context = context;
    }
    
    public ResearchReport getReport() {
        return report;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
    
    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
    
    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
    
    public LocalDateTime getCompletedAt() {
        return completedAt;
    }
    
    public void setCompletedAt(LocalDateTime completedAt) {
        this.completedAt = completedAt;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    public String getUserId() {
        return userId;
    }
    
    public void setUserId(String userId) {
        this.userId = userId;
    }
    
    public String getProjectId() {
        return projectId;
    }
    
    public void setProjectId(String projectId) {
        this.projectId = projectId;
    }
    
    public Map<String, Object> getConfig() {
        return config;
    }
    
    public void setConfig(Map<String, Object> config) {
        this.config = config;
    }
    
    @Override
    public String toString() {
        return "ResearchTask{" +
                "taskId='" + taskId + '\'' +
                ", topic='" + topic + '\'' +
                ", status=" + status +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                '}';
    }
}
