package io.leavesfly.tinyai.agent.research.v2.model;

import java.time.LocalDateTime;
import java.util.*;

/**
 * 智能体任务
 * 表示由特定智能体执行的任务
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class AgentTask {
    
    /**
     * 任务ID
     */
    private String taskId;
    
    /**
     * 智能体类型
     */
    private AgentType agentType;
    
    /**
     * 关联的研究问题ID
     */
    private String questionId;
    
    /**
     * 任务状态
     */
    private TaskStatus status;
    
    /**
     * 任务输入参数
     */
    private Map<String, Object> input;
    
    /**
     * 任务输出结果
     */
    private Map<String, Object> output;
    
    /**
     * 任务上下文
     */
    private Map<String, Object> context;
    
    /**
     * 开始时间
     */
    private LocalDateTime startTime;
    
    /**
     * 结束时间
     */
    private LocalDateTime endTime;
    
    /**
     * 错误信息
     */
    private String errorMessage;
    
    /**
     * 依赖的任务ID列表
     */
    private List<String> dependencies;
    
    /**
     * 优先级
     */
    private int priority;
    
    public AgentTask() {
        this.taskId = UUID.randomUUID().toString();
        this.status = TaskStatus.SUBMITTED;
        this.input = new HashMap<>();
        this.output = new HashMap<>();
        this.context = new HashMap<>();
        this.dependencies = new ArrayList<>();
        this.priority = 5;
    }
    
    public AgentTask(AgentType agentType, String questionId) {
        this();
        this.agentType = agentType;
        this.questionId = questionId;
    }
    
    /**
     * 启动任务
     */
    public void start() {
        this.status = TaskStatus.EXECUTING;
        this.startTime = LocalDateTime.now();
    }
    
    /**
     * 完成任务
     */
    public void complete() {
        this.status = TaskStatus.COMPLETED;
        this.endTime = LocalDateTime.now();
    }
    
    /**
     * 任务失败
     */
    public void fail(String errorMessage) {
        this.status = TaskStatus.FAILED;
        this.errorMessage = errorMessage;
        this.endTime = LocalDateTime.now();
    }
    
    /**
     * 获取执行时长（秒）
     */
    public long getDurationSeconds() {
        if (startTime == null) {
            return 0;
        }
        LocalDateTime end = endTime != null ? endTime : LocalDateTime.now();
        return java.time.Duration.between(startTime, end).getSeconds();
    }
    
    /**
     * 添加输入参数
     */
    public void putInput(String key, Object value) {
        this.input.put(key, value);
    }
    
    /**
     * 添加输出结果
     */
    public void putOutput(String key, Object value) {
        this.output.put(key, value);
    }
    
    /**
     * 获取输入参数
     */
    @SuppressWarnings("unchecked")
    public <T> T getInput(String key) {
        return (T) this.input.get(key);
    }
    
    /**
     * 获取输出结果
     */
    @SuppressWarnings("unchecked")
    public <T> T getOutput(String key) {
        return (T) this.output.get(key);
    }
    
    // Getters and Setters
    
    public String getTaskId() {
        return taskId;
    }
    
    public void setTaskId(String taskId) {
        this.taskId = taskId;
    }
    
    public AgentType getAgentType() {
        return agentType;
    }
    
    public void setAgentType(AgentType agentType) {
        this.agentType = agentType;
    }
    
    public String getQuestionId() {
        return questionId;
    }
    
    public void setQuestionId(String questionId) {
        this.questionId = questionId;
    }
    
    public TaskStatus getStatus() {
        return status;
    }
    
    public void setStatus(TaskStatus status) {
        this.status = status;
    }
    
    public Map<String, Object> getInput() {
        return input;
    }
    
    public void setInput(Map<String, Object> input) {
        this.input = input;
    }
    
    public Map<String, Object> getOutput() {
        return output;
    }
    
    public void setOutput(Map<String, Object> output) {
        this.output = output;
    }
    
    public Map<String, Object> getContext() {
        return context;
    }
    
    public void setContext(Map<String, Object> context) {
        this.context = context;
    }
    
    public LocalDateTime getStartTime() {
        return startTime;
    }
    
    public void setStartTime(LocalDateTime startTime) {
        this.startTime = startTime;
    }
    
    public LocalDateTime getEndTime() {
        return endTime;
    }
    
    public void setEndTime(LocalDateTime endTime) {
        this.endTime = endTime;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    public List<String> getDependencies() {
        return dependencies;
    }
    
    public void setDependencies(List<String> dependencies) {
        this.dependencies = dependencies;
    }
    
    public int getPriority() {
        return priority;
    }
    
    public void setPriority(int priority) {
        this.priority = priority;
    }
    
    @Override
    public String toString() {
        return "AgentTask{" +
                "taskId='" + taskId + '\'' +
                ", agentType=" + agentType +
                ", status=" + status +
                ", questionId='" + questionId + '\'' +
                '}';
    }
}
