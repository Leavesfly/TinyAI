package io.leavesfly.tinyai.agent.multi;

import java.time.LocalDateTime;
import java.util.*;

/**
 * 任务类
 * 定义Agent执行的任务结构
 * 
 * @author 山泽
 */
public class AgentTask {
    
    private String id;                           // 任务唯一标识
    private String title;                        // 任务标题
    private String description;                  // 任务描述
    private String assignedTo;                   // 分配给的Agent ID
    private String createdBy;                    // 创建者Agent ID
    private TaskStatus status;                   // 任务状态
    private int priority;                        // 优先级 (1-10, 10最高)
    private LocalDateTime deadline;              // 截止时间
    private Object result;                       // 任务结果
    private List<AgentTask> subtasks;           // 子任务列表
    private List<String> dependencies;           // 依赖的任务ID列表
    private Map<String, Object> metadata;        // 元数据
    private LocalDateTime createdAt;             // 创建时间
    private LocalDateTime updatedAt;             // 更新时间
    
    // 构造函数
    public AgentTask() {
        this.id = UUID.randomUUID().toString();
        this.status = TaskStatus.PENDING;
        this.priority = 1;
        this.subtasks = new ArrayList<>();
        this.dependencies = new ArrayList<>();
        this.metadata = new HashMap<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public AgentTask(String title, String description) {
        this();
        this.title = title;
        this.description = description;
    }
    
    public AgentTask(String title, String description, String createdBy) {
        this(title, description);
        this.createdBy = createdBy;
    }
    
    // Getter 和 Setter 方法
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getTitle() {
        return title;
    }
    
    public void setTitle(String title) {
        this.title = title;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public String getAssignedTo() {
        return assignedTo;
    }
    
    public void setAssignedTo(String assignedTo) {
        this.assignedTo = assignedTo;
    }
    
    public String getCreatedBy() {
        return createdBy;
    }
    
    public void setCreatedBy(String createdBy) {
        this.createdBy = createdBy;
    }
    
    public TaskStatus getStatus() {
        return status;
    }
    
    public void setStatus(TaskStatus status) {
        this.status = status;
        this.updatedAt = LocalDateTime.now();
    }
    
    public int getPriority() {
        return priority;
    }
    
    public void setPriority(int priority) {
        this.priority = Math.max(1, Math.min(10, priority)); // 限制在1-10范围内
    }
    
    public LocalDateTime getDeadline() {
        return deadline;
    }
    
    public void setDeadline(LocalDateTime deadline) {
        this.deadline = deadline;
    }
    
    public Object getResult() {
        return result;
    }
    
    public void setResult(Object result) {
        this.result = result;
        this.updatedAt = LocalDateTime.now();
    }
    
    public List<AgentTask> getSubtasks() {
        return subtasks;
    }
    
    public void setSubtasks(List<AgentTask> subtasks) {
        this.subtasks = subtasks != null ? subtasks : new ArrayList<>();
    }
    
    public List<String> getDependencies() {
        return dependencies;
    }
    
    public void setDependencies(List<String> dependencies) {
        this.dependencies = dependencies != null ? dependencies : new ArrayList<>();
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata != null ? metadata : new HashMap<>();
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
    
    /**
     * 添加子任务
     */
    public void addSubtask(AgentTask subtask) {
        this.subtasks.add(subtask);
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 添加依赖任务
     */
    public void addDependency(String taskId) {
        if (!this.dependencies.contains(taskId)) {
            this.dependencies.add(taskId);
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 添加元数据
     */
    public void addMetadata(String key, Object value) {
        this.metadata.put(key, value);
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 检查任务是否可以开始执行（所有依赖已完成）
     */
    public boolean canStart(Map<String, AgentTask> allTasks) {
        if (this.status != TaskStatus.PENDING) {
            return false;
        }
        
        for (String depId : dependencies) {
            AgentTask depTask = allTasks.get(depId);
            if (depTask == null || depTask.getStatus() != TaskStatus.COMPLETED) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * 转换为Map格式
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("id", id);
        map.put("title", title);
        map.put("description", description);
        map.put("assignedTo", assignedTo);
        map.put("createdBy", createdBy);
        map.put("status", status.getValue());
        map.put("priority", priority);
        map.put("deadline", deadline != null ? deadline.toString() : null);
        map.put("result", result);
        map.put("subtaskCount", subtasks.size());
        map.put("dependencyCount", dependencies.size());
        map.put("metadata", metadata);
        map.put("createdAt", createdAt.toString());
        map.put("updatedAt", updatedAt.toString());
        return map;
    }
    
    @Override
    public String toString() {
        return String.format("AgentTask{id='%s', title='%s', status=%s, priority=%d, assignedTo='%s'}",
                id, title, status, priority, assignedTo);
    }
}