package io.leavesfly.tinyai.agent.manus;

import java.time.LocalDateTime;
import java.util.*;

/**
 * 计划类
 * 计划驱动模式的核心组件
 * 
 * @author 山泽
 */
public class Plan {
    
    private String id;                          // 计划ID
    private String title;                       // 计划标题
    private String description;                 // 计划描述
    private String goal;                        // 计划目标
    private List<Task> tasks;                   // 任务列表
    private String status;                      // 计划状态：planning, executing, completed, failed
    private LocalDateTime createdAt;            // 创建时间
    private LocalDateTime startedAt;            // 开始时间
    private LocalDateTime completedAt;          // 完成时间
    private Map<String, Object> metadata;       // 元数据
    private int currentTaskIndex;               // 当前任务索引
    private boolean allowParallel;              // 是否允许并行执行
    
    /**
     * 构造函数
     */
    public Plan() {
        this.id = generatePlanId();
        this.tasks = new ArrayList<>();
        this.status = "planning";
        this.createdAt = LocalDateTime.now();
        this.metadata = new HashMap<>();
        this.currentTaskIndex = 0;
        this.allowParallel = false;
    }
    
    public Plan(String title, String goal) {
        this();
        this.title = title;
        this.goal = goal;
    }
    
    public Plan(String title, String description, String goal) {
        this(title, goal);
        this.description = description;
    }
    
    /**
     * 生成计划ID
     */
    private String generatePlanId() {
        return "plan_" + System.currentTimeMillis() + "_" + (int)(Math.random() * 1000);
    }
    
    /**
     * 添加任务
     */
    public void addTask(Task task) {
        this.tasks.add(task);
    }
    
    public void addTask(String description, String type) {
        this.tasks.add(new Task(description, type));
    }
    
    public void addTask(String description, String type, Map<String, Object> parameters) {
        this.tasks.add(new Task(description, type, parameters));
    }
    
    /**
     * 开始执行计划
     */
    public void start() {
        this.status = "executing";
        this.startedAt = LocalDateTime.now();
        this.currentTaskIndex = 0;
    }
    
    /**
     * 获取下一个待执行的任务
     */
    public Task getNextTask() {
        if (currentTaskIndex >= tasks.size()) {
            return null;
        }
        
        // 如果允许并行，返回所有待执行的任务
        if (allowParallel) {
            for (int i = currentTaskIndex; i < tasks.size(); i++) {
                Task task = tasks.get(i);
                if ("pending".equals(task.getStatus())) {
                    return task;
                }
            }
            return null;
        } else {
            // 顺序执行
            Task currentTask = tasks.get(currentTaskIndex);
            if (currentTask.isCompleted() || currentTask.isFailed()) {
                currentTaskIndex++;
                return getNextTask();
            }
            return currentTask;
        }
    }
    
    /**
     * 标记任务完成并移动到下一个
     */
    public void completeCurrentTask(Object result) {
        if (currentTaskIndex < tasks.size()) {
            Task currentTask = tasks.get(currentTaskIndex);
            currentTask.complete(result);
            
            if (!allowParallel) {
                currentTaskIndex++;
            }
            
            // 检查计划是否完成
            checkPlanCompletion();
        }
    }
    
    /**
     * 标记当前任务失败
     */
    public void failCurrentTask(String errorMessage) {
        if (currentTaskIndex < tasks.size()) {
            Task currentTask = tasks.get(currentTaskIndex);
            currentTask.fail(errorMessage);
            
            // 根据策略决定是否继续执行
            if (!allowParallel) {
                // 计划失败
                this.status = "failed";
                this.completedAt = LocalDateTime.now();
            }
        }
    }
    
    /**
     * 检查计划是否完成
     */
    private void checkPlanCompletion() {
        boolean allCompleted = true;
        boolean hasFailed = false;
        
        for (Task task : tasks) {
            if (task.isFailed()) {
                hasFailed = true;
            } else if (!task.isCompleted()) {
                allCompleted = false;
            }
        }
        
        if (hasFailed && !allowParallel) {
            this.status = "failed";
            this.completedAt = LocalDateTime.now();
        } else if (allCompleted) {
            this.status = "completed";
            this.completedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 获取计划执行进度（0.0 - 1.0）
     */
    public double getProgress() {
        if (tasks.isEmpty()) {
            return 0.0;
        }
        
        long completedTasks = tasks.stream()
                .mapToLong(task -> task.isCompleted() ? 1 : 0)
                .sum();
        
        return (double) completedTasks / tasks.size();
    }
    
    /**
     * 获取计划统计信息
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total_tasks", tasks.size());
        stats.put("completed_tasks", (int) tasks.stream().mapToLong(task -> task.isCompleted() ? 1 : 0).sum());
        stats.put("failed_tasks", (int) tasks.stream().mapToLong(task -> task.isFailed() ? 1 : 0).sum());
        stats.put("running_tasks", (int) tasks.stream().mapToLong(task -> task.isRunning() ? 1 : 0).sum());
        stats.put("progress", getProgress());
        stats.put("status", status);
        
        if (startedAt != null) {
            long executionTime = java.time.Duration.between(startedAt, 
                completedAt != null ? completedAt : LocalDateTime.now()).toMillis();
            stats.put("execution_time_ms", executionTime);
        }
        
        return stats;
    }
    
    /**
     * 判断计划是否完成
     */
    public boolean isCompleted() {
        return "completed".equals(status);
    }
    
    /**
     * 判断计划是否失败
     */
    public boolean isFailed() {
        return "failed".equals(status);
    }
    
    /**
     * 判断计划是否正在执行
     */
    public boolean isExecuting() {
        return "executing".equals(status);
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
    
    public String getGoal() {
        return goal;
    }
    
    public void setGoal(String goal) {
        this.goal = goal;
    }
    
    public List<Task> getTasks() {
        return new ArrayList<>(tasks);
    }
    
    public void setTasks(List<Task> tasks) {
        this.tasks = tasks != null ? new ArrayList<>(tasks) : new ArrayList<>();
    }
    
    public String getStatus() {
        return status;
    }
    
    public void setStatus(String status) {
        this.status = status;
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
    
    public Map<String, Object> getMetadata() {
        return new HashMap<>(metadata);
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata != null ? new HashMap<>(metadata) : new HashMap<>();
    }
    
    public void addMetadata(String key, Object value) {
        this.metadata.put(key, value);
    }
    
    public int getCurrentTaskIndex() {
        return currentTaskIndex;
    }
    
    public boolean isAllowParallel() {
        return allowParallel;
    }
    
    public void setAllowParallel(boolean allowParallel) {
        this.allowParallel = allowParallel;
    }
    
    @Override
    public String toString() {
        return String.format("Plan{id='%s', title='%s', status='%s', tasks=%d, progress=%.2f}", 
                           id, title, status, tasks.size(), getProgress());
    }
}