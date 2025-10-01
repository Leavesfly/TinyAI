package io.leavesfly.tinyai.agent.multi;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Agent性能指标类
 * 用于跟踪和统计Agent的工作表现
 * 
 * @author 山泽
 */
public class AgentMetrics {
    
    // 任务相关指标
    private final AtomicInteger tasksCompleted = new AtomicInteger(0);      // 完成任务数
    private final AtomicInteger tasksAssigned = new AtomicInteger(0);       // 分配任务数
    private final AtomicInteger tasksFailed = new AtomicInteger(0);         // 失败任务数
    
    // 消息相关指标
    private final AtomicInteger messagesSent = new AtomicInteger(0);        // 发送消息数
    private final AtomicInteger messagesReceived = new AtomicInteger(0);    // 接收消息数
    
    // 时间相关指标
    private final AtomicLong totalExecutionTime = new AtomicLong(0);        // 总执行时间(毫秒)
    private volatile double averageResponseTime = 0.0;                      // 平均响应时间(秒)
    
    // 错误相关指标
    private final AtomicInteger errorCount = new AtomicInteger(0);          // 错误次数
    
    // 工作时间统计
    private volatile long startTime = 0;                                    // 启动时间
    private volatile long totalActiveTime = 0;                              // 总活跃时间
    
    /**
     * 记录任务完成
     */
    public void recordTaskCompleted(long executionTimeMs) {
        tasksCompleted.incrementAndGet();
        totalExecutionTime.addAndGet(executionTimeMs);
        updateAverageResponseTime();
    }
    
    /**
     * 记录任务分配
     */
    public void recordTaskAssigned() {
        tasksAssigned.incrementAndGet();
    }
    
    /**
     * 记录任务失败
     */
    public void recordTaskFailed() {
        tasksFailed.incrementAndGet();
    }
    
    /**
     * 记录发送消息
     */
    public void recordMessageSent() {
        messagesSent.incrementAndGet();
    }
    
    /**
     * 记录接收消息
     */
    public void recordMessageReceived() {
        messagesReceived.incrementAndGet();
    }
    
    /**
     * 记录错误
     */
    public void recordError() {
        errorCount.incrementAndGet();
    }
    
    /**
     * 设置启动时间
     */
    public void setStartTime(long startTime) {
        this.startTime = startTime;
    }
    
    /**
     * 更新总活跃时间
     */
    public void updateActiveTime() {
        if (startTime > 0) {
            totalActiveTime = System.currentTimeMillis() - startTime;
        }
    }
    
    /**
     * 更新平均响应时间
     */
    private void updateAverageResponseTime() {
        int completed = tasksCompleted.get();
        if (completed > 0) {
            averageResponseTime = totalExecutionTime.get() / (double) completed / 1000.0; // 转换为秒
        }
    }
    
    /**
     * 计算成功率
     */
    public double getSuccessRate() {
        int total = tasksCompleted.get() + tasksFailed.get();
        if (total == 0) {
            return 0.0;
        }
        return (double) tasksCompleted.get() / total;
    }
    
    /**
     * 计算任务完成率
     */
    public double getCompletionRate() {
        int assigned = tasksAssigned.get();
        if (assigned == 0) {
            return 0.0;
        }
        return (double) tasksCompleted.get() / assigned;
    }
    
    /**
     * 计算平均处理速度（任务/小时）
     */
    public double getTasksPerHour() {
        if (totalActiveTime <= 0) {
            return 0.0;
        }
        double hours = totalActiveTime / (1000.0 * 60 * 60); // 转换为小时
        return tasksCompleted.get() / hours;
    }
    
    // Getter 方法
    public int getTasksCompleted() {
        return tasksCompleted.get();
    }
    
    public int getTasksAssigned() {
        return tasksAssigned.get();
    }
    
    public int getTasksFailed() {
        return tasksFailed.get();
    }
    
    public int getMessagesSent() {
        return messagesSent.get();
    }
    
    public int getMessagesReceived() {
        return messagesReceived.get();
    }
    
    public long getTotalExecutionTime() {
        return totalExecutionTime.get();
    }
    
    public double getAverageResponseTime() {
        return averageResponseTime;
    }
    
    public int getErrorCount() {
        return errorCount.get();
    }
    
    public long getStartTime() {
        return startTime;
    }
    
    public long getTotalActiveTime() {
        return totalActiveTime;
    }
    
    /**
     * 重置所有指标
     */
    public void reset() {
        tasksCompleted.set(0);
        tasksAssigned.set(0);
        tasksFailed.set(0);
        messagesSent.set(0);
        messagesReceived.set(0);
        totalExecutionTime.set(0);
        averageResponseTime = 0.0;
        errorCount.set(0);
        startTime = 0;
        totalActiveTime = 0;
    }
    
    @Override
    public String toString() {
        updateActiveTime();
        return String.format(
            "AgentMetrics{" +
            "tasksCompleted=%d, tasksAssigned=%d, tasksFailed=%d, " +
            "messagesSent=%d, messagesReceived=%d, " +
            "avgResponseTime=%.2fs, successRate=%.2f%%, " +
            "errorCount=%d, activeTimeHours=%.2f}",
            tasksCompleted.get(), tasksAssigned.get(), tasksFailed.get(),
            messagesSent.get(), messagesReceived.get(),
            averageResponseTime, getSuccessRate() * 100,
            errorCount.get(), totalActiveTime / (1000.0 * 60 * 60)
        );
    }
}