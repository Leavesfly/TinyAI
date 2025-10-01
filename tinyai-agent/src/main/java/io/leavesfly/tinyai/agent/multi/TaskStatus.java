package io.leavesfly.tinyai.agent.multi;

/**
 * 任务状态枚举
 * 定义任务的生命周期状态
 * 
 * @author 山泽
 */
public enum TaskStatus {
    
    /**
     * 等待中 - 任务已创建但尚未开始
     */
    PENDING("pending"),
    
    /**
     * 进行中 - 任务正在执行
     */
    IN_PROGRESS("in_progress"),
    
    /**
     * 已完成 - 任务成功完成
     */
    COMPLETED("completed"),
    
    /**
     * 失败 - 任务执行失败
     */
    FAILED("failed"),
    
    /**
     * 已取消 - 任务被取消
     */
    CANCELLED("cancelled");
    
    private final String value;
    
    TaskStatus(String value) {
        this.value = value;
    }
    
    public String getValue() {
        return value;
    }
    
    @Override
    public String toString() {
        return value;
    }
    
    /**
     * 根据字符串值获取对应的枚举
     */
    public static TaskStatus fromValue(String value) {
        for (TaskStatus status : values()) {
            if (status.value.equals(value)) {
                return status;
            }
        }
        throw new IllegalArgumentException("未知的任务状态: " + value);
    }
}