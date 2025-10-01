package io.leavesfly.tinyai.agent.multi;

/**
 * 消息类型枚举
 * 定义Agent间通信的消息类型
 * 
 * @author 山泽
 */
public enum MessageType {
    
    /**
     * 文本消息 - 普通的对话消息
     */
    TEXT("text"),
    
    /**
     * 任务消息 - 分配任务的消息
     */
    TASK("task"),
    
    /**
     * 结果消息 - 返回任务执行结果
     */
    RESULT("result"),
    
    /**
     * 错误消息 - 报告错误状态
     */
    ERROR("error"),
    
    /**
     * 系统消息 - 系统级别的控制消息
     */
    SYSTEM("system"),
    
    /**
     * 广播消息 - 向所有Agent广播的消息
     */
    BROADCAST("broadcast");
    
    private final String value;
    
    MessageType(String value) {
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
    public static MessageType fromValue(String value) {
        for (MessageType type : values()) {
            if (type.value.equals(value)) {
                return type;
            }
        }
        throw new IllegalArgumentException("未知的消息类型: " + value);
    }
}