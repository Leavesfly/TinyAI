package io.leavesfly.tinyai.agent.multi;

/**
 * Agent状态枚举
 * 定义Agent的各种工作状态
 * 
 * @author 山泽
 */
public enum AgentState {
    
    /**
     * 空闲状态 - Agent可以接受新任务
     */
    IDLE("idle"),
    
    /**
     * 忙碌状态 - Agent正在执行任务
     */
    BUSY("busy"),
    
    /**
     * 思考状态 - Agent正在处理和分析信息
     */
    THINKING("thinking"),
    
    /**
     * 通信状态 - Agent正在与其他Agent通信
     */
    COMMUNICATING("communicating"),
    
    /**
     * 错误状态 - Agent遇到了错误
     */
    ERROR("error"),
    
    /**
     * 离线状态 - Agent已停止工作
     */
    OFFLINE("offline");
    
    private final String value;
    
    AgentState(String value) {
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
    public static AgentState fromValue(String value) {
        for (AgentState state : values()) {
            if (state.value.equals(value)) {
                return state;
            }
        }
        throw new IllegalArgumentException("未知的Agent状态: " + value);
    }
}