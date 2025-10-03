package io.leavesfly.tinyai.agent.manus;

/**
 * Agent状态枚举
 * 
 * @author 山泽
 */
public enum AgentState {
    /**
     * 空闲状态
     */
    IDLE("空闲"),
    
    /**
     * 思考状态
     */
    THINKING("思考"),
    
    /**
     * 行动状态
     */
    ACTING("行动"),
    
    /**
     * 观察状态
     */
    OBSERVING("观察"),
    
    /**
     * 反思状态
     */
    REFLECTING("反思"),
    
    /**
     * 计划状态
     */
    PLANNING("计划"),
    
    /**
     * 完成状态
     */
    DONE("完成"),
    
    /**
     * 错误状态
     */
    ERROR("错误");
    
    private final String description;
    
    AgentState(String description) {
        this.description = description;
    }
    
    public String getDescription() {
        return description;
    }
    
    @Override
    public String toString() {
        return description;
    }
}