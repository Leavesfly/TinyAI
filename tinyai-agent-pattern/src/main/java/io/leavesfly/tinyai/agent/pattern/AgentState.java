package io.leavesfly.tinyai.agent.pattern;

/**
 * Agent状态枚举
 * 定义了Agent在执行过程中的不同状态
 * @author 山泽
 */
public enum AgentState {
    /** 思考状态 */
    THINKING("thinking"),
    
    /** 行动状态 */
    ACTING("acting"),
    
    /** 观察状态 */
    OBSERVING("observing"),
    
    /** 反思状态 */
    REFLECTING("reflecting"),
    
    /** 规划状态 */
    PLANNING("planning"),
    
    /** 完成状态 */
    DONE("done");
    
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
}