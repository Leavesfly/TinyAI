package io.leavesfly.tinyai.agent.manus;

/**
 * OpenManus执行模式枚举
 * 
 * @author 山泽
 */
public enum ExecutionMode {
    /**
     * 直接Agent模式
     * 使用基础的ReAct模式进行推理和行动
     */
    DIRECT_AGENT("直接Agent模式"),
    
    /**
     * Flow编排模式
     * 根据查询类型选择合适的Flow进行执行
     */
    FLOW_ORCHESTRATION("Flow编排模式");
    
    private final String description;
    
    ExecutionMode(String description) {
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