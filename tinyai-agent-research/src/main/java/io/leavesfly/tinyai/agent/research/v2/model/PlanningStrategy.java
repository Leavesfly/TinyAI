package io.leavesfly.tinyai.agent.research.v2.model;

/**
 * 研究规划策略枚举
 * 定义不同的研究计划制定策略
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public enum PlanningStrategy {
    
    /**
     * 广度优先策略 - 优先探索主题的多个维度
     */
    BREADTH_FIRST("广度优先", "优先探索主题的多个维度，适合探索性研究"),
    
    /**
     * 深度优先策略 - 专注单一方向的深度挖掘
     */
    DEPTH_FIRST("深度优先", "专注单一方向的深度挖掘，适合专项研究"),
    
    /**
     * 混合策略 - 平衡广度和深度
     */
    HYBRID("混合策略", "平衡广度和深度，适合复杂研究任务"),
    
    /**
     * 重要性驱动策略 - 优先处理关键问题
     */
    IMPORTANCE_DRIVEN("重要性驱动", "优先处理关键问题，适合时间受限场景"),
    
    /**
     * 自适应策略 - 根据中间结果动态调整
     */
    ADAPTIVE("自适应策略", "根据中间结果动态调整，适合不确定性高的研究");
    
    /**
     * 策略名称
     */
    private final String name;
    
    /**
     * 策略描述
     */
    private final String description;
    
    PlanningStrategy(String name, String description) {
        this.name = name;
        this.description = description;
    }
    
    public String getName() {
        return name;
    }
    
    public String getDescription() {
        return description;
    }
}
