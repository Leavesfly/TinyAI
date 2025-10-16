package io.leavesfly.tinyai.agent.research.v2.model;

/**
 * 智能体类型枚举
 * 定义V2系统中的智能体角色和职责
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public enum AgentType {
    
    /**
     * 主控智能体 - 负责整体研究流程的协调和调度
     */
    MASTER("主控智能体", "负责整体研究流程的协调和调度"),
    
    /**
     * 规划智能体 - 制定研究计划和任务分解
     */
    PLANNER("规划智能体", "制定研究计划和任务分解"),
    
    /**
     * 执行智能体 - 协调并行任务执行
     */
    EXECUTOR("执行智能体", "协调并行任务执行"),
    
    /**
     * 检索智能体 - 负责信息检索和数据收集
     */
    SEARCHER("检索智能体", "负责信息检索和数据收集"),
    
    /**
     * 分析智能体 - 负责深度分析和推理
     */
    ANALYZER("分析智能体", "负责深度分析和推理"),
    
    /**
     * 验证智能体 - 负责结果验证和质量检查
     */
    VALIDATOR("验证智能体", "负责结果验证和质量检查"),
    
    /**
     * 综合智能体 - 负责信息整合和综合
     */
    SYNTHESIZER("综合智能体", "负责信息整合和综合"),
    
    /**
     * 写作智能体 - 负责研究报告生成
     */
    WRITER("写作智能体", "负责研究报告生成");
    
    /**
     * 智能体名称
     */
    private final String name;
    
    /**
     * 智能体职责描述
     */
    private final String responsibility;
    
    AgentType(String name, String responsibility) {
        this.name = name;
        this.responsibility = responsibility;
    }
    
    public String getName() {
        return name;
    }
    
    public String getResponsibility() {
        return responsibility;
    }
    
    /**
     * 判断是否为核心编排智能体
     */
    public boolean isCoreOrchestrator() {
        return this == MASTER || this == PLANNER || this == EXECUTOR;
    }
    
    /**
     * 判断是否为专业执行智能体
     */
    public boolean isSpecialist() {
        return this == SEARCHER || this == ANALYZER || 
               this == VALIDATOR || this == SYNTHESIZER || this == WRITER;
    }
}
