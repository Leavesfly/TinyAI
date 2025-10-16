package io.leavesfly.tinyai.agent.research.v2.model;

/**
 * 问题类型枚举
 * 定义研究问题的分类和特征
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public enum QuestionType {
    
    /**
     * 事实性问题 - 需要查找和验证具体事实
     */
    FACTUAL("事实性问题", "需要查找和验证具体事实", AgentType.SEARCHER),
    
    /**
     * 分析性问题 - 需要深度分析和推理
     */
    ANALYTICAL("分析性问题", "需要深度分析和推理", AgentType.ANALYZER),
    
    /**
     * 对比性问题 - 需要对比和评估多个选项
     */
    COMPARATIVE("对比性问题", "需要对比和评估多个选项", AgentType.ANALYZER),
    
    /**
     * 探索性问题 - 需要广泛探索和发现
     */
    EXPLORATORY("探索性问题", "需要广泛探索和发现", AgentType.SEARCHER),
    
    /**
     * 综合性问题 - 需要整合多个维度的信息
     */
    SYNTHESIS("综合性问题", "需要整合多个维度的信息", AgentType.SYNTHESIZER),
    
    /**
     * 评估性问题 - 需要评判和验证
     */
    EVALUATIVE("评估性问题", "需要评判和验证", AgentType.VALIDATOR),
    
    /**
     * 创造性问题 - 需要创新和构想
     */
    CREATIVE("创造性问题", "需要创新和构想", AgentType.WRITER);
    
    /**
     * 问题类型名称
     */
    private final String name;
    
    /**
     * 问题类型描述
     */
    private final String description;
    
    /**
     * 推荐的智能体类型
     */
    private final AgentType recommendedAgent;
    
    QuestionType(String name, String description, AgentType recommendedAgent) {
        this.name = name;
        this.description = description;
        this.recommendedAgent = recommendedAgent;
    }
    
    public String getName() {
        return name;
    }
    
    public String getDescription() {
        return description;
    }
    
    public AgentType getRecommendedAgent() {
        return recommendedAgent;
    }
    
    /**
     * 判断是否需要检索能力
     */
    public boolean requiresRetrieval() {
        return this == FACTUAL || this == EXPLORATORY || this == COMPARATIVE;
    }
    
    /**
     * 判断是否需要深度分析
     */
    public boolean requiresDeepAnalysis() {
        return this == ANALYTICAL || this == COMPARATIVE || 
               this == SYNTHESIS || this == EVALUATIVE;
    }
}
