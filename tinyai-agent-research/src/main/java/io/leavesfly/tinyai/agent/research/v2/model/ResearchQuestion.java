package io.leavesfly.tinyai.agent.research.v2.model;

import java.util.*;

/**
 * 研究问题
 * 表示研究计划中的一个子问题
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ResearchQuestion {
    
    /**
     * 问题ID
     */
    private String questionId;
    
    /**
     * 问题内容
     */
    private String content;
    
    /**
     * 问题类型
     */
    private QuestionType type;
    
    /**
     * 优先级（1-10，10最高）
     */
    private int priority;
    
    /**
     * 依赖的问题ID列表
     */
    private List<String> dependencies;
    
    /**
     * 推荐的智能体类型
     */
    private AgentType recommendedAgent;
    
    /**
     * 预估耗时（秒）
     */
    private long estimatedDurationSeconds;
    
    /**
     * 附加元数据
     */
    private Map<String, Object> metadata;
    
    public ResearchQuestion() {
        this.questionId = UUID.randomUUID().toString();
        this.dependencies = new ArrayList<>();
        this.metadata = new HashMap<>();
        this.priority = 5;
    }
    
    public ResearchQuestion(String content, QuestionType type) {
        this();
        this.content = content;
        this.type = type;
        this.recommendedAgent = type.getRecommendedAgent();
    }
    
    /**
     * 添加依赖问题
     */
    public void addDependency(String questionId) {
        if (!this.dependencies.contains(questionId)) {
            this.dependencies.add(questionId);
        }
    }
    
    /**
     * 判断是否有依赖
     */
    public boolean hasDependencies() {
        return !dependencies.isEmpty();
    }
    
    // Getters and Setters
    
    public String getQuestionId() {
        return questionId;
    }
    
    public void setQuestionId(String questionId) {
        this.questionId = questionId;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    public QuestionType getType() {
        return type;
    }
    
    public void setType(QuestionType type) {
        this.type = type;
    }
    
    public int getPriority() {
        return priority;
    }
    
    public void setPriority(int priority) {
        this.priority = Math.max(1, Math.min(10, priority));
    }
    
    public List<String> getDependencies() {
        return dependencies;
    }
    
    public void setDependencies(List<String> dependencies) {
        this.dependencies = dependencies;
    }
    
    public AgentType getRecommendedAgent() {
        return recommendedAgent;
    }
    
    public void setRecommendedAgent(AgentType recommendedAgent) {
        this.recommendedAgent = recommendedAgent;
    }
    
    public long getEstimatedDurationSeconds() {
        return estimatedDurationSeconds;
    }
    
    public void setEstimatedDurationSeconds(long estimatedDurationSeconds) {
        this.estimatedDurationSeconds = estimatedDurationSeconds;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    @Override
    public String toString() {
        return "ResearchQuestion{" +
                "questionId='" + questionId + '\'' +
                ", content='" + content + '\'' +
                ", type=" + type +
                ", priority=" + priority +
                '}';
    }
}
