package io.leavesfly.tinyai.agent.research;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 研究阶段结果
 * 封装单个研究阶段的执行结果
 * 
 * @author 山泽
 */
public class ResearchPhaseResult {
    
    /** 研究阶段 */
    private final ResearchPhase phase;
    
    /** 步骤列表 */
    private List<ResearchStep> steps;
    
    /** 洞察列表 */
    private List<ResearchInsight> insights;
    
    /** 调用的工具 */
    private List<String> toolsCalled;
    
    /** 新生成的知识 */
    private List<Map<String, Object>> newKnowledge;
    
    /** 元数据 */
    private Map<String, Object> metadata;
    
    /**
     * 构造函数
     */
    public ResearchPhaseResult(ResearchPhase phase) {
        this.phase = phase;
        this.steps = new ArrayList<>();
        this.insights = new ArrayList<>();
        this.toolsCalled = new ArrayList<>();
        this.newKnowledge = new ArrayList<>();
        this.metadata = new HashMap<>();
    }
    
    // Getter和Setter方法
    public ResearchPhase getPhase() {
        return phase;
    }
    
    public List<ResearchStep> getSteps() {
        return steps;
    }
    
    public void setSteps(List<ResearchStep> steps) {
        this.steps = steps != null ? steps : new ArrayList<>();
    }
    
    public List<ResearchInsight> getInsights() {
        return insights;
    }
    
    public void setInsights(List<ResearchInsight> insights) {
        this.insights = insights != null ? insights : new ArrayList<>();
    }
    
    public List<String> getToolsCalled() {
        return toolsCalled;
    }
    
    public void setToolsCalled(List<String> toolsCalled) {
        this.toolsCalled = toolsCalled != null ? toolsCalled : new ArrayList<>();
    }
    
    public List<Map<String, Object>> getNewKnowledge() {
        return newKnowledge;
    }
    
    public void setNewKnowledge(List<Map<String, Object>> newKnowledge) {
        this.newKnowledge = newKnowledge != null ? newKnowledge : new ArrayList<>();
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata != null ? metadata : new HashMap<>();
    }
    
    /**
     * 添加步骤
     */
    public void addStep(ResearchStep step) {
        this.steps.add(step);
    }
    
    /**
     * 添加洞察
     */
    public void addInsight(ResearchInsight insight) {
        this.insights.add(insight);
    }
    
    /**
     * 添加工具调用
     */
    public void addToolCall(String toolName) {
        this.toolsCalled.add(toolName);
    }
    
    /**
     * 添加新知识
     */
    public void addKnowledge(String content, String source, double confidence) {
        Map<String, Object> knowledge = new HashMap<>();
        knowledge.put("content", content);
        knowledge.put("source", source);
        knowledge.put("confidence", confidence);
        this.newKnowledge.add(knowledge);
    }
    
    /**
     * 添加元数据
     */
    public void addMetadata(String key, Object value) {
        this.metadata.put(key, value);
    }
    
    @Override
    public String toString() {
        return String.format("ResearchPhaseResult{phase=%s, steps=%d, insights=%d, tools=%d}", 
            phase, steps.size(), insights.size(), toolsCalled.size());
    }
}