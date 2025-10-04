package io.leavesfly.tinyai.agent.research;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 研究上下文
 * 管理研究过程中的状态和数据
 * 
 * @author 山泽
 */
public class ResearchContext {
    
    /** 研究查询 */
    private final ResearchQuery query;
    
    /** 各阶段结果 */
    private final Map<ResearchPhase, ResearchPhaseResult> phaseResults;
    
    /** 上下文元数据 */
    private final Map<String, Object> metadata;
    
    /**
     * 构造函数
     */
    public ResearchContext(ResearchQuery query) {
        this.query = query;
        this.phaseResults = new HashMap<>();
        this.metadata = new HashMap<>();
    }
    
    /**
     * 添加阶段结果
     */
    public void addPhaseResult(ResearchPhase phase, ResearchPhaseResult result) {
        phaseResults.put(phase, result);
    }
    
    /**
     * 获取阶段结果
     */
    public ResearchPhaseResult getPhaseResult(ResearchPhase phase) {
        return phaseResults.get(phase);
    }
    
    /**
     * 获取所有研究步骤
     */
    public List<ResearchStep> getAllSteps() {
        return phaseResults.values().stream()
                .flatMap(result -> result.getSteps().stream())
                .collect(Collectors.toList());
    }
    
    /**
     * 获取所有洞察
     */
    public List<ResearchInsight> getAllInsights() {
        return phaseResults.values().stream()
                .flatMap(result -> result.getInsights().stream())
                .collect(Collectors.toList());
    }
    
    /**
     * 获取所有调用的工具
     */
    public List<String> getAllToolsCalled() {
        return phaseResults.values().stream()
                .flatMap(result -> result.getToolsCalled().stream())
                .collect(Collectors.toList());
    }
    
    /**
     * 获取所有新知识
     */
    public List<Map<String, Object>> getAllNewKnowledge() {
        return phaseResults.values().stream()
                .flatMap(result -> result.getNewKnowledge().stream())
                .collect(Collectors.toList());
    }
    
    /**
     * 添加元数据
     */
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }
    
    /**
     * 获取元数据
     */
    public Object getMetadata(String key) {
        return metadata.get(key);
    }
    
    /**
     * 转换为Map格式
     */
    public Map<String, Object> toMap() {
        Map<String, Object> contextMap = new HashMap<>();
        contextMap.put("query", query);
        contextMap.put("steps", getAllSteps());
        contextMap.put("insights", getAllInsights());
        contextMap.put("toolsCalled", getAllToolsCalled());
        contextMap.put("newKnowledge", getAllNewKnowledge());
        contextMap.put("metadata", metadata);
        
        return contextMap;
    }
    
    // Getter方法
    public ResearchQuery getQuery() {
        return query;
    }
    
    public Map<ResearchPhase, ResearchPhaseResult> getPhaseResults() {
        return phaseResults;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
}