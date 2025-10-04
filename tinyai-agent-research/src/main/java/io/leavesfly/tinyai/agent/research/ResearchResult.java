package io.leavesfly.tinyai.agent.research;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 研究结果
 * 封装完整研究流程的最终结果
 * 
 * @author 山泽
 */
public class ResearchResult {
    
    /** 研究上下文 */
    private final ResearchContext context;
    
    /** 最终答案 */
    private String finalAnswer;
    
    /** 总体置信度 */
    private double totalConfidence;
    
    /** 质量评分 */
    private double qualityScore;
    
    /** 结果时间戳 */
    private LocalDateTime timestamp;
    
    /** 处理统计 */
    private Map<String, Object> statistics;
    
    /**
     * 构造函数
     */
    public ResearchResult(ResearchContext context) {
        this.context = context;
        this.timestamp = LocalDateTime.now();
        this.statistics = new HashMap<>();
        
        // 计算结果指标
        calculateMetrics();
    }
    
    /**
     * 计算研究指标
     */
    private void calculateMetrics() {
        List<ResearchStep> allSteps = context.getAllSteps();
        
        // 计算总体置信度
        this.totalConfidence = allSteps.stream()
                .mapToDouble(ResearchStep::getConfidence)
                .average()
                .orElse(0.0);
        
        // 生成最终答案
        this.finalAnswer = generateFinalAnswer();
        
        // 评估质量评分
        this.qualityScore = assessQualityScore();
        
        // 统计信息
        calculateStatistics();
    }
    
    /**
     * 生成最终答案
     */
    private String generateFinalAnswer() {
        // 获取结论阶段的步骤
        ResearchPhaseResult conclusionResult = context.getPhaseResult(ResearchPhase.CONCLUSION);
        
        if (conclusionResult != null && !conclusionResult.getSteps().isEmpty()) {
            return conclusionResult.getSteps().get(0).getContent();
        }
        
        // 如果没有结论，生成默认答案
        ResearchQuery query = context.getQuery();
        int keyFindings = context.getAllInsights().size();
        
        return String.format(
            "基于深度研究分析，对问题'%s'进行了全面调研。通过多阶段研究流程，" +
            "发现了 %d 个关键洞察，为理解该问题提供了多维度的视角。",
            query.getQuery(), keyFindings);
    }
    
    /**
     * 评估质量评分
     */
    private double assessQualityScore() {
        List<ResearchStep> allSteps = context.getAllSteps();
        List<ResearchInsight> allInsights = context.getAllInsights();
        List<String> allTools = context.getAllToolsCalled();
        
        // 多维度评估
        Map<String, Double> factors = new HashMap<>();
        factors.put("completeness", Math.min(1.0, allSteps.size() / 15.0));  // 完整性
        factors.put("depth", Math.min(1.0, allSteps.stream()
            .filter(step -> "thought".equals(step.getStepType()))
            .count() / 8.0));  // 思考深度
        factors.put("diversity", Math.min(1.0, allTools.stream()
            .collect(Collectors.toSet()).size() / 3.0));  // 工具多样性
        factors.put("insights", Math.min(1.0, allInsights.size() / 3.0));  // 洞察数量
        factors.put("confidence", totalConfidence);  // 平均置信度
        
        // 加权平均
        Map<String, Double> weights = Map.of(
            "completeness", 0.2,
            "depth", 0.25,
            "diversity", 0.2,
            "insights", 0.2,
            "confidence", 0.15
        );
        
        double qualityScore = factors.entrySet().stream()
                .mapToDouble(entry -> entry.getValue() * weights.get(entry.getKey()))
                .sum();
        
        return Math.round(qualityScore * 1000.0) / 1000.0;  // 保留3位小数
    }
    
    /**
     * 计算统计信息
     */
    private void calculateStatistics() {
        statistics.put("totalSteps", context.getAllSteps().size());
        statistics.put("totalInsights", context.getAllInsights().size());
        statistics.put("uniqueTools", context.getAllToolsCalled().stream()
            .collect(Collectors.toSet()).size());
        statistics.put("phasesCompleted", context.getPhaseResults().size());
        statistics.put("averageConfidence", totalConfidence);
        statistics.put("qualityScore", qualityScore);
        
        // 按阶段统计
        Map<String, Integer> phaseStats = new HashMap<>();
        for (ResearchPhase phase : context.getPhaseResults().keySet()) {
            ResearchPhaseResult result = context.getPhaseResult(phase);
            phaseStats.put(phase.name(), result.getSteps().size());
        }
        statistics.put("phaseSteps", phaseStats);
    }
    
    /**
     * 转换为详细的Map格式
     */
    public Map<String, Object> toDetailedMap() {
        Map<String, Object> result = new HashMap<>();
        
        // 基本信息
        result.put("query", context.getQuery().getQuery());
        result.put("domain", context.getQuery().getDomain());
        result.put("finalAnswer", finalAnswer);
        result.put("totalConfidence", totalConfidence);
        result.put("qualityScore", qualityScore);
        result.put("timestamp", timestamp.toString());
        
        // 统计信息
        result.putAll(statistics);
        
        // 关键洞察
        List<String> keyInsights = context.getAllInsights().stream()
                .map(ResearchInsight::getContent)
                .collect(Collectors.toList());
        result.put("keyInsights", keyInsights);
        
        // 详细步骤
        List<Map<String, Object>> detailedSteps = context.getAllSteps().stream()
                .map(step -> {
                    Map<String, Object> stepMap = new HashMap<>();
                    stepMap.put("phase", step.getPhase().name());
                    stepMap.put("type", step.getStepType());
                    stepMap.put("content", step.getContent());
                    stepMap.put("confidence", step.getConfidence());
                    return stepMap;
                })
                .collect(Collectors.toList());
        result.put("detailedSteps", detailedSteps);
        
        return result;
    }
    
    /**
     * 转换为简化的Map格式
     */
    public Map<String, Object> toSummaryMap() {
        Map<String, Object> summary = new HashMap<>();
        
        summary.put("query", context.getQuery().getQuery());
        summary.put("finalAnswer", finalAnswer);
        summary.put("totalConfidence", totalConfidence);
        summary.put("qualityScore", qualityScore);
        summary.put("keyInsightsCount", context.getAllInsights().size());
        summary.put("totalSteps", context.getAllSteps().size());
        summary.put("timestamp", timestamp.toString());
        
        return summary;
    }
    
    // Getter方法
    public ResearchContext getContext() {
        return context;
    }
    
    public String getFinalAnswer() {
        return finalAnswer;
    }
    
    public double getTotalConfidence() {
        return totalConfidence;
    }
    
    public double getQualityScore() {
        return qualityScore;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public Map<String, Object> getStatistics() {
        return statistics;
    }
    
    @Override
    public String toString() {
        return String.format("ResearchResult{confidence=%.2f, quality=%.3f, steps=%d, insights=%d}", 
            totalConfidence, qualityScore, 
            context.getAllSteps().size(), context.getAllInsights().size());
    }
}