package io.leavesfly.tinyai.agent.research;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * 智能推理器
 * 自适应选择推理策略并执行推理任务
 * 
 * @author 山泽
 */
public class IntelligentReasoner {
    
    /** 推理策略映射 */
    private final Map<ReasoningMode, BiFunction<ResearchQuery, Map<String, Object>, List<String>>> reasoningPatterns;
    
    /** 性能历史记录 */
    private final Map<ReasoningMode, List<Double>> performanceHistory;
    
    /** 推理器配置 */
    private final Map<String, Object> config;
    
    /**
     * 构造函数
     */
    public IntelligentReasoner() {
        this.reasoningPatterns = new HashMap<>();
        this.performanceHistory = new HashMap<>();
        this.config = new HashMap<>();
        
        // 初始化推理策略
        initializeReasoningPatterns();
        
        // 初始化性能历史
        for (ReasoningMode mode : ReasoningMode.values()) {
            performanceHistory.put(mode, new ArrayList<>());
        }
        
        // 设置默认配置
        initializeConfig();
    }
    
    /**
     * 智能选择推理模式
     * 基于查询特征和历史性能选择最适合的推理策略
     */
    public ReasoningMode selectReasoningMode(ResearchQuery query, Map<String, Object> context) {
        // 基于查询复杂度和紧急度的基础选择
        if (query.getComplexity() <= 2 && query.getUrgency() >= 4) {
            return ReasoningMode.QUICK;
        }
        
        if (query.getDepthRequired() >= 4) {
            return ReasoningMode.THOROUGH;
        }
        
        // 基于查询内容的关键词匹配
        String queryText = query.getQuery().toLowerCase();
        
        if (containsKeywords(queryText, "创新", "新", "发明", "原创")) {
            return ReasoningMode.CREATIVE;
        }
        
        if (containsKeywords(queryText, "分析", "比较", "评估", "对比")) {
            return ReasoningMode.ANALYTICAL;
        }
        
        if (containsKeywords(queryText, "系统", "结构", "框架", "体系")) {
            return ReasoningMode.SYSTEMATIC;
        }
        
        // 考虑历史性能，选择表现最好的模式
        ReasoningMode bestMode = getBestPerformingMode();
        if (bestMode != null) {
            return bestMode;
        }
        
        // 默认选择系统推理模式
        return ReasoningMode.SYSTEMATIC;
    }
    
    /**
     * 执行推理
     */
    public List<String> reason(ResearchQuery query, Map<String, Object> context, ReasoningMode mode) {
        if (mode == null) {
            mode = selectReasoningMode(query, context);
        }
        
        BiFunction<ResearchQuery, Map<String, Object>, List<String>> reasoningFunction = 
            reasoningPatterns.get(mode);
        
        if (reasoningFunction == null) {
            throw new IllegalArgumentException("不支持的推理模式: " + mode);
        }
        
        return reasoningFunction.apply(query, context);
    }
    
    /**
     * 执行推理（自动选择模式）
     */
    public List<String> reason(ResearchQuery query, Map<String, Object> context) {
        return reason(query, context, null);
    }
    
    /**
     * 记录推理性能
     */
    public void recordPerformance(ReasoningMode mode, double score) {
        List<Double> scores = performanceHistory.get(mode);
        scores.add(score);
        
        // 保持历史记录在合理范围内
        if (scores.size() > 100) {
            scores.remove(0);
        }
    }
    
    /**
     * 获取推理模式的平均性能
     */
    public double getAveragePerformance(ReasoningMode mode) {
        List<Double> scores = performanceHistory.get(mode);
        if (scores.isEmpty()) {
            return 0.0;
        }
        
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * 获取性能最佳的推理模式
     */
    private ReasoningMode getBestPerformingMode() {
        ReasoningMode bestMode = null;
        double bestScore = 0.0;
        
        for (ReasoningMode mode : ReasoningMode.values()) {
            double avgScore = getAveragePerformance(mode);
            if (avgScore > bestScore) {
                bestScore = avgScore;
                bestMode = mode;
            }
        }
        
        return bestMode;
    }
    
    /**
     * 检查查询中是否包含指定关键词
     */
    private boolean containsKeywords(String text, String... keywords) {
        for (String keyword : keywords) {
            if (text.contains(keyword)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 初始化推理策略
     */
    private void initializeReasoningPatterns() {
        reasoningPatterns.put(ReasoningMode.QUICK, this::quickReasoning);
        reasoningPatterns.put(ReasoningMode.THOROUGH, this::thoroughReasoning);
        reasoningPatterns.put(ReasoningMode.CREATIVE, this::creativeReasoning);
        reasoningPatterns.put(ReasoningMode.ANALYTICAL, this::analyticalReasoning);
        reasoningPatterns.put(ReasoningMode.SYSTEMATIC, this::systematicReasoning);
    }
    
    /**
     * 快速推理模式
     */
    private List<String> quickReasoning(ResearchQuery query, Map<String, Object> context) {
        List<String> steps = new ArrayList<>();
        steps.add("快速分析问题: " + query.getQuery());
        steps.add("识别核心关键词和概念");
        steps.add("调用已有知识进行直接匹配");
        steps.add("生成初步答案");
        
        return steps;
    }
    
    /**
     * 彻底推理模式
     */
    private List<String> thoroughReasoning(ResearchQuery query, Map<String, Object> context) {
        List<String> steps = new ArrayList<>();
        steps.add("深入分析问题的多个维度: " + query.getQuery());
        steps.add("分解问题为多个子问题");
        steps.add("系统性收集相关信息");
        steps.add("多角度分析每个子问题");
        steps.add("综合分析结果");
        steps.add("验证推理逻辑");
        steps.add("形成全面结论");
        
        return steps;
    }
    
    /**
     * 创意推理模式
     */
    private List<String> creativeReasoning(ResearchQuery query, Map<String, Object> context) {
        List<String> steps = new ArrayList<>();
        steps.add("从创新角度重新审视问题: " + query.getQuery());
        steps.add("寻找非传统的思考角度");
        steps.add("联想相关但不直接的领域知识");
        steps.add("生成多个假设性方案");
        steps.add("评估创新方案的可行性");
        steps.add("整合最有潜力的创新想法");
        
        return steps;
    }
    
    /**
     * 分析推理模式
     */
    private List<String> analyticalReasoning(ResearchQuery query, Map<String, Object> context) {
        List<String> steps = new ArrayList<>();
        steps.add("系统分析问题结构: " + query.getQuery());
        steps.add("识别变量和影响因素");
        steps.add("建立因果关系模型");
        steps.add("量化分析各因素权重");
        steps.add("对比不同方案的优劣");
        steps.add("得出基于数据的结论");
        
        return steps;
    }
    
    /**
     * 系统推理模式
     */
    private List<String> systematicReasoning(ResearchQuery query, Map<String, Object> context) {
        List<String> steps = new ArrayList<>();
        steps.add("系统性地梳理问题: " + query.getQuery());
        steps.add("构建问题的概念框架");
        steps.add("按逻辑顺序收集信息");
        steps.add("建立知识结构图");
        steps.add("进行结构化分析");
        steps.add("形成系统性结论");
        
        return steps;
    }
    
    /**
     * 初始化配置
     */
    private void initializeConfig() {
        config.put("maxReasoningSteps", 10);
        config.put("confidenceThreshold", 0.7);
        config.put("performanceWeight", 0.3);
    }
    
    // Getter方法
    public Map<ReasoningMode, List<Double>> getPerformanceHistory() {
        return performanceHistory;
    }
    
    public Map<String, Object> getConfig() {
        return config;
    }
    
    /**
     * 获取推理器统计信息
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        
        for (ReasoningMode mode : ReasoningMode.values()) {
            Map<String, Object> modeStats = new HashMap<>();
            List<Double> scores = performanceHistory.get(mode);
            
            modeStats.put("totalRuns", scores.size());
            modeStats.put("averagePerformance", getAveragePerformance(mode));
            
            stats.put(mode.name(), modeStats);
        }
        
        return stats;
    }
}