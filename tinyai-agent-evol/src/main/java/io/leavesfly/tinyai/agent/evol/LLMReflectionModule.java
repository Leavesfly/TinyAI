package io.leavesfly.tinyai.agent.evol;

import java.util.*;
import java.util.stream.Collectors;
import java.util.concurrent.CompletableFuture;

/**
 * LLM增强的反思模块
 * 结合大语言模型的深度分析能力，提供更智能的经验反思和模式识别
 * 
 * @author 山泽
 */
public class LLMReflectionModule extends ReflectionModule {
    
    /**
     * LLM增强的反思记录类
     */
    public static class LLMReflectionRecord extends ReflectionRecord {
        private String llmDeepAnalysis;           // LLM深度分析
        private String insightSummary;            // 洞察摘要
        private double analyticalConfidence;      // 分析置信度
        private List<String> suggestedActions;    // 建议行动
        private Map<String, Object> enrichedMetadata; // 丰富的元数据
        
        public LLMReflectionRecord(String experienceId, String reflection, String reflectionType,
                                 String llmDeepAnalysis, String insightSummary, 
                                 double analyticalConfidence, List<String> suggestedActions) {
            super(experienceId, reflection, reflectionType);
            this.llmDeepAnalysis = llmDeepAnalysis;
            this.insightSummary = insightSummary;
            this.analyticalConfidence = analyticalConfidence;
            this.suggestedActions = suggestedActions != null ? suggestedActions : new ArrayList<>();
            this.enrichedMetadata = new HashMap<>();
        }
        
        // Getters and Setters
        public String getLlmDeepAnalysis() { return llmDeepAnalysis; }
        public String getInsightSummary() { return insightSummary; }
        public double getAnalyticalConfidence() { return analyticalConfidence; }
        public List<String> getSuggestedActions() { return suggestedActions; }
        public Map<String, Object> getEnrichedMetadata() { return enrichedMetadata; }
        public void setEnrichedMetadata(String key, Object value) { enrichedMetadata.put(key, value); }
    }
    
    /**
     * LLM增强的模式类
     */
    public static class LLMPattern extends Pattern {
        private String llmInterpretation;         // LLM解释
        private String predictiveInsight;         // 预测性洞察
        private double semanticRelevance;         // 语义相关性
        private List<String> relatedConcepts;     // 相关概念
        private String improvementSuggestion;     // 改进建议
        
        public LLMPattern(String type, String description, double strength,
                        String llmInterpretation, String predictiveInsight, double semanticRelevance) {
            super(type, description, strength);
            this.llmInterpretation = llmInterpretation;
            this.predictiveInsight = predictiveInsight;
            this.semanticRelevance = semanticRelevance;
            this.relatedConcepts = new ArrayList<>();
        }
        
        // Getters and Setters
        public String getLlmInterpretation() { return llmInterpretation; }
        public String getPredictiveInsight() { return predictiveInsight; }
        public double getSemanticRelevance() { return semanticRelevance; }
        public List<String> getRelatedConcepts() { return relatedConcepts; }
        public String getImprovementSuggestion() { return improvementSuggestion; }
        public void setImprovementSuggestion(String suggestion) { this.improvementSuggestion = suggestion; }
        public void addRelatedConcept(String concept) { relatedConcepts.add(concept); }
    }
    
    /** LLM模拟器 */
    private final EvolLLMSimulator llmSimulator;
    
    /** LLM增强的反思历史 */
    private final List<LLMReflectionRecord> llmReflectionHistory;
    
    /** LLM增强的模式缓存 */
    private final Map<String, List<LLMPattern>> llmPatternCache;
    
    /** 异步处理开关 */
    private boolean enableAsyncProcessing;
    
    /** LLM分析置信度阈值 */
    private double llmAnalysisThreshold;
    
    public LLMReflectionModule() {
        this(new EvolLLMSimulator(), true, 0.6);
    }
    
    public LLMReflectionModule(EvolLLMSimulator llmSimulator, boolean enableAsync, double analysisThreshold) {
        super();
        this.llmSimulator = llmSimulator;
        this.llmReflectionHistory = new ArrayList<>();
        this.llmPatternCache = new HashMap<>();
        this.enableAsyncProcessing = enableAsync;
        this.llmAnalysisThreshold = analysisThreshold;
    }
    
    /**
     * LLM增强的经验反思
     */
    @Override
    public String reflectOnExperience(Experience experience) {
        // 基础反思
        String basicReflection = super.reflectOnExperience(experience);
        
        // LLM增强反思
        if (llmSimulator != null) {
            try {
                // 收集上下文信息
                List<Experience> historicalExperiences = getRecentHistoricalExperiences(20);
                Map<String, Object> performanceMetrics = createPerformanceMetrics(experience, historicalExperiences);
                
                // 生成LLM深度反思
                String llmDeepAnalysis = generateLLMDeepReflection(experience, historicalExperiences, performanceMetrics);
                
                // 提取洞察摘要
                String insightSummary = extractInsightSummary(llmDeepAnalysis);
                
                // 计算分析置信度
                double analyticalConfidence = calculateAnalyticalConfidence(llmDeepAnalysis, experience);
                
                // 提取建议行动
                List<String> suggestedActions = extractSuggestedActions(llmDeepAnalysis);
                
                // 创建LLM反思记录
                if (analyticalConfidence >= llmAnalysisThreshold) {
                    String experienceId = String.valueOf(experience.hashCode());
                    String reflectionType = experience.isSuccess() ? "llm_success_analysis" : "llm_failure_analysis";
                    
                    LLMReflectionRecord llmRecord = new LLMReflectionRecord(
                        experienceId, basicReflection, reflectionType,
                        llmDeepAnalysis, insightSummary, analyticalConfidence, suggestedActions
                    );
                    
                    // 添加丰富的元数据
                    llmRecord.setEnrichedMetadata("experience_reward", experience.getReward());
                    llmRecord.setEnrichedMetadata("context_complexity", assessContextComplexity(experience.getContext()));
                    llmRecord.setEnrichedMetadata("historical_similarity", calculateHistoricalSimilarity(experience, historicalExperiences));
                    
                    llmReflectionHistory.add(llmRecord);
                    
                    // 合并基础反思和LLM反思
                    return combineReflections(basicReflection, llmDeepAnalysis, insightSummary);
                }
                
            } catch (Exception e) {
                System.err.println("LLM反思过程中出现异常: " + e.getMessage());
            }
        }
        
        return basicReflection;
    }
    
    /**
     * LLM增强的模式识别
     */
    public List<LLMPattern> identifyLLMPatterns(List<Experience> recentExperiences) {
        List<LLMPattern> llmPatterns = new ArrayList<>();
        
        // 基础模式识别
        List<Pattern> basicPatterns = super.identifyPatterns(recentExperiences);
        
        // LLM增强每个模式
        if (llmSimulator != null && !recentExperiences.isEmpty()) {
            for (Pattern basicPattern : basicPatterns) {
                try {
                    // 生成LLM解释
                    String llmInterpretation = generatePatternInterpretation(basicPattern, recentExperiences);
                    
                    // 生成预测性洞察
                    String predictiveInsight = generatePredictiveInsight(basicPattern, recentExperiences);
                    
                    // 计算语义相关性
                    double semanticRelevance = calculateSemanticRelevance(basicPattern, recentExperiences);
                    
                    // 创建LLM增强模式
                    LLMPattern llmPattern = new LLMPattern(
                        basicPattern.getType(),
                        basicPattern.getDescription(),
                        basicPattern.getStrength(),
                        llmInterpretation,
                        predictiveInsight,
                        semanticRelevance
                    );
                    
                    // 添加相关概念
                    List<String> relatedConcepts = identifyRelatedConcepts(basicPattern, recentExperiences);
                    relatedConcepts.forEach(llmPattern::addRelatedConcept);
                    
                    // 生成改进建议
                    String improvementSuggestion = generateImprovementSuggestion(basicPattern, recentExperiences);
                    llmPattern.setImprovementSuggestion(improvementSuggestion);
                    
                    // 复制基础模式的证据和元数据
                    basicPattern.getEvidences().forEach(llmPattern::addEvidence);
                    llmPattern.getMetadata().putAll(basicPattern.getMetadata());
                    
                    llmPatterns.add(llmPattern);
                    
                } catch (Exception e) {
                    System.err.println("LLM模式增强过程中出现异常: " + e.getMessage());
                    // 继续处理其他模式
                }
            }
        }
        
        // 缓存LLM模式结果
        String cacheKey = "llm_patterns_" + recentExperiences.size();
        llmPatternCache.put(cacheKey, llmPatterns);
        
        return llmPatterns;
    }
    
    /**
     * 异步LLM反思
     */
    public CompletableFuture<String> reflectOnExperienceAsync(Experience experience) {
        if (!enableAsyncProcessing || llmSimulator == null) {
            return CompletableFuture.completedFuture(reflectOnExperience(experience));
        }
        
        return CompletableFuture.supplyAsync(() -> {
            try {
                return reflectOnExperience(experience);
            } catch (Exception e) {
                return "异步反思过程中出现异常: " + e.getMessage();
            }
        });
    }
    
    /**
     * 智能洞察提取
     */
    public List<String> extractIntelligentInsights(List<Experience> experiences, String focusArea) {
        List<String> insights = new ArrayList<>();
        
        if (llmSimulator != null && !experiences.isEmpty()) {
            try {
                // 基于焦点领域生成洞察
                String prompt = String.format("分析%d个经验记录，重点关注%s领域的洞察", 
                                             experiences.size(), focusArea);
                String llmInsights = llmSimulator.generateEvolResponse(
                    prompt, 
                    summarizeExperiences(experiences), 
                    "pattern_analysis"
                );
                
                // 解析LLM生成的洞察
                insights.addAll(parseInsightsFromLLMResponse(llmInsights));
                
            } catch (Exception e) {
                System.err.println("智能洞察提取过程中出现异常: " + e.getMessage());
            }
        }
        
        // 如果LLM洞察不足，添加传统洞察
        if (insights.size() < 3) {
            insights.addAll(generateTraditionalInsights(experiences, focusArea));
        }
        
        return insights.stream().distinct().collect(Collectors.toList());
    }
    
    /**
     * 元学习分析
     */
    public String generateMetaLearningAnalysis(List<Experience> experiences) {
        if (llmSimulator == null || experiences.isEmpty()) {
            return "元学习分析需要LLM支持和充足的经验数据";
        }
        
        try {
            // 分析学习过程本身
            String learningProgressSummary = analyzeLearningProgress(experiences);
            
            // 生成元学习洞察
            String metaPrompt = String.format(
                "基于学习进展分析：%s，进行元学习分析，重点关注学习策略的有效性和改进方向",
                learningProgressSummary
            );
            
            return llmSimulator.generateEvolResponse(
                metaPrompt,
                createMetaLearningContext(experiences),
                "deep_reflection"
            );
            
        } catch (Exception e) {
            return "元学习分析过程中出现异常: " + e.getMessage();
        }
    }
    
    // ================================
    // LLM增强的具体实现方法
    // ================================
    
    private String generateLLMDeepReflection(Experience experience, List<Experience> historical, 
                                           Map<String, Object> performanceMetrics) {
        if (enableAsyncProcessing) {
            try {
                CompletableFuture<String> reflectionTask = llmSimulator.generateEvolResponseAsync(
                    createReflectionPrompt(experience),
                    createReflectionContext(experience, historical, performanceMetrics),
                    "deep_reflection"
                );
                return reflectionTask.get(); // 简化处理，实际应用中可考虑超时
            } catch (Exception e) {
                System.err.println("异步LLM反思失败，切换到同步模式: " + e.getMessage());
            }
        }
        
        return llmSimulator.generateDeepReflection(experience, historical, performanceMetrics);
    }
    
    private String createReflectionPrompt(Experience experience) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("深度分析以下经验:");
        prompt.append("\\n任务: ").append(experience.getTask());
        prompt.append("\\n动作: ").append(experience.getAction());
        prompt.append("\\n结果: ").append(experience.isSuccess() ? "成功" : "失败");
        prompt.append("\\n奖励: ").append(experience.getReward());
        prompt.append("\\n\\n请提供深入的因果分析和学习建议。");
        return prompt.toString();
    }
    
    private String createReflectionContext(Experience experience, List<Experience> historical, 
                                         Map<String, Object> performanceMetrics) {
        Map<String, Object> context = new HashMap<>();
        context.put("target_experience", experience.toMap());
        context.put("historical_count", historical.size());
        context.put("performance_metrics", performanceMetrics);
        context.put("context_summary", summarizeContext(experience.getContext()));
        return context.toString();
    }
    
    private String extractInsightSummary(String llmAnalysis) {
        // 从LLM分析中提取关键洞察
        if (llmAnalysis.contains("洞察")) {
            int start = llmAnalysis.indexOf("洞察");
            int end = Math.min(start + 150, llmAnalysis.length());
            return llmAnalysis.substring(start, end).trim();
        }
        
        // 提取第一句作为摘要
        String[] sentences = llmAnalysis.split("[。！？]");
        return sentences.length > 0 ? sentences[0] + "。" : "深度分析已完成";
    }
    
    private double calculateAnalyticalConfidence(String llmAnalysis, Experience experience) {
        double confidence = 0.5; // 基础置信度
        
        // 基于分析长度调整
        if (llmAnalysis.length() > 200) confidence += 0.1;
        if (llmAnalysis.length() > 500) confidence += 0.1;
        
        // 基于关键词密度调整
        String[] keyTerms = {"分析", "因为", "建议", "洞察", "模式", "策略"};
        long keyTermCount = Arrays.stream(keyTerms)
            .filter(llmAnalysis::contains)
            .count();
        confidence += keyTermCount * 0.05;
        
        // 基于经验质量调整
        if (experience.getReward() > 0.8 || experience.getReward() < -0.5) {
            confidence += 0.1; // 极端结果的分析更有价值
        }
        
        return Math.min(0.95, confidence);
    }
    
    private List<String> extractSuggestedActions(String llmAnalysis) {
        List<String> actions = new ArrayList<>();
        
        // 查找建议相关的内容
        String[] patterns = {"建议", "应该", "可以", "推荐", "尝试"};
        
        for (String pattern : patterns) {
            if (llmAnalysis.contains(pattern)) {
                int start = llmAnalysis.indexOf(pattern);
                int end = Math.min(start + 100, llmAnalysis.length());
                String suggestion = llmAnalysis.substring(start, end);
                actions.add(suggestion.split("[。！？]")[0]);
            }
        }
        
        // 如果没有找到具体建议，提供通用建议
        if (actions.isEmpty()) {
            actions.add("继续积累相关经验");
            actions.add("优化决策策略");
            actions.add("加强模式识别能力");
        }
        
        return actions.stream().distinct().limit(5).collect(Collectors.toList());
    }
    
    private Map<String, Object> createPerformanceMetrics(Experience experience, List<Experience> historical) {
        Map<String, Object> metrics = new HashMap<>();
        
        if (!historical.isEmpty()) {
            double avgReward = historical.stream().mapToDouble(Experience::getReward).average().orElse(0.0);
            long successCount = historical.stream().filter(Experience::isSuccess).count();
            double successRate = (double) successCount / historical.size();
            
            metrics.put("avg_reward", avgReward);
            metrics.put("success_rate", successRate);
            metrics.put("total_experiences", historical.size());
            metrics.put("current_reward", experience.getReward());
        }
        
        return metrics;
    }
    
    private double assessContextComplexity(Map<String, Object> context) {
        double complexity = 0.0;
        
        // 基于上下文要素数量
        complexity += context.size() * 0.1;
        
        // 基于值的复杂性
        for (Object value : context.values()) {
            if (value instanceof Map) {
                complexity += 0.2; // 嵌套结构增加复杂性
            } else if (value instanceof List) {
                complexity += 0.15; // 列表结构
            } else if (value instanceof String && ((String) value).length() > 50) {
                complexity += 0.1; // 长字符串
            }
        }
        
        return Math.min(1.0, complexity);
    }
    
    private double calculateHistoricalSimilarity(Experience experience, List<Experience> historical) {
        if (historical.isEmpty()) return 0.0;
        
        return historical.stream()
            .mapToDouble(hist -> calculateExperienceSimilarity(experience, hist))
            .max().orElse(0.0);
    }
    
    private double calculateExperienceSimilarity(Experience exp1, Experience exp2) {
        double similarity = 0.0;
        
        // 任务相似度
        if (exp1.getTask().equals(exp2.getTask())) {
            similarity += 0.4;
        } else if (exp1.getTask().contains(exp2.getTask()) || exp2.getTask().contains(exp1.getTask())) {
            similarity += 0.2;
        }
        
        // 动作相似度
        if (exp1.getAction().equals(exp2.getAction())) {
            similarity += 0.3;
        }
        
        // 结果相似度
        if (exp1.isSuccess() == exp2.isSuccess()) {
            similarity += 0.2;
        }
        
        // 奖励相似度
        double rewardDiff = Math.abs(exp1.getReward() - exp2.getReward());
        similarity += Math.max(0, 0.1 * (1 - rewardDiff));
        
        return similarity;
    }
    
    private String combineReflections(String basicReflection, String llmAnalysis, String insightSummary) {
        StringBuilder combined = new StringBuilder();
        combined.append("[基础反思] ").append(basicReflection);
        combined.append("\\n\\n[深度洞察] ").append(insightSummary);
        
        if (llmAnalysis.length() > 200) {
            combined.append("\\n\\n[详细分析] ").append(llmAnalysis.substring(0, 200)).append("...");
        } else {
            combined.append("\\n\\n[详细分析] ").append(llmAnalysis);
        }
        
        return combined.toString();
    }
    
    private String generatePatternInterpretation(Pattern pattern, List<Experience> experiences) {
        return llmSimulator.generatePatternAnalysis(experiences, pattern.getType());
    }
    
    private String generatePredictiveInsight(Pattern pattern, List<Experience> experiences) {
        String prompt = String.format("基于模式'%s'，预测未来可能的发展趋势和影响", pattern.getDescription());
        return llmSimulator.generateEvolResponse(prompt, 
            summarizeExperiences(experiences), "pattern_analysis");
    }
    
    private double calculateSemanticRelevance(Pattern pattern, List<Experience> experiences) {
        // 简化的语义相关性计算
        double relevance = 0.5;
        
        // 基于模式强度
        relevance += pattern.getStrength() * 0.3;
        
        // 基于证据数量
        relevance += Math.min(0.2, pattern.getEvidences().size() * 0.05);
        
        return Math.min(1.0, relevance);
    }
    
    private List<String> identifyRelatedConcepts(Pattern pattern, List<Experience> experiences) {
        Set<String> concepts = new HashSet<>();
        
        // 从经验中提取概念
        for (Experience exp : experiences) {
            concepts.add(exp.getTask());
            concepts.add(exp.getAction());
        }
        
        // 基于模式类型添加相关概念
        switch (pattern.getType()) {
            case "success_action":
                concepts.add("成功模式");
                concepts.add("有效策略");
                break;
            case "failure_combo":
                concepts.add("失败模式");
                concepts.add("风险因素");
                break;
            case "improvement_trend":
                concepts.add("学习进展");
                concepts.add("能力提升");
                break;
        }
        
        return new ArrayList<>(concepts).subList(0, Math.min(5, concepts.size()));
    }
    
    private String generateImprovementSuggestion(Pattern pattern, List<Experience> experiences) {
        String prompt = String.format("基于识别到的模式'%s'，提供具体的改进建议", pattern.getDescription());
        return llmSimulator.generateEvolResponse(prompt, 
            summarizeExperiences(experiences), "decision_advice");
    }
    
    private List<Experience> getRecentHistoricalExperiences(int count) {
        // 简化实现：返回空列表，实际应用中应从上层获取
        return new ArrayList<>();
    }
    
    private String summarizeExperiences(List<Experience> experiences) {
        if (experiences.isEmpty()) return "无经验数据";
        
        long successCount = experiences.stream().filter(Experience::isSuccess).count();
        double avgReward = experiences.stream().mapToDouble(Experience::getReward).average().orElse(0.0);
        
        return String.format("经验总数: %d, 成功率: %.1f%%, 平均奖励: %.2f", 
                           experiences.size(), 
                           (double) successCount / experiences.size() * 100, 
                           avgReward);
    }
    
    private String summarizeContext(Map<String, Object> context) {
        if (context.isEmpty()) return "空上下文";
        
        StringBuilder summary = new StringBuilder();
        context.entrySet().stream().limit(3).forEach(entry -> 
            summary.append(entry.getKey()).append("=").append(entry.getValue()).append("; ")
        );
        
        return summary.toString();
    }
    
    private List<String> parseInsightsFromLLMResponse(String llmResponse) {
        List<String> insights = new ArrayList<>();
        
        // 按行分割并查找洞察相关内容
        String[] lines = llmResponse.split("\\n");
        for (String line : lines) {
            if (line.contains("洞察") || line.contains("发现") || line.contains("观察")) {
                insights.add(line.trim());
            }
        }
        
        // 如果没有找到格式化的洞察，提取关键句子
        if (insights.isEmpty()) {
            String[] sentences = llmResponse.split("[。！？]");
            for (int i = 0; i < Math.min(3, sentences.length); i++) {
                if (sentences[i].trim().length() > 10) {
                    insights.add(sentences[i].trim());
                }
            }
        }
        
        return insights;
    }
    
    private List<String> generateTraditionalInsights(List<Experience> experiences, String focusArea) {
        List<String> insights = new ArrayList<>();
        
        if (!experiences.isEmpty()) {
            long successCount = experiences.stream().filter(Experience::isSuccess).count();
            double successRate = (double) successCount / experiences.size();
            
            insights.add(String.format("当前成功率为%.1f%%，%s", 
                       successRate * 100,
                       successRate > 0.7 ? "表现良好" : "需要改进"));
            
            // 最常用的动作
            Map<String, Long> actionCounts = experiences.stream()
                .collect(Collectors.groupingBy(Experience::getAction, Collectors.counting()));
            
            actionCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .ifPresent(entry -> insights.add(String.format("最常使用的动作是'%s'，使用了%d次", 
                                                 entry.getKey(), entry.getValue())));
        }
        
        return insights;
    }
    
    private String analyzeLearningProgress(List<Experience> experiences) {
        if (experiences.size() < 5) {
            return "经验数据不足，无法分析学习进展";
        }
        
        // 分阶段分析学习进展
        int midPoint = experiences.size() / 2;
        List<Experience> early = experiences.subList(0, midPoint);
        List<Experience> recent = experiences.subList(midPoint, experiences.size());
        
        double earlySuccessRate = early.stream().filter(Experience::isSuccess).count() / (double) early.size();
        double recentSuccessRate = recent.stream().filter(Experience::isSuccess).count() / (double) recent.size();
        
        double improvement = recentSuccessRate - earlySuccessRate;
        
        if (improvement > 0.1) {
            return String.format("学习进展良好，成功率从%.1f%%提升到%.1f%%", 
                               earlySuccessRate * 100, recentSuccessRate * 100);
        } else if (improvement < -0.1) {
            return String.format("学习进展出现退步，成功率从%.1f%%下降到%.1f%%", 
                               earlySuccessRate * 100, recentSuccessRate * 100);
        } else {
            return String.format("学习进展稳定，成功率维持在%.1f%%左右", 
                               recentSuccessRate * 100);
        }
    }
    
    private String createMetaLearningContext(List<Experience> experiences) {
        Map<String, Object> context = new HashMap<>();
        context.put("total_experiences", experiences.size());
        context.put("experience_summary", summarizeExperiences(experiences));
        context.put("learning_trajectory", analyzeLearningProgress(experiences));
        return context.toString();
    }
    
    // ================================
    // LLM增强的Getter方法
    // ================================
    
    /**
     * 获取LLM反思历史摘要
     */
    public List<LLMReflectionRecord> getLLMReflectionSummary(int limit) {
        return llmReflectionHistory.stream()
            .sorted((r1, r2) -> Long.compare(r2.getTimestamp(), r1.getTimestamp()))
            .limit(limit)
            .collect(Collectors.toList());
    }
    
    /**
     * 获取高置信度的LLM洞察
     */
    public List<LLMReflectionRecord> getHighConfidenceInsights(double minConfidence) {
        return llmReflectionHistory.stream()
            .filter(record -> record.getAnalyticalConfidence() >= minConfidence)
            .sorted((r1, r2) -> Double.compare(r2.getAnalyticalConfidence(), r1.getAnalyticalConfidence()))
            .collect(Collectors.toList());
    }
    
    /**
     * 获取LLM模式缓存
     */
    public Map<String, List<LLMPattern>> getLLMPatternCache() {
        return new HashMap<>(llmPatternCache);
    }
    
    /**
     * 获取LLM相关统计信息
     */
    public Map<String, Object> getLLMStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("llm_enabled", llmSimulator != null);
        stats.put("llm_reflection_count", llmReflectionHistory.size());
        stats.put("async_processing_enabled", enableAsyncProcessing);
        stats.put("analysis_threshold", llmAnalysisThreshold);
        
        if (!llmReflectionHistory.isEmpty()) {
            double avgConfidence = llmReflectionHistory.stream()
                .mapToDouble(LLMReflectionRecord::getAnalyticalConfidence)
                .average().orElse(0.0);
            stats.put("avg_analytical_confidence", avgConfidence);
        }
        
        return stats;
    }
    
    // Getters and Setters
    public boolean isAsyncProcessingEnabled() { return enableAsyncProcessing; }
    public void setAsyncProcessingEnabled(boolean enabled) { this.enableAsyncProcessing = enabled; }
    
    public double getLlmAnalysisThreshold() { return llmAnalysisThreshold; }
    public void setLlmAnalysisThreshold(double threshold) { 
        this.llmAnalysisThreshold = Math.max(0.0, Math.min(1.0, threshold)); 
    }
    
    public EvolLLMSimulator getLlmSimulator() { return llmSimulator; }
}