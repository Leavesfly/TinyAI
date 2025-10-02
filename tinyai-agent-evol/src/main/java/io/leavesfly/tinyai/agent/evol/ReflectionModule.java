package io.leavesfly.tinyai.agent.evol;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 反思模块
 * 用于分析经验，识别模式，生成学习洞察
 * 
 * @author 山泽
 */
public class ReflectionModule {
    
    /**
     * 反思记录类
     */
    public static class ReflectionRecord {
        private String experienceId;
        private String reflection;
        private long timestamp;
        private String reflectionType;
        private double confidence;
        
        public ReflectionRecord(String experienceId, String reflection, String reflectionType) {
            this.experienceId = experienceId;
            this.reflection = reflection;
            this.reflectionType = reflectionType;
            this.timestamp = System.currentTimeMillis();
            this.confidence = 0.8; // 默认置信度
        }
        
        // Getters and Setters
        public String getExperienceId() { return experienceId; }
        public String getReflection() { return reflection; }
        public long getTimestamp() { return timestamp; }
        public String getReflectionType() { return reflectionType; }
        public double getConfidence() { return confidence; }
        public void setConfidence(double confidence) { this.confidence = confidence; }
    }
    
    /**
     * 模式识别结果类
     */
    public static class Pattern {
        private String type;
        private String description;
        private double strength;
        private List<String> evidences;
        private Map<String, Object> metadata;
        
        public Pattern(String type, String description, double strength) {
            this.type = type;
            this.description = description;
            this.strength = strength;
            this.evidences = new ArrayList<>();
            this.metadata = new HashMap<>();
        }
        
        public void addEvidence(String evidence) {
            evidences.add(evidence);
        }
        
        // Getters and Setters
        public String getType() { return type; }
        public String getDescription() { return description; }
        public double getStrength() { return strength; }
        public List<String> getEvidences() { return evidences; }
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(String key, Object value) { metadata.put(key, value); }
    }
    
    /** 反思历史记录 */
    private final List<ReflectionRecord> reflectionHistory;
    
    /** 模式缓存 */
    private final Map<String, List<Pattern>> patternCache;
    
    /** 反思模板 */
    private final List<String> reflectionTemplates;
    
    public ReflectionModule() {
        this.reflectionHistory = new ArrayList<>();
        this.patternCache = new HashMap<>();
        this.reflectionTemplates = initializeReflectionTemplates();
    }
    
    /**
     * 初始化反思模板
     * 
     * @return 反思模板列表
     */
    private List<String> initializeReflectionTemplates() {
        return Arrays.asList(
            "为什么任务 '%s' 的结果是%s？关键因素是什么？",
            "在处理 '%s' 时，我可以如何改进执行策略？",
            "这次经验教会了我什么关于 '%s' 的新知识？",
            "动作 '%s' 在什么情况下最有效？",
            "失败的原因是什么？如何避免类似问题？",
            "成功的关键要素有哪些？如何复制这种成功？",
            "上下文 %s 对结果有什么影响？",
            "这个经验与之前的经验有什么关联？"
        );
    }
    
    /**
     * 对单个经验进行反思
     * 
     * @param experience 经验对象
     * @return 反思内容
     */
    public String reflectOnExperience(Experience experience) {
        String reflection = generateReflection(experience);
        
        // 记录反思
        String experienceId = String.valueOf(experience.hashCode());
        String reflectionType = experience.isSuccess() ? "success_analysis" : "failure_analysis";
        ReflectionRecord record = new ReflectionRecord(experienceId, reflection, reflectionType);
        
        reflectionHistory.add(record);
        
        // 限制反思历史大小
        if (reflectionHistory.size() > 500) {
            reflectionHistory.remove(0);
        }
        
        return reflection;
    }
    
    /**
     * 生成具体的反思内容
     * 
     * @param experience 经验对象
     * @return 反思内容
     */
    private String generateReflection(Experience experience) {
        StringBuilder reflection = new StringBuilder();
        
        if (experience.isSuccess()) {
            reflection.append("成功分析：");
            reflection.append(String.format("任务'%s'成功完成，", experience.getTask()));
            reflection.append(String.format("关键动作'%s'在上下文%s中表现良好。", 
                            experience.getAction(), summarizeContext(experience.getContext())));
            
            // 分析成功因素
            String successFactors = analyzeSuccessFactors(experience);
            if (!successFactors.isEmpty()) {
                reflection.append(" 成功因素：").append(successFactors);
            }
            
        } else {
            reflection.append("失败分析：");
            reflection.append(String.format("任务'%s'执行失败，", experience.getTask()));
            reflection.append(String.format("动作'%s'在当前上下文中不够有效。", experience.getAction()));
            
            // 分析失败原因
            String failureReasons = analyzeFailureReasons(experience);
            if (!failureReasons.isEmpty()) {
                reflection.append(" 可能原因：").append(failureReasons);
            }
            
            // 改进建议
            String improvements = suggestImprovements(experience);
            if (!improvements.isEmpty()) {
                reflection.append(" 改进建议：").append(improvements);
            }
        }
        
        return reflection.toString();
    }
    
    /**
     * 总结上下文信息
     * 
     * @param context 上下文映射
     * @return 上下文摘要
     */
    private String summarizeContext(Map<String, Object> context) {
        if (context.isEmpty()) {
            return "空上下文";
        }
        
        StringBuilder summary = new StringBuilder();
        List<String> keyItems = context.entrySet().stream()
            .limit(3)  // 只显示前3个关键项
            .map(entry -> entry.getKey() + "=" + entry.getValue())
            .collect(Collectors.toList());
            
        summary.append("{").append(String.join(", ", keyItems));
        if (context.size() > 3) {
            summary.append(", ...");
        }
        summary.append("}");
        
        return summary.toString();
    }
    
    /**
     * 分析成功因素
     * 
     * @param experience 成功的经验
     * @return 成功因素描述
     */
    private String analyzeSuccessFactors(Experience experience) {
        List<String> factors = new ArrayList<>();
        
        // 分析上下文因素
        Map<String, Object> context = experience.getContext();
        if (context.containsKey("difficulty") && "beginner".equals(context.get("difficulty"))) {
            factors.add("任务难度适中");
        }
        if (context.containsKey("confidence")) {
            double confidence = ((Number) context.get("confidence")).doubleValue();
            if (confidence > 0.8) {
                factors.add("高置信度条件");
            }
        }
        
        // 分析动作类型
        String action = experience.getAction();
        if ("search".equals(action)) {
            factors.add("有效的信息搜索");
        } else if ("plan".equals(action)) {
            factors.add("良好的规划策略");
        } else if ("analyze".equals(action)) {
            factors.add("深入的分析过程");
        }
        
        // 分析奖励值
        if (experience.getReward() > 0.8) {
            factors.add("高质量的执行结果");
        }
        
        return String.join("、", factors);
    }
    
    /**
     * 分析失败原因
     * 
     * @param experience 失败的经验
     * @return 失败原因描述
     */
    private String analyzeFailureReasons(Experience experience) {
        List<String> reasons = new ArrayList<>();
        
        // 分析上下文因素
        Map<String, Object> context = experience.getContext();
        if (context.containsKey("difficulty") && "hard".equals(context.get("difficulty"))) {
            reasons.add("任务难度过高");
        }
        
        // 分析结果类型
        Object result = experience.getResult();
        if (result instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> resultMap = (Map<String, Object>) result;
            if (resultMap.containsKey("error")) {
                reasons.add("执行过程中出现错误");
            }
            if (resultMap.containsKey("confidence")) {
                double confidence = ((Number) resultMap.get("confidence")).doubleValue();
                if (confidence < 0.5) {
                    reasons.add("置信度过低");
                }
            }
        }
        
        // 分析奖励值
        if (experience.getReward() < -0.5) {
            reasons.add("执行结果质量差");
        }
        
        return String.join("、", reasons);
    }
    
    /**
     * 提出改进建议
     * 
     * @param experience 失败的经验
     * @return 改进建议
     */
    private String suggestImprovements(Experience experience) {
        List<String> suggestions = new ArrayList<>();
        
        String action = experience.getAction();
        String task = experience.getTask();
        
        // 基于动作类型的建议
        if ("search".equals(action)) {
            suggestions.add("尝试更精确的搜索关键词");
        } else if ("calculate".equals(action)) {
            suggestions.add("检查计算表达式的正确性");
        } else if ("analyze".equals(action)) {
            suggestions.add("增加分析的深度和广度");
        } else if ("plan".equals(action)) {
            suggestions.add("制定更详细的执行计划");
        }
        
        // 基于任务类型的建议
        if (task.contains("搜索")) {
            suggestions.add("考虑使用组合搜索策略");
        } else if (task.contains("计算")) {
            suggestions.add("验证输入数据的准确性");
        } else if (task.contains("分析")) {
            suggestions.add("收集更多相关数据");
        }
        
        // 通用建议
        if (experience.getReward() < 0) {
            suggestions.add("寻找替代方案");
        }
        
        return String.join("、", suggestions);
    }
    
    /**
     * 识别经验模式
     * 
     * @param recentExperiences 最近的经验列表
     * @return 识别到的模式列表
     */
    public List<Pattern> identifyPatterns(List<Experience> recentExperiences) {
        List<Pattern> patterns = new ArrayList<>();
        
        // 识别成功模式
        patterns.addAll(identifySuccessPatterns(recentExperiences));
        
        // 识别失败模式
        patterns.addAll(identifyFailurePatterns(recentExperiences));
        
        // 识别时间模式
        patterns.addAll(identifyTemporalPatterns(recentExperiences));
        
        // 识别上下文模式
        patterns.addAll(identifyContextPatterns(recentExperiences));
        
        // 缓存模式结果
        String cacheKey = "recent_" + recentExperiences.size();
        patternCache.put(cacheKey, patterns);
        
        return patterns;
    }
    
    /**
     * 识别成功模式
     * 
     * @param experiences 经验列表
     * @return 成功模式列表
     */
    private List<Pattern> identifySuccessPatterns(List<Experience> experiences) {
        List<Pattern> patterns = new ArrayList<>();
        
        // 统计成功的动作
        Map<String, Integer> successfulActions = new HashMap<>();
        List<Experience> successfulExperiences = experiences.stream()
            .filter(Experience::isSuccess)
            .collect(Collectors.toList());
            
        for (Experience exp : successfulExperiences) {
            successfulActions.merge(exp.getAction(), 1, Integer::sum);
        }
        
        // 识别高成功率动作模式
        for (Map.Entry<String, Integer> entry : successfulActions.entrySet()) {
            String action = entry.getKey();
            int count = entry.getValue();
            
            if (count >= 3) {  // 至少成功3次
                double successRate = calculateActionSuccessRate(action, experiences);
                if (successRate > 0.7) {  // 成功率超过70%
                    Pattern pattern = new Pattern("success_action", 
                        String.format("高成功率动作模式：'%s'（成功率%.1f%%，成功%d次）", 
                                    action, successRate * 100, count), 
                        successRate);
                    pattern.addEvidence(String.format("动作'%s'在%d次执行中成功%d次", action, 
                                       countActionUsage(action, experiences), count));
                    patterns.add(pattern);
                }
            }
        }
        
        return patterns;
    }
    
    /**
     * 识别失败模式
     * 
     * @param experiences 经验列表
     * @return 失败模式列表
     */
    private List<Pattern> identifyFailurePatterns(List<Experience> experiences) {
        List<Pattern> patterns = new ArrayList<>();
        
        // 统计失败的任务-动作组合
        Map<String, Integer> failureCombo = new HashMap<>();
        List<Experience> failedExperiences = experiences.stream()
            .filter(exp -> !exp.isSuccess())
            .collect(Collectors.toList());
            
        for (Experience exp : failedExperiences) {
            String key = exp.getTask() + ":" + exp.getAction();
            failureCombo.merge(key, 1, Integer::sum);
        }
        
        // 识别常见失败模式
        for (Map.Entry<String, Integer> entry : failureCombo.entrySet()) {
            String combo = entry.getKey();
            int count = entry.getValue();
            
            if (count >= 2) {  // 至少失败2次
                Pattern pattern = new Pattern("failure_combo", 
                    String.format("常见失败模式：%s（失败%d次）", combo, count), 
                    (double) count / experiences.size());
                pattern.addEvidence(String.format("组合'%s'重复失败%d次", combo, count));
                patterns.add(pattern);
            }
        }
        
        return patterns;
    }
    
    /**
     * 识别时间模式
     * 
     * @param experiences 经验列表
     * @return 时间模式列表
     */
    private List<Pattern> identifyTemporalPatterns(List<Experience> experiences) {
        List<Pattern> patterns = new ArrayList<>();
        
        if (experiences.size() < 5) {
            return patterns;  // 数据不足
        }
        
        // 分析性能趋势
        List<Double> recentRewards = experiences.stream()
            .map(Experience::getReward)
            .collect(Collectors.toList());
            
        if (recentRewards.size() >= 5) {
            double earlyAvg = recentRewards.subList(0, recentRewards.size() / 2).stream()
                .mapToDouble(Double::doubleValue).average().orElse(0.0);
            double lateAvg = recentRewards.subList(recentRewards.size() / 2, recentRewards.size()).stream()
                .mapToDouble(Double::doubleValue).average().orElse(0.0);
                
            if (lateAvg > earlyAvg + 0.2) {
                Pattern pattern = new Pattern("improvement_trend", 
                    String.format("性能改善趋势：平均奖励从%.2f提升到%.2f", earlyAvg, lateAvg), 
                    lateAvg - earlyAvg);
                pattern.addEvidence(String.format("最近表现比早期表现提升%.2f", lateAvg - earlyAvg));
                patterns.add(pattern);
            } else if (earlyAvg > lateAvg + 0.2) {
                Pattern pattern = new Pattern("decline_trend", 
                    String.format("性能下降趋势：平均奖励从%.2f下降到%.2f", earlyAvg, lateAvg), 
                    earlyAvg - lateAvg);
                pattern.addEvidence(String.format("最近表现比早期表现下降%.2f", earlyAvg - lateAvg));
                patterns.add(pattern);
            }
        }
        
        return patterns;
    }
    
    /**
     * 识别上下文模式
     * 
     * @param experiences 经验列表
     * @return 上下文模式列表
     */
    private List<Pattern> identifyContextPatterns(List<Experience> experiences) {
        List<Pattern> patterns = new ArrayList<>();
        
        // 分析上下文对成功率的影响
        Map<String, List<Boolean>> contextSuccessMap = new HashMap<>();
        
        for (Experience exp : experiences) {
            for (Map.Entry<String, Object> contextEntry : exp.getContext().entrySet()) {
                String contextKey = contextEntry.getKey() + "=" + contextEntry.getValue();
                contextSuccessMap.computeIfAbsent(contextKey, k -> new ArrayList<>())
                    .add(exp.isSuccess());
            }
        }
        
        // 识别影响显著的上下文因素
        for (Map.Entry<String, List<Boolean>> entry : contextSuccessMap.entrySet()) {
            String contextKey = entry.getKey();
            List<Boolean> results = entry.getValue();
            
            if (results.size() >= 3) {  // 至少出现3次
                double successRate = results.stream()
                    .mapToDouble(success -> success ? 1.0 : 0.0)
                    .average().orElse(0.0);
                    
                if (successRate > 0.8 || successRate < 0.2) {  // 高成功率或高失败率
                    String description = successRate > 0.8 ? 
                        String.format("有利上下文：%s（成功率%.1f%%）", contextKey, successRate * 100) :
                        String.format("不利上下文：%s（成功率%.1f%%）", contextKey, successRate * 100);
                        
                    Pattern pattern = new Pattern("context_influence", description, 
                        Math.abs(successRate - 0.5) * 2);  // 强度基于偏离中性的程度
                    pattern.addEvidence(String.format("在%d次出现中，成功%d次", 
                                       results.size(), (int) (results.size() * successRate)));
                    patterns.add(pattern);
                }
            }
        }
        
        return patterns;
    }
    
    /**
     * 计算动作的成功率
     * 
     * @param action 动作名称
     * @param experiences 经验列表
     * @return 成功率
     */
    private double calculateActionSuccessRate(String action, List<Experience> experiences) {
        List<Experience> actionExperiences = experiences.stream()
            .filter(exp -> action.equals(exp.getAction()))
            .collect(Collectors.toList());
            
        if (actionExperiences.isEmpty()) {
            return 0.0;
        }
        
        long successCount = actionExperiences.stream()
            .mapToLong(exp -> exp.isSuccess() ? 1 : 0)
            .sum();
            
        return (double) successCount / actionExperiences.size();
    }
    
    /**
     * 统计动作使用次数
     * 
     * @param action 动作名称
     * @param experiences 经验列表
     * @return 使用次数
     */
    private int countActionUsage(String action, List<Experience> experiences) {
        return (int) experiences.stream()
            .filter(exp -> action.equals(exp.getAction()))
            .count();
    }
    
    /**
     * 获取反思摘要
     * 
     * @param limit 返回记录数量限制
     * @return 反思摘要列表
     */
    public List<ReflectionRecord> getReflectionSummary(int limit) {
        return reflectionHistory.stream()
            .sorted((r1, r2) -> Long.compare(r2.getTimestamp(), r1.getTimestamp()))
            .limit(limit)
            .collect(Collectors.toList());
    }
    
    /**
     * 清理旧的反思记录
     * 
     * @param ageThreshold 年龄阈值（毫秒）
     */
    public void cleanupOldReflections(long ageThreshold) {
        long currentTime = System.currentTimeMillis();
        reflectionHistory.removeIf(record -> 
            (currentTime - record.getTimestamp()) > ageThreshold);
    }
    
    // Getters
    
    public List<ReflectionRecord> getReflectionHistory() {
        return new ArrayList<>(reflectionHistory);
    }
    
    public int getReflectionCount() {
        return reflectionHistory.size();
    }
}