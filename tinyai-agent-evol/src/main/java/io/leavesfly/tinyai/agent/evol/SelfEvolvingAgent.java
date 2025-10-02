package io.leavesfly.tinyai.agent.evol;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.logging.Logger;
import java.io.*;

/**
 * 自进化自学习Agent
 * 具有经验记忆、策略优化、反思改进、知识图谱构建等自进化能力
 * 
 * @author 山泽
 */
public class SelfEvolvingAgent {
    
    private static final Logger logger = Logger.getLogger(SelfEvolvingAgent.class.getName());
    
    /**
     * 感知结果类
     */
    public static class PerceptionResult {
        private Map<String, Object> currentContext;
        private List<Experience> relevantExperiences;
        private List<Strategy> applicableStrategies;
        private double uncertaintyLevel;
        
        public PerceptionResult(Map<String, Object> currentContext, List<Experience> relevantExperiences,
                              List<Strategy> applicableStrategies, double uncertaintyLevel) {
            this.currentContext = currentContext;
            this.relevantExperiences = relevantExperiences;
            this.applicableStrategies = applicableStrategies;
            this.uncertaintyLevel = uncertaintyLevel;
        }
        
        // Getters
        public Map<String, Object> getCurrentContext() { return currentContext; }
        public List<Experience> getRelevantExperiences() { return relevantExperiences; }
        public List<Strategy> getApplicableStrategies() { return applicableStrategies; }
        public double getUncertaintyLevel() { return uncertaintyLevel; }
    }
    
    /**
     * 性能记录类
     */
    public static class PerformanceRecord {
        private long timestamp;
        private double successRate;
        private int totalTasks;
        
        public PerformanceRecord(long timestamp, double successRate, int totalTasks) {
            this.timestamp = timestamp;
            this.successRate = successRate;
            this.totalTasks = totalTasks;
        }
        
        // Getters
        public long getTimestamp() { return timestamp; }
        public double getSuccessRate() { return successRate; }
        public int getTotalTasks() { return totalTasks; }
    }
    
    /**
     * 任务处理结果类
     */
    public static class TaskResult {
        private String task;
        private String action;
        private Object result;
        private boolean success;
        private double reward;
        private String learningInsights;
        
        public TaskResult(String task, String action, Object result, boolean success, double reward, String learningInsights) {
            this.task = task;
            this.action = action;
            this.result = result;
            this.success = success;
            this.reward = reward;
            this.learningInsights = learningInsights;
        }
        
        // Getters
        public String getTask() { return task; }
        public String getAction() { return action; }
        public Object getResult() { return result; }
        public boolean isSuccess() { return success; }
        public double getReward() { return reward; }
        public String getLearningInsights() { return learningInsights; }
        
        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("task", task);
            map.put("action", action);
            map.put("result", result);
            map.put("success", success);
            map.put("reward", reward);
            map.put("learningInsights", learningInsights);
            return map;
        }
    }
    
    // 核心组件
    private final String name;
    private final List<Experience> experiences;
    private final Map<String, Strategy> strategies;
    private final KnowledgeGraph knowledgeGraph;
    private final ReflectionModule reflectionModule;
    
    // 学习参数
    private double learningRate;
    private double explorationRate;
    private int memorySize;
    
    // 性能指标
    private final List<PerformanceRecord> performanceHistory;
    private int totalTasks;
    private int successfulTasks;
    
    // 工具库
    private final Map<String, Function<Map<String, Object>, Map<String, Object>>> availableTools;
    
    public SelfEvolvingAgent(String name) {
        this.name = name;
        this.experiences = Collections.synchronizedList(new ArrayList<>());
        this.strategies = new ConcurrentHashMap<>();
        this.knowledgeGraph = new KnowledgeGraph();
        this.reflectionModule = new ReflectionModule();
        
        // 初始化学习参数
        this.learningRate = 0.1;
        this.explorationRate = 0.2;
        this.memorySize = 1000;
        
        // 初始化性能指标
        this.performanceHistory = Collections.synchronizedList(new ArrayList<>());
        this.totalTasks = 0;
        this.successfulTasks = 0;
        
        // 初始化工具库
        this.availableTools = new ConcurrentHashMap<>();
        initializeBasicTools();
        
        // 初始化基础策略
        initializeBaseStrategies();
        
        logger.info("自进化Agent '" + name + "' 初始化完成");
    }
    
    /**
     * 初始化基础工具
     */
    private void initializeBasicTools() {
        // 搜索工具
        availableTools.put("search", context -> {
            String query = (String) context.getOrDefault("query", "默认查询");
            Map<String, Object> result = new HashMap<>();
            result.put("results", Arrays.asList(
                "搜索结果1 for " + query,
                "搜索结果2 for " + query,
                "搜索结果3 for " + query
            ));
            result.put("confidence", 0.5 + Math.random() * 0.5);
            return result;
        });
        
        // 计算工具
        availableTools.put("calculate", context -> {
            String expression = (String) context.getOrDefault("expression", "1+1");
            Map<String, Object> result = new HashMap<>();
            try {
                double value = evaluateSimpleExpression(expression);
                result.put("result", value);
                result.put("success", true);
            } catch (Exception e) {
                result.put("result", null);
                result.put("success", false);
                result.put("error", e.getMessage());
            }
            return result;
        });
        
        // 分析工具
        availableTools.put("analyze", context -> {
            Object data = context.getOrDefault("data", context);
            Map<String, Object> result = new HashMap<>();
            result.put("analysis", "分析结果：数据包含" + data.toString().length() + "个字符");
            result.put("insights", Arrays.asList("洞察1：数据特征分析", "洞察2：趋势识别"));
            result.put("confidence", 0.6 + Math.random() * 0.3);
            return result;
        });
        
        // 规划工具
        availableTools.put("plan", context -> {
            String goal = (String) context.getOrDefault("goal", "默认目标");
            Map<String, Object> result = new HashMap<>();
            List<String> steps = new ArrayList<>();
            for (int i = 1; i <= 3; i++) {
                steps.add("步骤" + i + ": 处理" + goal + "的第" + i + "部分");
            }
            result.put("plan", steps);
            result.put("estimated_effort", 1 + (int)(Math.random() * 10));
            result.put("success_probability", 0.7 + Math.random() * 0.25);
            return result;
        });
    }
    
    /**
     * 简单的数学表达式计算
     */
    private double evaluateSimpleExpression(String expression) {
        expression = expression.replaceAll("\\s+", "");
        
        if (expression.equals("1+1")) return 2.0;
        if (expression.equals("2*3")) return 6.0;
        if (expression.equals("10/2")) return 5.0;
        if (expression.contains("*")) {
            String[] parts = expression.split("\\*");
            if (parts.length == 2) {
                return Double.parseDouble(parts[0]) * Double.parseDouble(parts[1]);
            }
        }
        if (expression.contains("+")) {
            String[] parts = expression.split("\\+");
            if (parts.length == 2) {
                return Double.parseDouble(parts[0]) + Double.parseDouble(parts[1]);
            }
        }
        
        return Math.random() * 100;
    }
    
    /**
     * 初始化基础策略
     */
    private void initializeBaseStrategies() {
        Map<String, Object> exploreConditions = new HashMap<>();
        exploreConditions.put("uncertainty", "high");
        Strategy exploreStrategy = new Strategy(
            "探索策略",
            "在不确定情况下进行探索",
            exploreConditions,
            Arrays.asList("search", "analyze"),
            0.5,
            0
        );
        strategies.put(exploreStrategy.getName(), exploreStrategy);
        
        Map<String, Object> exploitConditions = new HashMap<>();
        exploitConditions.put("confidence", "high");
        Strategy exploitStrategy = new Strategy(
            "利用策略",
            "使用已知有效的方法",
            exploitConditions,
            Arrays.asList("plan", "calculate"),
            0.8,
            0
        );
        strategies.put(exploitStrategy.getName(), exploitStrategy);
    }
    
    /**
     * 环境感知
     */
    public PerceptionResult perceiveEnvironment(Map<String, Object> context) {
        List<Experience> relevantExperiences = findRelevantExperiences(context);
        List<Strategy> applicableStrategies = findApplicableStrategies(context);
        double uncertaintyLevel = assessUncertainty(context);
        
        return new PerceptionResult(context, relevantExperiences, applicableStrategies, uncertaintyLevel);
    }
    
    /**
     * 查找相关经验
     */
    private List<Experience> findRelevantExperiences(Map<String, Object> context) {
        List<Experience> relevantExperiences = new ArrayList<>();
        
        int startIndex = Math.max(0, experiences.size() - 50);
        for (int i = startIndex; i < experiences.size(); i++) {
            Experience exp = experiences.get(i);
            double similarity = calculateContextSimilarity(exp.getContext(), context);
            if (similarity > 0.5) {
                relevantExperiences.add(exp);
            }
        }
        
        return relevantExperiences.stream()
                .sorted((e1, e2) -> Double.compare(e2.getReward(), e1.getReward()))
                .limit(10)
                .collect(Collectors.toList());
    }
    
    /**
     * 计算上下文相似度
     */
    private double calculateContextSimilarity(Map<String, Object> context1, Map<String, Object> context2) {
        Set<String> commonKeys = new HashSet<>(context1.keySet());
        commonKeys.retainAll(context2.keySet());
        
        if (commonKeys.isEmpty()) {
            return 0.0;
        }
        
        List<Double> similarityScores = new ArrayList<>();
        for (String key : commonKeys) {
            Object value1 = context1.get(key);
            Object value2 = context2.get(key);
            
            if (value1 instanceof String && value2 instanceof String) {
                double similarity = calculateStringSimilarity((String) value1, (String) value2);
                similarityScores.add(similarity);
            } else if (Objects.equals(value1, value2)) {
                similarityScores.add(1.0);
            } else {
                similarityScores.add(0.0);
            }
        }
        
        return similarityScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * 计算字符串相似度
     */
    private double calculateStringSimilarity(String str1, String str2) {
        Set<String> words1 = new HashSet<>(Arrays.asList(str1.split("\\s+")));
        Set<String> words2 = new HashSet<>(Arrays.asList(str2.split("\\s+")));
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
    /**
     * 查找适用策略
     */
    private List<Strategy> findApplicableStrategies(Map<String, Object> context) {
        return strategies.values().stream()
                .filter(strategy -> strategy.matchesContext(context))
                .sorted((s1, s2) -> Double.compare(s2.getSuccessRate(), s1.getSuccessRate()))
                .collect(Collectors.toList());
    }
    
    /**
     * 评估不确定性
     */
    private double assessUncertainty(Map<String, Object> context) {
        List<Double> uncertaintyFactors = new ArrayList<>();
        
        String task = (String) context.get("task");
        if (task != null) {
            double taskComplexity = Math.min(task.split("\\s+").length / 10.0, 1.0);
            uncertaintyFactors.add(taskComplexity);
        }
        
        List<Experience> relevantExperiences = findRelevantExperiences(context);
        double experienceFactor = Math.max(0.0, 1.0 - relevantExperiences.size() / 10.0);
        uncertaintyFactors.add(experienceFactor);
        
        return uncertaintyFactors.stream().mapToDouble(Double::doubleValue).average().orElse(0.5);
    }
    
    /**
     * 决策行动
     */
    public String decideAction(PerceptionResult perception) {
        List<Strategy> applicableStrategies = perception.getApplicableStrategies();
        List<Experience> relevantExperiences = perception.getRelevantExperiences();
        
        if (!applicableStrategies.isEmpty() && Math.random() > explorationRate) {
            Strategy bestStrategy = applicableStrategies.get(0);
            List<String> actions = bestStrategy.getActions();
            return actions.get((int) (Math.random() * actions.size()));
        } else {
            if (!relevantExperiences.isEmpty()) {
                List<String> successfulActions = relevantExperiences.stream()
                        .filter(Experience::isSuccess)
                        .map(Experience::getAction)
                        .collect(Collectors.toList());
                        
                if (!successfulActions.isEmpty()) {
                    return successfulActions.get((int) (Math.random() * successfulActions.size()));
                }
            }
            
            List<String> availableActions = new ArrayList<>(availableTools.keySet());
            return availableActions.get((int) (Math.random() * availableActions.size()));
        }
    }
    
    /**
     * 执行动作
     */
    public Object executeAction(String action, Map<String, Object> context) {
        Function<Map<String, Object>, Map<String, Object>> tool = availableTools.get(action);
        if (tool != null) {
            return tool.apply(context);
        } else {
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", "未知动作: " + action);
            return errorResult;
        }
    }
    
    /**
     * 评估结果
     */
    public double[] evaluateResult(Object result, Object expectedOutcome) {
        boolean success = false;
        double reward = 0.0;
        
        if (result instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> resultMap = (Map<String, Object>) result;
            
            if (resultMap.containsKey("error")) {
                success = false;
                reward = -1.0;
            } else if (resultMap.containsKey("success")) {
                success = (Boolean) resultMap.get("success");
                reward = success ? 1.0 : -0.5;
            } else if (resultMap.containsKey("confidence")) {
                double confidence = ((Number) resultMap.get("confidence")).doubleValue();
                success = confidence > 0.7;
                reward = success ? confidence : -0.3;
            } else {
                success = true;
                reward = 0.5;
            }
        } else {
            success = true;
            reward = 0.3;
        }
        
        return new double[]{success ? 1.0 : 0.0, reward};
    }
    
    /**
     * 从经验中学习
     */
    public void learnFromExperience(Experience experience) {
        experiences.add(experience);
        
        if (experiences.size() > memorySize) {
            experiences.remove(0);
        }
        
        updateKnowledgeGraph(experience);
        
        String reflection = reflectionModule.reflectOnExperience(experience);
        experience.setReflection(reflection);
        
        updateStrategies(experience);
        
        totalTasks++;
        if (experience.isSuccess()) {
            successfulTasks++;
        }
        
        double currentSuccessRate = (double) successfulTasks / totalTasks;
        performanceHistory.add(new PerformanceRecord(
            System.currentTimeMillis(), currentSuccessRate, totalTasks));
        
        adjustLearningParameters();
        
        logger.info(String.format("学习完成 - 任务: %s, 成功: %s, 奖励: %.2f", 
                  experience.getTask(), experience.isSuccess(), experience.getReward()));
    }
    
    /**
     * 更新知识图谱
     */
    private void updateKnowledgeGraph(Experience experience) {
        String taskConcept = "task:" + experience.getTask();
        String actionConcept = "action:" + experience.getAction();
        
        Map<String, Object> taskProperties = new HashMap<>();
        taskProperties.put("type", "task");
        taskProperties.put("description", experience.getTask());
        knowledgeGraph.addConcept(taskConcept, taskProperties);
        
        Map<String, Object> actionProperties = new HashMap<>();
        actionProperties.put("type", "action");
        actionProperties.put("description", experience.getAction());
        knowledgeGraph.addConcept(actionConcept, actionProperties);
        
        String relationType = experience.isSuccess() ? "succeeds_with" : "fails_with";
        double weight = experience.getReward();
        
        knowledgeGraph.addRelation(taskConcept, actionConcept, relationType, weight);
    }
    
    /**
     * 更新策略
     */
    private void updateStrategies(Experience experience) {
        String taskType = experience.getTask().contains(":") ? 
                         experience.getTask().split(":")[0] : experience.getTask();
        String strategyName = "策略_" + taskType + "_" + experience.getAction();
        
        Strategy strategy = strategies.get(strategyName);
        if (strategy != null) {
            strategy.updateSuccessRate(experience.isSuccess(), learningRate);
        } else {
            double initialSuccessRate = experience.isSuccess() ? 1.0 : 0.0;
            Map<String, Object> conditions = new HashMap<>();
            conditions.put("task_type", taskType);
            
            Strategy newStrategy = new Strategy(
                strategyName,
                String.format("针对%s任务的%s策略", taskType, experience.getAction()),
                conditions,
                Arrays.asList(experience.getAction()),
                initialSuccessRate,
                1
            );
            strategies.put(strategyName, newStrategy);
        }
    }
    
    /**
     * 自适应调整学习参数
     */
    private void adjustLearningParameters() {
        if (performanceHistory.size() >= 10) {
            List<PerformanceRecord> recent = performanceHistory.subList(
                performanceHistory.size() - 10, performanceHistory.size());
            double avgSuccessRate = recent.stream()
                    .mapToDouble(PerformanceRecord::getSuccessRate)
                    .average().orElse(0.0);
            
            if (avgSuccessRate < 0.6) {
                explorationRate = Math.min(0.5, explorationRate + 0.05);
            } else if (avgSuccessRate > 0.8) {
                explorationRate = Math.max(0.1, explorationRate - 0.02);
            }
        }
    }
    
    /**
     * 处理任务的主要接口
     */
    public TaskResult processTask(String task, Map<String, Object> context) {
        if (context == null) {
            context = new HashMap<>();
        }
        context.put("task", task);
        
        logger.info("处理任务: " + task);
        
        PerceptionResult perception = perceiveEnvironment(context);
        String action = decideAction(perception);
        Object result = executeAction(action, context);
        double[] evaluation = evaluateResult(result, null);
        
        boolean success = evaluation[0] == 1.0;
        double reward = evaluation[1];
        
        Experience experience = new Experience(task, context, action, result, success, reward);
        learnFromExperience(experience);
        
        if (totalTasks % 50 == 0) {
            selfEvolve();
        }
        
        return new TaskResult(task, action, result, success, reward, experience.getReflection());
    }
    
    /**
     * 自我进化过程
     */
    public void selfEvolve() {
        logger.info("开始自我进化过程...");
        
        List<Experience> recentExperiences = experiences.stream()
                .skip(Math.max(0, experiences.size() - 100))
                .collect(Collectors.toList());
        List<ReflectionModule.Pattern> patterns = reflectionModule.identifyPatterns(recentExperiences);
        
        logger.info("识别到的模式数量: " + patterns.size());
        
        optimizeStrategies();
        integrateKnowledge();
        expandCapabilities();
        
        logger.info("自我进化完成");
    }
    
    /**
     * 策略优化
     */
    private void optimizeStrategies() {
        List<String> strategiesToRemove = strategies.entrySet().stream()
                .filter(entry -> {
                    Strategy strategy = entry.getValue();
                    return strategy.getUsageCount() > 10 && strategy.getSuccessRate() < 0.3;
                })
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        
        strategiesToRemove.forEach(name -> {
            strategies.remove(name);
            logger.info("移除低效策略: " + name);
        });
    }
    
    /**
     * 知识整合
     */
    private void integrateKnowledge() {
        List<String> concepts = new ArrayList<>(knowledgeGraph.getNodes().keySet());
        
        for (int i = 0; i < concepts.size(); i++) {
            String concept1 = concepts.get(i);
            for (int j = i + 1; j < concepts.size(); j++) {
                String concept2 = concepts.get(j);
                double similarity = knowledgeGraph.getConceptSimilarity(concept1, concept2);
                
                if (similarity > 0.8) {
                    knowledgeGraph.addRelation(concept1, concept2, "similar_to", similarity);
                }
            }
        }
    }
    
    /**
     * 能力扩展
     */
    private void expandCapabilities() {
        List<Experience> successfulExperiences = experiences.stream()
                .skip(Math.max(0, experiences.size() - 50))
                .filter(Experience::isSuccess)
                .collect(Collectors.toList());
        
        Map<String, Integer> actionSequences = new HashMap<>();
        for (int i = 0; i < successfulExperiences.size() - 1; i++) {
            Experience current = successfulExperiences.get(i);
            Experience next = successfulExperiences.get(i + 1);
            
            if (current.getTimestamp() < next.getTimestamp()) {
                String sequence = current.getAction() + "->" + next.getAction();
                actionSequences.merge(sequence, 1, Integer::sum);
            }
        }
        
        actionSequences.entrySet().stream()
                .filter(entry -> entry.getValue() >= 3)
                .forEach(entry -> {
                    String[] actions = entry.getKey().split("->");
                    String comboName = "combo_" + actions[0] + "_" + actions[1];
                    
                    if (!availableTools.containsKey(comboName)) {
                        availableTools.put(comboName, createComboTool(actions[0], actions[1]));
                        logger.info("发现新的工具组合: " + comboName);
                    }
                });
    }
    
    /**
     * 创建组合工具
     */
    private Function<Map<String, Object>, Map<String, Object>> createComboTool(String action1, String action2) {
        return context -> {
            Map<String, Object> result1 = availableTools.get(action1).apply(context);
            Map<String, Object> enhancedContext = new HashMap<>(context);
            enhancedContext.put("previous_result", result1);
            Map<String, Object> result2 = availableTools.get(action2).apply(enhancedContext);
            
            Map<String, Object> comboResult = new HashMap<>();
            comboResult.put("sequence_results", Arrays.asList(result1, result2));
            comboResult.put("final_result", result2);
            comboResult.put("combo_success", true);
            return comboResult;
        };
    }
    
    /**
     * 获取性能摘要
     */
    public Map<String, Object> getPerformanceSummary() {
        if (performanceHistory.isEmpty()) {
            Map<String, Object> summary = new HashMap<>();
            summary.put("message", "暂无性能数据");
            return summary;
        }
        
        PerformanceRecord latest = performanceHistory.get(performanceHistory.size() - 1);
        
        String trend = "insufficient_data";
        if (performanceHistory.size() >= 10) {
            double recentAvg = performanceHistory.stream()
                    .skip(performanceHistory.size() - 10)
                    .mapToDouble(PerformanceRecord::getSuccessRate)
                    .average().orElse(0.0);
            double overallAvg = performanceHistory.stream()
                    .mapToDouble(PerformanceRecord::getSuccessRate)
                    .average().orElse(0.0);
            trend = recentAvg > overallAvg ? "improving" : "declining";
        }
        
        Map<String, Object> summary = new HashMap<>();
        summary.put("total_tasks", totalTasks);
        summary.put("successful_tasks", successfulTasks);
        summary.put("current_success_rate", latest.getSuccessRate());
        summary.put("trend", trend);
        summary.put("strategies_count", strategies.size());
        summary.put("experiences_count", experiences.size());
        summary.put("exploration_rate", explorationRate);
        summary.put("knowledge_concepts", knowledgeGraph.getNodes().size());
        
        return summary;
    }
    
    // Getters
    public String getName() { return name; }
    public int getTotalTasks() { return totalTasks; }
    public int getSuccessfulTasks() { return successfulTasks; }
    public double getExplorationRate() { return explorationRate; }
    public List<Experience> getExperiences() { return new ArrayList<>(experiences); }
    public Map<String, Strategy> getStrategies() { return new HashMap<>(strategies); }
    public KnowledgeGraph getKnowledgeGraph() { return knowledgeGraph; }
    public ReflectionModule getReflectionModule() { return reflectionModule; }
}