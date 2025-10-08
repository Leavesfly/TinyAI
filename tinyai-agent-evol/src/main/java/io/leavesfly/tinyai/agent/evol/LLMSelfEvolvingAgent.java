package io.leavesfly.tinyai.agent.evol;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.logging.Logger;

/**
 * 基于LLM的自进化智能体
 * 结合大语言模型的推理能力与自主学习机制，实现更智能的自进化行为
 * 
 * @author 山泽
 */
public class LLMSelfEvolvingAgent {
    
    private static final Logger logger = Logger.getLogger(LLMSelfEvolvingAgent.class.getName());
    
    /**
     * LLM增强的感知结果类
     */
    public static class EnhancedPerceptionResult {
        private Map<String, Object> currentContext;
        private List<Experience> relevantExperiences;
        private List<Strategy> applicableStrategies;
        private double uncertaintyLevel;
        private String llmAnalysis;              // LLM的智能分析
        private String strategicInsights;        // 策略洞察
        private double confidenceScore;          // 置信度评分
        
        public EnhancedPerceptionResult(Map<String, Object> currentContext, List<Experience> relevantExperiences,
                                      List<Strategy> applicableStrategies, double uncertaintyLevel, 
                                      String llmAnalysis, String strategicInsights, double confidenceScore) {
            this.currentContext = currentContext;
            this.relevantExperiences = relevantExperiences;
            this.applicableStrategies = applicableStrategies;
            this.uncertaintyLevel = uncertaintyLevel;
            this.llmAnalysis = llmAnalysis;
            this.strategicInsights = strategicInsights;
            this.confidenceScore = confidenceScore;
        }
        
        // Getters
        public Map<String, Object> getCurrentContext() { return currentContext; }
        public List<Experience> getRelevantExperiences() { return relevantExperiences; }
        public List<Strategy> getApplicableStrategies() { return applicableStrategies; }
        public double getUncertaintyLevel() { return uncertaintyLevel; }
        public String getLlmAnalysis() { return llmAnalysis; }
        public String getStrategicInsights() { return strategicInsights; }
        public double getConfidenceScore() { return confidenceScore; }
    }
    
    /**
     * LLM增强的任务结果类
     */
    public static class EnhancedTaskResult {
        private String task;
        private String action;
        private Object result;
        private boolean success;
        private double reward;
        private String learningInsights;
        private String llmReflection;           // LLM深度反思
        private String improvementAdvice;       // 改进建议
        private List<String> discoveredPatterns; // 发现的模式
        
        public EnhancedTaskResult(String task, String action, Object result, boolean success, 
                                double reward, String learningInsights, String llmReflection, 
                                String improvementAdvice, List<String> discoveredPatterns) {
            this.task = task;
            this.action = action;
            this.result = result;
            this.success = success;
            this.reward = reward;
            this.learningInsights = learningInsights;
            this.llmReflection = llmReflection;
            this.improvementAdvice = improvementAdvice;
            this.discoveredPatterns = discoveredPatterns != null ? discoveredPatterns : new ArrayList<>();
        }
        
        // Getters
        public String getTask() { return task; }
        public String getAction() { return action; }
        public Object getResult() { return result; }
        public boolean isSuccess() { return success; }
        public double getReward() { return reward; }
        public String getLearningInsights() { return learningInsights; }
        public String getLlmReflection() { return llmReflection; }
        public String getImprovementAdvice() { return improvementAdvice; }
        public List<String> getDiscoveredPatterns() { return discoveredPatterns; }
        
        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("task", task);
            map.put("action", action);
            map.put("result", result);
            map.put("success", success);
            map.put("reward", reward);
            map.put("learningInsights", learningInsights);
            map.put("llmReflection", llmReflection);
            map.put("improvementAdvice", improvementAdvice);
            map.put("discoveredPatterns", discoveredPatterns);
            return map;
        }
    }
    
    // 核心组件
    private final String name;
    private final List<Experience> experiences;
    private final Map<String, Strategy> strategies;
    private final KnowledgeGraph knowledgeGraph;
    private final ReflectionModule reflectionModule;
    private final EvolLLMSimulator llmSimulator;      // LLM模拟器
    
    // 学习参数
    private double learningRate;
    private double explorationRate;
    private int memorySize;
    
    // LLM增强参数
    private double llmConfidenceThreshold;            // LLM建议采纳阈值
    private boolean enableAsyncLLM;                   // 是否启用异步LLM处理
    private int llmCacheSize;                         // LLM结果缓存大小
    
    // 性能指标
    private final List<SelfEvolvingAgent.PerformanceRecord> performanceHistory;
    private int totalTasks;
    private int successfulTasks;
    private int llmAssistedDecisions;                 // LLM辅助决策数量
    
    // 工具库
    private final Map<String, Function<Map<String, Object>, Map<String, Object>>> availableTools;
    
    // LLM增强缓存
    private final Map<String, String> llmResponseCache;
    
    public LLMSelfEvolvingAgent(String name) {
        this(name, true);
    }
    
    public LLMSelfEvolvingAgent(String name, boolean enableLLM) {
        this.name = name;
        this.experiences = Collections.synchronizedList(new ArrayList<>());
        this.strategies = new ConcurrentHashMap<>();
        this.knowledgeGraph = new KnowledgeGraph();
        this.reflectionModule = new ReflectionModule();
        this.llmSimulator = enableLLM ? new EvolLLMSimulator() : null;
        
        // 初始化学习参数
        this.learningRate = 0.1;
        this.explorationRate = 0.2;
        this.memorySize = 1000;
        
        // 初始化LLM增强参数
        this.llmConfidenceThreshold = 0.7;
        this.enableAsyncLLM = true;
        this.llmCacheSize = 100;
        
        // 初始化性能指标
        this.performanceHistory = Collections.synchronizedList(new ArrayList<>());
        this.totalTasks = 0;
        this.successfulTasks = 0;
        this.llmAssistedDecisions = 0;
        
        // 初始化工具库
        this.availableTools = new ConcurrentHashMap<>();
        this.llmResponseCache = new ConcurrentHashMap<>();
        
        initializeEnhancedTools();
        initializeBaseStrategies();
        
        logger.info("LLM增强自进化Agent '" + name + "' 初始化完成，LLM功能：" + (enableLLM ? "启用" : "禁用"));
    }
    
    /**
     * 初始化LLM增强的工具库
     */
    private void initializeEnhancedTools() {
        // 智能搜索工具
        availableTools.put("intelligent_search", context -> {
            String query = (String) context.getOrDefault("query", "默认查询");
            Map<String, Object> result = new HashMap<>();
            
            // 基础搜索结果
            List<String> basicResults = Arrays.asList(
                "搜索结果1 for " + query,
                "搜索结果2 for " + query,
                "高相关性结果 for " + query
            );
            
            // LLM增强分析
            if (llmSimulator != null) {
                String llmAnalysis = llmSimulator.generateEvolResponse(
                    "分析搜索查询：" + query, 
                    context.toString(), 
                    "strategic_reasoning"
                );
                result.put("llm_analysis", llmAnalysis);
            }
            
            result.put("results", basicResults);
            result.put("confidence", 0.6 + Math.random() * 0.3);
            result.put("enhanced", llmSimulator != null);
            return result;
        });
        
        // 智能计算工具
        availableTools.put("smart_calculate", context -> {
            String expression = (String) context.getOrDefault("expression", "1+1");
            Map<String, Object> result = new HashMap<>();
            
            try {
                double value = evaluateExpression(expression);
                result.put("result", value);
                result.put("success", true);
                
                // LLM提供计算洞察
                if (llmSimulator != null) {
                    String insight = llmSimulator.generateEvolResponse(
                        "分析计算过程：" + expression + " = " + value,
                        context.toString(),
                        "pattern_analysis"
                    );
                    result.put("calculation_insight", insight);
                }
                
            } catch (Exception e) {
                result.put("result", null);
                result.put("success", false);
                result.put("error", e.getMessage());
            }
            
            return result;
        });
        
        // 深度分析工具
        availableTools.put("deep_analyze", context -> {
            Object data = context.getOrDefault("data", context);
            Map<String, Object> result = new HashMap<>();
            
            // 基础分析
            String basicAnalysis = "数据包含" + data.toString().length() + "个字符的信息";
            result.put("basic_analysis", basicAnalysis);
            
            // LLM深度分析
            if (llmSimulator != null) {
                String deepAnalysis = llmSimulator.generateEvolResponse(
                    "深度分析数据：" + data.toString().substring(0, Math.min(100, data.toString().length())),
                    context.toString(),
                    "knowledge_inference"
                );
                result.put("deep_analysis", deepAnalysis);
                result.put("insights", Arrays.asList("LLM洞察1：数据模式识别", "LLM洞察2：关联性分析"));
            } else {
                result.put("insights", Arrays.asList("洞察1：数据特征分析", "洞察2：趋势识别"));
            }
            
            result.put("confidence", 0.7 + Math.random() * 0.25);
            return result;
        });
        
        // 智能规划工具
        availableTools.put("strategic_plan", context -> {
            String goal = (String) context.getOrDefault("goal", "默认目标");
            Map<String, Object> result = new HashMap<>();
            
            // 基础规划
            List<String> basicSteps = Arrays.asList(
                "步骤1: 分析" + goal + "的需求",
                "步骤2: 制定执行策略", 
                "步骤3: 实施并监控进展"
            );
            
            // LLM智能规划
            if (llmSimulator != null) {
                String strategicAdvice = llmSimulator.generateEvolResponse(
                    "制定实现目标的战略规划：" + goal,
                    context.toString(),
                    "decision_advice"
                );
                result.put("strategic_advice", strategicAdvice);
                
                // 增强规划步骤
                List<String> enhancedSteps = Arrays.asList(
                    "智能步骤1: 基于LLM的需求深度分析",
                    "智能步骤2: 多维度策略评估与选择",
                    "智能步骤3: 自适应执行与持续优化"
                );
                result.put("enhanced_plan", enhancedSteps);
            }
            
            result.put("plan", basicSteps);
            result.put("estimated_effort", 1 + (int)(Math.random() * 10));
            result.put("success_probability", 0.75 + Math.random() * 0.2);
            return result;
        });
    }
    
    /**
     * LLM增强的环境感知
     */
    public EnhancedPerceptionResult perceiveEnvironmentWithLLM(Map<String, Object> context) {
        // 传统感知
        List<Experience> relevantExperiences = findRelevantExperiences(context);
        List<Strategy> applicableStrategies = findApplicableStrategies(context);
        double uncertaintyLevel = assessUncertainty(context);
        
        // LLM增强分析
        String llmAnalysis = "";
        String strategicInsights = "";
        double confidenceScore = 0.5;
        
        if (llmSimulator != null) {
            try {
                // 生成LLM分析
                String contextSummary = summarizeContext(context, relevantExperiences, applicableStrategies);
                
                if (enableAsyncLLM) {
                    CompletableFuture<String> analysisTask = llmSimulator.generateStrategicReasoningAsync(
                        context, getAvailableActionNames(), relevantExperiences);
                    llmAnalysis = analysisTask.get(); // 简化处理，实际应用中可考虑超时
                } else {
                    llmAnalysis = llmSimulator.generateStrategicReasoning(
                        context, getAvailableActionNames(), relevantExperiences);
                }
                
                // 提取策略洞察
                strategicInsights = extractStrategicInsights(llmAnalysis, context);
                confidenceScore = calculateLLMConfidence(llmAnalysis, relevantExperiences.size());
                
            } catch (Exception e) {
                logger.warning("LLM分析过程中出现异常: " + e.getMessage());
                llmAnalysis = "LLM分析暂时不可用，使用传统方法进行环境感知";
                strategicInsights = "建议基于历史经验进行决策";
                confidenceScore = 0.4;
            }
        }
        
        return new EnhancedPerceptionResult(context, relevantExperiences, applicableStrategies, 
                                          uncertaintyLevel, llmAnalysis, strategicInsights, confidenceScore);
    }
    
    /**
     * LLM增强的决策制定
     */
    public String decideActionWithLLM(EnhancedPerceptionResult perception) {
        List<Strategy> applicableStrategies = perception.getApplicableStrategies();
        List<Experience> relevantExperiences = perception.getRelevantExperiences();
        
        // LLM辅助决策
        if (llmSimulator != null && perception.getConfidenceScore() > llmConfidenceThreshold) {
            try {
                // 构建决策上下文
                Map<String, Object> decisionContext = new HashMap<>(perception.getCurrentContext());
                decisionContext.put("llm_analysis", perception.getLlmAnalysis());
                decisionContext.put("strategic_insights", perception.getStrategicInsights());
                
                // 计算各动作的成功率
                Map<String, Double> actionSuccessRates = calculateActionSuccessRates(getAvailableActionNames(), relevantExperiences);
                
                // 获取LLM决策建议
                String decisionAdvice = llmSimulator.generateDecisionAdvice(
                    (String) perception.getCurrentContext().get("task"),
                    decisionContext,
                    getAvailableActionNames(),
                    actionSuccessRates
                );
                
                // 从LLM建议中提取推荐动作
                String recommendedAction = extractRecommendedAction(decisionAdvice, getAvailableActionNames());
                
                if (recommendedAction != null && availableTools.containsKey(recommendedAction)) {
                    llmAssistedDecisions++;
                    logger.info("采用LLM推荐动作: " + recommendedAction);
                    return recommendedAction;
                }
                
            } catch (Exception e) {
                logger.warning("LLM决策过程中出现异常: " + e.getMessage());
            }
        }
        
        // 回退到传统决策逻辑
        return decideActionTraditional(applicableStrategies, relevantExperiences);
    }
    
    /**
     * 传统决策逻辑（作为回退机制）
     */
    private String decideActionTraditional(List<Strategy> applicableStrategies, List<Experience> relevantExperiences) {
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
            
            List<String> availableActions = getAvailableActionNames();
            return availableActions.get((int) (Math.random() * availableActions.size()));
        }
    }
    
    /**
     * 处理任务的主要接口（LLM增强版本）
     */
    public EnhancedTaskResult processTaskWithLLM(String task, Map<String, Object> context) {
        if (context == null) {
            context = new HashMap<>();
        }
        context.put("task", task);
        
        logger.info("处理任务（LLM增强）: " + task);
        
        // LLM增强的环境感知
        EnhancedPerceptionResult perception = perceiveEnvironmentWithLLM(context);
        
        // LLM增强的决策制定
        String action = decideActionWithLLM(perception);
        
        // 执行动作
        Object result = executeAction(action, context);
        double[] evaluation = evaluateResult(result, null);
        
        boolean success = evaluation[0] == 1.0;
        double reward = evaluation[1];
        
        // 创建经验记录
        Experience experience = new Experience(task, context, action, result, success, reward);
        
        // LLM增强的反思和学习
        String llmReflection = "";
        String improvementAdvice = "";
        List<String> discoveredPatterns = new ArrayList<>();
        
        if (llmSimulator != null) {
            try {
                // 生成深度反思
                Map<String, Object> performanceMetrics = getPerformanceSummary();
                llmReflection = llmSimulator.generateDeepReflection(
                    experience, 
                    getRecentExperiences(20), 
                    performanceMetrics
                );
                
                // 提取改进建议
                improvementAdvice = extractImprovementAdvice(llmReflection);
                
                // 识别新模式
                discoveredPatterns = identifyNewPatterns(experience, getRecentExperiences(10));
                
            } catch (Exception e) {
                logger.warning("LLM反思过程中出现异常: " + e.getMessage());
                llmReflection = "LLM反思暂时不可用";
                improvementAdvice = "建议基于传统反思机制进行改进";
            }
        }
        
        // 学习和进化
        learnFromExperienceWithLLM(experience, llmReflection, improvementAdvice, discoveredPatterns);
        
        // 定期触发自进化
        if (totalTasks % 25 == 0) {  // 更频繁的进化触发
            selfEvolveWithLLM();
        }
        
        return new EnhancedTaskResult(task, action, result, success, reward, 
                                    experience.getReflection(), llmReflection, 
                                    improvementAdvice, discoveredPatterns);
    }
    
    /**
     * LLM增强的学习过程
     */
    public void learnFromExperienceWithLLM(Experience experience, String llmReflection, 
                                         String improvementAdvice, List<String> discoveredPatterns) {
        // 传统学习过程
        experiences.add(experience);
        
        if (experiences.size() > memorySize) {
            experiences.remove(0);
        }
        
        // LLM增强的知识图谱更新
        updateKnowledgeGraphWithLLM(experience, llmReflection);
        
        // 传统反思（作为基础）
        String basicReflection = reflectionModule.reflectOnExperience(experience);
        
        // 合并LLM反思
        String combinedReflection = combineReflections(basicReflection, llmReflection);
        experience.setReflection(combinedReflection);
        
        // 更新策略（考虑LLM建议）
        updateStrategiesWithLLM(experience, improvementAdvice, discoveredPatterns);
        
        // 更新性能指标
        totalTasks++;
        if (experience.isSuccess()) {
            successfulTasks++;
        }
        
        double currentSuccessRate = (double) successfulTasks / totalTasks;
        performanceHistory.add(new SelfEvolvingAgent.PerformanceRecord(
            System.currentTimeMillis(), currentSuccessRate, totalTasks));
        
        // LLM辅助的参数调整
        adjustLearningParametersWithLLM();
        
        logger.info(String.format("LLM增强学习完成 - 任务: %s, 成功: %s, 奖励: %.2f, LLM辅助: %s", 
                  experience.getTask(), experience.isSuccess(), experience.getReward(), 
                  llmReflection != null && !llmReflection.isEmpty() ? "是" : "否"));
    }
    
    /**
     * LLM增强的自进化过程
     */
    public void selfEvolveWithLLM() {
        logger.info("开始LLM增强的自我进化过程...");
        
        List<Experience> recentExperiences = getRecentExperiences(50);
        
        // 传统模式识别
        List<ReflectionModule.Pattern> basicPatterns = reflectionModule.identifyPatterns(recentExperiences);
        
        // LLM增强的模式分析
        if (llmSimulator != null && !recentExperiences.isEmpty()) {
            try {
                for (ReflectionModule.Pattern pattern : basicPatterns) {
                    String llmPatternAnalysis = llmSimulator.generatePatternAnalysis(
                        recentExperiences, pattern.getType());
                    pattern.setMetadata("llm_analysis", llmPatternAnalysis);
                }
            } catch (Exception e) {
                logger.warning("LLM模式分析过程中出现异常: " + e.getMessage());
            }
        }
        
        logger.info("识别到的模式数量: " + basicPatterns.size());
        
        // 执行优化
        optimizeStrategiesWithLLM();
        integrateKnowledgeWithLLM();
        expandCapabilitiesWithLLM();
        
        logger.info("LLM增强的自我进化完成");
    }
    
    // ================================
    // LLM增强的具体实现方法
    // ================================
    
    private CompletableFuture<String> generateStrategicReasoningAsync(Map<String, Object> context, 
                                                                    List<String> actions, 
                                                                    List<Experience> experiences) {
        return llmSimulator.generateStrategicReasoningAsync(context, actions, experiences);
    }
    
    private String summarizeContext(Map<String, Object> context, List<Experience> experiences, 
                                  List<Strategy> strategies) {
        StringBuilder summary = new StringBuilder();
        summary.append("上下文包含").append(context.size()).append("个要素，");
        summary.append("相关经验").append(experiences.size()).append("条，");
        summary.append("可用策略").append(strategies.size()).append("个");
        return summary.toString();
    }
    
    private String extractStrategicInsights(String llmAnalysis, Map<String, Object> context) {
        // 简化实现：从LLM分析中提取关键洞察
        if (llmAnalysis.contains("高成功率")) {
            return "识别到高成功率模式，建议利用优势策略";
        } else if (llmAnalysis.contains("风险")) {
            return "检测到潜在风险，建议采用保守策略";
        } else {
            return "建议保持当前策略并持续监控";
        }
    }
    
    private double calculateLLMConfidence(String llmAnalysis, int experienceCount) {
        double baseConfidence = 0.5;
        
        // 基于经验数量调整
        baseConfidence += Math.min(0.3, experienceCount * 0.02);
        
        // 基于LLM分析质量调整
        if (llmAnalysis.contains("建议") && llmAnalysis.contains("分析")) {
            baseConfidence += 0.1;
        }
        
        return Math.min(0.95, baseConfidence);
    }
    
    private Map<String, Double> calculateActionSuccessRates(List<String> actions, List<Experience> experiences) {
        Map<String, Double> rates = new HashMap<>();
        
        for (String action : actions) {
            List<Experience> actionExps = experiences.stream()
                .filter(exp -> action.equals(exp.getAction()))
                .collect(Collectors.toList());
                
            if (!actionExps.isEmpty()) {
                double successRate = actionExps.stream()
                    .mapToDouble(exp -> exp.isSuccess() ? 1.0 : 0.0)
                    .average().orElse(0.5);
                rates.put(action, successRate);
            } else {
                rates.put(action, 0.5); // 默认值
            }
        }
        
        return rates;
    }
    
    private String extractRecommendedAction(String decisionAdvice, List<String> availableActions) {
        // 简化实现：在决策建议中查找推荐的动作
        for (String action : availableActions) {
            if (decisionAdvice.toLowerCase().contains(action.toLowerCase())) {
                return action;
            }
        }
        return null;
    }
    
    private String extractImprovementAdvice(String llmReflection) {
        // 从LLM反思中提取改进建议
        if (llmReflection.contains("改进策略")) {
            int start = llmReflection.indexOf("改进策略");
            int end = Math.min(start + 200, llmReflection.length());
            return llmReflection.substring(start, end);
        }
        return "继续当前学习路径，注重经验积累";
    }
    
    private List<String> identifyNewPatterns(Experience experience, List<Experience> recentExperiences) {
        List<String> patterns = new ArrayList<>();
        
        // 简化模式识别
        String action = experience.getAction();
        long actionCount = recentExperiences.stream()
            .filter(exp -> action.equals(exp.getAction()))
            .count();
            
        if (actionCount >= 3 && experience.isSuccess()) {
            patterns.add("成功动作模式: " + action);
        }
        
        if (experience.getReward() > 0.8) {
            patterns.add("高奖励模式: " + experience.getTask() + "->" + action);
        }
        
        return patterns;
    }
    
    private void updateKnowledgeGraphWithLLM(Experience experience, String llmReflection) {
        // 传统知识图谱更新
        String taskConcept = "task:" + experience.getTask();
        String actionConcept = "action:" + experience.getAction();
        
        Map<String, Object> taskProperties = new HashMap<>();
        taskProperties.put("type", "task");
        taskProperties.put("description", experience.getTask());
        if (llmReflection != null && !llmReflection.isEmpty()) {
            taskProperties.put("llm_insight", llmReflection.substring(0, Math.min(100, llmReflection.length())));
        }
        knowledgeGraph.addConcept(taskConcept, taskProperties);
        
        Map<String, Object> actionProperties = new HashMap<>();
        actionProperties.put("type", "action");
        actionProperties.put("description", experience.getAction());
        knowledgeGraph.addConcept(actionConcept, actionProperties);
        
        String relationType = experience.isSuccess() ? "succeeds_with" : "fails_with";
        double weight = experience.getReward();
        
        knowledgeGraph.addRelation(taskConcept, actionConcept, relationType, weight);
        
        // LLM增强：添加语义关联
        if (llmSimulator != null && llmReflection != null) {
            try {
                String inferenceResult = llmSimulator.generateKnowledgeInference(
                    taskConcept, actionConcept, Arrays.asList(experience), "semantic_relation");
                // 基于推理结果可以添加更多的知识关联
            } catch (Exception e) {
                logger.warning("LLM知识推理过程中出现异常: " + e.getMessage());
            }
        }
    }
    
    private String combineReflections(String basicReflection, String llmReflection) {
        if (llmReflection == null || llmReflection.isEmpty()) {
            return basicReflection;
        }
        
        return basicReflection + "\n\n[LLM深度洞察] " + llmReflection;
    }
    
    private void updateStrategiesWithLLM(Experience experience, String improvementAdvice, List<String> patterns) {
        // 传统策略更新
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
        
        // LLM增强：基于改进建议创建新策略
        if (improvementAdvice != null && !improvementAdvice.isEmpty() && !patterns.isEmpty()) {
            for (String pattern : patterns) {
                if (pattern.contains("成功")) {
                    // 为成功模式创建强化策略
                    String patternStrategyName = "LLM_强化_" + pattern.hashCode();
                    if (!strategies.containsKey(patternStrategyName)) {
                        Map<String, Object> patternConditions = new HashMap<>();
                        patternConditions.put("pattern_based", true);
                        patternConditions.put("llm_derived", true);
                        
                        Strategy patternStrategy = new Strategy(
                            patternStrategyName,
                            "基于LLM识别的成功模式策略: " + pattern,
                            patternConditions,
                            Arrays.asList(experience.getAction()),
                            0.8, // 高初始成功率
                            1
                        );
                        strategies.put(patternStrategyName, patternStrategy);
                    }
                }
            }
        }
    }
    
    private void adjustLearningParametersWithLLM() {
        // 传统参数调整
        if (performanceHistory.size() >= 10) {
            List<SelfEvolvingAgent.PerformanceRecord> recent = performanceHistory.subList(
                performanceHistory.size() - 10, performanceHistory.size());
            double avgSuccessRate = recent.stream()
                    .mapToDouble(SelfEvolvingAgent.PerformanceRecord::getSuccessRate)
                    .average().orElse(0.0);
            
            if (avgSuccessRate < 0.6) {
                explorationRate = Math.min(0.5, explorationRate + 0.05);
            } else if (avgSuccessRate > 0.8) {
                explorationRate = Math.max(0.1, explorationRate - 0.02);
            }
        }
        
        // LLM增强：动态调整LLM参与度
        if (totalTasks > 0) {
            double llmContribution = (double) llmAssistedDecisions / totalTasks;
            if (llmContribution > 0.8 && explorationRate < 0.3) {
                // 如果过度依赖LLM，增加传统探索
                explorationRate += 0.1;
                logger.info("检测到过度依赖LLM，增加传统探索");
            }
        }
    }
    
    private void optimizeStrategiesWithLLM() {
        // 传统策略优化
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
        
        // LLM增强：识别需要合并的相似策略
        if (llmSimulator != null && strategies.size() > 5) {
            // 简化实现：基于名称相似度合并策略
            List<String> strategyNames = new ArrayList<>(strategies.keySet());
            for (int i = 0; i < strategyNames.size() - 1; i++) {
                for (int j = i + 1; j < strategyNames.size(); j++) {
                    String name1 = strategyNames.get(i);
                    String name2 = strategyNames.get(j);
                    
                    if (calculateNameSimilarity(name1, name2) > 0.8) {
                        // 合并相似策略
                        Strategy strategy1 = strategies.get(name1);
                        Strategy strategy2 = strategies.get(name2);
                        
                        if (strategy1 != null && strategy2 != null) {
                            // 保留表现更好的策略
                            if (strategy1.getSuccessRate() >= strategy2.getSuccessRate()) {
                                strategies.remove(name2);
                                logger.info("LLM建议合并策略: 保留 " + name1 + "，移除 " + name2);
                            } else {
                                strategies.remove(name1);
                                logger.info("LLM建议合并策略: 保留 " + name2 + "，移除 " + name1);
                            }
                        }
                    }
                }
            }
        }
    }
    
    private void integrateKnowledgeWithLLM() {
        List<String> concepts = new ArrayList<>(knowledgeGraph.getNodes().keySet());
        
        // 传统知识整合
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
        
        // LLM增强：语义知识推理
        if (llmSimulator != null && concepts.size() > 2) {
            try {
                // 随机选择概念对进行LLM推理
                for (int i = 0; i < Math.min(5, concepts.size() / 2); i++) {
                    String concept1 = concepts.get((int) (Math.random() * concepts.size()));
                    String concept2 = concepts.get((int) (Math.random() * concepts.size()));
                    
                    if (!concept1.equals(concept2)) {
                        List<Experience> relatedExps = getRecentExperiences(10);
                        String inferenceResult = llmSimulator.generateKnowledgeInference(
                            concept1, concept2, relatedExps, "semantic_relation");
                            
                        // 基于推理结果添加知识连接
                        if (inferenceResult.contains("相关") || inferenceResult.contains("关联")) {
                            knowledgeGraph.addRelation(concept1, concept2, "llm_inferred", 0.6);
                        }
                    }
                }
            } catch (Exception e) {
                logger.warning("LLM知识整合过程中出现异常: " + e.getMessage());
            }
        }
    }
    
    private void expandCapabilitiesWithLLM() {
        List<Experience> successfulExperiences = getRecentExperiences(50).stream()
                .filter(Experience::isSuccess)
                .collect(Collectors.toList());
        
        // 传统能力扩展
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
        
        // LLM增强：智能工具创建
        if (llmSimulator != null && successfulExperiences.size() > 10) {
            try {
                // 分析成功经验，识别潜在的新工具需求
                Map<String, Long> taskTypes = successfulExperiences.stream()
                    .collect(Collectors.groupingBy(
                        exp -> exp.getTask().split(":")[0],
                        Collectors.counting()
                    ));
                
                // 为高频任务类型创建专用工具
                taskTypes.entrySet().stream()
                    .filter(entry -> entry.getValue() >= 5)
                    .forEach(entry -> {
                        String taskType = entry.getKey();
                        String toolName = "llm_specialized_" + taskType.toLowerCase();
                        
                        if (!availableTools.containsKey(toolName)) {
                            availableTools.put(toolName, createLLMSpecializedTool(taskType));
                            logger.info("LLM建议创建专用工具: " + toolName);
                        }
                    });
                    
            } catch (Exception e) {
                logger.warning("LLM能力扩展过程中出现异常: " + e.getMessage());
            }
        }
    }
    
    // ================================
    // 工具方法
    // ================================
    
    private List<String> getAvailableActionNames() {
        return new ArrayList<>(availableTools.keySet());
    }
    
    private List<Experience> getRecentExperiences(int count) {
        int startIndex = Math.max(0, experiences.size() - count);
        return new ArrayList<>(experiences.subList(startIndex, experiences.size()));
    }
    
    private double evaluateExpression(String expression) {
        // 简化的表达式计算
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
    
    private double calculateNameSimilarity(String name1, String name2) {
        // 简化的名称相似度计算
        Set<String> words1 = new HashSet<>(Arrays.asList(name1.split("_")));
        Set<String> words2 = new HashSet<>(Arrays.asList(name2.split("_")));
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
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
    
    private Function<Map<String, Object>, Map<String, Object>> createLLMSpecializedTool(String taskType) {
        return context -> {
            Map<String, Object> result = new HashMap<>();
            result.put("specialized_for", taskType);
            result.put("llm_enhanced", true);
            result.put("result", "专用工具处理结果 for " + taskType);
            
            if (llmSimulator != null) {
                try {
                    String analysis = llmSimulator.generateEvolResponse(
                        "专用处理：" + taskType,
                        context.toString(),
                        "strategic_reasoning"
                    );
                    result.put("llm_analysis", analysis);
                } catch (Exception e) {
                    result.put("llm_analysis", "LLM分析暂时不可用");
                }
            }
            
            result.put("confidence", 0.8 + Math.random() * 0.15);
            return result;
        };
    }
    
    // 继承原有方法以保持兼容性
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
    
    private double calculateStringSimilarity(String str1, String str2) {
        Set<String> words1 = new HashSet<>(Arrays.asList(str1.split("\\s+")));
        Set<String> words2 = new HashSet<>(Arrays.asList(str2.split("\\s+")));
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
    private List<Strategy> findApplicableStrategies(Map<String, Object> context) {
        return strategies.values().stream()
                .filter(strategy -> strategy.matchesContext(context))
                .sorted((s1, s2) -> Double.compare(s2.getSuccessRate(), s1.getSuccessRate()))
                .collect(Collectors.toList());
    }
    
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
    
    private Object executeAction(String action, Map<String, Object> context) {
        Function<Map<String, Object>, Map<String, Object>> tool = availableTools.get(action);
        if (tool != null) {
            return tool.apply(context);
        } else {
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", "未知动作: " + action);
            return errorResult;
        }
    }
    
    private double[] evaluateResult(Object result, Object expectedOutcome) {
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
            
            // LLM增强结果额外加分
            if (resultMap.containsKey("llm_enhanced") && (Boolean) resultMap.get("llm_enhanced")) {
                reward += 0.1; // LLM增强奖励
            }
            
        } else {
            success = true;
            reward = 0.3;
        }
        
        return new double[]{success ? 1.0 : 0.0, reward};
    }
    
    private void initializeBaseStrategies() {
        Map<String, Object> exploreConditions = new HashMap<>();
        exploreConditions.put("uncertainty", "high");
        Strategy exploreStrategy = new Strategy(
            "LLM增强探索策略",
            "在不确定情况下结合LLM进行智能探索",
            exploreConditions,
            Arrays.asList("intelligent_search", "deep_analyze"),
            0.6,
            0
        );
        strategies.put(exploreStrategy.getName(), exploreStrategy);
        
        Map<String, Object> exploitConditions = new HashMap<>();
        exploitConditions.put("confidence", "high");
        Strategy exploitStrategy = new Strategy(
            "LLM增强利用策略",
            "使用LLM优化的已知有效方法",
            exploitConditions,
            Arrays.asList("strategic_plan", "smart_calculate"),
            0.8,
            0
        );
        strategies.put(exploitStrategy.getName(), exploitStrategy);
    }
    
    /**
     * 获取增强的性能摘要
     */
    public Map<String, Object> getEnhancedPerformanceSummary() {
        Map<String, Object> summary = getPerformanceSummary();
        
        // 添加LLM相关指标
        summary.put("llm_enabled", llmSimulator != null);
        summary.put("llm_assisted_decisions", llmAssistedDecisions);
        
        if (totalTasks > 0) {
            summary.put("llm_assistance_rate", (double) llmAssistedDecisions / totalTasks);
        }
        
        summary.put("llm_confidence_threshold", llmConfidenceThreshold);
        summary.put("async_llm_enabled", enableAsyncLLM);
        
        return summary;
    }
    
    public Map<String, Object> getPerformanceSummary() {
        if (performanceHistory.isEmpty()) {
            Map<String, Object> summary = new HashMap<>();
            summary.put("message", "暂无性能数据");
            return summary;
        }
        
        SelfEvolvingAgent.PerformanceRecord latest = performanceHistory.get(performanceHistory.size() - 1);
        
        String trend = "insufficient_data";
        if (performanceHistory.size() >= 10) {
            double recentAvg = performanceHistory.stream()
                    .skip(performanceHistory.size() - 10)
                    .mapToDouble(SelfEvolvingAgent.PerformanceRecord::getSuccessRate)
                    .average().orElse(0.0);
            double overallAvg = performanceHistory.stream()
                    .mapToDouble(SelfEvolvingAgent.PerformanceRecord::getSuccessRate)
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
    public boolean isLLMEnabled() { return llmSimulator != null; }
    public int getTotalTasks() { return totalTasks; }
    public int getSuccessfulTasks() { return successfulTasks; }
    public int getLlmAssistedDecisions() { return llmAssistedDecisions; }
    public double getExplorationRate() { return explorationRate; }
    public double getLlmConfidenceThreshold() { return llmConfidenceThreshold; }
    public List<Experience> getExperiences() { return new ArrayList<>(experiences); }
    public Map<String, Strategy> getStrategies() { return new HashMap<>(strategies); }
    public KnowledgeGraph getKnowledgeGraph() { return knowledgeGraph; }
    public ReflectionModule getReflectionModule() { return reflectionModule; }
    public EvolLLMSimulator getLlmSimulator() { return llmSimulator; }
    
    // Setters for configuration
    public void setLlmConfidenceThreshold(double threshold) { 
        this.llmConfidenceThreshold = Math.max(0.0, Math.min(1.0, threshold)); 
    }
    
    public void setEnableAsyncLLM(boolean enable) { 
        this.enableAsyncLLM = enable; 
    }
    
    public void setExplorationRate(double rate) { 
        this.explorationRate = Math.max(0.0, Math.min(1.0, rate)); 
    }
}