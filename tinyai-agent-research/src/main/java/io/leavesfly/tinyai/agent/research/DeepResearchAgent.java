package io.leavesfly.tinyai.agent.research;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.util.*;

/**
 * 深度研究Agent - 主类
 * 基于LLM驱动的高级研究智能体，具备多阶段推理、知识图谱构建、自适应学习等能力
 * 
 * @author 山泽
 */
public class DeepResearchAgent {
    
    /** Agent名称 */
    private final String name;
    
    /** 默认研究领域 */
    private final String domain;
    
    /** 核心组件 */
    private final KnowledgeGraph knowledgeGraph;
    private final IntelligentReasoner reasoner;
    private final ResearchPipeline pipeline;
    
    /** 研究历史 */
    private final List<Map<String, Object>> researchHistory;
    
    /** 性能指标 */
    private final Map<String, Object> performanceMetrics;
    
    /** 学习和适应参数 */
    private double learningRate;
    private double confidenceThreshold;
    private int maxResearchDepth;
    
    /**
     * 构造函数
     */
    public DeepResearchAgent(String name, String domain) {
        this.name = name;
        this.domain = domain;
        
        // 初始化核心组件
        this.knowledgeGraph = new KnowledgeGraph();
        this.reasoner = new IntelligentReasoner();
        this.pipeline = new ResearchPipeline(knowledgeGraph, reasoner);
        
        // 初始化历史和指标
        this.researchHistory = new ArrayList<>();
        this.performanceMetrics = new HashMap<>();
        
        // 设置默认参数
        this.learningRate = 0.1;
        this.confidenceThreshold = 0.7;
        this.maxResearchDepth = 5;
        
        // 初始化性能指标
        initializePerformanceMetrics();
    }
    
    /**
     * 便利构造函数
     */
    public DeepResearchAgent(String name) {
        this(name, "general");
    }
    
    /**
     * 默认构造函数
     */
    public DeepResearchAgent() {
        this("DeepResearch Agent", "general");
    }
    
    /**
     * 执行深度研究
     */
    public Map<String, Object> research(String query, String domain, int complexity, 
                                      int depthRequired, int urgency) {
        
        // 构建研究查询
        ResearchQuery researchQuery = new ResearchQuery(
            query, 
            domain != null ? domain : this.domain, 
            complexity, 
            urgency, 
            depthRequired
        );
        
        System.out.println("🔍 开始深度研究: " + query);
        System.out.println("🎨 研究配置: 复杂度=" + complexity + ", 深度=" + depthRequired + ", 紧急度=" + urgency);
        
        // 选择推理模式
        ReasoningMode reasoningMode = reasoner.selectReasoningMode(researchQuery, new HashMap<>());
        System.out.println("🧠 选择推理模式: " + reasoningMode.getCode());
        
        // 执行研究管道
        ResearchResult researchResult = pipeline.executeResearch(researchQuery);
        
        // 处理和包装结果
        Map<String, Object> finalResult = processResearchResult(researchQuery, researchResult, reasoningMode);
        
        // 更新学习指标
        updateLearningMetrics(researchQuery, researchResult, reasoningMode);
        
        // 保存研究历史
        saveResearchHistory(researchQuery, finalResult);
        
        return finalResult;
    }
    
    /**
     * 执行研究（简化版本）
     */
    public Map<String, Object> research(String query, int complexity, int depthRequired, int urgency) {
        return research(query, null, complexity, depthRequired, urgency);
    }
    
    /**
     * 执行研究（最简版本）
     */
    public Map<String, Object> research(String query) {
        return research(query, null, 3, 3, 2);
    }
    
    /**
     * 处理研究结果
     */
    private Map<String, Object> processResearchResult(ResearchQuery query, ResearchResult result, 
                                                    ReasoningMode reasoningMode) {
        
        Map<String, Object> finalResult = new HashMap<>();
        
        // 基本信息
        finalResult.put("query", query.getQuery());
        finalResult.put("domain", query.getDomain());
        finalResult.put("reasoningMode", reasoningMode.getCode());
        
        // 研究结果
        finalResult.put("finalAnswer", result.getFinalAnswer());
        finalResult.put("totalConfidence", result.getTotalConfidence());
        finalResult.put("qualityScore", result.getQualityScore());
        
        // 关键洞察
        List<String> keyInsights = result.getContext().getAllInsights().stream()
                .map(ResearchInsight::getContent)
                .limit(5)  // 最多显示5个洞察
                .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
        finalResult.put("keyInsights", keyInsights);
        
        // 统计信息
        finalResult.put("researchSteps", result.getContext().getAllSteps().size());
        finalResult.put("toolsUsed", result.getContext().getAllToolsCalled().stream()
                .collect(HashSet::new, HashSet::add, HashSet::addAll).size());
        finalResult.put("knowledgeNodesConsulted", 0);  // 暂时设为0，后续可以从上下文中获取
        finalResult.put("phasesCompleted", ResearchPhase.values().length);
        
        // 详细步骤
        List<Map<String, Object>> detailedSteps = result.getContext().getAllSteps().stream()
                .map(step -> {
                    Map<String, Object> stepMap = new HashMap<>();
                    stepMap.put("phase", step.getPhase().getCode());
                    stepMap.put("type", step.getStepType());
                    stepMap.put("content", step.getContent());
                    stepMap.put("confidence", step.getConfidence());
                    return stepMap;
                })
                .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
        finalResult.put("detailedSteps", detailedSteps);
        
        // 时间戳
        finalResult.put("timestamp", LocalDateTime.now().toString());
        
        return finalResult;
    }
    
    /**
     * 更新学习指标
     */
    private void updateLearningMetrics(ResearchQuery query, ResearchResult result, ReasoningMode reasoningMode) {
        // 更新总体指标
        incrementCounter("totalResearchCount");
        
        // 更新平均置信度
        updateAverageConfidence(result.getTotalConfidence());
        
        // 更新领域专业度
        updateDomainExpertise(query.getDomain(), result.getTotalConfidence());
        
        // 更新推理模式使用统计
        incrementCounter("reasoningModeUsage." + reasoningMode.getCode());
        
        // 记录推理器性能
        reasoner.recordPerformance(reasoningMode, result.getQualityScore());
    }
    
    /**
     * 添加领域知识
     */
    public String addDomainKnowledge(String content, String domain, String nodeType, double confidence) {
        String nodeId = KnowledgeGraph.generateNodeId(content + domain + System.currentTimeMillis());
        
        KnowledgeNode node = new KnowledgeNode(nodeId, content, nodeType, domain, confidence);
        knowledgeGraph.addNode(node);
        
        System.out.println("✅ 已添加知识节点: " + content.substring(0, Math.min(50, content.length())) + "...");
        
        return nodeId;
    }
    
    /**
     * 添加领域知识（简化版本）
     */
    public String addDomainKnowledge(String content, String domain, String nodeType) {
        return addDomainKnowledge(content, domain, nodeType, 0.8);
    }
    
    /**
     * 添加领域知识（最简版本）
     */
    public String addDomainKnowledge(String content, String domain) {
        return addDomainKnowledge(content, domain, "concept", 0.8);
    }
    
    /**
     * 探索性研究主题
     */
    public Map<String, Object> exploreResearchTopic(String topic, int maxDepth) {
        System.out.println("🤝 开始探索性研究: " + topic);
        
        List<Map<String, Object>> explorationResults = new ArrayList<>();
        
        // 生成多个相关问题
        List<String> relatedQuestions = generateRelatedQuestions(topic);
        
        for (int i = 0; i < Math.min(relatedQuestions.size(), maxDepth); i++) {
            String question = relatedQuestions.get(i);
            System.out.println("🔎 探索问题 " + (i + 1) + ": " + question);
            
            Map<String, Object> result = research(question, 2, 2, 1);  // 中等复杂度，低紧急度
            
            Map<String, Object> explorationItem = new HashMap<>();
            explorationItem.put("question", question);
            explorationItem.put("result", result);
            explorationResults.add(explorationItem);
        }
        
        Map<String, Object> explorationSummary = new HashMap<>();
        explorationSummary.put("topic", topic);
        explorationSummary.put("explorationResults", explorationResults);
        explorationSummary.put("totalQuestionsExplored", explorationResults.size());
        explorationSummary.put("timestamp", LocalDateTime.now().toString());
        
        return explorationSummary;
    }
    
    /**
     * 协作式研究
     */
    public Map<String, Object> collaborativeResearch(String mainQuery, List<String> perspectives) {
        System.out.println("🤝 开始协作式研究: " + mainQuery);
        System.out.println("👁️ 研究视角: " + String.join(", ", perspectives));
        
        List<Map<String, Object>> perspectiveResults = new ArrayList<>();
        
        for (String perspective : perspectives) {
            String perspectiveQuery = "从" + perspective + "视角分析: " + mainQuery;
            System.out.println("🔍 研究视角: " + perspective);
            
            Map<String, Object> result = research(perspectiveQuery, 3, 3, 2);
            
            Map<String, Object> perspectiveItem = new HashMap<>();
            perspectiveItem.put("perspective", perspective);
            perspectiveItem.put("query", perspectiveQuery);
            perspectiveItem.put("result", result);
            perspectiveResults.add(perspectiveItem);
        }
        
        // 综合分析
        Map<String, Object> synthesis = synthesizePerspectives(mainQuery, perspectiveResults);
        
        Map<String, Object> collaborationResult = new HashMap<>();
        collaborationResult.put("mainQuery", mainQuery);
        collaborationResult.put("perspectives", perspectives);
        collaborationResult.put("perspectiveResults", perspectiveResults);
        collaborationResult.put("synthesis", synthesis);
        collaborationResult.put("timestamp", LocalDateTime.now().toString());
        
        return collaborationResult;
    }
    
    /**
     * 获取知识概览
     */
    public Map<String, Object> getKnowledgeOverview(String domain) {
        String targetDomain = domain != null ? domain : this.domain;
        return knowledgeGraph.getDomainOverview(targetDomain);
    }
    
    /**
     * 获取性能报告
     */
    public Map<String, Object> getPerformanceReport() {
        Map<String, Object> report = new HashMap<>();
        
        report.put("agentName", name);
        report.put("primaryDomain", domain);
        report.put("performanceMetrics", new HashMap<>(performanceMetrics));
        report.put("totalKnowledgeNodes", knowledgeGraph.getNodes().size());
        report.put("researchHistoryCount", researchHistory.size());
        report.put("knowledgeDomains", new ArrayList<>(knowledgeGraph.getDomains().keySet()));
        
        if (!researchHistory.isEmpty()) {
            Map<String, Object> lastResearch = researchHistory.get(researchHistory.size() - 1);
            report.put("lastResearch", lastResearch.get("timestamp"));
        }
        
        return report;
    }
    
    /**
     * 生成相关问题
     */
    private List<String> generateRelatedQuestions(String topic) {
        return Arrays.asList(
            topic + "的核心原理是什么？",
            topic + "在实际中有哪些应用？",
            topic + "的发展趋势如何？",
            topic + "面临的主要挑战是什么？",
            topic + "与其他领域有什么关联？"
        );
    }
    
    /**
     * 综合多视角分析
     */
    private Map<String, Object> synthesizePerspectives(String mainQuery, 
                                                     List<Map<String, Object>> perspectiveResults) {
        
        // 提取共同主题
        List<String> commonThemes = new ArrayList<>();
        List<String> differences = new ArrayList<>();
        
        // 简化的综合分析
        if (perspectiveResults.size() > 1) {
            commonThemes.add("多个视角都认识到了问题的复杂性");
            commonThemes.add("需要跨领域的综合考虑");
            differences.add("不同视角的关注重点存在差异");
        }
        
        Map<String, Object> synthesis = new HashMap<>();
        synthesis.put("synthesisConclusion", 
            "通过多视角协作研究，对问题'" + mainQuery + "'形成了更全面的理解");
        synthesis.put("commonThemes", commonThemes);
        synthesis.put("keyDifferences", differences);
        synthesis.put("synthesisConfidence", 0.8);
        
        return synthesis;
    }
    
    /**
     * 初始化性能指标
     */
    private void initializePerformanceMetrics() {
        performanceMetrics.put("totalResearchCount", 0);
        performanceMetrics.put("avgConfidence", 0.0);
        performanceMetrics.put("domainExpertise", new HashMap<String, Double>());
        performanceMetrics.put("reasoningModeUsage", new HashMap<String, Integer>());
    }
    
    /**
     * 增加计数器
     */
    @SuppressWarnings("unchecked")
    private void incrementCounter(String path) {
        String[] parts = path.split("\\.");
        Map<String, Object> current = performanceMetrics;
        
        for (int i = 0; i < parts.length - 1; i++) {
            current = (Map<String, Object>) current.computeIfAbsent(parts[i], k -> new HashMap<>());
        }
        
        String key = parts[parts.length - 1];
        current.put(key, (Integer) current.getOrDefault(key, 0) + 1);
    }
    
    /**
     * 更新平均置信度
     */
    private void updateAverageConfidence(double currentConfidence) {
        double oldAvg = (Double) performanceMetrics.get("avgConfidence");
        int count = (Integer) performanceMetrics.get("totalResearchCount");
        double newAvg = (oldAvg * (count - 1) + currentConfidence) / count;
        performanceMetrics.put("avgConfidence", newAvg);
    }
    
    /**
     * 更新领域专业度
     */
    @SuppressWarnings("unchecked")
    private void updateDomainExpertise(String domain, double confidence) {
        Map<String, Double> domainExpertise = (Map<String, Double>) performanceMetrics.get("domainExpertise");
        double oldExpertise = domainExpertise.getOrDefault(domain, 0.0);
        double newExpertise = oldExpertise * (1 - learningRate) + confidence * learningRate;
        domainExpertise.put(domain, newExpertise);
    }
    
    /**
     * 保存研究历史
     */
    private void saveResearchHistory(ResearchQuery query, Map<String, Object> result) {
        Map<String, Object> historyItem = new HashMap<>();
        historyItem.put("query", query);
        historyItem.put("result", result);
        historyItem.put("timestamp", LocalDateTime.now());
        
        researchHistory.add(historyItem);
        
        // 保持历史记录在合理范围内
        if (researchHistory.size() > 1000) {
            researchHistory.remove(0);
        }
    }
    
    // Getter和Setter方法
    public String getName() {
        return name;
    }
    
    public String getDomain() {
        return domain;
    }
    
    public KnowledgeGraph getKnowledgeGraph() {
        return knowledgeGraph;
    }
    
    public IntelligentReasoner getReasoner() {
        return reasoner;
    }
    
    public ResearchPipeline getPipeline() {
        return pipeline;
    }
    
    public List<Map<String, Object>> getResearchHistory() {
        return researchHistory;
    }
    
    public Map<String, Object> getPerformanceMetrics() {
        return performanceMetrics;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    public void setConfidenceThreshold(double confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
    
    public int getMaxResearchDepth() {
        return maxResearchDepth;
    }
    
    public void setMaxResearchDepth(int maxResearchDepth) {
        this.maxResearchDepth = maxResearchDepth;
    }
}