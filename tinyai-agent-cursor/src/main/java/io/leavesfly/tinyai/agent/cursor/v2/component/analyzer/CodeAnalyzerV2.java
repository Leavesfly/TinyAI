package io.leavesfly.tinyai.agent.cursor.v2.component.analyzer;

import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.LLMGateway;
import io.leavesfly.tinyai.agent.cursor.v2.tool.ToolOrchestrator;

import java.util.*;

/**
 * 增强版代码分析器（V2）
 * 基于LLM的深度代码分析，提供多维度的代码质量评估
 * 
 * 核心功能：
 * 1. 代码质量分析（代码规范、设计模式、最佳实践）
 * 2. 潜在问题检测（Bug、安全漏洞、性能问题）
 * 3. 代码复杂度评估（圈复杂度、认知复杂度）
 * 4. 改进建议生成（具体的修改建议和示例代码）
 * 5. 代码评分（综合评分和分项评分）
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class CodeAnalyzerV2 {
    
    /**
     * LLM网关
     */
    private final LLMGateway llmGateway;
    
    /**
     * 上下文引擎
     */
    private final ContextEngine contextEngine;
    
    /**
     * 工具编排器
     */
    private final ToolOrchestrator toolOrchestrator;
    
    /**
     * 默认分析模型
     */
    private String analysisModel = "deepseek-chat";
    
    /**
     * 分析温度（较低的温度使分析更准确）
     */
    private double analysisTemperature = 0.3;
    
    public CodeAnalyzerV2(LLMGateway llmGateway, ContextEngine contextEngine, 
                         ToolOrchestrator toolOrchestrator) {
        this.llmGateway = llmGateway;
        this.contextEngine = contextEngine;
        this.toolOrchestrator = toolOrchestrator;
    }
    
    /**
     * 综合代码分析
     * 
     * @param request 分析请求
     * @return 分析结果
     */
    public AnalysisResult analyze(AnalysisRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 1. 构建分析上下文
            Context context = contextEngine.buildAnalysisContext(
                request.getProjectId(),
                request.getCode(),
                "comprehensive"
            );
            
            // 2. 先使用工具进行基础分析
            ToolResult toolAnalysis = performToolAnalysis(request.getCode());
            
            // 3. 构建LLM分析提示词
            String analysisPrompt = buildAnalysisPrompt(request, toolAnalysis);
            
            // 4. 调用LLM进行深度分析
            ChatRequest chatRequest = ChatRequest.builder()
                .model(analysisModel)
                .addSystemMessage(context.buildSystemPrompt())
                .addSystemMessage(getAnalysisSystemPrompt())
                .addUserMessage(analysisPrompt)
                .temperature(analysisTemperature)
                .maxTokens(2000)
                .build();
            
            ChatResponse response = llmGateway.chat(chatRequest);
            
            // 5. 解析分析结果
            AnalysisResult result = parseAnalysisResult(response.getContent());
            result.setToolAnalysis(toolAnalysis);
            result.setAnalysisTime(System.currentTimeMillis() - startTime);
            
            return result;
            
        } catch (Exception e) {
            AnalysisResult errorResult = new AnalysisResult();
            errorResult.setSuccess(false);
            errorResult.setErrorMessage("分析失败: " + e.getMessage());
            errorResult.setAnalysisTime(System.currentTimeMillis() - startTime);
            return errorResult;
        }
    }
    
    /**
     * 快速质量检查
     * 
     * @param code 代码
     * @param projectId 项目ID
     * @return 质量检查结果
     */
    public QualityCheckResult quickCheck(String code, String projectId) {
        // 使用工具进行快速分析
        ToolResult toolResult = performToolAnalysis(code);
        
        // 转换为质量检查结果
        return convertToQualityCheck(toolResult);
    }
    
    /**
     * 安全性分析
     * 
     * @param code 代码
     * @param projectId 项目ID
     * @return 安全分析结果
     */
    public SecurityAnalysisResult analyzeSecurity(String code, String projectId) {
        Context context = contextEngine.buildAnalysisContext(projectId, code, "security");
        
        String securityPrompt = buildSecurityPrompt(code);
        
        ChatRequest request = ChatRequest.builder()
            .model(analysisModel)
            .addSystemMessage(context.buildSystemPrompt())
            .addSystemMessage("你是一个代码安全专家，专注于识别安全漏洞和潜在风险。")
            .addUserMessage(securityPrompt)
            .temperature(0.2)
            .maxTokens(1500)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        
        return parseSecurityResult(response.getContent());
    }
    
    /**
     * 性能分析
     * 
     * @param code 代码
     * @param projectId 项目ID
     * @return 性能分析结果
     */
    public PerformanceAnalysisResult analyzePerformance(String code, String projectId) {
        Context context = contextEngine.buildAnalysisContext(projectId, code, "performance");
        
        String performancePrompt = buildPerformancePrompt(code);
        
        ChatRequest request = ChatRequest.builder()
            .model(analysisModel)
            .addSystemMessage(context.buildSystemPrompt())
            .addSystemMessage("你是一个性能优化专家，专注于识别性能瓶颈和优化机会。")
            .addUserMessage(performancePrompt)
            .temperature(0.3)
            .maxTokens(1500)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        
        return parsePerformanceResult(response.getContent());
    }
    
    /**
     * 执行工具分析
     */
    private ToolResult performToolAnalysis(String code) {
        Map<String, Object> params = new HashMap<>();
        params.put("code", code);
        params.put("analysisType", "all");
        
        return toolOrchestrator.executeTool("code_analyzer", params);
    }
    
    /**
     * 构建分析提示词
     */
    private String buildAnalysisPrompt(AnalysisRequest request, ToolResult toolAnalysis) {
        StringBuilder prompt = new StringBuilder();
        
        prompt.append("请对以下代码进行全面分析：\n\n");
        prompt.append("```").append(request.getLanguage()).append("\n");
        prompt.append(request.getCode()).append("\n");
        prompt.append("```\n\n");
        
        if (toolAnalysis != null && toolAnalysis.isSuccess()) {
            prompt.append("基础分析结果：\n");
            prompt.append(toolAnalysis.getResult()).append("\n\n");
        }
        
        prompt.append("请从以下维度进行深度分析：\n");
        prompt.append("1. 代码质量（命名规范、代码风格、可读性）\n");
        prompt.append("2. 设计质量（设计模式、SOLID原则、架构合理性）\n");
        prompt.append("3. 潜在问题（Bug风险、边界条件、异常处理）\n");
        prompt.append("4. 性能考虑（时间复杂度、空间复杂度、优化建议）\n");
        prompt.append("5. 安全性（输入验证、权限检查、敏感信息）\n");
        prompt.append("6. 可维护性（模块化、注释、测试友好）\n\n");
        
        prompt.append("请提供：\n");
        prompt.append("- 具体的问题描述和位置\n");
        prompt.append("- 改进建议和最佳实践\n");
        prompt.append("- 示例代码（如果需要）\n");
        prompt.append("- 综合评分（0-100分）\n");
        
        return prompt.toString();
    }
    
    /**
     * 构建安全分析提示词
     */
    private String buildSecurityPrompt(String code) {
        return "请分析以下代码的安全性，识别潜在的安全漏洞：\n\n" +
               "```java\n" + code + "\n```\n\n" +
               "重点关注：\n" +
               "1. SQL注入风险\n" +
               "2. XSS攻击风险\n" +
               "3. 权限验证缺失\n" +
               "4. 敏感信息泄露\n" +
               "5. 加密问题\n" +
               "6. 输入验证不足\n\n" +
               "对每个问题提供：风险等级、具体位置、修复建议";
    }
    
    /**
     * 构建性能分析提示词
     */
    private String buildPerformancePrompt(String code) {
        return "请分析以下代码的性能问题和优化机会：\n\n" +
               "```java\n" + code + "\n```\n\n" +
               "重点关注：\n" +
               "1. 算法复杂度\n" +
               "2. 数据库查询优化\n" +
               "3. 内存使用\n" +
               "4. 并发性能\n" +
               "5. 缓存机会\n" +
               "6. 资源泄漏\n\n" +
               "对每个问题提供：影响程度、具体位置、优化方案";
    }
    
    /**
     * 获取分析系统提示词
     */
    private String getAnalysisSystemPrompt() {
        return "你是一个资深的代码审查专家，具备以下能力：\n" +
               "1. 深入理解代码设计原则和最佳实践\n" +
               "2. 识别潜在的Bug、性能问题和安全隐患\n" +
               "3. 提供具体、可操作的改进建议\n" +
               "4. 给出清晰的代码示例\n" +
               "请进行专业、全面、客观的代码分析。";
    }
    
    /**
     * 解析分析结果
     */
    private AnalysisResult parseAnalysisResult(String content) {
        AnalysisResult result = new AnalysisResult();
        result.setSuccess(true);
        result.setRawAnalysis(content);
        
        // 简单解析（实际应使用更复杂的解析逻辑）
        result.setSummary(extractSummary(content));
        result.setIssues(extractIssues(content));
        result.setSuggestions(extractSuggestions(content));
        result.setScore(extractScore(content));
        
        return result;
    }
    
    /**
     * 解析安全分析结果
     */
    private SecurityAnalysisResult parseSecurityResult(String content) {
        SecurityAnalysisResult result = new SecurityAnalysisResult();
        result.setRawAnalysis(content);
        result.setVulnerabilities(extractVulnerabilities(content));
        result.setRiskLevel(calculateRiskLevel(result.getVulnerabilities()));
        return result;
    }
    
    /**
     * 解析性能分析结果
     */
    private PerformanceAnalysisResult parsePerformanceResult(String content) {
        PerformanceAnalysisResult result = new PerformanceAnalysisResult();
        result.setRawAnalysis(content);
        result.setBottlenecks(extractBottlenecks(content));
        result.setOptimizations(extractOptimizations(content));
        return result;
    }
    
    /**
     * 转换为质量检查结果
     */
    private QualityCheckResult convertToQualityCheck(ToolResult toolResult) {
        QualityCheckResult result = new QualityCheckResult();
        
        if (toolResult.isSuccess()) {
            result.setPassed(true);
            result.setMessage(toolResult.getResult());
        } else {
            result.setPassed(false);
            result.setMessage(toolResult.getError());
        }
        
        return result;
    }
    
    // 辅助方法：提取摘要、问题、建议、评分等
    private String extractSummary(String content) {
        // TODO: 实现智能提取
        return content.length() > 200 ? content.substring(0, 200) + "..." : content;
    }
    
    private List<CodeIssue> extractIssues(String content) {
        // TODO: 实现智能提取
        return new ArrayList<>();
    }
    
    private List<String> extractSuggestions(String content) {
        // TODO: 实现智能提取
        return new ArrayList<>();
    }
    
    private int extractScore(String content) {
        // TODO: 实现智能提取
        return 75; // 默认评分
    }
    
    private List<SecurityVulnerability> extractVulnerabilities(String content) {
        // TODO: 实现智能提取
        return new ArrayList<>();
    }
    
    private String calculateRiskLevel(List<SecurityVulnerability> vulnerabilities) {
        if (vulnerabilities.isEmpty()) return "LOW";
        // TODO: 实现风险等级计算
        return "MEDIUM";
    }
    
    private List<PerformanceBottleneck> extractBottlenecks(String content) {
        // TODO: 实现智能提取
        return new ArrayList<>();
    }
    
    private List<PerformanceOptimization> extractOptimizations(String content) {
        // TODO: 实现智能提取
        return new ArrayList<>();
    }
    
    /**
     * 配置项
     */
    public void setAnalysisModel(String model) {
        this.analysisModel = model;
    }
    
    public void setAnalysisTemperature(double temperature) {
        this.analysisTemperature = temperature;
    }
    
    // ========== 内部类 ==========
    
    /**
     * 分析请求
     */
    public static class AnalysisRequest {
        private String code;
        private String language;
        private String projectId;
        private List<String> focusAreas;
        
        public AnalysisRequest(String code, String language, String projectId) {
            this.code = code;
            this.language = language;
            this.projectId = projectId;
            this.focusAreas = new ArrayList<>();
        }
        
        // Getters and Setters
        public String getCode() { return code; }
        public void setCode(String code) { this.code = code; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
        public List<String> getFocusAreas() { return focusAreas; }
        public void setFocusAreas(List<String> focusAreas) { this.focusAreas = focusAreas; }
    }
    
    /**
     * 分析结果
     */
    public static class AnalysisResult {
        private boolean success;
        private String summary;
        private List<CodeIssue> issues;
        private List<String> suggestions;
        private int score;
        private String rawAnalysis;
        private ToolResult toolAnalysis;
        private long analysisTime;
        private String errorMessage;
        
        public AnalysisResult() {
            this.issues = new ArrayList<>();
            this.suggestions = new ArrayList<>();
        }
        
        // Getters and Setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getSummary() { return summary; }
        public void setSummary(String summary) { this.summary = summary; }
        public List<CodeIssue> getIssues() { return issues; }
        public void setIssues(List<CodeIssue> issues) { this.issues = issues; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public int getScore() { return score; }
        public void setScore(int score) { this.score = score; }
        public String getRawAnalysis() { return rawAnalysis; }
        public void setRawAnalysis(String rawAnalysis) { this.rawAnalysis = rawAnalysis; }
        public ToolResult getToolAnalysis() { return toolAnalysis; }
        public void setToolAnalysis(ToolResult toolAnalysis) { this.toolAnalysis = toolAnalysis; }
        public long getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(long analysisTime) { this.analysisTime = analysisTime; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    }
    
    /**
     * 代码问题
     */
    public static class CodeIssue {
        private String type;
        private String severity;
        private String description;
        private int lineNumber;
        private String suggestion;
        
        // Getters and Setters
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getSeverity() { return severity; }
        public void setSeverity(String severity) { this.severity = severity; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public int getLineNumber() { return lineNumber; }
        public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
        public String getSuggestion() { return suggestion; }
        public void setSuggestion(String suggestion) { this.suggestion = suggestion; }
    }
    
    /**
     * 质量检查结果
     */
    public static class QualityCheckResult {
        private boolean passed;
        private String message;
        
        public boolean isPassed() { return passed; }
        public void setPassed(boolean passed) { this.passed = passed; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }
    
    /**
     * 安全分析结果
     */
    public static class SecurityAnalysisResult {
        private String rawAnalysis;
        private List<SecurityVulnerability> vulnerabilities;
        private String riskLevel;
        
        public SecurityAnalysisResult() {
            this.vulnerabilities = new ArrayList<>();
        }
        
        public String getRawAnalysis() { return rawAnalysis; }
        public void setRawAnalysis(String rawAnalysis) { this.rawAnalysis = rawAnalysis; }
        public List<SecurityVulnerability> getVulnerabilities() { return vulnerabilities; }
        public void setVulnerabilities(List<SecurityVulnerability> vulnerabilities) { 
            this.vulnerabilities = vulnerabilities; 
        }
        public String getRiskLevel() { return riskLevel; }
        public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }
    }
    
    /**
     * 安全漏洞
     */
    public static class SecurityVulnerability {
        private String type;
        private String severity;
        private String description;
        private String fixSuggestion;
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getSeverity() { return severity; }
        public void setSeverity(String severity) { this.severity = severity; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getFixSuggestion() { return fixSuggestion; }
        public void setFixSuggestion(String fixSuggestion) { this.fixSuggestion = fixSuggestion; }
    }
    
    /**
     * 性能分析结果
     */
    public static class PerformanceAnalysisResult {
        private String rawAnalysis;
        private List<PerformanceBottleneck> bottlenecks;
        private List<PerformanceOptimization> optimizations;
        
        public PerformanceAnalysisResult() {
            this.bottlenecks = new ArrayList<>();
            this.optimizations = new ArrayList<>();
        }
        
        public String getRawAnalysis() { return rawAnalysis; }
        public void setRawAnalysis(String rawAnalysis) { this.rawAnalysis = rawAnalysis; }
        public List<PerformanceBottleneck> getBottlenecks() { return bottlenecks; }
        public void setBottlenecks(List<PerformanceBottleneck> bottlenecks) { 
            this.bottlenecks = bottlenecks; 
        }
        public List<PerformanceOptimization> getOptimizations() { return optimizations; }
        public void setOptimizations(List<PerformanceOptimization> optimizations) { 
            this.optimizations = optimizations; 
        }
    }
    
    /**
     * 性能瓶颈
     */
    public static class PerformanceBottleneck {
        private String type;
        private String description;
        private String impact;
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getImpact() { return impact; }
        public void setImpact(String impact) { this.impact = impact; }
    }
    
    /**
     * 性能优化建议
     */
    public static class PerformanceOptimization {
        private String type;
        private String description;
        private String benefit;
        private String example;
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getBenefit() { return benefit; }
        public void setBenefit(String benefit) { this.benefit = benefit; }
        public String getExample() { return example; }
        public void setExample(String example) { this.example = example; }
    }
}
