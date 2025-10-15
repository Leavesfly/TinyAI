package io.leavesfly.tinyai.agent.cursor.v2.component.refactor;

import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.analyzer.CodeAnalyzerV2;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.LLMGateway;

import java.util.*;

/**
 * 智能重构助手（V2）
 * 基于LLM的智能代码重构，提供多种重构策略
 * 
 * 核心功能：
 * 1. 自动重构建议（识别重构机会）
 * 2. 代码优化（性能优化、可读性优化）
 * 3. 设计模式应用（识别并应用设计模式）
 * 4. 代码坏味道修复（消除代码坏味道）
 * 5. 批量重构（多文件重构）
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class RefactorAgentV2 {
    
    private final LLMGateway llmGateway;
    private final ContextEngine contextEngine;
    private final CodeAnalyzerV2 codeAnalyzer;
    
    private String refactorModel = "deepseek-chat";
    private double refactorTemperature = 0.3;
    
    public RefactorAgentV2(LLMGateway llmGateway, ContextEngine contextEngine,
                          CodeAnalyzerV2 codeAnalyzer) {
        this.llmGateway = llmGateway;
        this.contextEngine = contextEngine;
        this.codeAnalyzer = codeAnalyzer;
    }
    
    /**
     * 分析重构机会
     */
    public RefactorSuggestionsResult suggestRefactorings(String code, String projectId) {
        // 先分析代码
        CodeAnalyzerV2.AnalysisRequest analysisRequest = 
            new CodeAnalyzerV2.AnalysisRequest(code, "java", projectId);
        CodeAnalyzerV2.AnalysisResult analysis = codeAnalyzer.analyze(analysisRequest);
        
        // 构建重构建议提示词
        String prompt = String.format(
            "基于以下代码分析结果，提供重构建议：\n\n%s\n\n代码：\n```java\n%s\n```\n\n" +
            "请提供：\n1. 重构优先级\n2. 具体重构方案\n3. 预期收益",
            analysis.getSummary(), code
        );
        
        ChatRequest request = ChatRequest.builder()
            .model(refactorModel)
            .addSystemMessage("你是一个代码重构专家，擅长识别重构机会并提供可行方案。")
            .addUserMessage(prompt)
            .temperature(refactorTemperature)
            .maxTokens(1500)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        
        RefactorSuggestionsResult result = new RefactorSuggestionsResult();
        result.setSuggestions(parseSuggestions(response.getContent()));
        result.setAnalysisResult(analysis);
        
        return result;
    }
    
    /**
     * 执行重构
     */
    public RefactorResult refactor(RefactorRequest refactorRequest) {
        String prompt = buildRefactorPrompt(refactorRequest);
        
        ChatRequest request = ChatRequest.builder()
            .model(refactorModel)
            .addSystemMessage(getRefactorSystemPrompt())
            .addUserMessage(prompt)
            .temperature(0.2)
            .maxTokens(2000)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        String refactoredCode = extractCode(response.getContent());
        
        RefactorResult result = new RefactorResult();
        result.setSuccess(true);
        result.setRefactoredCode(refactoredCode);
        result.setExplanation(response.getContent());
        result.setChanges(compareCode(refactorRequest.getOriginalCode(), refactoredCode));
        
        return result;
    }
    
    /**
     * 应用设计模式
     */
    public DesignPatternResult applyDesignPattern(String code, String patternName, String projectId) {
        String prompt = String.format(
            "请将以下代码重构为%s设计模式：\n\n```java\n%s\n```\n\n" +
            "请提供：\n1. 重构后的完整代码\n2. 模式说明\n3. UML类图（文字描述）",
            patternName, code
        );
        
        ChatRequest request = ChatRequest.builder()
            .model(refactorModel)
            .addSystemMessage("你是一个设计模式专家，精通各种设计模式的应用。")
            .addUserMessage(prompt)
            .temperature(0.3)
            .maxTokens(2000)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        
        DesignPatternResult result = new DesignPatternResult();
        result.setRefactoredCode(extractCode(response.getContent()));
        result.setPatternExplanation(response.getContent());
        
        return result;
    }
    
    /**
     * 优化性能
     */
    public PerformanceOptimizationResult optimizePerformance(String code, String projectId) {
        // 先进行性能分析
        CodeAnalyzerV2.PerformanceAnalysisResult perfAnalysis = 
            codeAnalyzer.analyzePerformance(code, projectId);
        
        String prompt = String.format(
            "请优化以下代码的性能：\n\n```java\n%s\n```\n\n" +
            "性能分析结果：\n%s\n\n请提供优化后的代码和性能改进说明。",
            code, perfAnalysis.getRawAnalysis()
        );
        
        ChatRequest request = ChatRequest.builder()
            .model(refactorModel)
            .addSystemMessage("你是一个性能优化专家。")
            .addUserMessage(prompt)
            .temperature(0.2)
            .maxTokens(1500)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        
        PerformanceOptimizationResult result = new PerformanceOptimizationResult();
        result.setOptimizedCode(extractCode(response.getContent()));
        result.setOptimizations(perfAnalysis.getOptimizations());
        result.setExplanation(response.getContent());
        
        return result;
    }
    
    // ========== 私有方法 ==========
    
    private String buildRefactorPrompt(RefactorRequest request) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("请重构以下代码：\n\n");
        prompt.append("```java\n").append(request.getOriginalCode()).append("\n```\n\n");
        prompt.append("重构类型: ").append(request.getRefactorType()).append("\n");
        
        if (request.getTargetPattern() != null) {
            prompt.append("目标模式: ").append(request.getTargetPattern()).append("\n");
        }
        
        prompt.append("\n请提供：\n1. 重构后的代码\n2. 重构说明\n3. 注意事项");
        
        return prompt.toString();
    }
    
    private String getRefactorSystemPrompt() {
        return "你是一个代码重构专家，精通各种重构技术和最佳实践。" +
               "请确保重构后的代码：\n" +
               "1. 功能完全等价\n" +
               "2. 更易读易维护\n" +
               "3. 遵循最佳实践\n" +
               "4. 包含必要注释";
    }
    
    private String extractCode(String response) {
        int start = response.indexOf("```");
        if (start == -1) return response.trim();
        
        start = response.indexOf("\n", start) + 1;
        int end = response.indexOf("```", start);
        
        return end == -1 ? response.trim() : response.substring(start, end).trim();
    }
    
    private List<String> parseSuggestions(String content) {
        // TODO: 实现智能解析
        return Arrays.asList(content.split("\n\n"));
    }
    
    private List<CodeChange> compareCode(String original, String refactored) {
        // TODO: 实现代码差异比较
        List<CodeChange> changes = new ArrayList<>();
        CodeChange change = new CodeChange();
        change.setType("REFACTORED");
        change.setDescription("代码已重构");
        changes.add(change);
        return changes;
    }
    
    public void setRefactorModel(String model) {
        this.refactorModel = model;
    }
    
    public void setRefactorTemperature(double temperature) {
        this.refactorTemperature = temperature;
    }
    
    // ========== 内部类 ==========
    
    public static class RefactorRequest {
        private String originalCode;
        private String refactorType;
        private String targetPattern;
        private String projectId;
        
        public String getOriginalCode() { return originalCode; }
        public void setOriginalCode(String code) { this.originalCode = code; }
        public String getRefactorType() { return refactorType; }
        public void setRefactorType(String type) { this.refactorType = type; }
        public String getTargetPattern() { return targetPattern; }
        public void setTargetPattern(String pattern) { this.targetPattern = pattern; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class RefactorResult {
        private boolean success;
        private String refactoredCode;
        private String explanation;
        private List<CodeChange> changes;
        
        public RefactorResult() {
            this.changes = new ArrayList<>();
        }
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getRefactoredCode() { return refactoredCode; }
        public void setRefactoredCode(String code) { this.refactoredCode = code; }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
        public List<CodeChange> getChanges() { return changes; }
        public void setChanges(List<CodeChange> changes) { this.changes = changes; }
    }
    
    public static class CodeChange {
        private String type;
        private String description;
        private int lineNumber;
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public int getLineNumber() { return lineNumber; }
        public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    }
    
    public static class RefactorSuggestionsResult {
        private List<String> suggestions;
        private CodeAnalyzerV2.AnalysisResult analysisResult;
        
        public RefactorSuggestionsResult() {
            this.suggestions = new ArrayList<>();
        }
        
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public CodeAnalyzerV2.AnalysisResult getAnalysisResult() { return analysisResult; }
        public void setAnalysisResult(CodeAnalyzerV2.AnalysisResult result) { 
            this.analysisResult = result; 
        }
    }
    
    public static class DesignPatternResult {
        private String refactoredCode;
        private String patternExplanation;
        
        public String getRefactoredCode() { return refactoredCode; }
        public void setRefactoredCode(String code) { this.refactoredCode = code; }
        public String getPatternExplanation() { return patternExplanation; }
        public void setPatternExplanation(String explanation) { 
            this.patternExplanation = explanation; 
        }
    }
    
    public static class PerformanceOptimizationResult {
        private String optimizedCode;
        private List<CodeAnalyzerV2.PerformanceOptimization> optimizations;
        private String explanation;
        
        public PerformanceOptimizationResult() {
            this.optimizations = new ArrayList<>();
        }
        
        public String getOptimizedCode() { return optimizedCode; }
        public void setOptimizedCode(String code) { this.optimizedCode = code; }
        public List<CodeAnalyzerV2.PerformanceOptimization> getOptimizations() { 
            return optimizations; 
        }
        public void setOptimizations(List<CodeAnalyzerV2.PerformanceOptimization> optimizations) { 
            this.optimizations = optimizations; 
        }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
    }
}
