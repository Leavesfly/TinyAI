package io.leavesfly.tinyai.agent.cursor.v2.component.debug;

import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.LLMGateway;

import java.util.*;

/**
 * 智能调试助手（V2）
 * 基于LLM的智能代码调试，提供问题诊断和修复建议
 * 
 * 核心功能：
 * 1. 错误诊断（分析错误信息和堆栈）
 * 2. 修复建议（提供具体的修复方案）
 * 3. 根因分析（深入分析问题根源）
 * 4. 预防建议（提供预防类似问题的建议）
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class DebugAgentV2 {
    
    private final LLMGateway llmGateway;
    private final ContextEngine contextEngine;
    
    private String debugModel = "deepseek-chat";
    
    public DebugAgentV2(LLMGateway llmGateway, ContextEngine contextEngine) {
        this.llmGateway = llmGateway;
        this.contextEngine = contextEngine;
    }
    
    /**
     * 诊断错误
     */
    public DiagnosisResult diagnose(DiagnosisRequest request) {
        Context context = contextEngine.buildAnalysisContext(
            request.getProjectId(),
            request.getCode(),
            "debug"
        );
        
        String prompt = buildDiagnosisPrompt(request);
        
        ChatRequest chatRequest = ChatRequest.builder()
            .model(debugModel)
            .addSystemMessage(context.buildSystemPrompt())
            .addSystemMessage("你是一个调试专家，擅长分析和解决各种代码问题。")
            .addUserMessage(prompt)
            .temperature(0.3)
            .maxTokens(1500)
            .build();
        
        ChatResponse response = llmGateway.chat(chatRequest);
        
        return parseDiagnosisResult(response.getContent(), request);
    }
    
    /**
     * 提供修复建议
     */
    public FixSuggestionResult suggestFix(String code, String errorMessage, String projectId) {
        String prompt = String.format(
            "代码：\n```java\n%s\n```\n\n错误信息：\n%s\n\n请提供：\n1. 问题原因\n2. 修复方案\n3. 修复后的代码",
            code, errorMessage
        );
        
        ChatRequest request = ChatRequest.builder()
            .model(debugModel)
            .addSystemMessage("你是一个代码修复专家。")
            .addUserMessage(prompt)
            .temperature(0.2)
            .maxTokens(1000)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        
        FixSuggestionResult result = new FixSuggestionResult();
        result.setRootCause(extractRootCause(response.getContent()));
        result.setFixSuggestions(extractFixSuggestions(response.getContent()));
        result.setFixedCode(extractCode(response.getContent()));
        
        return result;
    }
    
    private String buildDiagnosisPrompt(DiagnosisRequest request) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("请诊断以下问题：\n\n");
        prompt.append("代码：\n```java\n").append(request.getCode()).append("\n```\n\n");
        
        if (request.getErrorMessage() != null) {
            prompt.append("错误信息：\n").append(request.getErrorMessage()).append("\n\n");
        }
        
        if (request.getStackTrace() != null) {
            prompt.append("堆栈跟踪：\n").append(request.getStackTrace()).append("\n\n");
        }
        
        prompt.append("请提供详细的诊断分析和解决方案。");
        
        return prompt.toString();
    }
    
    private DiagnosisResult parseDiagnosisResult(String content, DiagnosisRequest request) {
        DiagnosisResult result = new DiagnosisResult();
        result.setProblemDescription(extractProblemDescription(content));
        result.setRootCause(extractRootCause(content));
        result.setFixSuggestions(extractFixSuggestions(content));
        result.setPreventionTips(extractPreventionTips(content));
        return result;
    }
    
    private String extractCode(String content) {
        int start = content.indexOf("```");
        if (start == -1) return "";
        start = content.indexOf("\n", start) + 1;
        int end = content.indexOf("```", start);
        return end == -1 ? "" : content.substring(start, end).trim();
    }
    
    private String extractProblemDescription(String content) {
        return content.length() > 200 ? content.substring(0, 200) : content;
    }
    
    private String extractRootCause(String content) {
        return ""; // TODO: 实现智能提取
    }
    
    private List<String> extractFixSuggestions(String content) {
        return new ArrayList<>(); // TODO: 实现智能提取
    }
    
    private List<String> extractPreventionTips(String content) {
        return new ArrayList<>(); // TODO: 实现智能提取
    }
    
    public void setDebugModel(String model) {
        this.debugModel = model;
    }
    
    // ========== 内部类 ==========
    
    public static class DiagnosisRequest {
        private String code;
        private String errorMessage;
        private String stackTrace;
        private String projectId;
        
        public String getCode() { return code; }
        public void setCode(String code) { this.code = code; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        public String getStackTrace() { return stackTrace; }
        public void setStackTrace(String stackTrace) { this.stackTrace = stackTrace; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class DiagnosisResult {
        private String problemDescription;
        private String rootCause;
        private List<String> fixSuggestions;
        private List<String> preventionTips;
        
        public DiagnosisResult() {
            this.fixSuggestions = new ArrayList<>();
            this.preventionTips = new ArrayList<>();
        }
        
        public String getProblemDescription() { return problemDescription; }
        public void setProblemDescription(String description) { this.problemDescription = description; }
        public String getRootCause() { return rootCause; }
        public void setRootCause(String cause) { this.rootCause = cause; }
        public List<String> getFixSuggestions() { return fixSuggestions; }
        public void setFixSuggestions(List<String> suggestions) { this.fixSuggestions = suggestions; }
        public List<String> getPreventionTips() { return preventionTips; }
        public void setPreventionTips(List<String> tips) { this.preventionTips = tips; }
    }
    
    public static class FixSuggestionResult {
        private String rootCause;
        private List<String> fixSuggestions;
        private String fixedCode;
        
        public FixSuggestionResult() {
            this.fixSuggestions = new ArrayList<>();
        }
        
        public String getRootCause() { return rootCause; }
        public void setRootCause(String cause) { this.rootCause = cause; }
        public List<String> getFixSuggestions() { return fixSuggestions; }
        public void setFixSuggestions(List<String> suggestions) { this.fixSuggestions = suggestions; }
        public String getFixedCode() { return fixedCode; }
        public void setFixedCode(String code) { this.fixedCode = code; }
    }
}
