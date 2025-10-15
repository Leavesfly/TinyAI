package io.leavesfly.tinyai.agent.cursor.v2.controller.dto;

import io.leavesfly.tinyai.agent.cursor.v2.component.analyzer.CodeAnalyzerV2;
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.service.SessionService;
import io.leavesfly.tinyai.agent.cursor.v2.tool.ToolOrchestrator;

import java.util.List;
import java.util.Map;

/**
 * API响应类集合
 * 
 * @author TinyAI  
 * @since 2.0.0
 */
public class ApiResponses {
    
    // ========== 分析相关响应 ==========
    
    public static class AnalysisResponse {
        private String sessionId;
        private boolean success;
        private String summary;
        private int score;
        private List<Object> issues;
        private List<String> suggestions;
        private long analysisTime;
        
        // Getters and Setters
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getSummary() { return summary; }
        public void setSummary(String summary) { this.summary = summary; }
        public int getScore() { return score; }
        public void setScore(int score) { this.score = score; }
        public List<Object> getIssues() { return issues; }
        public void setIssues(List<Object> issues) { this.issues = issues; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public long getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(long analysisTime) { this.analysisTime = analysisTime; }
    }
    
    public static class SecurityAnalysisResponse {
        private List<Object> vulnerabilities;
        private String riskLevel;
        private String rawAnalysis;
        
        public List<Object> getVulnerabilities() { return vulnerabilities; }
        public void setVulnerabilities(List<Object> vulnerabilities) { this.vulnerabilities = vulnerabilities; }
        public String getRiskLevel() { return riskLevel; }
        public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }
        public String getRawAnalysis() { return rawAnalysis; }
        public void setRawAnalysis(String rawAnalysis) { this.rawAnalysis = rawAnalysis; }
    }
    
    public static class PerformanceAnalysisResponse {
        private List<CodeAnalyzerV2.PerformanceBottleneck> bottlenecks;
        private List<CodeAnalyzerV2.PerformanceOptimization> optimizations;
        private String rawAnalysis;
        
        public List<CodeAnalyzerV2.PerformanceBottleneck> getBottlenecks() { return bottlenecks; }
        public void setBottlenecks(List<CodeAnalyzerV2.PerformanceBottleneck> bottlenecks) { 
            this.bottlenecks = bottlenecks; 
        }
        public List<CodeAnalyzerV2.PerformanceOptimization> getOptimizations() { return optimizations; }
        public void setOptimizations(List<CodeAnalyzerV2.PerformanceOptimization> optimizations) { 
            this.optimizations = optimizations; 
        }
        public String getRawAnalysis() { return rawAnalysis; }
        public void setRawAnalysis(String rawAnalysis) { this.rawAnalysis = rawAnalysis; }
    }
    
    // ========== 生成相关响应 ==========
    
    public static class CompletionResponse {
        private String sessionId;
        private boolean success;
        private String completionText;
        private long generationTime;
        
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getCompletionText() { return completionText; }
        public void setCompletionText(String completionText) { this.completionText = completionText; }
        public long getGenerationTime() { return generationTime; }
        public void setGenerationTime(long generationTime) { this.generationTime = generationTime; }
    }
    
    public static class FunctionGenerationResponse {
        private boolean success;
        private String functionCode;
        private String explanation;
        private long generationTime;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getFunctionCode() { return functionCode; }
        public void setFunctionCode(String functionCode) { this.functionCode = functionCode; }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
        public long getGenerationTime() { return generationTime; }
        public void setGenerationTime(long generationTime) { this.generationTime = generationTime; }
    }
    
    public static class TestGenerationResponse {
        private boolean success;
        private String testCode;
        private List<String> testCases;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getTestCode() { return testCode; }
        public void setTestCode(String testCode) { this.testCode = testCode; }
        public List<String> getTestCases() { return testCases; }
        public void setTestCases(List<String> testCases) { this.testCases = testCases; }
    }
    
    // ========== 重构相关响应 ==========
    
    public static class RefactorSuggestionsResponse {
        private List<String> suggestions;
        private int analysisScore;
        
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public int getAnalysisScore() { return analysisScore; }
        public void setAnalysisScore(int analysisScore) { this.analysisScore = analysisScore; }
    }
    
    public static class RefactorResponse {
        private boolean success;
        private String refactoredCode;
        private String explanation;
        private List<Object> changes;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getRefactoredCode() { return refactoredCode; }
        public void setRefactoredCode(String refactoredCode) { this.refactoredCode = refactoredCode; }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
        public List<Object> getChanges() { return changes; }
        public void setChanges(List<Object> changes) { this.changes = changes; }
    }
    
    // ========== 调试相关响应 ==========
    
    public static class DiagnosisResponse {
        private String problemDescription;
        private String rootCause;
        private List<String> fixSuggestions;
        private List<String> preventionTips;
        
        public String getProblemDescription() { return problemDescription; }
        public void setProblemDescription(String problemDescription) { this.problemDescription = problemDescription; }
        public String getRootCause() { return rootCause; }
        public void setRootCause(String rootCause) { this.rootCause = rootCause; }
        public List<String> getFixSuggestions() { return fixSuggestions; }
        public void setFixSuggestions(List<String> fixSuggestions) { this.fixSuggestions = fixSuggestions; }
        public List<String> getPreventionTips() { return preventionTips; }
        public void setPreventionTips(List<String> preventionTips) { this.preventionTips = preventionTips; }
    }
    
    public static class FixSuggestionResponse {
        private String rootCause;
        private List<String> fixSuggestions;
        private String fixedCode;
        
        public String getRootCause() { return rootCause; }
        public void setRootCause(String rootCause) { this.rootCause = rootCause; }
        public List<String> getFixSuggestions() { return fixSuggestions; }
        public void setFixSuggestions(List<String> fixSuggestions) { this.fixSuggestions = fixSuggestions; }
        public String getFixedCode() { return fixedCode; }
        public void setFixedCode(String fixedCode) { this.fixedCode = fixedCode; }
    }
    
    // ========== 会话相关响应 ==========
    
    public static class SessionResponse {
        private String sessionId;
        private long createdAt;
        
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public long getCreatedAt() { return createdAt; }
        public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }
    }
    
    public static class SessionInfoResponse {
        private String sessionId;
        private String userId;
        private String projectId;
        private long createdAt;
        private long lastAccessAt;
        private int messageCount;
        
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
        public long getCreatedAt() { return createdAt; }
        public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }
        public long getLastAccessAt() { return lastAccessAt; }
        public void setLastAccessAt(long lastAccessAt) { this.lastAccessAt = lastAccessAt; }
        public int getMessageCount() { return messageCount; }
        public void setMessageCount(int messageCount) { this.messageCount = messageCount; }
    }
    
    // ========== 统计相关响应 ==========
    
    public static class StatsResponse {
        private MemoryManager.MemoryStats memoryStats;
        private RAGEngine.RAGStats ragStats;
        private ToolOrchestrator.OrchestratorStats toolStats;
        private SessionService.SessionStats sessionStats;
        
        public MemoryManager.MemoryStats getMemoryStats() { return memoryStats; }
        public void setMemoryStats(MemoryManager.MemoryStats memoryStats) { this.memoryStats = memoryStats; }
        public RAGEngine.RAGStats getRagStats() { return ragStats; }
        public void setRagStats(RAGEngine.RAGStats ragStats) { this.ragStats = ragStats; }
        public ToolOrchestrator.OrchestratorStats getToolStats() { return toolStats; }
        public void setToolStats(ToolOrchestrator.OrchestratorStats toolStats) { this.toolStats = toolStats; }
        public SessionService.SessionStats getSessionStats() { return sessionStats; }
        public void setSessionStats(SessionService.SessionStats sessionStats) { this.sessionStats = sessionStats; }
    }
    
    // ========== API请求类 ==========
    
    public static class SecurityAnalysisApiRequest {
        private String code;
        private String projectId;
        
        public String getCode() { return code; }
        public void setCode(String code) { this.code = code; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class PerformanceAnalysisApiRequest {
        private String code;
        private String projectId;
        
        public String getCode() { return code; }
        public void setCode(String code) { this.code = code; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class CompletionApiRequest {
        private String userId;
        private String projectId;
        private String filePath;
        private String prefix;
        private String suffix;
        private String language;
        private int line;
        private int column;
        
        // Getters and Setters
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        public String getPrefix() { return prefix; }
        public void setPrefix(String prefix) { this.prefix = prefix; }
        public String getSuffix() { return suffix; }
        public void setSuffix(String suffix) { this.suffix = suffix; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public int getLine() { return line; }
        public void setLine(int line) { this.line = line; }
        public int getColumn() { return column; }
        public void setColumn(int column) { this.column = column; }
    }
    
    public static class FunctionGenerationApiRequest {
        private String projectId;
        private String functionName;
        private String description;
        private Map<String, String> parameters;
        private String returnType;
        private String language;
        
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
        public String getFunctionName() { return functionName; }
        public void setFunctionName(String functionName) { this.functionName = functionName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Map<String, String> getParameters() { return parameters; }
        public void setParameters(Map<String, String> parameters) { this.parameters = parameters; }
        public String getReturnType() { return returnType; }
        public void setReturnType(String returnType) { this.returnType = returnType; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
    }
    
    public static class TestGenerationApiRequest {
        private String sourceCode;
        private String className;
        private String projectId;
        
        public String getSourceCode() { return sourceCode; }
        public void setSourceCode(String sourceCode) { this.sourceCode = sourceCode; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class RefactorSuggestionsApiRequest {
        private String code;
        private String projectId;
        
        public String getCode() { return code; }
        public void setCode(String code) { this.code = code; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class RefactorApiRequest {
        private String originalCode;
        private String refactorType;
        private String targetPattern;
        private String projectId;
        
        public String getOriginalCode() { return originalCode; }
        public void setOriginalCode(String originalCode) { this.originalCode = originalCode; }
        public String getRefactorType() { return refactorType; }
        public void setRefactorType(String refactorType) { this.refactorType = refactorType; }
        public String getTargetPattern() { return targetPattern; }
        public void setTargetPattern(String targetPattern) { this.targetPattern = targetPattern; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class DiagnosisApiRequest {
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
    
    public static class FixSuggestionApiRequest {
        private String code;
        private String errorMessage;
        private String projectId;
        
        public String getCode() { return code; }
        public void setCode(String code) { this.code = code; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
    
    public static class SessionCreateRequest {
        private String userId;
        private String projectId;
        
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
    }
}
