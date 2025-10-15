package io.leavesfly.tinyai.agent.cursor.v2.controller;

import io.leavesfly.tinyai.agent.cursor.v2.component.analyzer.CodeAnalyzerV2;
import io.leavesfly.tinyai.agent.cursor.v2.component.debug.DebugAgentV2;
import io.leavesfly.tinyai.agent.cursor.v2.component.generator.CodeGeneratorV2;
import io.leavesfly.tinyai.agent.cursor.v2.component.refactor.RefactorAgentV2;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.CodeIntelligenceService;
import io.leavesfly.tinyai.agent.cursor.v2.service.SessionService;
import io.leavesfly.tinyai.agent.cursor.v2.controller.dto.ApiResponses.*;

import java.util.*;

/**
 * Cursor V2 统一控制器
 * 提供RESTful API接口，对外暴露所有AI编程助手功能
 * 
 * 核心功能：
 * 1. 代码分析API
 * 2. 代码生成API
 * 3. 代码重构API
 * 4. 代码调试API
 * 5. 会话管理API
 * 6. 流式响应支持
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class CursorV2Controller {
    
    /**
     * 代码智能服务
     */
    private final CodeIntelligenceService intelligenceService;
    
    /**
     * 会话服务
     */
    private final SessionService sessionService;
    
    public CursorV2Controller(CodeIntelligenceService intelligenceService,
                             SessionService sessionService) {
        this.intelligenceService = intelligenceService;
        this.sessionService = sessionService;
    }
    
    // ========== 代码分析API ==========
    
    /**
     * POST /api/v2/analyze
     * 综合代码分析
     */
    public ApiResponse<AnalysisResponse> analyzeCode(AnalysisApiRequest request) {
        try {
            // 验证请求
            validateRequest(request);
            
            // 获取或创建会话
            String sessionId = getOrCreateSession(request.getUserId(), request.getProjectId());
            
            // 执行分析
            CodeAnalyzerV2.AnalysisRequest analysisRequest = new CodeAnalyzerV2.AnalysisRequest(
                request.getCode(),
                request.getLanguage(),
                request.getProjectId()
            );
            
            CodeAnalyzerV2.AnalysisResult result = 
                intelligenceService.getCodeAnalyzer().analyze(analysisRequest);
            
            // 保存到会话历史
            sessionService.addAnalysisHistory(sessionId, request, result);
            
            // 构建响应
            AnalysisResponse response = new AnalysisResponse();
            response.setSessionId(sessionId);
            response.setSuccess(result.isSuccess());
            response.setSummary(result.getSummary());
            response.setScore(result.getScore());
            response.setIssues(convertIssues(result.getIssues()));
            response.setSuggestions(result.getSuggestions());
            response.setAnalysisTime(result.getAnalysisTime());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("代码分析失败: " + e.getMessage());
        }
    }
    
    /**
     * POST /api/v2/analyze/security
     * 安全性分析
     */
    public ApiResponse<SecurityAnalysisResponse> analyzeSecurity(SecurityAnalysisApiRequest request) {
        try {
            validateRequest(request);
            
            CodeAnalyzerV2.SecurityAnalysisResult result = 
                intelligenceService.getCodeAnalyzer().analyzeSecurity(
                    request.getCode(),
                    request.getProjectId()
                );
            
            SecurityAnalysisResponse response = new SecurityAnalysisResponse();
            response.setVulnerabilities(convertVulnerabilities(result.getVulnerabilities()));
            response.setRiskLevel(result.getRiskLevel());
            response.setRawAnalysis(result.getRawAnalysis());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("安全分析失败: " + e.getMessage());
        }
    }
    
    /**
     * POST /api/v2/analyze/performance
     * 性能分析
     */
    public ApiResponse<PerformanceAnalysisResponse> analyzePerformance(
            PerformanceAnalysisApiRequest request) {
        try {
            validateRequest(request);
            
            CodeAnalyzerV2.PerformanceAnalysisResult result = 
                intelligenceService.getCodeAnalyzer().analyzePerformance(
                    request.getCode(),
                    request.getProjectId()
                );
            
            PerformanceAnalysisResponse response = new PerformanceAnalysisResponse();
            response.setBottlenecks(result.getBottlenecks());
            response.setOptimizations(result.getOptimizations());
            response.setRawAnalysis(result.getRawAnalysis());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("性能分析失败: " + e.getMessage());
        }
    }
    
    // ========== 代码生成API ==========
    
    /**
     * POST /api/v2/generate/complete
     * 代码补全
     */
    public ApiResponse<CompletionResponse> completeCode(CompletionApiRequest request) {
        try {
            validateRequest(request);
            
            String sessionId = getOrCreateSession(request.getUserId(), request.getProjectId());
            
            CodeGeneratorV2.CompletionRequest completionRequest = 
                new CodeGeneratorV2.CompletionRequest();
            completionRequest.setProjectId(request.getProjectId());
            completionRequest.setFilePath(request.getFilePath());
            completionRequest.setPrefix(request.getPrefix());
            completionRequest.setSuffix(request.getSuffix());
            completionRequest.setLanguage(request.getLanguage());
            completionRequest.setLine(request.getLine());
            completionRequest.setColumn(request.getColumn());
            
            CodeGeneratorV2.CompletionResult result = 
                intelligenceService.getCodeGenerator().complete(completionRequest);
            
            CompletionResponse response = new CompletionResponse();
            response.setSessionId(sessionId);
            response.setSuccess(result.isSuccess());
            response.setCompletionText(result.getCompletionText());
            response.setGenerationTime(result.getGenerationTime());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("代码补全失败: " + e.getMessage());
        }
    }
    
    /**
     * POST /api/v2/generate/function
     * 生成函数
     */
    public ApiResponse<FunctionGenerationResponse> generateFunction(
            FunctionGenerationApiRequest request) {
        try {
            validateRequest(request);
            
            CodeGeneratorV2.FunctionGenerationRequest funcRequest = 
                new CodeGeneratorV2.FunctionGenerationRequest();
            funcRequest.setProjectId(request.getProjectId());
            funcRequest.setFunctionName(request.getFunctionName());
            funcRequest.setDescription(request.getDescription());
            funcRequest.setParameters(request.getParameters());
            funcRequest.setReturnType(request.getReturnType());
            funcRequest.setLanguage(request.getLanguage());
            
            CodeGeneratorV2.FunctionGenerationResult result = 
                intelligenceService.getCodeGenerator().generateFunction(funcRequest);
            
            FunctionGenerationResponse response = new FunctionGenerationResponse();
            response.setSuccess(result.isSuccess());
            response.setFunctionCode(result.getFunctionCode());
            response.setExplanation(result.getExplanation());
            response.setGenerationTime(result.getGenerationTime());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("函数生成失败: " + e.getMessage());
        }
    }
    
    /**
     * POST /api/v2/generate/test
     * 生成测试代码
     */
    public ApiResponse<TestGenerationResponse> generateTest(TestGenerationApiRequest request) {
        try {
            validateRequest(request);
            
            CodeGeneratorV2.TestGenerationResult result = 
                intelligenceService.getCodeGenerator().generateTest(
                    request.getSourceCode(),
                    request.getClassName(),
                    request.getProjectId()
                );
            
            TestGenerationResponse response = new TestGenerationResponse();
            response.setSuccess(result.isSuccess());
            response.setTestCode(result.getTestCode());
            response.setTestCases(result.getTestCases());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("测试生成失败: " + e.getMessage());
        }
    }
    
    // ========== 代码重构API ==========
    
    /**
     * POST /api/v2/refactor/suggest
     * 获取重构建议
     */
    public ApiResponse<RefactorSuggestionsResponse> suggestRefactorings(
            RefactorSuggestionsApiRequest request) {
        try {
            validateRequest(request);
            
            RefactorAgentV2.RefactorSuggestionsResult result = 
                intelligenceService.getRefactorAgent().suggestRefactorings(
                    request.getCode(),
                    request.getProjectId()
                );
            
            RefactorSuggestionsResponse response = new RefactorSuggestionsResponse();
            response.setSuggestions(result.getSuggestions());
            response.setAnalysisScore(result.getAnalysisResult().getScore());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("重构建议失败: " + e.getMessage());
        }
    }
    
    /**
     * POST /api/v2/refactor/execute
     * 执行重构
     */
    public ApiResponse<RefactorResponse> executeRefactor(RefactorApiRequest request) {
        try {
            validateRequest(request);
            
            RefactorAgentV2.RefactorRequest refactorRequest = 
                new RefactorAgentV2.RefactorRequest();
            refactorRequest.setOriginalCode(request.getOriginalCode());
            refactorRequest.setRefactorType(request.getRefactorType());
            refactorRequest.setTargetPattern(request.getTargetPattern());
            refactorRequest.setProjectId(request.getProjectId());
            
            RefactorAgentV2.RefactorResult result = 
                intelligenceService.getRefactorAgent().refactor(refactorRequest);
            
            RefactorResponse response = new RefactorResponse();
            response.setSuccess(result.isSuccess());
            response.setRefactoredCode(result.getRefactoredCode());
            response.setExplanation(result.getExplanation());
            response.setChanges(convertChanges(result.getChanges()));
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("重构执行失败: " + e.getMessage());
        }
    }
    
    // ========== 代码调试API ==========
    
    /**
     * POST /api/v2/debug/diagnose
     * 诊断错误
     */
    public ApiResponse<DiagnosisResponse> diagnoseError(DiagnosisApiRequest request) {
        try {
            validateRequest(request);
            
            DebugAgentV2.DiagnosisRequest diagRequest = new DebugAgentV2.DiagnosisRequest();
            diagRequest.setCode(request.getCode());
            diagRequest.setErrorMessage(request.getErrorMessage());
            diagRequest.setStackTrace(request.getStackTrace());
            diagRequest.setProjectId(request.getProjectId());
            
            DebugAgentV2.DiagnosisResult result = 
                intelligenceService.getDebugAgent().diagnose(diagRequest);
            
            DiagnosisResponse response = new DiagnosisResponse();
            response.setProblemDescription(result.getProblemDescription());
            response.setRootCause(result.getRootCause());
            response.setFixSuggestions(result.getFixSuggestions());
            response.setPreventionTips(result.getPreventionTips());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("错误诊断失败: " + e.getMessage());
        }
    }
    
    /**
     * POST /api/v2/debug/fix
     * 获取修复建议
     */
    public ApiResponse<FixSuggestionResponse> suggestFix(FixSuggestionApiRequest request) {
        try {
            validateRequest(request);
            
            DebugAgentV2.FixSuggestionResult result = 
                intelligenceService.getDebugAgent().suggestFix(
                    request.getCode(),
                    request.getErrorMessage(),
                    request.getProjectId()
                );
            
            FixSuggestionResponse response = new FixSuggestionResponse();
            response.setRootCause(result.getRootCause());
            response.setFixSuggestions(result.getFixSuggestions());
            response.setFixedCode(result.getFixedCode());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("修复建议失败: " + e.getMessage());
        }
    }
    
    // ========== 会话管理API ==========
    
    /**
     * POST /api/v2/session/create
     * 创建会话
     */
    public ApiResponse<SessionResponse> createSession(SessionCreateRequest request) {
        try {
            validateRequest(request);
            
            String sessionId = sessionService.createSession(
                request.getUserId(),
                request.getProjectId()
            );
            
            SessionResponse response = new SessionResponse();
            response.setSessionId(sessionId);
            response.setCreatedAt(System.currentTimeMillis());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("会话创建失败: " + e.getMessage());
        }
    }
    
    /**
     * GET /api/v2/session/{sessionId}
     * 获取会话信息
     */
    public ApiResponse<SessionInfoResponse> getSession(String sessionId) {
        try {
            SessionService.SessionInfo info = sessionService.getSessionInfo(sessionId);
            
            if (info == null) {
                return ApiResponse.error("会话不存在");
            }
            
            SessionInfoResponse response = new SessionInfoResponse();
            response.setSessionId(info.getSessionId());
            response.setUserId(info.getUserId());
            response.setProjectId(info.getProjectId());
            response.setCreatedAt(info.getCreatedAt());
            response.setLastAccessAt(info.getLastAccessAt());
            response.setMessageCount(info.getMessageCount());
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("获取会话失败: " + e.getMessage());
        }
    }
    
    /**
     * DELETE /api/v2/session/{sessionId}
     * 删除会话
     */
    public ApiResponse<Void> deleteSession(String sessionId) {
        try {
            sessionService.deleteSession(sessionId);
            return ApiResponse.success(null);
        } catch (Exception e) {
            return ApiResponse.error("删除会话失败: " + e.getMessage());
        }
    }
    
    /**
     * GET /api/v2/stats
     * 获取统计信息
     */
    public ApiResponse<StatsResponse> getStats() {
        try {
            CodeIntelligenceService.ServiceStats serviceStats = 
                intelligenceService.getStats();
            SessionService.SessionStats sessionStats = 
                sessionService.getStats();
            
            StatsResponse response = new StatsResponse();
            response.setMemoryStats(serviceStats.memoryStats);
            response.setRagStats(serviceStats.ragStats);
            response.setToolStats(serviceStats.toolStats);
            response.setSessionStats(sessionStats);
            
            return ApiResponse.success(response);
            
        } catch (Exception e) {
            return ApiResponse.error("获取统计失败: " + e.getMessage());
        }
    }
    
    // ========== 辅助方法 ==========
    
    private void validateRequest(Object request) {
        if (request == null) {
            throw new IllegalArgumentException("请求不能为空");
        }
        // TODO: 实现更详细的请求验证
    }
    
    private String getOrCreateSession(String userId, String projectId) {
        // 尝试获取现有会话
        String sessionId = sessionService.findActiveSession(userId, projectId);
        
        if (sessionId == null) {
            // 创建新会话
            sessionId = sessionService.createSession(userId, projectId);
        }
        
        return sessionId;
    }
    
    private List<Object> convertIssues(List<CodeAnalyzerV2.CodeIssue> issues) {
        List<Object> dtos = new ArrayList<>();
        for (CodeAnalyzerV2.CodeIssue issue : issues) {
            IssueDto dto = new IssueDto();
            dto.setType(issue.getType());
            dto.setSeverity(issue.getSeverity());
            dto.setDescription(issue.getDescription());
            dto.setLineNumber(issue.getLineNumber());
            dto.setSuggestion(issue.getSuggestion());
            dtos.add(dto);
        }
        return dtos;
    }
    
    private List<Object> convertVulnerabilities(
            List<CodeAnalyzerV2.SecurityVulnerability> vulnerabilities) {
        List<Object> dtos = new ArrayList<>();
        for (CodeAnalyzerV2.SecurityVulnerability vuln : vulnerabilities) {
            VulnerabilityDto dto = new VulnerabilityDto();
            dto.setType(vuln.getType());
            dto.setSeverity(vuln.getSeverity());
            dto.setDescription(vuln.getDescription());
            dto.setFixSuggestion(vuln.getFixSuggestion());
            dtos.add(dto);
        }
        return dtos;
    }
    
    private List<Object> convertChanges(List<RefactorAgentV2.CodeChange> changes) {
        List<Object> dtos = new ArrayList<>();
        for (RefactorAgentV2.CodeChange change : changes) {
            ChangeDto dto = new ChangeDto();
            dto.setType(change.getType());
            dto.setDescription(change.getDescription());
            dto.setLineNumber(change.getLineNumber());
            dtos.add(dto);
        }
        return dtos;
    }
    
    // ========== API请求/响应类 ==========
    
    // 基础API响应
    public static class ApiResponse<T> {
        private boolean success;
        private T data;
        private String errorMessage;
        private long timestamp;
        
        public ApiResponse() {
            this.timestamp = System.currentTimeMillis();
        }
        
        public static <T> ApiResponse<T> success(T data) {
            ApiResponse<T> response = new ApiResponse<>();
            response.setSuccess(true);
            response.setData(data);
            return response;
        }
        
        public static <T> ApiResponse<T> error(String message) {
            ApiResponse<T> response = new ApiResponse<>();
            response.setSuccess(false);
            response.setErrorMessage(message);
            return response;
        }
        
        // Getters and Setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public T getData() { return data; }
        public void setData(T data) { this.data = data; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    }
    
    // DTO类（部分示例，实际应创建独立的DTO包）
    public static class IssueDto {
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
    
    public static class VulnerabilityDto {
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
    
    public static class ChangeDto {
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
    
    // API请求类（简化版，实际应创建完整的请求类）
    public static class AnalysisApiRequest {
        private String userId;
        private String projectId;
        private String code;
        private String language;
        
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
        public String getCode() { return code; }
        public void setCode(String code) { this.code = code; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
    }
    
    // 其他请求响应类省略，实际开发中应完整定义
    // 为简洁起见，这里只定义基本结构
}
