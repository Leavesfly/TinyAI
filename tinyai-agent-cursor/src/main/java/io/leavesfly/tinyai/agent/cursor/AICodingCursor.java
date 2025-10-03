package io.leavesfly.tinyai.agent.cursor;

import io.leavesfly.tinyai.agent.Message;
import io.leavesfly.tinyai.agent.SimplifiedAdvancedAgent;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * AI Coding Cursor 主系统 - 智能编程助手
 * 整合代码分析、生成、重构、调试等功能，提供统一的智能编程辅助服务
 * 
 * @author 山泽
 */
public class AICodingCursor {
    
    private final String name;
    private final CodeAnalyzer analyzer;
    private final CodeGenerator generator;
    private final RefactorAgent refactorAgent;
    private final DebugAgent debugAgent;
    
    // 系统状态和配置
    private final Map<String, Object> preferences;
    private final List<Message> sessionHistory;
    private final Map<String, Object> currentContext;
    private final SimplifiedAdvancedAgent llmAgent;
    
    // 性能和统计信息
    private final Map<String, Integer> operationStats;
    private final LocalDateTime startTime;
    
    /**
     * 构造函数
     * @param name 系统名称
     */
    public AICodingCursor(String name) {
        this.name = name != null ? name : "AI Coding Cursor";
        
        // 初始化核心组件
        this.analyzer = new CodeAnalyzer();
        this.generator = new CodeGenerator();
        this.refactorAgent = new RefactorAgent(analyzer);
        this.debugAgent = new DebugAgent(analyzer);
        
        // 初始化系统状态
        this.preferences = new ConcurrentHashMap<>();
        this.sessionHistory = Collections.synchronizedList(new ArrayList<>());
        this.currentContext = new ConcurrentHashMap<>();
        this.operationStats = new ConcurrentHashMap<>();
        this.startTime = LocalDateTime.now();
        
        // 初始化LLM代理
        this.llmAgent = new SimplifiedAdvancedAgent("CursorLLM", 
            "你是一个专业的Java编程助手，能够帮助用户进行代码分析、生成、重构和调试。" +
            "请用中文回答问题，提供准确、实用的编程建议。");
        
        // 设置默认偏好
        initializeDefaultPreferences();
        
        System.out.println("🚀 " + this.name + " 智能编程助手已启动!");
        System.out.println("💡 支持功能：代码分析、生成、重构、调试、智能对话");
    }
    
    /**
     * 默认构造函数
     */
    public AICodingCursor() {
        this("AI Coding Cursor");
    }
    
    /**
     * 初始化默认偏好设置
     */
    private void initializeDefaultPreferences() {
        preferences.put("language", "java");
        preferences.put("style", "standard");
        preferences.put("auto_refactor", true);
        preferences.put("debug_level", "detailed");
        preferences.put("max_suggestions", 10);
        preferences.put("enable_ai_chat", true);
    }
    
    /**
     * 分析代码
     * @param code 待分析的代码
     * @return 分析结果
     */
    public Map<String, Object> analyzeCode(String code) {
        long startTime = System.currentTimeMillis();
        System.out.println("🔍 正在分析代码...");
        
        try {
            Map<String, Object> analysis = analyzer.analyzeJavaCode(code);
            
            // 记录操作统计
            operationStats.merge("analyze", 1, Integer::sum);
            
            // 记录到会话历史
            recordOperation("analyze", "代码分析", analysis);
            
            // 更新当前上下文
            currentContext.put("last_analysis", analysis);
            currentContext.put("last_code", code);
            
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("✅ 代码分析完成 (耗时: " + duration + "ms)");
            
            return analysis;
            
        } catch (Exception e) {
            System.err.println("❌ 代码分析失败: " + e.getMessage());
            operationStats.merge("analyze_error", 1, Integer::sum);
            
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("message", e.getMessage());
            return errorResult;
        }
    }
    
    /**
     * 生成代码
     * @param request 生成请求
     * @return 生成的代码
     */
    public String generateCode(String request) {
        long startTime = System.currentTimeMillis();
        System.out.println("🤖 正在生成代码: " + request);
        
        try {
            String generatedCode = generator.generateFromRequest(request);
            
            // 记录操作统计
            operationStats.merge("generate", 1, Integer::sum);
            
            // 记录到会话历史
            recordOperation("generate", request, generatedCode);
            
            // 更新当前上下文
            currentContext.put("last_generated_code", generatedCode);
            currentContext.put("last_request", request);
            
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("✅ 代码生成完成 (耗时: " + duration + "ms)");
            
            return generatedCode;
            
        } catch (Exception e) {
            System.err.println("❌ 代码生成失败: " + e.getMessage());
            operationStats.merge("generate_error", 1, Integer::sum);
            return "// 代码生成失败: " + e.getMessage();
        }
    }
    
    /**
     * 获取重构建议
     * @param code 待重构的代码
     * @return 重构建议列表
     */
    public List<RefactorSuggestion> suggestRefactor(String code) {
        long startTime = System.currentTimeMillis();
        System.out.println("🔧 正在分析重构机会...");
        
        try {
            List<RefactorSuggestion> suggestions = refactorAgent.analyzeRefactorOpportunities(code);
            
            // 记录操作统计
            operationStats.merge("refactor", 1, Integer::sum);
            
            // 记录到会话历史
            recordOperation("refactor", "重构分析", suggestions.size() + " 个建议");
            
            // 更新当前上下文
            currentContext.put("last_refactor_suggestions", suggestions);
            
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("✅ 重构分析完成，发现 " + suggestions.size() + " 个建议 (耗时: " + duration + "ms)");
            
            return suggestions;
            
        } catch (Exception e) {
            System.err.println("❌ 重构分析失败: " + e.getMessage());
            operationStats.merge("refactor_error", 1, Integer::sum);
            return new ArrayList<>();
        }
    }
    
    /**
     * 调试代码
     * @param code 待调试的代码
     * @param errorMessage 可选的错误消息
     * @return 调试结果
     */
    public Map<String, Object> debugCode(String code, String errorMessage) {
        long startTime = System.currentTimeMillis();
        System.out.println("🐛 正在进行智能调试...");
        
        try {
            Map<String, Object> debugResult = debugAgent.diagnoseError(code, errorMessage);
            
            // 记录操作统计
            operationStats.merge("debug", 1, Integer::sum);
            
            // 记录到会话历史
            String resultSummary = (Boolean) debugResult.get("error_found") ? 
                "发现错误: " + debugResult.get("error_type") : "未发现明显错误";
            recordOperation("debug", "错误诊断", resultSummary);
            
            // 更新当前上下文
            currentContext.put("last_debug_result", debugResult);
            
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("✅ 调试分析完成 (耗时: " + duration + "ms)");
            
            return debugResult;
            
        } catch (Exception e) {
            System.err.println("❌ 调试分析失败: " + e.getMessage());
            operationStats.merge("debug_error", 1, Integer::sum);
            
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("message", e.getMessage());
            return errorResult;
        }
    }
    
    /**
     * 调试代码（简化版本）
     * @param code 待调试的代码
     * @return 调试结果
     */
    public Map<String, Object> debugCode(String code) {
        return debugCode(code, null);
    }
    
    /**
     * 综合代码审查
     * @param code 待审查的代码
     * @return 审查报告
     */
    public Map<String, Object> reviewCode(String code) {
        long startTime = System.currentTimeMillis();
        System.out.println("📋 正在进行代码审查...");
        
        try {
            // 执行综合分析
            Map<String, Object> analysis = analyzeCode(code);
            List<RefactorSuggestion> refactorSuggestions = suggestRefactor(code);
            Map<String, Object> debugInfo = debugCode(code);
            
            // 计算代码质量评分
            double qualityScore = calculateCodeQualityScore(analysis, refactorSuggestions, debugInfo);
            
            // 生成综合建议
            List<String> recommendations = generateRecommendations(analysis, refactorSuggestions, debugInfo);
            
            // 构建审查报告
            Map<String, Object> review = new HashMap<>();
            review.put("overall_score", qualityScore);
            review.put("analysis", analysis);
            review.put("refactor_suggestions", refactorSuggestions);
            review.put("debug_info", debugInfo);
            review.put("recommendations", recommendations);
            review.put("review_time", LocalDateTime.now().toString());
            
            // 记录操作统计
            operationStats.merge("review", 1, Integer::sum);
            
            // 记录到会话历史
            recordOperation("review", "代码审查", "质量评分: " + String.format("%.1f", qualityScore));
            
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("✅ 代码审查完成，质量评分: " + String.format("%.1f", qualityScore) + " (耗时: " + duration + "ms)");
            
            return review;
            
        } catch (Exception e) {
            System.err.println("❌ 代码审查失败: " + e.getMessage());
            operationStats.merge("review_error", 1, Integer::sum);
            
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("message", e.getMessage());
            return errorResult;
        }
    }
    
    /**
     * 智能对话功能
     * @param userInput 用户输入
     * @return AI回复
     */
    public String chat(String userInput) {
        if (!isAIChatEnabled()) {
            return "AI对话功能已禁用，请在设置中启用。";
        }
        
        long startTime = System.currentTimeMillis();
        System.out.println("💬 正在处理对话请求...");
        
        try {
            // 构建上下文信息
            String contextualInput = buildContextualInput(userInput);
            
            // 调用LLM代理
            String response = llmAgent.processMessage(contextualInput);
            
            // 记录操作统计
            operationStats.merge("chat", 1, Integer::sum);
            
            // 记录到会话历史
            recordOperation("chat", userInput, response);
            
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("✅ 对话处理完成 (耗时: " + duration + "ms)");
            
            return response;
            
        } catch (Exception e) {
            System.err.println("❌ 对话处理失败: " + e.getMessage());
            operationStats.merge("chat_error", 1, Integer::sum);
            return "抱歉，我遇到了一些问题：" + e.getMessage();
        }
    }
    
    /**
     * 构建上下文相关的输入
     */
    private String buildContextualInput(String userInput) {
        StringBuilder contextBuilder = new StringBuilder();
        
        // 添加当前上下文信息
        if (currentContext.containsKey("last_code")) {
            contextBuilder.append("当前正在处理的代码上下文已加载。\n");
        }
        
        if (currentContext.containsKey("last_analysis")) {
            contextBuilder.append("最近的代码分析结果可供参考。\n");
        }
        
        contextBuilder.append("用户问题：").append(userInput);
        
        return contextBuilder.toString();
    }
    
    /**
     * 计算代码质量评分
     */
    private double calculateCodeQualityScore(Map<String, Object> analysis, 
                                           List<RefactorSuggestion> refactorSuggestions, 
                                           Map<String, Object> debugInfo) {
        double score = 100.0;
        
        // 检查语法有效性
        if (!(Boolean) analysis.getOrDefault("syntax_valid", true)) {
            score -= 30.0;
        }
        
        // 根据代码问题扣分
        @SuppressWarnings("unchecked")
        List<CodeIssue> issues = (List<CodeIssue>) analysis.getOrDefault("issues", new ArrayList<>());
        for (CodeIssue issue : issues) {
            switch (issue.getSeverity().toLowerCase()) {
                case "critical":
                    score -= 20.0;
                    break;
                case "high":
                    score -= 10.0;
                    break;
                case "medium":
                    score -= 5.0;
                    break;
                case "low":
                    score -= 2.0;
                    break;
            }
        }
        
        // 根据复杂度扣分
        Integer complexity = (Integer) analysis.getOrDefault("complexity", 0);
        if (complexity > 15) {
            score -= 15.0;
        } else if (complexity > 10) {
            score -= 10.0;
        } else if (complexity > 5) {
            score -= 5.0;
        }
        
        // 根据重构建议扣分
        if (refactorSuggestions != null) {
            for (RefactorSuggestion suggestion : refactorSuggestions) {
                if (suggestion.isHighPriority()) {
                    score -= 8.0;
                } else {
                    score -= 3.0;
                }
            }
        }
        
        // 根据调试发现的问题扣分
        if ((Boolean) debugInfo.getOrDefault("error_found", false)) {
            score -= 15.0;
        }
        
        // 确保评分在0-100范围内
        return Math.max(0.0, Math.min(100.0, score));
    }
    
    /**
     * 生成综合建议
     */
    private List<String> generateRecommendations(Map<String, Object> analysis, 
                                               List<RefactorSuggestion> refactorSuggestions, 
                                               Map<String, Object> debugInfo) {
        List<String> recommendations = new ArrayList<>();
        
        // 语法建议
        if (!(Boolean) analysis.getOrDefault("syntax_valid", true)) {
            recommendations.add("首先修复语法错误，确保代码可以编译");
        }
        
        // 重构建议
        if (refactorSuggestions != null && !refactorSuggestions.isEmpty()) {
            long highPriorityCount = refactorSuggestions.stream()
                    .mapToLong(s -> s.isHighPriority() ? 1 : 0)
                    .sum();
            if (highPriorityCount > 0) {
                recommendations.add("发现 " + highPriorityCount + " 个高优先级重构建议，建议优先处理");
            }
            recommendations.add("考虑应用重构建议以提高代码质量");
        }
        
        // 调试建议
        if ((Boolean) debugInfo.getOrDefault("error_found", false)) {
            recommendations.add("修复已识别的潜在错误和风险");
        }
        
        // 代码质量建议
        @SuppressWarnings("unchecked")
        Map<String, Object> metrics = (Map<String, Object>) analysis.getOrDefault("metrics", new HashMap<>());
        Integer commentLines = (Integer) metrics.getOrDefault("comment_lines", 0);
        if (commentLines == 0) {
            recommendations.add("增加代码注释以提高可读性");
        }
        
        Integer complexity = (Integer) analysis.getOrDefault("complexity", 0);
        if (complexity > 10) {
            recommendations.add("简化复杂的逻辑结构，降低圈复杂度");
        }
        
        // 如果没有明显问题，给出正面反馈
        if (recommendations.isEmpty()) {
            recommendations.add("代码质量良好，继续保持！");
            recommendations.add("可以考虑添加单元测试以提高代码健壮性");
        }
        
        return recommendations;
    }
    
    /**
     * 记录操作到会话历史
     */
    private void recordOperation(String operation, String input, Object result) {
        try {
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("operation", operation);
            metadata.put("input", input);
            metadata.put("result_summary", result.toString());
            metadata.put("timestamp", LocalDateTime.now());
            
            Message message = new Message("system", "操作记录: " + operation, metadata);
            sessionHistory.add(message);
            
            // 限制历史记录长度
            if (sessionHistory.size() > 100) {
                sessionHistory.remove(0);
            }
            
        } catch (Exception e) {
            System.err.println("记录操作历史失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取系统状态
     */
    public Map<String, Object> getSystemStatus() {
        Map<String, Object> status = new HashMap<>();
        
        status.put("name", name);
        status.put("start_time", startTime.toString());
        status.put("uptime_minutes", java.time.Duration.between(startTime, LocalDateTime.now()).toMinutes());
        status.put("session_operations", sessionHistory.size());
        status.put("operation_stats", new HashMap<>(operationStats));
        status.put("preferences", new HashMap<>(preferences));
        status.put("cache_size", analyzer.getCacheSize());
        status.put("ai_chat_enabled", isAIChatEnabled());
        
        return status;
    }
    
    /**
     * 获取操作统计
     */
    public Map<String, Integer> getOperationStats() {
        return new HashMap<>(operationStats);
    }
    
    /**
     * 获取会话历史
     */
    public List<Message> getSessionHistory() {
        return new ArrayList<>(sessionHistory);
    }
    
    /**
     * 清空会话历史
     */
    public void clearSessionHistory() {
        sessionHistory.clear();
        currentContext.clear();
        System.out.println("🗑️ 会话历史已清空");
    }
    
    /**
     * 更新偏好设置
     */
    public void updatePreferences(Map<String, Object> newPreferences) {
        if (newPreferences != null) {
            preferences.putAll(newPreferences);
            System.out.println("⚙️ 偏好设置已更新");
        }
    }
    
    /**
     * 获取偏好设置
     */
    public Map<String, Object> getPreferences() {
        return new HashMap<>(preferences);
    }
    
    /**
     * 检查AI对话是否启用
     */
    private boolean isAIChatEnabled() {
        return (Boolean) preferences.getOrDefault("enable_ai_chat", true);
    }
    
    /**
     * 启用/禁用AI对话
     */
    public void setAIChatEnabled(boolean enabled) {
        preferences.put("enable_ai_chat", enabled);
        System.out.println("💬 AI对话功能已" + (enabled ? "启用" : "禁用"));
    }
    
    /**
     * 获取帮助信息
     */
    public String getHelp() {
        return "🚀 AI Coding Cursor 智能编程助手\n\n" +
               "📋 主要功能：\n" +
               "• analyzeCode(code) - 分析代码结构和质量\n" +
               "• generateCode(request) - 根据需求生成代码\n" +
               "• suggestRefactor(code) - 提供重构建议\n" +
               "• debugCode(code) - 诊断和修复错误\n" +
               "• reviewCode(code) - 综合代码审查\n" +
               "• chat(message) - 智能对话功能\n\n" +
               "⚙️ 系统管理：\n" +
               "• getSystemStatus() - 查看系统状态\n" +
               "• updatePreferences(prefs) - 更新设置\n" +
               "• clearSessionHistory() - 清空历史\n" +
               "• getHelp() - 查看帮助信息\n\n" +
               "作者：山泽 | 版本：1.0.0";
    }
    
    // Getter 方法
    public String getName() {
        return name;
    }
    
    public CodeAnalyzer getAnalyzer() {
        return analyzer;
    }
    
    public CodeGenerator getGenerator() {
        return generator;
    }
    
    public RefactorAgent getRefactorAgent() {
        return refactorAgent;
    }
    
    public DebugAgent getDebugAgent() {
        return debugAgent;
    }
}