package io.leavesfly.tinyai.agent.cursor.v2.component.generator;

import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.LLMGateway;

import java.util.*;

/**
 * 增强版代码生成器（V2）
 * 基于LLM的智能代码生成，支持多种生成场景
 * 
 * 核心功能：
 * 1. 代码补全（智能补全、上下文感知）
 * 2. 函数生成（根据描述生成完整函数）
 * 3. 类生成（生成完整的类结构）
 * 4. 测试代码生成（生成单元测试）
 * 5. 文档生成（JavaDoc、注释）
 * 6. 代码转换（语言转换、重构）
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class CodeGeneratorV2 {
    
    /**
     * LLM网关
     */
    private final LLMGateway llmGateway;
    
    /**
     * 上下文引擎
     */
    private final ContextEngine contextEngine;
    
    /**
     * 默认生成模型
     */
    private String generationModel = "deepseek-coder";
    
    /**
     * 生成温度
     */
    private double generationTemperature = 0.2;
    
    /**
     * 最大生成Token数
     */
    private int maxTokens = 1000;
    
    public CodeGeneratorV2(LLMGateway llmGateway, ContextEngine contextEngine) {
        this.llmGateway = llmGateway;
        this.contextEngine = contextEngine;
    }
    
    /**
     * 智能代码补全
     * 
     * @param request 补全请求
     * @return 补全结果
     */
    public CompletionResult complete(CompletionRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 构建补全上下文
            Context context = contextEngine.buildCompletionContext(
                request.getProjectId(),
                request.getFilePath(),
                request.getPrefix(),
                request.getSuffix(),
                new Context.CursorPosition(request.getLine(), request.getColumn())
            );
            
            // 构建补全提示词
            String prompt = buildCompletionPrompt(request, context);
            
            // 调用LLM补全
            String completion = llmGateway.complete(
                prompt,
                request.getSuffix(),
                request.getLanguage(),
                maxTokens
            );
            
            CompletionResult result = new CompletionResult();
            result.setSuccess(true);
            result.setCompletionText(completion);
            result.setGenerationTime(System.currentTimeMillis() - startTime);
            
            return result;
            
        } catch (Exception e) {
            CompletionResult errorResult = new CompletionResult();
            errorResult.setSuccess(false);
            errorResult.setErrorMessage("补全失败: " + e.getMessage());
            errorResult.setGenerationTime(System.currentTimeMillis() - startTime);
            return errorResult;
        }
    }
    
    /**
     * 生成函数
     * 
     * @param request 函数生成请求
     * @return 生成结果
     */
    public FunctionGenerationResult generateFunction(FunctionGenerationRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 构建上下文
            Context context = contextEngine.buildAnalysisContext(
                request.getProjectId(),
                "",
                "generation"
            );
            
            // 构建生成提示词
            String prompt = buildFunctionPrompt(request);
            
            // 调用LLM生成
            ChatRequest chatRequest = ChatRequest.builder()
                .model(generationModel)
                .addSystemMessage(context.buildSystemPrompt())
                .addSystemMessage(getFunctionGenerationSystemPrompt())
                .addUserMessage(prompt)
                .temperature(generationTemperature)
                .maxTokens(maxTokens)
                .build();
            
            ChatResponse response = llmGateway.chat(chatRequest);
            
            // 提取生成的代码
            String generatedCode = extractCodeFromResponse(response.getContent());
            
            FunctionGenerationResult result = new FunctionGenerationResult();
            result.setSuccess(true);
            result.setFunctionCode(generatedCode);
            result.setExplanation(response.getContent());
            result.setGenerationTime(System.currentTimeMillis() - startTime);
            
            return result;
            
        } catch (Exception e) {
            FunctionGenerationResult errorResult = new FunctionGenerationResult();
            errorResult.setSuccess(false);
            errorResult.setErrorMessage("函数生成失败: " + e.getMessage());
            errorResult.setGenerationTime(System.currentTimeMillis() - startTime);
            return errorResult;
        }
    }
    
    /**
     * 生成类
     * 
     * @param request 类生成请求
     * @return 生成结果
     */
    public ClassGenerationResult generateClass(ClassGenerationRequest request) {
        String prompt = buildClassPrompt(request);
        
        ChatRequest chatRequest = ChatRequest.builder()
            .model(generationModel)
            .addSystemMessage(getClassGenerationSystemPrompt())
            .addUserMessage(prompt)
            .temperature(generationTemperature)
            .maxTokens(2000)
            .build();
        
        ChatResponse response = llmGateway.chat(chatRequest);
        String generatedCode = extractCodeFromResponse(response.getContent());
        
        ClassGenerationResult result = new ClassGenerationResult();
        result.setSuccess(true);
        result.setClassCode(generatedCode);
        result.setExplanation(response.getContent());
        
        return result;
    }
    
    /**
     * 生成单元测试
     * 
     * @param sourceCode 源代码
     * @param className 类名
     * @param projectId 项目ID
     * @return 测试代码
     */
    public TestGenerationResult generateTest(String sourceCode, String className, String projectId) {
        String prompt = buildTestPrompt(sourceCode, className);
        
        ChatRequest request = ChatRequest.builder()
            .model(generationModel)
            .addSystemMessage(getTestGenerationSystemPrompt())
            .addUserMessage(prompt)
            .temperature(0.3)
            .maxTokens(2000)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        String testCode = extractCodeFromResponse(response.getContent());
        
        TestGenerationResult result = new TestGenerationResult();
        result.setSuccess(true);
        result.setTestCode(testCode);
        result.setTestCases(extractTestCases(response.getContent()));
        
        return result;
    }
    
    /**
     * 生成文档
     * 
     * @param code 代码
     * @param docType 文档类型（javadoc/comment/readme）
     * @return 文档内容
     */
    public DocumentationResult generateDocumentation(String code, String docType) {
        String prompt = buildDocumentationPrompt(code, docType);
        
        ChatRequest request = ChatRequest.builder()
            .model(generationModel)
            .addSystemMessage("你是一个技术文档专家，善于编写清晰、准确的代码文档。")
            .addUserMessage(prompt)
            .temperature(0.4)
            .maxTokens(1500)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        
        DocumentationResult result = new DocumentationResult();
        result.setSuccess(true);
        result.setDocumentation(response.getContent());
        
        return result;
    }
    
    /**
     * 代码转换
     * 
     * @param sourceCode 源代码
     * @param fromLanguage 源语言
     * @param toLanguage 目标语言
     * @return 转换后的代码
     */
    public CodeConversionResult convertCode(String sourceCode, String fromLanguage, String toLanguage) {
        String prompt = String.format(
            "请将以下%s代码转换为%s代码，保持相同的功能和逻辑：\n\n```%s\n%s\n```",
            fromLanguage, toLanguage, fromLanguage, sourceCode
        );
        
        ChatRequest request = ChatRequest.builder()
            .model(generationModel)
            .addSystemMessage("你是一个多语言编程专家，精通代码转换。")
            .addUserMessage(prompt)
            .temperature(0.2)
            .maxTokens(2000)
            .build();
        
        ChatResponse response = llmGateway.chat(request);
        String convertedCode = extractCodeFromResponse(response.getContent());
        
        CodeConversionResult result = new CodeConversionResult();
        result.setSuccess(true);
        result.setConvertedCode(convertedCode);
        result.setExplanation(response.getContent());
        
        return result;
    }
    
    // ========== 私有辅助方法 ==========
    
    private String buildCompletionPrompt(CompletionRequest request, Context context) {
        StringBuilder prompt = new StringBuilder();
        
        prompt.append("当前文件: ").append(request.getFilePath()).append("\n");
        prompt.append("语言: ").append(request.getLanguage()).append("\n\n");
        
        prompt.append("光标前代码:\n```\n");
        prompt.append(request.getPrefix());
        prompt.append("\n```\n\n");
        
        if (request.getSuffix() != null && !request.getSuffix().isEmpty()) {
            prompt.append("光标后代码:\n```\n");
            prompt.append(request.getSuffix());
            prompt.append("\n```\n\n");
        }
        
        prompt.append("请提供智能代码补全，只返回补全的代码，不要解释。");
        
        return prompt.toString();
    }
    
    private String buildFunctionPrompt(FunctionGenerationRequest request) {
        StringBuilder prompt = new StringBuilder();
        
        prompt.append("请生成一个").append(request.getLanguage()).append("函数：\n\n");
        prompt.append("函数名: ").append(request.getFunctionName()).append("\n");
        prompt.append("功能描述: ").append(request.getDescription()).append("\n");
        
        if (!request.getParameters().isEmpty()) {
            prompt.append("\n参数:\n");
            for (Map.Entry<String, String> param : request.getParameters().entrySet()) {
                prompt.append("- ").append(param.getKey()).append(": ").append(param.getValue()).append("\n");
            }
        }
        
        if (request.getReturnType() != null) {
            prompt.append("\n返回类型: ").append(request.getReturnType()).append("\n");
        }
        
        prompt.append("\n请生成完整的函数代码，包括必要的注释和错误处理。");
        
        return prompt.toString();
    }
    
    private String buildClassPrompt(ClassGenerationRequest request) {
        return String.format(
            "请生成一个%s类：\n\n类名: %s\n描述: %s\n\n请包含：\n- 必要的字段\n- 构造函数\n- Getter/Setter方法\n- 核心业务方法\n- JavaDoc注释",
            request.getLanguage(), request.getClassName(), request.getDescription()
        );
    }
    
    private String buildTestPrompt(String sourceCode, String className) {
        return String.format(
            "请为以下类生成JUnit单元测试：\n\n```java\n%s\n```\n\n" +
            "测试类名: %sTest\n\n" +
            "请包含：\n- 正常情况测试\n- 边界条件测试\n- 异常情况测试\n- Mock依赖（如需要）",
            sourceCode, className
        );
    }
    
    private String buildDocumentationPrompt(String code, String docType) {
        return String.format(
            "请为以下代码生成%s文档：\n\n```java\n%s\n```",
            docType, code
        );
    }
    
    private String getFunctionGenerationSystemPrompt() {
        return "你是一个专业的程序员，精通各种编程语言。" +
               "请生成清晰、高效、符合最佳实践的代码。" +
               "代码应该包含适当的注释、错误处理和边界检查。";
    }
    
    private String getClassGenerationSystemPrompt() {
        return "你是一个软件架构师，擅长设计清晰的类结构。" +
               "请生成符合SOLID原则、设计模式和最佳实践的代码。";
    }
    
    private String getTestGenerationSystemPrompt() {
        return "你是一个测试工程师，擅长编写全面的单元测试。" +
               "请生成高覆盖率、有意义的测试用例。";
    }
    
    private String extractCodeFromResponse(String response) {
        // 提取代码块
        int start = response.indexOf("```");
        if (start == -1) return response.trim();
        
        start = response.indexOf("\n", start) + 1;
        int end = response.indexOf("```", start);
        
        if (end == -1) return response.trim();
        
        return response.substring(start, end).trim();
    }
    
    private List<String> extractTestCases(String response) {
        // TODO: 实现智能提取测试用例
        return new ArrayList<>();
    }
    
    // ========== 配置方法 ==========
    
    public void setGenerationModel(String model) {
        this.generationModel = model;
    }
    
    public void setGenerationTemperature(double temperature) {
        this.generationTemperature = temperature;
    }
    
    public void setMaxTokens(int maxTokens) {
        this.maxTokens = maxTokens;
    }
    
    // ========== 内部类 ==========
    
    /**
     * 补全请求
     */
    public static class CompletionRequest {
        private String projectId;
        private String filePath;
        private String prefix;
        private String suffix;
        private String language;
        private int line;
        private int column;
        
        // Getters and Setters
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
    
    /**
     * 补全结果
     */
    public static class CompletionResult {
        private boolean success;
        private String completionText;
        private long generationTime;
        private String errorMessage;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getCompletionText() { return completionText; }
        public void setCompletionText(String completionText) { this.completionText = completionText; }
        public long getGenerationTime() { return generationTime; }
        public void setGenerationTime(long generationTime) { this.generationTime = generationTime; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    }
    
    /**
     * 函数生成请求
     */
    public static class FunctionGenerationRequest {
        private String projectId;
        private String functionName;
        private String description;
        private Map<String, String> parameters;
        private String returnType;
        private String language;
        
        public FunctionGenerationRequest() {
            this.parameters = new HashMap<>();
            this.language = "java";
        }
        
        // Getters and Setters
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
    
    /**
     * 函数生成结果
     */
    public static class FunctionGenerationResult {
        private boolean success;
        private String functionCode;
        private String explanation;
        private long generationTime;
        private String errorMessage;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getFunctionCode() { return functionCode; }
        public void setFunctionCode(String functionCode) { this.functionCode = functionCode; }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
        public long getGenerationTime() { return generationTime; }
        public void setGenerationTime(long generationTime) { this.generationTime = generationTime; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    }
    
    /**
     * 类生成请求
     */
    public static class ClassGenerationRequest {
        private String className;
        private String description;
        private String language;
        
        public ClassGenerationRequest() {
            this.language = "java";
        }
        
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
    }
    
    /**
     * 类生成结果
     */
    public static class ClassGenerationResult {
        private boolean success;
        private String classCode;
        private String explanation;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getClassCode() { return classCode; }
        public void setClassCode(String classCode) { this.classCode = classCode; }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
    }
    
    /**
     * 测试生成结果
     */
    public static class TestGenerationResult {
        private boolean success;
        private String testCode;
        private List<String> testCases;
        
        public TestGenerationResult() {
            this.testCases = new ArrayList<>();
        }
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getTestCode() { return testCode; }
        public void setTestCode(String testCode) { this.testCode = testCode; }
        public List<String> getTestCases() { return testCases; }
        public void setTestCases(List<String> testCases) { this.testCases = testCases; }
    }
    
    /**
     * 文档生成结果
     */
    public static class DocumentationResult {
        private boolean success;
        private String documentation;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getDocumentation() { return documentation; }
        public void setDocumentation(String documentation) { this.documentation = documentation; }
    }
    
    /**
     * 代码转换结果
     */
    public static class CodeConversionResult {
        private boolean success;
        private String convertedCode;
        private String explanation;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getConvertedCode() { return convertedCode; }
        public void setConvertedCode(String convertedCode) { this.convertedCode = convertedCode; }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
    }
}
