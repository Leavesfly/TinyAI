package io.leavesfly.tinyai.agent.context;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 高级LLM Agent
 * 集成记忆管理、RAG、工具调用和上下文工程
 * 
 * @author 山泽
 */
public class AdvancedAgent {
    
    private final String name;                      // Agent名称
    private final String systemPrompt;             // 系统提示
    private final String currentSessionId;         // 当前会话ID
    
    // 核心组件
    private final MemoryManager memoryManager;     // 记忆管理器
    private final RAGSystem ragSystem;             // RAG系统
    private final ToolRegistry toolRegistry;       // 工具注册表
    private final ContextEngine contextEngine;     // 上下文引擎
    private final LLMSimulator llmSimulator;       // LLM模拟器
    
    // 对话状态
    private final List<Message> conversationHistory;  // 对话历史
    
    // 构造函数
    public AdvancedAgent(String name) {
        this(name, "", 4000);
    }
    
    public AdvancedAgent(String name, String systemPrompt) {
        this(name, systemPrompt, 4000);
    }
    
    public AdvancedAgent(String name, String systemPrompt, int maxContextLength) {
        this.name = name;
        this.systemPrompt = systemPrompt;
        this.currentSessionId = AgentHelper.generateSessionId();
        
        // 初始化核心组件
        this.memoryManager = new MemoryManager();
        this.ragSystem = new RAGSystem();
        this.toolRegistry = new ToolRegistry();
        this.contextEngine = new ContextEngine(maxContextLength);
        this.llmSimulator = new LLMSimulator();
        
        // 初始化对话历史
        this.conversationHistory = new ArrayList<>();
        
        // 设置系统提示
        if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
            this.contextEngine.addSystemPrompt(systemPrompt);
        }
        
        // 注册默认工具
        registerDefaultTools();
    }
    
    /**
     * 处理用户消息的主要方法
     */
    public String processMessage(String userInput) {
        if (userInput == null || userInput.trim().isEmpty()) {
            return "请提供有效的输入。";
        }
        
        // 1. 记录用户输入
        Message userMessage = new Message("user", userInput);
        conversationHistory.add(userMessage);
        contextEngine.addMessage(userMessage);
        
        // 2. 记录到情节记忆
        memoryManager.addMemory("用户说: " + userInput, "episodic", 0.5);
        
        // 3. 检索相关记忆
        List<Memory> relevantMemories = memoryManager.retrieveMemories(userInput, 3);
        
        // 4. 检索相关知识（RAG）
        String ragContext = ragSystem.getContext(userInput, 800);
        
        // 5. 获取工具信息
        String toolsInfo = getToolsInfoString();
        
        // 6. 构建完整上下文
        String fullContext = contextEngine.buildContext(
            userInput,
            relevantMemories,
            ragContext.trim().isEmpty() ? null : ragContext,
            toolsInfo
        );
        
        // 7. 生成响应（使用LLM模拟器）
        String response = generateLLMResponse(userInput, fullContext);
        
        // 8. 记录助手响应
        Message assistantMessage = new Message("assistant", response);
        conversationHistory.add(assistantMessage);
        contextEngine.addMessage(assistantMessage);
        
        // 9. 记录到工作记忆
        memoryManager.addMemory("我回复: " + response, "working", 0.4);
        
        // 10. 定期整合记忆
        if (conversationHistory.size() % 10 == 0) {
            memoryManager.consolidateMemories();
        }
        
        return response;
    }
    
    /**
     * 添加知识到RAG系统
     */
    public void addKnowledge(String content, String docId, Map<String, Object> metadata) {
        ragSystem.addDocument(docId != null ? docId : AgentHelper.generateDocumentId(content), content, metadata);
        
        // 同时添加到语义记忆
        Map<String, Object> defaultMetadata = new HashMap<>();
        defaultMetadata.put("type", "knowledge");
        memoryManager.addMemory(content, "semantic", 0.8, 
            metadata != null ? metadata : defaultMetadata);
    }
    
    public void addKnowledge(String content) {
        addKnowledge(content, null, null);
    }
    
    /**
     * 注册工具
     */
    public void registerTool(String name, Function<Map<String, Object>, Object> function, 
                           String description, Map<String, Object> parameters) {
        toolRegistry.register(name, function, description, parameters);
    }
    
    public void registerTool(String name, Function<Map<String, Object>, Object> function, String description) {
        toolRegistry.register(name, function, description);
    }
    
    /**
     * 获取Agent统计信息
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("name", name);
        stats.put("session_id", currentSessionId);
        stats.put("conversation_length", conversationHistory.size());
        stats.put("memory_stats", memoryManager.getMemoryStats());
        stats.put("rag_stats", ragSystem.getStats());
        stats.put("tool_count", toolRegistry.getToolCount());
        stats.put("context_stats", contextEngine.getContextStats());
        return stats;
    }
    
    /**
     * 导出对话历史
     */
    public List<Map<String, Object>> exportConversation() {
        return conversationHistory.stream()
                .map(msg -> {
                    Map<String, Object> msgMap = new HashMap<>();
                    msgMap.put("role", msg.getRole());
                    msgMap.put("content", msg.getContent());
                    msgMap.put("timestamp", msg.getTimestamp().toString());
                    msgMap.put("metadata", msg.getMetadata());
                    return msgMap;
                })
                .collect(Collectors.toList());
    }
    
    /**
     * 清空对话历史
     */
    public void clearConversation() {
        conversationHistory.clear();
        contextEngine.clearConversationHistory();
    }
    
    /**
     * 注册默认工具
     */
    private void registerDefaultTools() {
        // 计算器工具
        Map<String, Object> calculatorParams = new HashMap<>();
        Map<String, Object> operationParam = new HashMap<>();
        operationParam.put("type", "string");
        operationParam.put("enum", Arrays.asList("add", "subtract", "multiply", "divide"));
        calculatorParams.put("operation", operationParam);
        
        Map<String, Object> aParam = new HashMap<>();
        aParam.put("type", "number");
        calculatorParams.put("a", aParam);
        
        Map<String, Object> bParam = new HashMap<>();
        bParam.put("type", "number");
        calculatorParams.put("b", bParam);
        
        registerTool("calculator", AgentHelper::calculatorTool, 
            "执行数学计算：加法、减法、乘法、除法", calculatorParams);
        
        // 时间工具
        registerTool("time", AgentHelper::timeTool, "获取当前时间信息");
        
        // 笔记工具
        Map<String, Object> noteParams = new HashMap<>();
        Map<String, Object> actionParam = new HashMap<>();
        actionParam.put("type", "string");
        actionParam.put("enum", Arrays.asList("create", "list", "get", "delete"));
        noteParams.put("action", actionParam);
        
        Map<String, Object> contentParam = new HashMap<>();
        contentParam.put("type", "string");
        contentParam.put("description", "笔记内容（创建时必需）");
        noteParams.put("content", contentParam);
        
        Map<String, Object> noteIdParam = new HashMap<>();
        noteIdParam.put("type", "string");
        noteIdParam.put("description", "笔记ID（获取/删除时必需）");
        noteParams.put("note_id", noteIdParam);
        
        registerTool("note", AgentHelper::noteTool, 
            "管理笔记：创建、查看、列出、删除笔记", noteParams);
    }
    
    /**
     * 获取工具信息字符串
     */
    private String getToolsInfoString() {
        return toolRegistry.listTools().stream()
                .map(tool -> String.format("- %s: %s", tool.getName(), tool.getDescription()))
                .collect(Collectors.joining("\n"));
    }
    
    // Getter 方法
    public String getName() {
        return name;
    }
    
    public String getSystemPrompt() {
        return systemPrompt;
    }
    
    public String getCurrentSessionId() {
        return currentSessionId;
    }
    
    public List<Message> getConversationHistory() {
        return new ArrayList<>(conversationHistory);
    }
    
    public MemoryManager getMemoryManager() {
        return memoryManager;
    }
    
    public RAGSystem getRagSystem() {
        return ragSystem;
    }
    
    public ToolRegistry getToolRegistry() {
        return toolRegistry;
    }
    
    public ContextEngine getContextEngine() {
        return contextEngine;
    }
    
    public LLMSimulator getLLMSimulator() {
        return llmSimulator;
    }
    
    /**
     * 生成LLM响应
     */
    private String generateLLMResponse(String userInput, String fullContext) {
        try {
            // 构建消息列表
            List<Map<String, String>> messages = new ArrayList<>();
            
            // 添加系统消息
            if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
                Map<String, String> systemMessage = new HashMap<>();
                systemMessage.put("role", "system");
                systemMessage.put("content", systemPrompt);
                messages.add(systemMessage);
            }
            
            // 添加上下文信息（如果有的话）
            if (fullContext != null && !fullContext.trim().isEmpty()) {
                Map<String, String> contextMessage = new HashMap<>();
                contextMessage.put("role", "system");
                contextMessage.put("content", "上下文信息：\n" + fullContext);
                messages.add(contextMessage);
            }
            
            // 添加用户消息
            Map<String, String> userMessage = new HashMap<>();
            userMessage.put("role", "user");
            userMessage.put("content", userInput);
            messages.add(userMessage);
            
            // 调用LLM模拟器生成回复
            return llmSimulator.chatCompletion(messages, "general");
            
        } catch (Exception e) {
            return "抱歉，我在处理您的请求时遇到了技术问题：" + e.getMessage();
        }
    }
}