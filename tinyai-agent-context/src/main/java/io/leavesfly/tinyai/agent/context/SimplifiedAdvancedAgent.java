package io.leavesfly.tinyai.agent.context;


import java.util.*;

/**
 * 简化版AdvancedAgent，仅使用LLMSimulator功能
 * 避免数据库依赖问题，专注展示LLM集成
 * 
 * @author 山泽
 */
public class SimplifiedAdvancedAgent {
    
    private final String name;
    private final String systemPrompt;
    private final LLMSimulator llmSimulator;
    private final List<Message> conversationHistory;
    
    public SimplifiedAdvancedAgent(String name) {
        this(name, "你是一个智能助手，能够帮助用户解决各种问题。");
    }
    
    public SimplifiedAdvancedAgent(String name, String systemPrompt) {
        this.name = name;
        this.systemPrompt = systemPrompt;
        this.llmSimulator = new LLMSimulator();
        this.conversationHistory = new ArrayList<>();
    }
    
    /**
     * 处理用户消息的主要方法
     */
    public String processMessage(String userInput) {
        if (userInput == null || userInput.trim().isEmpty()) {
            return "请提供有效的输入。";
        }
        
        // 记录用户输入
        Message userMessage = new Message("user", userInput);
        conversationHistory.add(userMessage);
        
        // 生成LLM响应
        String response = generateLLMResponse(userInput);
        
        // 记录助手响应
        Message assistantMessage = new Message("assistant", response);
        conversationHistory.add(assistantMessage);
        
        return response;
    }
    
    /**
     * 生成LLM响应
     */
    private String generateLLMResponse(String userInput) {
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
            
            // 添加历史对话（最近5轮）
            int historyStart = Math.max(0, conversationHistory.size() - 10);
            for (int i = historyStart; i < conversationHistory.size(); i++) {
                Message msg = conversationHistory.get(i);
                Map<String, String> historyMessage = new HashMap<>();
                historyMessage.put("role", msg.getRole());
                historyMessage.put("content", msg.getContent());
                messages.add(historyMessage);
            }
            
            // 添加当前用户消息
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
    
    // Getter 方法
    public String getName() {
        return name;
    }
    
    public String getSystemPrompt() {
        return systemPrompt;
    }
    
    public LLMSimulator getLLMSimulator() {
        return llmSimulator;
    }
    
    public List<Message> getConversationHistory() {
        return new ArrayList<>(conversationHistory);
    }
    
    /**
     * 清空对话历史
     */
    public void clearConversation() {
        conversationHistory.clear();
    }
    
    /**
     * 导出对话历史
     */
    public List<Map<String, Object>> exportConversation() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (Message msg : conversationHistory) {
            Map<String, Object> msgMap = new HashMap<>();
            msgMap.put("role", msg.getRole());
            msgMap.put("content", msg.getContent());
            msgMap.put("timestamp", msg.getTimestamp().toString());
            msgMap.put("metadata", msg.getMetadata());
            result.add(msgMap);
        }
        return result;
    }
    
    /**
     * 获取基本统计信息
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("name", name);
        stats.put("conversation_length", conversationHistory.size());
        stats.put("llm_model", llmSimulator.getModelName());
        stats.put("llm_temperature", llmSimulator.getTemperature());
        stats.put("llm_max_tokens", llmSimulator.getMaxTokens());
        return stats;
    }
}