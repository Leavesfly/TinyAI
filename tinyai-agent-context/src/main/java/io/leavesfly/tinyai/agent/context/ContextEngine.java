package io.leavesfly.tinyai.agent.context;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 上下文工程引擎
 * 负责构建和管理对话上下文
 * 
 * @author 山泽
 */
public class ContextEngine {
    
    private final int maxContextLength;                 // 最大上下文长度
    private final List<Message> conversationHistory;   // 对话历史
    private final List<String> systemPrompts;          // 系统提示
    
    // 构造函数
    public ContextEngine() {
        this(4000);
    }
    
    public ContextEngine(int maxContextLength) {
        this.maxContextLength = maxContextLength;
        this.conversationHistory = new ArrayList<>();
        this.systemPrompts = new ArrayList<>();
    }
    
    /**
     * 添加系统提示
     */
    public void addSystemPrompt(String prompt) {
        if (prompt != null && !prompt.trim().isEmpty()) {
            systemPrompts.add(prompt.trim());
        }
    }
    
    /**
     * 添加消息到对话历史
     */
    public void addMessage(Message message) {
        if (message != null) {
            conversationHistory.add(message);
        }
    }
    
    /**
     * 构建完整上下文
     * 
     * @param currentQuery 当前查询
     * @param relevantMemories 相关记忆
     * @param ragContext RAG检索到的上下文
     * @param toolsInfo 工具信息
     * @return 构建的完整上下文
     */
    public String buildContext(String currentQuery, List<Memory> relevantMemories, 
                             String ragContext, String toolsInfo) {
        List<String> contextParts = new ArrayList<>();
        
        // 1. 系统提示
        if (!systemPrompts.isEmpty()) {
            contextParts.add("系统指令：\n" + String.join("\n", systemPrompts));
        }
        
        // 2. 工具信息
        if (toolsInfo != null && !toolsInfo.trim().isEmpty()) {
            contextParts.add("可用工具：\n" + toolsInfo);
        }
        
        // 3. 相关记忆
        if (relevantMemories != null && !relevantMemories.isEmpty()) {
            List<String> memoryTexts = relevantMemories.stream()
                    .map(memory -> String.format("[%s记忆] %s", memory.getMemoryType(), memory.getContent()))
                    .collect(Collectors.toList());
            contextParts.add("相关记忆：\n" + String.join("\n", memoryTexts));
        }
        
        // 4. RAG上下文
        if (ragContext != null && !ragContext.trim().isEmpty()) {
            contextParts.add("相关文档：\n" + ragContext);
        }
        
        // 5. 对话历史（压缩）
        String compressedHistory = compressConversationHistory();
        if (!compressedHistory.isEmpty()) {
            contextParts.add("对话历史：\n" + compressedHistory);
        }
        
        // 6. 当前查询
        if (currentQuery != null && !currentQuery.trim().isEmpty()) {
            contextParts.add("当前问题：" + currentQuery);
        }
        
        // 组合并截断
        String fullContext = String.join("\n\n", contextParts);
        
        if (fullContext.length() > maxContextLength) {
            fullContext = truncateContext(fullContext);
        }
        
        return fullContext;
    }
    
    /**
     * 构建简化上下文（重载方法）
     */
    public String buildContext(String currentQuery) {
        return buildContext(currentQuery, null, null, null);
    }
    
    public String buildContext(String currentQuery, List<Memory> relevantMemories) {
        return buildContext(currentQuery, relevantMemories, null, null);
    }
    
    /**
     * 压缩对话历史
     */
    private String compressConversationHistory() {
        if (conversationHistory.isEmpty()) {
            return "";
        }
        
        // 保留最近的几轮对话
        int maxMessages = 6;  // 最近3轮对话
        List<Message> recentMessages = conversationHistory.stream()
                .skip(Math.max(0, conversationHistory.size() - maxMessages))
                .collect(Collectors.toList());
        
        List<String> compressed = new ArrayList<>();
        
        for (Message msg : recentMessages) {
            String role = getRoleDisplayName(msg.getRole());
            String content = msg.getContent();
            
            // 截断过长内容
            if (content.length() > 200) {
                content = content.substring(0, 197) + "...";
            }
            
            compressed.add(role + ": " + content);
        }
        
        return String.join("\n", compressed);
    }
    
    /**
     * 获取角色显示名称
     */
    private String getRoleDisplayName(String role) {
        if (role == null) {
            return "未知";
        }
        
        switch (role.toLowerCase()) {
            case "user":
                return "用户";
            case "assistant":
                return "助手";
            case "system":
                return "系统";
            case "tool":
                return "工具";
            default:
                return role;
        }
    }
    
    /**
     * 截断上下文以适应长度限制
     */
    private String truncateContext(String context) {
        if (context.length() <= maxContextLength) {
            return context;
        }
        
        String[] lines = context.split("\n");
        
        // 找到重要部分的索引
        int systemEnd = findSectionEnd(lines, "系统指令：");
        int currentQueryStart = findSectionStart(lines, "当前问题：");
        
        // 如果找不到当前问题，则使用最后一行
        if (currentQueryStart == -1) {
            currentQueryStart = lines.length - 1;
        }
        
        // 保留系统指令和当前查询
        List<String> importantParts = new ArrayList<>();
        
        // 添加系统指令部分
        for (int i = 0; i <= systemEnd && i < lines.length; i++) {
            importantParts.add(lines[i]);
        }
        
        // 添加当前查询部分
        for (int i = currentQueryStart; i < lines.length; i++) {
            importantParts.add(lines[i]);
        }
        
        String importantText = String.join("\n", importantParts);
        
        // 计算剩余空间
        int remainingSpace = maxContextLength - importantText.length();
        
        if (remainingSpace > 100) {  // 如果还有空间，添加其他内容
            List<String> middleParts = new ArrayList<>();
            for (int i = systemEnd + 1; i < currentQueryStart; i++) {
                middleParts.add(lines[i]);
            }
            
            String middleText = String.join("\n", middleParts);
            
            if (middleText.length() <= remainingSpace) {
                return context;
            } else {
                // 截断中间部分
                String truncatedMiddle = middleText.substring(0, remainingSpace - 50) + "\n\n[内容被截断...]\n";
                
                List<String> result = new ArrayList<>();
                for (int i = 0; i <= systemEnd && i < lines.length; i++) {
                    result.add(lines[i]);
                }
                result.add(truncatedMiddle);
                for (int i = currentQueryStart; i < lines.length; i++) {
                    result.add(lines[i]);
                }
                
                return String.join("\n", result);
            }
        }
        
        return importantText;
    }
    
    /**
     * 查找指定部分的结束位置
     */
    private int findSectionEnd(String[] lines, String sectionStart) {
        for (int i = 0; i < lines.length; i++) {
            if (lines[i].startsWith(sectionStart)) {
                // 找到下一个部分的开始或数组结束
                for (int j = i + 1; j < lines.length; j++) {
                    if (lines[j].contains("：") && !lines[j].startsWith(" ") && !lines[j].startsWith("\t")) {
                        return j - 1;
                    }
                }
                return lines.length - 1;
            }
        }
        return 0;
    }
    
    /**
     * 查找指定部分的开始位置
     */
    private int findSectionStart(String[] lines, String sectionStart) {
        for (int i = 0; i < lines.length; i++) {
            if (lines[i].startsWith(sectionStart)) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * 清空对话历史
     */
    public void clearConversationHistory() {
        conversationHistory.clear();
    }
    
    /**
     * 清空系统提示
     */
    public void clearSystemPrompts() {
        systemPrompts.clear();
    }
    
    /**
     * 获取对话历史
     */
    public List<Message> getConversationHistory() {
        return new ArrayList<>(conversationHistory);
    }
    
    /**
     * 获取系统提示
     */
    public List<String> getSystemPrompts() {
        return new ArrayList<>(systemPrompts);
    }
    
    /**
     * 获取最大上下文长度
     */
    public int getMaxContextLength() {
        return maxContextLength;
    }
    
    /**
     * 获取当前上下文统计信息
     */
    public Map<String, Object> getContextStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("max_context_length", maxContextLength);
        stats.put("conversation_history_count", conversationHistory.size());
        stats.put("system_prompts_count", systemPrompts.size());
        
        if (!conversationHistory.isEmpty()) {
            int totalLength = conversationHistory.stream()
                    .mapToInt(msg -> msg.getContent().length())
                    .sum();
            stats.put("total_conversation_length", totalLength);
            stats.put("average_message_length", totalLength / conversationHistory.size());
        }
        
        return stats;
    }
}