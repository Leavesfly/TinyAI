package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 上下文数据结构
 * 整合多源上下文信息，用于增强LLM的理解能力
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class Context {
    
    /**
     * 当前文件内容
     */
    private String currentFileContent;
    
    /**
     * 当前文件路径
     */
    private String currentFilePath;
    
    /**
     * 光标位置（用于代码补全）
     */
    private CursorPosition cursorPosition;
    
    /**
     * 相关代码片段（来自RAG检索）
     */
    private List<CodeSnippet> relatedCodeSnippets;
    
    /**
     * 会话历史消息
     */
    private List<Message> conversationHistory;
    
    /**
     * 项目规则
     */
    private List<String> projectRules;
    
    /**
     * 长期记忆
     */
    private List<Memory> longTermMemories;
    
    /**
     * 扩展上下文（键值对）
     */
    private Map<String, Object> extraContext;
    
    public Context() {
        this.relatedCodeSnippets = new ArrayList<>();
        this.conversationHistory = new ArrayList<>();
        this.projectRules = new ArrayList<>();
        this.longTermMemories = new ArrayList<>();
        this.extraContext = new HashMap<>();
    }
    
    /**
     * 添加相关代码片段
     */
    public void addCodeSnippet(CodeSnippet snippet) {
        this.relatedCodeSnippets.add(snippet);
    }
    
    /**
     * 添加会话历史
     */
    public void addConversationMessage(Message message) {
        this.conversationHistory.add(message);
    }
    
    /**
     * 添加项目规则
     */
    public void addProjectRule(String rule) {
        this.projectRules.add(rule);
    }
    
    /**
     * 添加长期记忆
     */
    public void addLongTermMemory(Memory memory) {
        this.longTermMemories.add(memory);
    }
    
    /**
     * 设置扩展上下文
     */
    public void putExtra(String key, Object value) {
        this.extraContext.put(key, value);
    }
    
    /**
     * 获取扩展上下文
     */
    public Object getExtra(String key) {
        return this.extraContext.get(key);
    }
    
    /**
     * 构建系统提示词（融合所有上下文信息）
     */
    public String buildSystemPrompt() {
        StringBuilder prompt = new StringBuilder();
        
        // 项目规则
        if (!projectRules.isEmpty()) {
            prompt.append("# Project Rules\n");
            for (String rule : projectRules) {
                prompt.append("- ").append(rule).append("\n");
            }
            prompt.append("\n");
        }
        
        // 当前文件
        if (currentFilePath != null) {
            prompt.append("# Current File: ").append(currentFilePath).append("\n");
            if (currentFileContent != null) {
                prompt.append("```\n").append(currentFileContent).append("\n```\n\n");
            }
        }
        
        // 相关代码片段
        if (!relatedCodeSnippets.isEmpty()) {
            prompt.append("# Related Code Snippets\n");
            for (int i = 0; i < relatedCodeSnippets.size(); i++) {
                CodeSnippet snippet = relatedCodeSnippets.get(i);
                prompt.append("## Snippet ").append(i + 1).append(": ").append(snippet.getFilePath()).append("\n");
                prompt.append("```\n").append(snippet.getContent()).append("\n```\n\n");
            }
        }
        
        return prompt.toString();
    }
    
    // Getters and Setters
    
    public String getCurrentFileContent() {
        return currentFileContent;
    }
    
    public void setCurrentFileContent(String currentFileContent) {
        this.currentFileContent = currentFileContent;
    }
    
    public String getCurrentFilePath() {
        return currentFilePath;
    }
    
    public void setCurrentFilePath(String currentFilePath) {
        this.currentFilePath = currentFilePath;
    }
    
    public CursorPosition getCursorPosition() {
        return cursorPosition;
    }
    
    public void setCursorPosition(CursorPosition cursorPosition) {
        this.cursorPosition = cursorPosition;
    }
    
    public List<CodeSnippet> getRelatedCodeSnippets() {
        return relatedCodeSnippets;
    }
    
    public void setRelatedCodeSnippets(List<CodeSnippet> relatedCodeSnippets) {
        this.relatedCodeSnippets = relatedCodeSnippets;
    }
    
    public List<Message> getConversationHistory() {
        return conversationHistory;
    }
    
    public void setConversationHistory(List<Message> conversationHistory) {
        this.conversationHistory = conversationHistory;
    }
    
    public List<String> getProjectRules() {
        return projectRules;
    }
    
    public void setProjectRules(List<String> projectRules) {
        this.projectRules = projectRules;
    }
    
    public List<Memory> getLongTermMemories() {
        return longTermMemories;
    }
    
    public void setLongTermMemories(List<Memory> longTermMemories) {
        this.longTermMemories = longTermMemories;
    }
    
    public Map<String, Object> getExtraContext() {
        return extraContext;
    }
    
    public void setExtraContext(Map<String, Object> extraContext) {
        this.extraContext = extraContext;
    }
    
    /**
     * 光标位置
     */
    public static class CursorPosition {
        private int line;
        private int column;
        
        public CursorPosition() {
        }
        
        public CursorPosition(int line, int column) {
            this.line = line;
            this.column = column;
        }
        
        public int getLine() {
            return line;
        }
        
        public void setLine(int line) {
            this.line = line;
        }
        
        public int getColumn() {
            return column;
        }
        
        public void setColumn(int column) {
            this.column = column;
        }
    }
    
    /**
     * 代码片段
     */
    public static class CodeSnippet {
        private String filePath;
        private String content;
        private double score; // 相似度评分
        private Map<String, Object> metadata;
        
        public CodeSnippet() {
        }
        
        public CodeSnippet(String filePath, String content) {
            this.filePath = filePath;
            this.content = content;
        }
        
        public CodeSnippet(String filePath, String content, double score) {
            this.filePath = filePath;
            this.content = content;
            this.score = score;
        }
        
        public String getFilePath() {
            return filePath;
        }
        
        public void setFilePath(String filePath) {
            this.filePath = filePath;
        }
        
        public String getContent() {
            return content;
        }
        
        public void setContent(String content) {
            this.content = content;
        }
        
        public double getScore() {
            return score;
        }
        
        public void setScore(double score) {
            this.score = score;
        }
        
        public Map<String, Object> getMetadata() {
            return metadata;
        }
        
        public void setMetadata(Map<String, Object> metadata) {
            this.metadata = metadata;
        }
    }
}
