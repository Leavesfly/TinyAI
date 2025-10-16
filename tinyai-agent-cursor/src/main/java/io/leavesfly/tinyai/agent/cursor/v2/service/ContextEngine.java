package io.leavesfly.tinyai.agent.cursor.v2.service;

import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;

import java.util.ArrayList;
import java.util.List;

/**
 * 上下文引擎
 * 整合多源上下文信息，构建完整的上下文环境
 * 
 * 上下文来源：
 * 1. 当前文件内容
 * 2. RAG检索的相关代码
 * 3. 会话历史记忆
 * 4. 项目规则和长期记忆
 * 5. 用户自定义上下文
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ContextEngine {
    
    /**
     * 记忆管理器
     */
    private final MemoryManager memoryManager;
    
    /**
     * RAG检索引擎
     */
    private final RAGEngine ragEngine;
    
    /**
     * 最大上下文长度（tokens，默认4000）
     */
    private int maxContextLength = 4000;
    
    /**
     * RAG检索Top-K数量
     */
    private int ragTopK = 5;
    
    /**
     * 会话历史消息数量
     */
    private int maxHistoryMessages = 10;
    
    public ContextEngine(MemoryManager memoryManager, RAGEngine ragEngine) {
        this.memoryManager = memoryManager;
        this.ragEngine = ragEngine;
    }
    
    /**
     * 构建上下文
     * 
     * @param request 上下文请求
     * @return 完整的上下文对象
     */
    public Context buildContext(ContextRequest request) {
        Context context = new Context();
        
        // 1. 当前文件内容
        if (request.currentFilePath != null) {
            context.setCurrentFilePath(request.currentFilePath);
            context.setCurrentFileContent(request.currentFileContent);
            context.setCursorPosition(request.cursorPosition);
        }
        
        // 2. RAG检索相关代码
        if (request.enableRAG && request.query != null) {
            List<Context.CodeSnippet> relatedSnippets = ragEngine.semanticSearch(
                request.query, 
                ragTopK, 
                request.projectId
            );
            
            for (Context.CodeSnippet snippet : relatedSnippets) {
                context.addCodeSnippet(snippet);
            }
        }
        
        // 3. 会话历史
        if (request.sessionId != null) {
            List<Memory> sessionMemories = memoryManager.retrieveSessionMemories(
                request.sessionId, 
                Memory.MemoryType.SHORT_TERM
            );
            
            // 转换为消息列表
            for (Memory memory : sessionMemories) {
                // 假设记忆内容是消息的JSON或文本表示
                // TODO: 实现更完善的消息序列化/反序列化
                context.addConversationMessage(Message.assistant(memory.getContent()));
            }
            
            // 限制历史消息数量
            List<Message> allMessages = context.getConversationHistory();
            if (allMessages.size() > maxHistoryMessages) {
                List<Message> recentMessages = allMessages.subList(
                    allMessages.size() - maxHistoryMessages, 
                    allMessages.size()
                );
                context.setConversationHistory(new ArrayList<>(recentMessages));
            }
        }
        
        // 4. 项目规则
        if (request.projectId != null) {
            List<Memory> projectRules = memoryManager.retrieveProjectMemories(
                request.projectId, 
                Memory.MemoryType.LONG_TERM
            );
            
            for (Memory memory : projectRules) {
                context.addProjectRule(memory.getContent());
            }
        }
        
        // 5. 长期记忆
        if (request.includeLongTermMemory && request.projectId != null) {
            List<Memory> longTermMemories = memoryManager.retrieveProjectMemories(
                request.projectId, 
                Memory.MemoryType.LONG_TERM
            );
            
            for (Memory memory : longTermMemories) {
                context.addLongTermMemory(memory);
            }
        }
        
        // 6. 自定义上下文
        if (request.extraContext != null) {
            for (String key : request.extraContext.keySet()) {
                context.putExtra(key, request.extraContext.get(key));
            }
        }
        
        // 7. 上下文优化（裁剪过长内容）
        optimizeContext(context);
        
        return context;
    }
    
    /**
     * 快速构建代码补全上下文
     */
    public Context buildCompletionContext(String filePath, String fileContent, 
                                         Context.CursorPosition cursorPosition, 
                                         String projectId) {
        ContextRequest request = new ContextRequest();
        request.currentFilePath = filePath;
        request.currentFileContent = fileContent;
        request.cursorPosition = cursorPosition;
        request.projectId = projectId;
        request.enableRAG = true;
        request.query = extractContextQuery(fileContent, cursorPosition);
        
        return buildContext(request);
    }
    
    /**
     * 快速构建对话上下文
     */
    public Context buildChatContext(String query, String sessionId, String projectId) {
        ContextRequest request = new ContextRequest();
        request.query = query;
        request.sessionId = sessionId;
        request.projectId = projectId;
        request.enableRAG = true;
        request.includeLongTermMemory = true;
        
        return buildContext(request);
    }
    
    /**
     * 优化上下文（控制长度）
     */
    private void optimizeContext(Context context) {
        // 估算当前上下文的token数（简单估算：字符数/4）
        int estimatedTokens = estimateTokens(context);
        
        if (estimatedTokens > maxContextLength) {
            // 优先级：当前文件 > 项目规则 > 相关代码 > 历史消息
            
            // 1. 裁剪相关代码片段
            List<Context.CodeSnippet> snippets = context.getRelatedCodeSnippets();
            while (estimatedTokens > maxContextLength && snippets.size() > 1) {
                snippets.remove(snippets.size() - 1);
                estimatedTokens = estimateTokens(context);
            }
            
            // 2. 裁剪历史消息
            List<Message> messages = context.getConversationHistory();
            while (estimatedTokens > maxContextLength && messages.size() > 3) {
                messages.remove(0);
                estimatedTokens = estimateTokens(context);
            }
            
            // 3. 裁剪长期记忆
            List<Memory> memories = context.getLongTermMemories();
            while (estimatedTokens > maxContextLength && memories.size() > 0) {
                memories.remove(memories.size() - 1);
                estimatedTokens = estimateTokens(context);
            }
        }
    }
    
    /**
     * 估算token数量
     */
    private int estimateTokens(Context context) {
        int tokens = 0;
        
        if (context.getCurrentFileContent() != null) {
            tokens += context.getCurrentFileContent().length() / 4;
        }
        
        for (Context.CodeSnippet snippet : context.getRelatedCodeSnippets()) {
            tokens += snippet.getContent().length() / 4;
        }
        
        for (Message message : context.getConversationHistory()) {
            if (message.getContent() != null) {
                tokens += message.getContent().length() / 4;
            }
        }
        
        for (String rule : context.getProjectRules()) {
            tokens += rule.length() / 4;
        }
        
        for (Memory memory : context.getLongTermMemories()) {
            tokens += memory.getContent().length() / 4;
        }
        
        return tokens;
    }
    
    /**
     * 提取上下文查询（从代码中提取关键信息作为查询）
     */
    private String extractContextQuery(String fileContent, Context.CursorPosition cursorPosition) {
        if (fileContent == null || cursorPosition == null) {
            return "";
        }
        
        // 提取光标附近的代码作为查询
        String[] lines = fileContent.split("\n");
        int line = cursorPosition.getLine();
        
        int start = Math.max(0, line - 5);
        int end = Math.min(lines.length, line + 5);
        
        StringBuilder query = new StringBuilder();
        for (int i = start; i < end; i++) {
            query.append(lines[i]).append("\n");
        }
        
        return query.toString();
    }
    
    /**
     * 保存上下文到记忆
     */
    public void saveContext(String sessionId, String projectId, Context context) {
        // 保存当前会话的上下文作为短期记忆
        if (sessionId != null && context.getCurrentFileContent() != null) {
            String memoryId = "ctx-" + System.currentTimeMillis();
            Memory memory = Memory.shortTerm(memoryId, sessionId, context.getCurrentFilePath());
            memoryManager.addMemory(memory);
        }
    }
    
    /**
     * 配置项
     */
    public void setMaxContextLength(int maxContextLength) {
        this.maxContextLength = maxContextLength;
    }
    
    public void setRagTopK(int ragTopK) {
        this.ragTopK = ragTopK;
    }
    
    public void setMaxHistoryMessages(int maxHistoryMessages) {
        this.maxHistoryMessages = maxHistoryMessages;
    }
    
    /**
     * 上下文请求
     */
    public static class ContextRequest {
        /** 查询文本 */
        public String query;
        
        /** 会话ID */
        public String sessionId;
        
        /** 项目ID */
        public String projectId;
        
        /** 当前文件路径 */
        public String currentFilePath;
        
        /** 当前文件内容 */
        public String currentFileContent;
        
        /** 光标位置 */
        public Context.CursorPosition cursorPosition;
        
        /** 是否启用RAG检索 */
        public boolean enableRAG = true;
        
        /** 是否包含长期记忆 */
        public boolean includeLongTermMemory = false;
        
        /** 自定义上下文 */
        public java.util.Map<String, Object> extraContext;
    }
}
