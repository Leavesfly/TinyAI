package io.leavesfly.tinyai.agent.cursor.v2.component;

import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.Context;
import io.leavesfly.tinyai.agent.cursor.v2.model.Context.CodeSnippet;
import io.leavesfly.tinyai.agent.cursor.v2.model.Context.CursorPosition;
import io.leavesfly.tinyai.agent.cursor.v2.model.Memory;
import io.leavesfly.tinyai.agent.cursor.v2.model.Memory.MemoryType;
import io.leavesfly.tinyai.agent.cursor.v2.model.Message;

import java.util.*;

/**
 * 上下文引擎
 * 负责编排和组织各种上下文信息，为LLM提供最优的上下文
 * 
 * 核心职责：
 * 1. 整合记忆管理器和RAG引擎的检索结果
 * 2. 根据任务类型智能选择上下文信息
 * 3. 控制上下文长度，避免超出模型限制
 * 4. 优化上下文质量，提高LLM响应准确性
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
     * 最大上下文Token数（默认8000）
     */
    private int maxContextTokens = 8000;
    
    /**
     * 每个代码片段预估Token数（平均）
     */
    private int avgTokensPerSnippet = 200;
    
    /**
     * 项目规则缓存
     */
    private final Map<String, List<String>> projectRulesCache;
    
    public ContextEngine(MemoryManager memoryManager, RAGEngine ragEngine) {
        this.memoryManager = memoryManager;
        this.ragEngine = ragEngine;
        this.projectRulesCache = new HashMap<>();
    }
    
    /**
     * 构建完整上下文（用于聊天场景）
     * 
     * @param sessionId 会话ID
     * @param projectId 项目ID
     * @param userQuery 用户查询
     * @param currentFile 当前文件路径
     * @param currentCode 当前代码内容
     * @return 完整的上下文对象
     */
    public Context buildChatContext(String sessionId, String projectId, String userQuery,
                                   String currentFile, String currentCode) {
        Context context = new Context();
        
        // 1. 设置当前文件信息
        if (currentFile != null && currentCode != null) {
            context.setCurrentFilePath(currentFile);
            context.setCurrentFileContent(currentCode);
        }
        
        // 2. 检索相关代码片段（RAG）
        List<CodeSnippet> relatedSnippets = ragEngine.semanticSearch(userQuery, 5, projectId);
        context.setRelatedCodeSnippets(relatedSnippets);
        
        // 3. 检索短期记忆（会话历史）
        List<Memory> shortTermMemories = memoryManager.retrieveSessionMemories(sessionId, MemoryType.SHORT_TERM);
        List<Message> conversationHistory = convertMemoriesToMessages(shortTermMemories);
        context.setConversationHistory(conversationHistory);
        
        // 4. 检索长期记忆（项目规则、用户偏好）
        List<Memory> longTermMemories = memoryManager.retrieveProjectMemories(projectId, MemoryType.LONG_TERM);
        context.setLongTermMemories(longTermMemories);
        
        // 5. 加载项目规则
        List<String> projectRules = loadProjectRules(projectId);
        context.setProjectRules(projectRules);
        
        // 6. 优化上下文长度
        optimizeContextLength(context);
        
        return context;
    }
    
    /**
     * 构建代码补全上下文
     * 
     * @param projectId 项目ID
     * @param filePath 文件路径
     * @param prefix 光标前代码
     * @param suffix 光标后代码
     * @param cursorPosition 光标位置
     * @return 代码补全上下文
     */
    public Context buildCompletionContext(String projectId, String filePath,
                                         String prefix, String suffix,
                                         CursorPosition cursorPosition) {
        Context context = new Context();
        
        // 1. 设置当前文件信息
        context.setCurrentFilePath(filePath);
        context.setCurrentFileContent(prefix + suffix);
        context.setCursorPosition(cursorPosition);
        
        // 2. 提取当前代码的关键词
        List<String> keywords = extractCodeKeywords(prefix);
        
        // 3. 检索相关代码片段（混合检索）
        List<CodeSnippet> relatedSnippets = ragEngine.hybridSearch(
            prefix, keywords, 3, projectId
        );
        context.setRelatedCodeSnippets(relatedSnippets);
        
        // 4. 加载项目规则（编码规范）
        List<String> projectRules = loadProjectRules(projectId);
        context.setProjectRules(projectRules);
        
        return context;
    }
    
    /**
     * 构建代码分析上下文
     * 
     * @param projectId 项目ID
     * @param targetCode 目标代码
     * @param analysisType 分析类型（review/refactor/debug等）
     * @return 代码分析上下文
     */
    public Context buildAnalysisContext(String projectId, String targetCode, String analysisType) {
        Context context = new Context();
        
        // 1. 设置目标代码
        context.setCurrentFileContent(targetCode);
        context.putExtra("analysisType", analysisType);
        
        // 2. 检索相似代码（作为参考）
        List<CodeSnippet> similarSnippets = ragEngine.semanticSearch(targetCode, 3, projectId);
        context.setRelatedCodeSnippets(similarSnippets);
        
        // 3. 加载项目规则
        List<String> projectRules = loadProjectRules(projectId);
        context.setProjectRules(projectRules);
        
        // 4. 检索相关长期记忆（代码模式、最佳实践）
        List<Memory> relevantMemories = memoryManager.retrieveProjectMemories(projectId, MemoryType.LONG_TERM);
        context.setLongTermMemories(relevantMemories);
        
        return context;
    }
    
    /**
     * 构建语义检索上下文
     * 
     * @param projectId 项目ID
     * @param query 查询文本
     * @param queryEmbedding 查询向量（可选）
     * @return 检索上下文
     */
    public Context buildSemanticSearchContext(String projectId, String query, double[] queryEmbedding) {
        Context context = new Context();
        
        // 1. RAG语义检索
        List<CodeSnippet> ragResults = ragEngine.semanticSearch(query, 10, projectId);
        context.setRelatedCodeSnippets(ragResults);
        
        // 2. 记忆语义检索
        if (queryEmbedding != null) {
            List<Memory> semanticMemories = memoryManager.retrieveSimilarMemories(
                queryEmbedding, 5, 0.7
            );
            context.setLongTermMemories(semanticMemories);
        }
        
        return context;
    }
    
    /**
     * 更新上下文（追加新信息）
     * 
     * @param context 现有上下文
     * @param newSnippets 新的代码片段
     * @param newMessages 新的对话消息
     */
    public void updateContext(Context context, List<CodeSnippet> newSnippets, List<Message> newMessages) {
        if (newSnippets != null) {
            for (CodeSnippet snippet : newSnippets) {
                context.addCodeSnippet(snippet);
            }
        }
        
        if (newMessages != null) {
            for (Message message : newMessages) {
                context.addConversationMessage(message);
            }
        }
        
        // 重新优化上下文长度
        optimizeContextLength(context);
    }
    
    /**
     * 优化上下文长度
     * 确保不超过模型的最大token限制
     */
    private void optimizeContextLength(Context context) {
        int estimatedTokens = estimateContextTokens(context);
        
        if (estimatedTokens <= maxContextTokens) {
            return; // 无需优化
        }
        
        // 策略1: 减少代码片段数量
        List<CodeSnippet> snippets = context.getRelatedCodeSnippets();
        if (snippets.size() > 3) {
            // 保留前3个最相关的
            context.setRelatedCodeSnippets(snippets.subList(0, 3));
            estimatedTokens = estimateContextTokens(context);
        }
        
        if (estimatedTokens <= maxContextTokens) {
            return;
        }
        
        // 策略2: 压缩会话历史
        List<Message> history = context.getConversationHistory();
        if (history.size() > 5) {
            // 保留最近5条
            context.setConversationHistory(
                history.subList(Math.max(0, history.size() - 5), history.size())
            );
            estimatedTokens = estimateContextTokens(context);
        }
        
        if (estimatedTokens <= maxContextTokens) {
            return;
        }
        
        // 策略3: 减少长期记忆数量
        List<Memory> longTermMemories = context.getLongTermMemories();
        if (longTermMemories.size() > 3) {
            context.setLongTermMemories(longTermMemories.subList(0, 3));
        }
    }
    
    /**
     * 估算上下文Token数
     */
    private int estimateContextTokens(Context context) {
        int tokens = 0;
        
        // 当前文件
        if (context.getCurrentFileContent() != null) {
            tokens += estimateTokens(context.getCurrentFileContent());
        }
        
        // 代码片段
        for (CodeSnippet snippet : context.getRelatedCodeSnippets()) {
            tokens += estimateTokens(snippet.getContent());
        }
        
        // 会话历史
        for (Message message : context.getConversationHistory()) {
            tokens += estimateTokens(message.getContent());
        }
        
        // 项目规则
        for (String rule : context.getProjectRules()) {
            tokens += estimateTokens(rule);
        }
        
        // 长期记忆
        for (Memory memory : context.getLongTermMemories()) {
            tokens += estimateTokens(memory.getContent());
        }
        
        return tokens;
    }
    
    /**
     * 估算文本Token数（简单按字符数/4估算）
     */
    private int estimateTokens(String text) {
        if (text == null) {
            return 0;
        }
        return text.length() / 4;
    }
    
    /**
     * 加载项目规则
     */
    private List<String> loadProjectRules(String projectId) {
        // 先从缓存获取
        if (projectRulesCache.containsKey(projectId)) {
            return projectRulesCache.get(projectId);
        }
        
        // 从记忆系统加载
        List<String> rules = new ArrayList<>();
        List<Memory> ruleMemories = memoryManager.retrieveProjectMemories(projectId, MemoryType.LONG_TERM);
        
        for (Memory memory : ruleMemories) {
            Object type = memory.getMetadata("type");
            if ("rule".equals(type)) {
                rules.add(memory.getContent());
            }
        }
        
        // 如果没有规则，添加默认规则
        if (rules.isEmpty()) {
            rules.add("Follow clean code principles");
            rules.add("Add appropriate comments for complex logic");
            rules.add("Handle exceptions properly");
        }
        
        // 缓存规则
        projectRulesCache.put(projectId, rules);
        
        return rules;
    }
    
    /**
     * 添加项目规则
     */
    public void addProjectRule(String projectId, String rule) {
        // 创建长期记忆
        Memory ruleMemory = Memory.longTerm(
            "rule_" + UUID.randomUUID().toString(),
            projectId,
            rule
        );
        ruleMemory.putMetadata("type", "rule");
        ruleMemory.setImportance(0.9); // 规则很重要
        
        memoryManager.addMemory(ruleMemory);
        
        // 更新缓存
        projectRulesCache.remove(projectId);
    }
    
    /**
     * 提取代码关键词
     */
    private List<String> extractCodeKeywords(String code) {
        if (code == null || code.isEmpty()) {
            return Collections.emptyList();
        }
        
        List<String> keywords = new ArrayList<>();
        
        // 提取类名、方法名等（简单的正则匹配）
        String[] words = code.split("[^a-zA-Z0-9_]+");
        
        for (String word : words) {
            if (word.length() > 3) {
                // 驼峰命名或全大写的词可能是类名/方法名
                if (Character.isUpperCase(word.charAt(0)) || word.equals(word.toUpperCase())) {
                    keywords.add(word);
                }
            }
        }
        
        return keywords.stream()
            .distinct()
            .limit(10)
            .collect(java.util.stream.Collectors.toList());
    }
    
    /**
     * 将记忆转换为消息
     */
    private List<Message> convertMemoriesToMessages(List<Memory> memories) {
        List<Message> messages = new ArrayList<>();
        
        for (Memory memory : memories) {
            // 从元数据提取消息信息
            Object roleObj = memory.getMetadata("role");
            if (roleObj != null) {
                try {
                    Message.Role role = Message.Role.fromValue(roleObj.toString());
                    Message message = new Message();
                    message.setRole(role);
                    message.setContent(memory.getContent());
                    messages.add(message);
                } catch (IllegalArgumentException e) {
                    // 忽略无效的角色
                }
            }
        }
        
        return messages;
    }
    
    /**
     * 保存会话消息到记忆
     */
    public void saveMessageToMemory(String sessionId, Message message, double importance) {
        Memory memory = Memory.shortTerm(
            "msg_" + UUID.randomUUID().toString(),
            sessionId,
            message.getContent()
        );
        memory.putMetadata("role", message.getRole().getValue());
        memory.setImportance(importance);
        
        memoryManager.addMemory(memory);
    }
    
    /**
     * 配置项
     */
    public void setMaxContextTokens(int maxTokens) {
        this.maxContextTokens = maxTokens;
    }
    
    public void setAvgTokensPerSnippet(int avgTokens) {
        this.avgTokensPerSnippet = avgTokens;
    }
    
    /**
     * 清除项目规则缓存
     */
    public void clearRulesCache(String projectId) {
        if (projectId == null) {
            projectRulesCache.clear();
        } else {
            projectRulesCache.remove(projectId);
        }
    }
    
    /**
     * 获取统计信息
     */
    public ContextStats getStats() {
        ContextStats stats = new ContextStats();
        stats.cachedProjectRules = projectRulesCache.size();
        stats.memoryStats = memoryManager.getStats();
        stats.ragStats = ragEngine.getStats();
        return stats;
    }
    
    /**
     * 上下文统计信息
     */
    public static class ContextStats {
        public int cachedProjectRules;
        public MemoryManager.MemoryStats memoryStats;
        public RAGEngine.RAGStats ragStats;
        
        @Override
        public String toString() {
            return "ContextStats{" +
                    "cachedRules=" + cachedProjectRules +
                    ", memory=" + memoryStats +
                    ", rag=" + ragStats +
                    '}';
        }
    }
}
