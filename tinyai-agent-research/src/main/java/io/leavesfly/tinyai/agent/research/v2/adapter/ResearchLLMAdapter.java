package io.leavesfly.tinyai.agent.research.v2.adapter;

import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.LLMGateway;
import io.leavesfly.tinyai.agent.cursor.v2.service.StreamCallback;

import java.util.*;

/**
 * 研究系统LLM适配器
 * 复用tinyai-agent-cursor的LLMGateway接口
 * 提供研究场景下的模型调用能力
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ResearchLLMAdapter {
    
    /**
     * LLM网关（复用cursor模块）
     */
    private final LLMGateway llmGateway;
    
    /**
     * 默认研究模型配置
     */
    private static final Map<String, String> RESEARCH_MODEL_CONFIG = new HashMap<>();
    
    static {
        // 研究规划 - 使用推理能力强的模型
        RESEARCH_MODEL_CONFIG.put("planning", "deepseek-reasoner");
        // 信息检索 - 使用通用模型
        RESEARCH_MODEL_CONFIG.put("search", "deepseek-chat");
        // 深度分析 - 使用推理模型
        RESEARCH_MODEL_CONFIG.put("analysis", "deepseek-reasoner");
        // 报告生成 - 使用长文本模型
        RESEARCH_MODEL_CONFIG.put("writing", "qwen-max");
    }
    
    public ResearchLLMAdapter(LLMGateway llmGateway) {
        this.llmGateway = llmGateway;
    }
    
    /**
     * 研究规划调用
     * 用于生成研究计划和问题分解
     */
    public ChatResponse planningChat(String topic, String context) {
        ChatRequest request = buildChatRequest(
            "planning",
            buildPlanningPrompt(topic, context),
            4000
        );
        return llmGateway.chat(request);
    }
    
    /**
     * 信息检索调用
     * 用于生成检索查询和筛选结果
     */
    public ChatResponse searchChat(String question, String retrievedContent) {
        ChatRequest request = buildChatRequest(
            "search",
            buildSearchPrompt(question, retrievedContent),
            2000
        );
        return llmGateway.chat(request);
    }
    
    /**
     * 深度分析调用
     * 用于执行多步推理和洞察提取
     */
    public ChatResponse analysisChat(String question, String data) {
        ChatRequest request = buildChatRequest(
            "analysis",
            buildAnalysisPrompt(question, data),
            6000
        );
        return llmGateway.chat(request);
    }
    
    /**
     * 报告生成调用
     * 用于撰写最终研究报告
     */
    public ChatResponse writingChat(String topic, String findings, String insights) {
        ChatRequest request = buildChatRequest(
            "writing",
            buildWritingPrompt(topic, findings, insights),
            8000
        );
        return llmGateway.chat(request);
    }
    
    /**
     * 流式分析调用
     */
    public void analysisStreamChat(String question, String data, StreamCallback callback) {
        ChatRequest request = buildChatRequest(
            "analysis",
            buildAnalysisPrompt(question, data),
            6000
        );
        llmGateway.chatStream(request, callback);
    }
    
    /**
     * 文本向量化（用于语义检索）
     */
    public double[] embedText(String text) {
        return llmGateway.embedSingle(text);
    }
    
    /**
     * 批量文本向量化
     */
    public List<double[]> embedTexts(List<String> texts) {
        return llmGateway.embed(texts);
    }
    
    /**
     * 构建聊天请求
     */
    private ChatRequest buildChatRequest(String taskType, String prompt, int maxTokens) {
        ChatRequest request = new ChatRequest();
        
        // 选择模型
        String modelName = RESEARCH_MODEL_CONFIG.getOrDefault(taskType, "deepseek-chat");
        request.setModel(modelName);
        
        // 构建消息
        Message systemMessage = new Message();
//        systemMessage.setRole("system");
        systemMessage.setContent(getSystemPromptForTaskType(taskType));
        
        Message userMessage = new Message();
//        userMessage.setRole("user");
        userMessage.setContent(prompt);
        
        request.setMessages(Arrays.asList(systemMessage, userMessage));
        request.setMaxTokens(maxTokens);
        request.setTemperature(0.7);
        
        return request;
    }
    
    /**
     * 获取任务类型的系统提示词
     */
    private String getSystemPromptForTaskType(String taskType) {
        switch (taskType) {
            case "planning":
                return "你是一个专业的研究规划助手，擅长将复杂主题分解为结构化的研究问题。";
            case "search":
                return "你是一个信息检索专家，擅长从海量信息中提取关键内容并去重整合。";
            case "analysis":
                return "你是一个深度分析专家，擅长多步推理、模式识别和洞察提取。";
            case "writing":
                return "你是一个专业的研究报告撰写者，擅长将复杂信息组织成清晰易懂的报告。";
            default:
                return "你是一个AI研究助手。";
        }
    }
    
    /**
     * 构建规划提示词
     */
    private String buildPlanningPrompt(String topic, String context) {
        return String.format(
            "研究主题: %s\n\n" +
            "上下文信息:\n%s\n\n" +
            "请将此研究主题分解为3-5个核心子问题，并说明它们之间的依赖关系。" +
            "每个子问题应该明确、可执行，并指定问题类型（事实性/分析性/综合性等）。",
            topic,
            context != null ? context : "无"
        );
    }
    
    /**
     * 构建检索提示词
     */
    private String buildSearchPrompt(String question, String retrievedContent) {
        return String.format(
            "研究问题: %s\n\n" +
            "检索到的内容:\n%s\n\n" +
            "请从以上内容中提取与问题相关的关键信息，去除冗余，并总结核心要点。",
            question,
            retrievedContent
        );
    }
    
    /**
     * 构建分析提示词
     */
    private String buildAnalysisPrompt(String question, String data) {
        return String.format(
            "研究问题: %s\n\n" +
            "数据和信息:\n%s\n\n" +
            "请对以上信息进行深度分析，包括：\n" +
            "1. 识别关键模式和趋势\n" +
            "2. 提取核心洞察\n" +
            "3. 发现潜在矛盾或知识缺口\n" +
            "4. 评估信息可信度",
            question,
            data
        );
    }
    
    /**
     * 构建写作提示词
     */
    private String buildWritingPrompt(String topic, String findings, String insights) {
        return String.format(
            "研究主题: %s\n\n" +
            "研究发现:\n%s\n\n" +
            "核心洞察:\n%s\n\n" +
            "请基于以上信息撰写一份结构化的研究报告，包括：\n" +
            "1. 执行摘要\n" +
            "2. 详细发现\n" +
            "3. 核心洞察\n" +
            "4. 结论和建议\n" +
            "5. 引用来源",
            topic,
            findings,
            insights
        );
    }
    
    /**
     * 获取底层网关
     */
    public LLMGateway getLLMGateway() {
        return llmGateway;
    }
}
