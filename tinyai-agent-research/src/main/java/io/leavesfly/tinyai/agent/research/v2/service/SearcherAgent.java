package io.leavesfly.tinyai.agent.research.v2.service;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.ResearchQuestion;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;

/**
 * 检索智能体
 * 负责信息检索和数据收集
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class SearcherAgent {
    
    private final ResearchLLMAdapter llmAdapter;
    
    public SearcherAgent(ResearchLLMAdapter llmAdapter) {
        this.llmAdapter = llmAdapter;
    }
    
    /**
     * 执行检索任务
     */
    public String search(ResearchQuestion question) {
        System.out.println("[SearcherAgent] 检索问题: " + question.getContent());
        
        // 模拟检索过程(实际应该调用搜索工具)
        String simulatedRetrievalContent = "关于\"" + question.getContent() + "\"的检索结果...\\n" +
            "这是模拟的检索内容。实际系统中，这里会调用网络搜索API、知识库检索等。";
        
        // 使用LLM处理和精炼检索结果
        ChatResponse response = llmAdapter.searchChat(
            question.getContent(),
            simulatedRetrievalContent
        );
        
        return response.getContent();
    }
}
