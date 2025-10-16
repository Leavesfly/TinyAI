package io.leavesfly.tinyai.agent.research.v2.service;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.ResearchQuestion;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;

import java.util.Map;

/**
 * 分析智能体
 * 负责深度分析和推理
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class AnalyzerAgent {
    
    private final ResearchLLMAdapter llmAdapter;
    
    public AnalyzerAgent(ResearchLLMAdapter llmAdapter) {
        this.llmAdapter = llmAdapter;
    }
    
    /**
     * 执行分析任务
     */
    public String analyze(ResearchQuestion question, Map<String, Object> previousResults) {
        System.out.println("[AnalyzerAgent] 分析问题: " + question.getContent());
        
        // 收集相关数据
        StringBuilder dataContext = new StringBuilder();
        if (previousResults != null && !previousResults.isEmpty()) {
            for (Map.Entry<String, Object> entry : previousResults.entrySet()) {
                dataContext.append(entry.getValue()).append("\\n\\n");
            }
        }
        
        // 使用LLM进行深度分析
        ChatResponse response = llmAdapter.analysisChat(
            question.getContent(),
            dataContext.toString()
        );
        
        return response.getContent();
    }
}
