package io.leavesfly.tinyai.agent.research.v2.service;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;

import java.util.Map;

/**
 * 执行智能体
 * 负责协调各专业智能体执行具体研究任务
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ExecutorAgent {
    
    private final ResearchLLMAdapter llmAdapter;
    private final ContextEngine contextEngine;
    private final MemoryManager memoryManager;
    private final SearcherAgent searcherAgent;
    private final AnalyzerAgent analyzerAgent;
    
    public ExecutorAgent(ResearchLLMAdapter llmAdapter, 
                        ContextEngine contextEngine,
                        MemoryManager memoryManager) {
        this.llmAdapter = llmAdapter;
        this.contextEngine = contextEngine;
        this.memoryManager = memoryManager;
        this.searcherAgent = new SearcherAgent(llmAdapter);
        this.analyzerAgent = new AnalyzerAgent(llmAdapter);
    }
    
    /**
     * 执行单个研究问题
     */
    public Object executeQuestion(ResearchQuestion question, Map<String, Object> previousResults) {
        System.out.println("[ExecutorAgent] 执行问题: " + question.getContent());
        
        try {
            // 根据问题类型选择智能体
            AgentType agent = question.getRecommendedAgent();
            
            switch (agent) {
                case SEARCHER:
                    return searcherAgent.search(question);
                    
                case ANALYZER:
                    return analyzerAgent.analyze(question, previousResults);
                    
                case SYNTHESIZER:
                    return synthesize(question, previousResults);
                    
                default:
                    return searcherAgent.search(question);
            }
            
        } catch (Exception e) {
            System.err.println("[ExecutorAgent] 问题执行失败: " + e.getMessage());
            return "执行失败: " + e.getMessage();
        }
    }
    
    /**
     * 综合结果
     */
    private Object synthesize(ResearchQuestion question, Map<String, Object> previousResults) {
        // 收集依赖问题的结果
        StringBuilder context = new StringBuilder();
        for (String depId : question.getDependencies()) {
            Object result = previousResults.get(depId);
            if (result != null) {
                context.append(result.toString()).append("\n\n");
            }
        }
        
        // 使用分析能力进行综合
        ChatResponse response = llmAdapter.analysisChat(
            question.getContent(),
            context.toString()
        );
        
        return response.getContent();
    }
}
