package io.leavesfly.tinyai.agent.research.v2.service;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.ResearchReport;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;

import java.util.Map;

/**
 * 写作智能体
 * 负责生成研究报告
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class WriterAgent {
    
    private final ResearchLLMAdapter llmAdapter;
    
    public WriterAgent(ResearchLLMAdapter llmAdapter) {
        this.llmAdapter = llmAdapter;
    }
    
    /**
     * 生成研究报告
     */
    public ResearchReport generateReport(String topic, 
                                        Map<String, Object> executionResults,
                                        Map<String, Object> analysisResults) {
        System.out.println("[WriterAgent] 生成研究报告: " + topic);
        
        // 整理发现和洞察
        StringBuilder findings = new StringBuilder();
        if (executionResults != null) {
            for (Map.Entry<String, Object> entry : executionResults.entrySet()) {
                findings.append("- ").append(entry.getValue()).append("\\n");
            }
        }
        
        StringBuilder insights = new StringBuilder();
        if (analysisResults != null && analysisResults.containsKey("keyFindings")) {
            Object keyFindings = analysisResults.get("keyFindings");
            insights.append(keyFindings.toString());
        }
        
        // 调用LLM生成报告
        ChatResponse response = llmAdapter.writingChat(
            topic,
            findings.toString(),
            insights.toString()
        );
        
        // 构建报告对象
        ResearchReport report = new ResearchReport();
        report.setTitle("研究报告: " + topic);
        report.setSummary("本报告针对\"" + topic + "\"进行了深入研究");
        report.setFullContent(response.getContent());
        report.setQualityScore(80.0);
        
        return report;
    }
}
