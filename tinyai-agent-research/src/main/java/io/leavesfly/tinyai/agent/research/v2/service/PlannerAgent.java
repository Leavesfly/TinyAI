package io.leavesfly.tinyai.agent.research.v2.service;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;

import java.util.*;

/**
 * 规划智能体
 * 负责分析研究主题并制定研究计划
 * 
 * 核心职责:
 * 1. 将研究主题分解为子问题
 * 2. 识别问题类型和推荐智能体
 * 3. 构建问题依赖关系(DAG)
 * 4. 估算执行时间和资源需求
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class PlannerAgent {
    
    private final ResearchLLMAdapter llmAdapter;
    
    public PlannerAgent(ResearchLLMAdapter llmAdapter) {
        this.llmAdapter = llmAdapter;
    }
    
    /**
     * 创建研究计划
     */
    public ResearchPlan createPlan(String topic, Map<String, Object> context) {
        System.out.println("[PlannerAgent] 为主题创建研究计划: " + topic);
        
        ResearchPlan plan = new ResearchPlan();
        plan.setStrategy(PlanningStrategy.HYBRID);
        
        // 使用LLM生成子问题
        String contextStr = context != null ? context.toString() : "";
        ChatResponse response = llmAdapter.planningChat(topic, contextStr);
        
        // 解析LLM响应生成问题列表
        List<ResearchQuestion> questions = parseQuestions(response.getContent());
        
        // 添加问题到计划
        for (ResearchQuestion q : questions) {
            plan.addQuestion(q);
        }
        
        // 构建依赖关系(简化版本,实际可由LLM生成)
        buildDependencies(plan, questions);
        
        // 估算深度
        plan.setEstimatedDepth(estimateDepth(plan));
        
        System.out.println(String.format("[PlannerAgent] 计划创建完成,共%d个问题", questions.size()));
        
        return plan;
    }
    
    /**
     * 解析问题列表
     */
    private List<ResearchQuestion> parseQuestions(String llmResponse) {
        List<ResearchQuestion> questions = new ArrayList<>();
        
        // 简单解析(实际应该更智能)
        String[] lines = llmResponse.split("\n");
        
        for (String line : lines) {
            if (line.trim().isEmpty()) continue;
            
            // 检测问题标记
            if (line.matches(".*[\\d]+[.、].*") || line.contains("?") || line.contains("？")) {
                ResearchQuestion q = new ResearchQuestion();
                q.setContent(line.trim());
                q.setType(inferQuestionType(line));
                q.setPriority(5);
                questions.add(q);
            }
        }
        
        // 如果没有解析到问题,创建默认问题
        if (questions.isEmpty()) {
            ResearchQuestion defaultQ = new ResearchQuestion();
            defaultQ.setContent("研究" + llmResponse.substring(0, Math.min(50, llmResponse.length())));
            defaultQ.setType(QuestionType.EXPLORATORY);
            questions.add(defaultQ);
        }
        
        return questions;
    }
    
    /**
     * 推断问题类型
     */
    private QuestionType inferQuestionType(String question) {
        String lower = question.toLowerCase();
        
        if (lower.contains("是什么") || lower.contains("定义") || lower.contains("概念")) {
            return QuestionType.FACTUAL;
        } else if (lower.contains("为什么") || lower.contains("原因") || lower.contains("分析")) {
            return QuestionType.ANALYTICAL;
        } else if (lower.contains("比较") || lower.contains("对比") || lower.contains("vs")) {
            return QuestionType.COMPARATIVE;
        } else if (lower.contains("综合") || lower.contains("整合") || lower.contains("总结")) {
            return QuestionType.SYNTHESIS;
        } else {
            return QuestionType.EXPLORATORY;
        }
    }
    
    /**
     * 构建依赖关系
     */
    private void buildDependencies(ResearchPlan plan, List<ResearchQuestion> questions) {
        // 简单策略:后面的问题可能依赖前面的基础问题
        for (int i = 1; i < questions.size(); i++) {
            ResearchQuestion current = questions.get(i);
            ResearchQuestion previous = questions.get(i - 1);
            
            // 综合性问题通常依赖其他问题
            if (current.getType() == QuestionType.SYNTHESIS) {
                for (int j = 0; j < i; j++) {
                    current.addDependency(questions.get(j).getQuestionId());
                    plan.addDependency(current.getQuestionId(), questions.get(j).getQuestionId());
                }
            }
        }
    }
    
    /**
     * 估算研究深度
     */
    private int estimateDepth(ResearchPlan plan) {
        List<List<ResearchQuestion>> levels = plan.getExecutionLevels();
        return levels.size();
    }
}
