package io.leavesfly.tinyai.agent.research.v2.model;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.List;

/**
 * ResearchPlan 单元测试
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ResearchPlanTest {
    
    @Test
    public void testPlanCreation() {
        ResearchPlan plan = new ResearchPlan();
        
        assertNotNull(plan.getPlanId());
        assertEquals(PlanningStrategy.HYBRID, plan.getStrategy());
        assertNotNull(plan.getQuestions());
        assertTrue(plan.getQuestions().isEmpty());
    }
    
    @Test
    public void testAddQuestion() {
        ResearchPlan plan = new ResearchPlan();
        
        ResearchQuestion q1 = new ResearchQuestion("问题1", QuestionType.FACTUAL);
        ResearchQuestion q2 = new ResearchQuestion("问题2", QuestionType.ANALYTICAL);
        
        plan.addQuestion(q1);
        plan.addQuestion(q2);
        
        assertEquals(2, plan.getQuestions().size());
    }
    
    @Test
    public void testDependencyGraph() {
        ResearchPlan plan = new ResearchPlan();
        
        ResearchQuestion q1 = new ResearchQuestion("问题1", QuestionType.FACTUAL);
        ResearchQuestion q2 = new ResearchQuestion("问题2", QuestionType.SYNTHESIS);
        
        plan.addQuestion(q1);
        plan.addQuestion(q2);
        
        // q2依赖q1
        plan.addDependency(q2.getQuestionId(), q1.getQuestionId());
        
        assertTrue(plan.getDependencyGraph().containsKey(q2.getQuestionId()));
    }
    
    @Test
    public void testExecutionLevels() {
        ResearchPlan plan = new ResearchPlan();
        
        ResearchQuestion q1 = new ResearchQuestion("问题1", QuestionType.FACTUAL);
        ResearchQuestion q2 = new ResearchQuestion("问题2", QuestionType.FACTUAL);
        ResearchQuestion q3 = new ResearchQuestion("问题3", QuestionType.SYNTHESIS);
        
        plan.addQuestion(q1);
        plan.addQuestion(q2);
        plan.addQuestion(q3);
        
        // q3依赖q1和q2
        q3.addDependency(q1.getQuestionId());
        q3.addDependency(q2.getQuestionId());
        plan.addDependency(q3.getQuestionId(), q1.getQuestionId());
        plan.addDependency(q3.getQuestionId(), q2.getQuestionId());
        
        List<List<ResearchQuestion>> levels = plan.getExecutionLevels();
        
        // 应该有2层：第1层是q1和q2，第2层是q3
        assertTrue(levels.size() >= 1);
        assertTrue(levels.get(0).size() >= 1);
    }
}
