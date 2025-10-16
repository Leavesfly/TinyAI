package io.leavesfly.tinyai.agent;

import org.junit.Test;

import io.leavesfly.tinyai.agent.context.AdvancedAgent;
import io.leavesfly.tinyai.agent.context.ContextEngine;
import io.leavesfly.tinyai.agent.context.LLMSimulator;
import io.leavesfly.tinyai.agent.context.Memory;
import io.leavesfly.tinyai.agent.context.MemoryManager;
import io.leavesfly.tinyai.agent.context.Message;
import io.leavesfly.tinyai.agent.context.RAGSystem;
import io.leavesfly.tinyai.agent.context.RetrievalResult;
import io.leavesfly.tinyai.agent.context.ToolCall;
import io.leavesfly.tinyai.agent.context.ToolRegistry;

import static org.junit.Assert.*;

import java.util.*;

/**
 * AdvancedAgent 单元测试
 * 
 * @author 山泽
 */
public class AdvancedAgentTest {
    
    @Test
    public void testAgentCreation() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        
        assertNotNull("Agent不应为null", agent);
        assertEquals("Agent名称应正确", "测试助手", agent.getName());
        assertNotNull("会话ID不应为null", agent.getCurrentSessionId());
        assertTrue("对话历史应为空", agent.getConversationHistory().isEmpty());
    }
    
    @Test
    public void testMessageProcessing() {
        AdvancedAgent agent = new AdvancedAgent("测试助手", "你是一个测试助手");
        
        String response = agent.processMessage("你好");
        
        assertNotNull("响应不应为null", response);
        assertFalse("响应不应为空", response.trim().isEmpty());
        assertEquals("对话历史应有2条消息", 2, agent.getConversationHistory().size());
    }
    
    @Test
    public void testKnowledgeAddition() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        
        agent.addKnowledge("这是一个测试知识", "test_doc_1", null);
        
        Map<String, Object> ragStats = agent.getRagSystem().getStats();
        assertEquals("应有1个文档", 1, ragStats.get("document_count"));
    }
    
    @Test
    public void testToolRegistration() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        
        // 测试默认工具
        assertTrue("应包含calculator工具", agent.getToolRegistry().hasTool("calculator"));
        assertTrue("应包含time工具", agent.getToolRegistry().hasTool("time"));
        assertTrue("应包含note工具", agent.getToolRegistry().hasTool("note"));
        
        // 注册自定义工具
        agent.registerTool("test_tool", args -> "测试结果", "测试工具");
        assertTrue("应包含test_tool", agent.getToolRegistry().hasTool("test_tool"));
    }
    
    @Test
    public void testMemoryManagement() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        MemoryManager memoryManager = agent.getMemoryManager();
        
        // 添加记忆
        String memoryId = memoryManager.addMemory("测试记忆内容", "working", 0.5);
        assertNotNull("记忆ID不应为null", memoryId);
        
        // 检索记忆
        List<Memory> memories = memoryManager.retrieveMemories("测试", 5);
        assertFalse("应能检索到记忆", memories.isEmpty());
        
        // 验证记忆内容
        Memory memory = memories.get(0);
        assertEquals("记忆内容应匹配", "测试记忆内容", memory.getContent());
        assertEquals("记忆类型应匹配", "working", memory.getMemoryType());
    }
    
    @Test
    public void testRAGSystem() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        RAGSystem ragSystem = agent.getRagSystem();
        
        // 添加文档
        ragSystem.addDocument("doc1", "人工智能是计算机科学的一个分支");
        ragSystem.addDocument("doc2", "机器学习是人工智能的子领域");
        
        // 测试检索
        List<RetrievalResult> results = ragSystem.retrieve("人工智能", 2);
        assertFalse("检索结果不应为空", results.isEmpty());
        assertTrue("应至少有一个结果", results.size() >= 1);
        
        // 测试相似度
        RetrievalResult firstResult = results.get(0);
        assertTrue("相似度应大于0", firstResult.getSimilarity() > 0);
    }
    
    @Test
    public void testToolExecution() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        ToolRegistry toolRegistry = agent.getToolRegistry();
        
        // 测试计算器工具
        Map<String, Object> calcArgs = new HashMap<>();
        calcArgs.put("operation", "add");
        calcArgs.put("a", 10);
        calcArgs.put("b", 5);
        
        ToolCall result = toolRegistry.callTool("calculator", calcArgs);
        assertTrue("工具调用应成功", result.isSuccess());
        assertNotNull("结果不应为null", result.getResult());
        
        // 测试时间工具
        ToolCall timeResult = toolRegistry.callTool("time", new HashMap<>());
        assertTrue("时间工具应成功", timeResult.isSuccess());
        assertNotNull("时间结果不应为null", timeResult.getResult());
    }
    
    @Test
    public void testContextEngine() {
        AdvancedAgent agent = new AdvancedAgent("测试助手", "你是一个智能助手");
        ContextEngine contextEngine = agent.getContextEngine();
        
        // 添加消息
        contextEngine.addMessage(new Message("user", "测试消息"));
        
        // 构建上下文
        String context = contextEngine.buildContext("当前查询");
        assertNotNull("上下文不应为null", context);
        assertTrue("上下文应包含系统指令", context.contains("系统指令"));
        assertTrue("上下文应包含当前查询", context.contains("当前查询"));
    }
    
    @Test
    public void testAgentStats() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        
        // 添加一些数据
        agent.addKnowledge("测试知识");
        agent.processMessage("测试消息");
        
        Map<String, Object> stats = agent.getStats();
        
        assertNotNull("统计信息不应为null", stats);
        assertTrue("应包含名称", stats.containsKey("name"));
        assertTrue("应包含会话ID", stats.containsKey("session_id"));
        assertTrue("应包含对话长度", stats.containsKey("conversation_length"));
        assertTrue("应包含记忆统计", stats.containsKey("memory_stats"));
        assertTrue("应包含RAG统计", stats.containsKey("rag_stats"));
        assertTrue("应包含工具数量", stats.containsKey("tool_count"));
    }
    
    @Test
    public void testConversationExport() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        
        agent.processMessage("你好");
        agent.processMessage("再见");
        
        List<Map<String, Object>> conversation = agent.exportConversation();
        
        assertEquals("应有4条消息（2条用户+2条助手）", 4, conversation.size());
        
        Map<String, Object> firstMessage = conversation.get(0);
        assertTrue("应包含角色", firstMessage.containsKey("role"));
        assertTrue("应包含内容", firstMessage.containsKey("content"));
        assertTrue("应包含时间戳", firstMessage.containsKey("timestamp"));
    }
    
    @Test
    public void testConversationClear() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        
        agent.processMessage("测试消息");
        assertFalse("对话历史不应为空", agent.getConversationHistory().isEmpty());
        
        agent.clearConversation();
        assertTrue("对话历史应为空", agent.getConversationHistory().isEmpty());
    }
    
    @Test
    public void testLLMSimulator() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        LLMSimulator llmSimulator = agent.getLLMSimulator();
        
        assertNotNull("LLM模拟器不应为 null", llmSimulator);
        assertEquals("模型名称应为 gpt-3.5-turbo", "gpt-3.5-turbo", llmSimulator.getModelName());
        assertEquals("温度参数应为 0.7", 0.7, llmSimulator.getTemperature(), 0.01);
        assertEquals("最大token数应为 2048", 2048, llmSimulator.getMaxTokens());
    }
    
    @Test
    public void testLLMResponseGeneration() {
        AdvancedAgent agent = new AdvancedAgent("测试助手", "你是一个测试助手");
        
        // 测试生成的响应不是简单的字符串拼接
        String response = agent.processMessage("你好，请分析一下数据");
        
        assertNotNull("响应不应为 null", response);
        assertFalse("响应不应为空", response.trim().isEmpty());
        // 验证不是简单的模板字符串拼接
        assertFalse("应使用 LLM 模拟器而不是简单拼接", 
            response.startsWith("我理解了你的问题："));
    }
    
    @Test
    public void testSystemPromptGeneration() {
        AdvancedAgent agent = new AdvancedAgent("测试助手");
        LLMSimulator llmSimulator = agent.getLLMSimulator();
        
        // 测试不同类型Agent的系统提示
        String[] agentTypes = {"analyst", "researcher", "coordinator", "executor", "critic"};
        
        for (String agentType : agentTypes) {
            String systemPrompt = llmSimulator.generateSystemPrompt(agentType, 
                agentType + "助手", agentType + "专家");
            
            assertNotNull(agentType + "的系统提示不应为null", systemPrompt);
            assertFalse(agentType + "的系统提示不应为空", systemPrompt.trim().isEmpty());
            assertTrue(agentType + "的系统提示应包含名称", 
                systemPrompt.contains(agentType + "助手"));
        }
    }
}