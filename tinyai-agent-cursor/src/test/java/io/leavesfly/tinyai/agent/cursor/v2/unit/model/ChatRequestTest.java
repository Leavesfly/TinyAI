package io.leavesfly.tinyai.agent.cursor.v2.unit.model;

import io.leavesfly.tinyai.agent.cursor.v2.model.ChatRequest;
import io.leavesfly.tinyai.agent.cursor.v2.model.Message;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * ChatRequest 数据模型单元测试
 *
 * @author leavesfly
 * @date 2025-01-15
 */
public class ChatRequestTest {

    @Test
    public void testBuilderBasicUsage() {
        ChatRequest request = ChatRequest.builder()
            .model("deepseek-chat")
            .addUserMessage("Hello")
            .temperature(0.7)
            .maxTokens(500)
            .build();
        
        assertEquals("deepseek-chat", request.getModel());
        assertEquals(1, request.getMessages().size());
        assertEquals("Hello", request.getMessages().get(0).getContent());
        assertEquals(0.7, request.getTemperature(), 0.001);
        assertEquals(Integer.valueOf(500), request.getMaxTokens());
    }

    @Test
    public void testBuilderWithSystemMessage() {
        ChatRequest request = ChatRequest.builder()
            .model("qwen-max")
            .addSystemMessage("You are a coding assistant")
            .addUserMessage("Write a hello world in Java")
            .build();
        
        assertEquals(2, request.getMessages().size());
        assertEquals(Message.Role.SYSTEM, request.getMessages().get(0).getRole());
        assertEquals(Message.Role.USER, request.getMessages().get(1).getRole());
    }

    @Test
    public void testBuilderWithAssistantMessage() {
        ChatRequest request = ChatRequest.builder()
            .model("deepseek-chat")
            .addUserMessage("Hello")
            .addMessage(Message.assistant("Hi there!"))
            .addUserMessage("How are you?")
            .build();
        
        assertEquals(3, request.getMessages().size());
        assertEquals(Message.Role.USER, request.getMessages().get(0).getRole());
        assertEquals(Message.Role.ASSISTANT, request.getMessages().get(1).getRole());
        assertEquals(Message.Role.USER, request.getMessages().get(2).getRole());
    }

    @Test
    public void testBuilderWithCustomMessage() {
        Message customMessage = Message.system("Custom system message");
        
        ChatRequest request = ChatRequest.builder()
            .model("qwen-turbo")
            .addMessage(customMessage)
            .build();
        
        assertEquals(1, request.getMessages().size());
        assertEquals(customMessage, request.getMessages().get(0));
    }

    @Test
    public void testBuilderWithTools() {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("type", "object");
        
        ToolDefinition tool = new ToolDefinition();
        tool.setType("function");
        
        ToolDefinition.FunctionDefinition function = new ToolDefinition.FunctionDefinition();
        function.setName("code_analyzer");
        function.setDescription("Analyze code quality");
        function.setParameters(parameters);
        tool.setFunction(function);
        
        ChatRequest request = ChatRequest.builder()
            .model("deepseek-chat")
            .addUserMessage("Analyze this code")
            .addTool(tool)
            .build();
        
        assertNotNull(request.getTools());
        assertEquals(1, request.getTools().size());
        assertEquals("code_analyzer", request.getTools().get(0).getFunction().getName());
    }

    @Test
    public void testBuilderWithAllParameters() {
        ChatRequest request = ChatRequest.builder()
            .model("qwen-max")
            .addSystemMessage("You are helpful")
            .addUserMessage("Test")
            .temperature(0.8)
            .topP(0.9)
            .maxTokens(1000)
            .stream(true)
            .build();
        
        assertEquals("qwen-max", request.getModel());
        assertEquals(2, request.getMessages().size());
        assertEquals(0.8, request.getTemperature(), 0.001);
        assertEquals(0.9, request.getTopP(), 0.001);
        assertEquals(Integer.valueOf(1000), request.getMaxTokens());
        assertTrue(request.getStream());
    }

    @Test
    public void testDefaultValues() {
        ChatRequest request = ChatRequest.builder()
            .model("deepseek-chat")
            .addUserMessage("Test")
            .build();
        
        assertFalse(request.getStream());
        assertNull(request.getTools());
        assertNull(request.getTopP());
        assertNull(request.getMaxTokens());
    }

    @Test
    public void testSettersAndGetters() {
        ChatRequest request = new ChatRequest();
        request.setModel("test-model");
        request.setTemperature(0.5);
        request.setStream(true);
        
        assertEquals("test-model", request.getModel());
        assertEquals(0.5, request.getTemperature(), 0.001);
        assertTrue(request.getStream());
    }
}
