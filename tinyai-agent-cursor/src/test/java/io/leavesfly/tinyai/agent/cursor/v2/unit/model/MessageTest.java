package io.leavesfly.tinyai.agent.cursor.v2.unit.model;

import io.leavesfly.tinyai.agent.cursor.v2.model.Message;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolCall;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Message 数据模型单元测试
 *
 * @author leavesfly
 * @date 2025-01-15
 */
public class MessageTest {

    @Test
    public void testSystemMessageCreation() {
        Message message = Message.system("You are a helpful assistant");
        
        assertEquals(Message.Role.SYSTEM, message.getRole());
        assertEquals("You are a helpful assistant", message.getContent());
        assertNull(message.getName());
        assertNull(message.getToolCalls());
        assertNull(message.getToolCallId());
    }

    @Test
    public void testUserMessageCreation() {
        Message message = Message.user("Hello, AI!");
        
        assertEquals(Message.Role.USER, message.getRole());
        assertEquals("Hello, AI!", message.getContent());
    }

    @Test
    public void testAssistantMessageCreation() {
        Message message = Message.assistant("Hello! How can I help you?");
        
        assertEquals(Message.Role.ASSISTANT, message.getRole());
        assertEquals("Hello! How can I help you?", message.getContent());
    }

    @Test
    public void testToolMessageCreation() {
        Message message = Message.tool("tool-call-123", "{\"result\": \"success\"}");
        
        assertEquals(Message.Role.TOOL, message.getRole());
        assertEquals("{\"result\": \"success\"}", message.getContent());
        assertEquals("tool-call-123", message.getToolCallId());
    }

    @Test
    public void testMessageWithToolCalls() {
        java.util.Map<String, Object> arguments = new java.util.HashMap<>();
        arguments.put("code", "public class Test {}");
        
        ToolCall.FunctionCall functionCall = new ToolCall.FunctionCall();
        functionCall.setName("code_analyzer");
        functionCall.setArguments(arguments);
        
        ToolCall toolCall = new ToolCall();
        toolCall.setId("call-001");
        toolCall.setType("function");
        toolCall.setFunction(functionCall);
        
        Message message = new Message();
        message.setRole(Message.Role.ASSISTANT);
        message.setToolCalls(Arrays.asList(toolCall));
        
        assertEquals(Message.Role.ASSISTANT, message.getRole());
        assertNotNull(message.getToolCalls());
        assertEquals(1, message.getToolCalls().size());
        assertEquals("call-001", message.getToolCalls().get(0).getId());
    }

    @Test
    public void testRoleFromValue() {
        assertEquals(Message.Role.SYSTEM, Message.Role.fromValue("system"));
        assertEquals(Message.Role.USER, Message.Role.fromValue("user"));
        assertEquals(Message.Role.ASSISTANT, Message.Role.fromValue("assistant"));
        assertEquals(Message.Role.TOOL, Message.Role.fromValue("tool"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRoleFromInvalidValue() {
        Message.Role.fromValue("invalid");
    }

    @Test
    public void testRoleGetValue() {
        assertEquals("system", Message.Role.SYSTEM.getValue());
        assertEquals("user", Message.Role.USER.getValue());
        assertEquals("assistant", Message.Role.ASSISTANT.getValue());
        assertEquals("tool", Message.Role.TOOL.getValue());
    }

    @Test
    public void testMessageSettersAndGetters() {
        Message message = new Message();
        message.setRole(Message.Role.USER);
        message.setContent("Test content");
        message.setName("TestUser");
        
        assertEquals(Message.Role.USER, message.getRole());
        assertEquals("Test content", message.getContent());
        assertEquals("TestUser", message.getName());
    }

    @Test
    public void testMessageToString() {
        Message message = Message.user("Hello");
        String result = message.toString();
        
        assertTrue(result.contains("USER"));
        assertTrue(result.contains("Hello"));
    }
}
