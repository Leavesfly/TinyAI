package io.leavesfly.tinyai.agent.mcp;

import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * MCP Server 测试类
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPServerTest {
    
    private MCPServer server;
    
    @Before
    public void setUp() {
        server = new MCPServer("Test Server", "1.0.0");
    }
    
    @Test
    public void testServerInitialization() {
        assertEquals("Test Server", server.getName());
        assertEquals("1.0.0", server.getVersion());
        assertNotNull(server.getResources());
        assertNotNull(server.getTools());
        assertNotNull(server.getPrompts());
    }
    
    @Test
    public void testRegisterResource() {
        Resource resource = new Resource("test://resource", "Test Resource", ResourceType.MEMORY);
        server.registerResource(resource);
        
        assertTrue(server.getResources().containsKey("test://resource"));
        assertEquals(1, server.getResources().size());
    }
    
    @Test
    public void testListResources() {
        Resource resource1 = new Resource("test://res1", "Resource 1", ResourceType.FILE);
        Resource resource2 = new Resource("test://res2", "Resource 2", ResourceType.DATABASE);
        
        server.registerResource(resource1);
        server.registerResource(resource2);
        
        List<Map<String, Object>> resources = server.listResources();
        assertEquals(2, resources.size());
    }
    
    @Test
    public void testGetResource() {
        Resource resource = new Resource("test://data", "Test Data", ResourceType.MEMORY);
        server.registerResource(resource);
        server.setResourceContent("test://data", "Test Content");
        
        ResourceContent content = server.getResource("test://data");
        assertNotNull(content);
        assertEquals("test://data", content.getUri());
        assertEquals("Test Content", content.getContent());
    }
    
    @Test
    public void testRegisterTool() {
        Tool tool = new Tool(
            "test_tool",
            "A test tool",
            ToolCategory.CUSTOM,
            new HashMap<>(),
            args -> "result"
        );
        
        server.registerTool(tool);
        assertTrue(server.getTools().containsKey("test_tool"));
    }
    
    @Test
    public void testCallTool() {
        Tool tool = new Tool(
            "add",
            "Add two numbers",
            ToolCategory.COMPUTATION,
            new HashMap<>(),
            args -> {
                int a = (Integer) args.get("a");
                int b = (Integer) args.get("b");
                return a + b;
            }
        );
        
        server.registerTool(tool);
        
        Map<String, Object> args = new HashMap<>();
        args.put("a", 10);
        args.put("b", 5);
        
        ToolCall toolCall = new ToolCall("add", args);
        ToolResult result = server.callTool(toolCall);
        
        assertFalse(result.isError());
        assertEquals(15, result.getContent());
    }
    
    @Test
    public void testCallNonExistentTool() {
        ToolCall toolCall = new ToolCall("non_existent", new HashMap<>());
        ToolResult result = server.callTool(toolCall);
        
        assertTrue(result.isError());
        assertNotNull(result.getErrorMessage());
    }
    
    @Test
    public void testRegisterPrompt() {
        Prompt prompt = new Prompt(
            "test_prompt",
            "A test prompt",
            "Hello {name}!"
        );
        
        server.registerPrompt(prompt);
        assertTrue(server.getPrompts().containsKey("test_prompt"));
    }
    
    @Test
    public void testGetPrompt() {
        Prompt prompt = new Prompt(
            "greeting",
            "Greeting prompt",
            "Hello {name}, you are {age} years old."
        );
        
        server.registerPrompt(prompt);
        
        Map<String, Object> params = new HashMap<>();
        params.put("name", "Alice");
        params.put("age", 25);
        
        String rendered = server.getPrompt("greeting", params);
        assertEquals("Hello Alice, you are 25 years old.", rendered);
    }
    
    @Test
    public void testHandleResourcesListRequest() {
        Resource resource = new Resource("test://res", "Test", ResourceType.FILE);
        server.registerResource(resource);
        
        MCPRequest request = new MCPRequest("resources/list", new HashMap<>());
        MCPResponse response = server.handleRequest(request);
        
        assertNotNull(response.getResult());
        assertNull(response.getError());
    }
    
    @Test
    public void testHandleResourcesReadRequest() {
        Resource resource = new Resource("test://data", "Test", ResourceType.MEMORY);
        server.registerResource(resource);
        server.setResourceContent("test://data", "Hello World");
        
        Map<String, Object> params = new HashMap<>();
        params.put("uri", "test://data");
        
        MCPRequest request = new MCPRequest("resources/read", params);
        MCPResponse response = server.handleRequest(request);
        
        assertNotNull(response.getResult());
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) response.getResult();
        assertEquals("test://data", result.get("uri"));
        assertEquals("Hello World", result.get("content"));
    }
    
    @Test
    public void testHandleToolsListRequest() {
        Tool tool = new Tool(
            "test",
            "Test tool",
            ToolCategory.CUSTOM,
            new HashMap<>(),
            args -> "ok"
        );
        server.registerTool(tool);
        
        MCPRequest request = new MCPRequest("tools/list", new HashMap<>());
        MCPResponse response = server.handleRequest(request);
        
        assertNotNull(response.getResult());
    }
    
    @Test
    public void testHandleInvalidMethod() {
        MCPRequest request = new MCPRequest("invalid/method", new HashMap<>());
        MCPResponse response = server.handleRequest(request);
        
        assertNull(response.getResult());
        assertNotNull(response.getError());
        assertEquals(-32601, response.getError().get("code"));
    }
    
    @Test
    public void testGetServerInfo() {
        server.registerResource(new Resource("test://r1", "R1", ResourceType.FILE));
        server.registerTool(new Tool("t1", "T1", ToolCategory.CUSTOM, new HashMap<>(), args -> "ok"));
        server.registerPrompt(new Prompt("p1", "P1", "template"));
        
        Map<String, Object> info = server.getServerInfo();
        
        assertEquals("Test Server", info.get("name"));
        assertEquals("1.0.0", info.get("version"));
        
        @SuppressWarnings("unchecked")
        Map<String, Object> capabilities = (Map<String, Object>) info.get("capabilities");
        assertEquals(1, capabilities.get("resources"));
        assertEquals(1, capabilities.get("tools"));
        assertEquals(1, capabilities.get("prompts"));
    }
}
