package io.leavesfly.tinyai.agent.mcp;

import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * MCP Client 测试类
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPClientTest {
    
    private MCPClient client;
    private MCPServer server;
    
    @Before
    public void setUp() {
        client = new MCPClient("test_client");
        server = new MCPServer("Test Server", "1.0.0");
        
        // 设置测试资源
        Resource resource = new Resource("test://data", "Test Data", ResourceType.MEMORY);
        server.registerResource(resource);
        server.setResourceContent("test://data", "Hello MCP");
        
        // 设置测试工具
        Tool tool = new Tool(
            "echo",
            "Echo tool",
            ToolCategory.CUSTOM,
            new HashMap<>(),
            args -> args.get("message")
        );
        server.registerTool(tool);
        
        // 设置测试提示词
        Prompt prompt = new Prompt(
            "greeting",
            "Greeting prompt",
            "Hello {name}!"
        );
        server.registerPrompt(prompt);
        
        // 连接到服务器
        client.connect("testserver", server);
    }
    
    @Test
    public void testClientInitialization() {
        assertEquals("test_client", client.getClientId());
        assertNotNull(client.getConnectedServers());
    }
    
    @Test
    public void testConnect() {
        MCPServer newServer = new MCPServer("New Server");
        client.connect("newserver", newServer);
        
        assertTrue(client.getConnectedServers().containsKey("newserver"));
        assertEquals(2, client.listServers().size());
    }
    
    @Test
    public void testDisconnect() {
        client.disconnect("testserver");
        assertFalse(client.getConnectedServers().containsKey("testserver"));
    }
    
    @Test
    public void testListServers() {
        List<String> servers = client.listServers();
        assertEquals(1, servers.size());
        assertTrue(servers.contains("testserver"));
    }
    
    @Test
    public void testListResources() {
        List<Map<String, Object>> resources = client.listResources("testserver");
        assertEquals(1, resources.size());
        assertEquals("Test Data", resources.get(0).get("name"));
    }
    
    @Test
    public void testReadResource() {
        Map<String, Object> content = client.readResource("testserver", "test://data");
        assertNotNull(content);
        assertEquals("test://data", content.get("uri"));
        assertEquals("Hello MCP", content.get("content"));
    }
    
    @Test
    public void testReadNonExistentResource() {
        Map<String, Object> content = client.readResource("testserver", "test://nonexistent");
        assertNull(content);
    }
    
    @Test
    public void testListTools() {
        List<Map<String, Object>> tools = client.listTools("testserver");
        assertEquals(1, tools.size());
        assertEquals("echo", tools.get(0).get("name"));
    }
    
    @Test
    public void testCallTool() {
        Map<String, Object> args = new HashMap<>();
        args.put("message", "Hello World");
        
        Map<String, Object> result = client.callTool("testserver", "echo", args);
        assertNotNull(result);
        assertFalse((Boolean) result.getOrDefault("isError", false));
        assertEquals("Hello World", result.get("content"));
    }
    
    @Test
    public void testCallNonExistentTool() {
        Map<String, Object> result = client.callTool("testserver", "nonexistent", new HashMap<>());
        assertNotNull(result);
        assertTrue((Boolean) result.get("isError"));
    }
    
    @Test
    public void testListPrompts() {
        List<Map<String, Object>> prompts = client.listPrompts("testserver");
        assertEquals(1, prompts.size());
        assertEquals("greeting", prompts.get(0).get("name"));
    }
    
    @Test
    public void testGetPrompt() {
        Map<String, Object> args = new HashMap<>();
        args.put("name", "Alice");
        
        String prompt = client.getPrompt("testserver", "greeting", args);
        assertEquals("Hello Alice!", prompt);
    }
    
    @Test
    public void testGetNonExistentPrompt() {
        String prompt = client.getPrompt("testserver", "nonexistent", new HashMap<>());
        assertNull(prompt);
    }
    
    @Test
    public void testCallToNonExistentServer() {
        List<Map<String, Object>> resources = client.listResources("nonexistent");
        assertTrue(resources.isEmpty());
    }
}
