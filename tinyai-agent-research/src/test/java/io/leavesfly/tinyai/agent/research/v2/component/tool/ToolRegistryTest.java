package io.leavesfly.tinyai.agent.research.v2.component.tool;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ToolRegistry 单元测试
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolRegistryTest {
    
    private ToolRegistry registry;
    
    @Before
    public void setUp() {
        registry = new ToolRegistry();
    }
    
    @Test
    public void testBuiltinToolsRegistered() {
        List<ToolRegistry.ToolInfo> tools = registry.listTools();
        
        // 应该有3个内置工具
        assertEquals(3, tools.size());
        
        // 验证工具名称
        boolean hasWebSearch = tools.stream()
            .anyMatch(t -> "web_search".equals(t.getName()));
        boolean hasDocReader = tools.stream()
            .anyMatch(t -> "document_read".equals(t.getName()));
        boolean hasCodeAnalyzer = tools.stream()
            .anyMatch(t -> "code_analyze".equals(t.getName()));
        
        assertTrue(hasWebSearch);
        assertTrue(hasDocReader);
        assertTrue(hasCodeAnalyzer);
    }
    
    @Test
    public void testGetTool() {
        Tool tool = registry.getTool("web_search");
        
        assertNotNull(tool);
        assertEquals("web_search", tool.getName());
        assertEquals("search", tool.getCategory());
    }
    
    @Test
    public void testExecuteWebSearch() {
        Map<String, Object> params = new HashMap<>();
        params.put("query", "Java并发编程");
        params.put("maxResults", 5);
        
        ToolResult result = registry.executeTool("web_search", params);
        
        assertTrue(result.isSuccess());
        assertNotNull(result.getData("results"));
    }
    
    @Test
    public void testExecuteWithInvalidTool() {
        Map<String, Object> params = new HashMap<>();
        
        ToolResult result = registry.executeTool("non_existent_tool", params);
        
        assertFalse(result.isSuccess());
        assertTrue(result.getMessage().contains("不存在"));
    }
    
    @Test
    public void testExecuteWithInvalidParams() {
        Map<String, Object> params = new HashMap<>();
        // 缺少必需参数query
        
        ToolResult result = registry.executeTool("web_search", params);
        
        assertFalse(result.isSuccess());
    }
    
    @Test
    public void testGetToolsByCategory() {
        List<Tool> searchTools = registry.getToolsByCategory("search");
        
        assertFalse(searchTools.isEmpty());
        assertEquals(1, searchTools.size());
        assertEquals("web_search", searchTools.get(0).getName());
    }
}
