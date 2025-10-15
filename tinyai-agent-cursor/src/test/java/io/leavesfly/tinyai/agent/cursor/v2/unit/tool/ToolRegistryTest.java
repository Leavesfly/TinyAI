package io.leavesfly.tinyai.agent.cursor.v2.unit.tool;

import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;
import io.leavesfly.tinyai.agent.cursor.v2.tool.Tool;
import io.leavesfly.tinyai.agent.cursor.v2.tool.ToolRegistry;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * ToolRegistry 单元测试
 *
 * @author leavesfly
 * @date 2025-01-15
 */
public class ToolRegistryTest {

    private ToolRegistry registry;

    @Before
    public void setUp() {
        registry = new ToolRegistry();
    }

    @Test
    public void testRegisterTool() {
        TestTool tool = new TestTool("test_tool", "Test tool", Tool.ToolCategory.GENERAL);
        
        registry.register(tool);
        
        assertTrue(registry.hasTool("test_tool"));
        assertNotNull(registry.getTool("test_tool"));
    }

    @Test
    public void testUnregisterTool() {
        TestTool tool = new TestTool("test_tool", "Test tool", Tool.ToolCategory.GENERAL);
        
        registry.register(tool);
        assertTrue(registry.hasTool("test_tool"));
        
        registry.unregister("test_tool");
        assertFalse(registry.hasTool("test_tool"));
    }

    @Test
    public void testGetToolNotFound() {
        Tool tool = registry.getTool("non_existent_tool");
        assertNull(tool);
    }

    @Test
    public void testGetToolsByCategory() {
        TestTool tool1 = new TestTool("tool1", "Tool 1", Tool.ToolCategory.CODE_ANALYSIS);
        TestTool tool2 = new TestTool("tool2", "Tool 2", Tool.ToolCategory.CODE_ANALYSIS);
        TestTool tool3 = new TestTool("tool3", "Tool 3", Tool.ToolCategory.RETRIEVAL);
        
        registry.register(tool1);
        registry.register(tool2);
        registry.register(tool3);
        
        List<Tool> codeTools = registry.getToolsByCategory(Tool.ToolCategory.CODE_ANALYSIS);
        assertEquals(2, codeTools.size());
        
        List<Tool> searchTools = registry.getToolsByCategory(Tool.ToolCategory.RETRIEVAL);
        assertEquals(1, searchTools.size());
    }

    @Test
    public void testGetAllTools() {
        assertTrue(registry.getAllTools().isEmpty());
        
        TestTool tool1 = new TestTool("tool1", "Tool 1", Tool.ToolCategory.GENERAL);
        TestTool tool2 = new TestTool("tool2", "Tool 2", Tool.ToolCategory.GENERAL);
        
        registry.register(tool1);
        registry.register(tool2);
        
        List<Tool> allTools = registry.getAllTools();
        assertEquals(2, allTools.size());
    }

    @Test
    public void testGetToolDefinitions() {
        TestTool tool = new TestTool("test_tool", "Test tool", Tool.ToolCategory.GENERAL);
        registry.register(tool);
        
        List<String> toolNames = java.util.Arrays.asList("test_tool");
        List<io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition> definitions = 
            registry.getToolDefinitions(toolNames);
        
        assertEquals(1, definitions.size());
        assertEquals("test_tool", definitions.get(0).getFunction().getName());
    }

    @Test
    public void testRegistryStats() {
        TestTool tool1 = new TestTool("tool1", "Tool 1", Tool.ToolCategory.CODE_ANALYSIS);
        TestTool tool2 = new TestTool("tool2", "Tool 2", Tool.ToolCategory.RETRIEVAL);
        
        registry.register(tool1);
        registry.register(tool2);
        
        ToolRegistry.RegistryStats stats = registry.getStats();
        assertEquals(2, stats.totalTools);
        assertTrue(stats.toolsByCategory.get(Tool.ToolCategory.CODE_ANALYSIS) >= 1);
    }

    @Test(expected = IllegalStateException.class)
    public void testToolDuplicateRegistration() {
        TestTool tool1 = new TestTool("same_tool", "Original tool", Tool.ToolCategory.GENERAL);
        TestTool tool2 = new TestTool("same_tool", "Duplicate tool", Tool.ToolCategory.GENERAL);
        
        registry.register(tool1);
        registry.register(tool2); // Should throw exception
    }

    /**
     * 测试工具实现
     */
    private static class TestTool implements Tool {
        private final String name;
        private final String description;
        private final ToolCategory category;

        public TestTool(String name, String description, ToolCategory category) {
            this.name = name;
            this.description = description;
            this.category = category;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public String getDescription() {
            return description;
        }

        @Override
        public io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition getDefinition() {
            io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition def = 
                new io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition();
            def.setType("function");
            
            io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition.FunctionDefinition func = 
                new io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition.FunctionDefinition();
            func.setName(name);
            func.setDescription(description);
            
            Map<String, Object> params = new HashMap<>();
            params.put("type", "object");
            func.setParameters(params);
            
            def.setFunction(func);
            return def;
        }

        @Override
        public ToolResult execute(Map<String, Object> parameters) {
            ToolResult result = new ToolResult();
            result.setSuccess(true);
            result.setResult("Executed: " + name);
            return result;
        }

        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            return true;
        }

        @Override
        public boolean requiresSandbox() {
            return false;
        }

        @Override
        public ToolCategory getCategory() {
            return category;
        }
    }
}
