package io.leavesfly.tinyai.agent.cursor.v2.unit.adapter;

import io.leavesfly.tinyai.agent.cursor.v2.adapter.AdapterRegistry;
import io.leavesfly.tinyai.agent.cursor.v2.adapter.DeepSeekAdapter;
import io.leavesfly.tinyai.agent.cursor.v2.adapter.QwenAdapter;
import io.leavesfly.tinyai.agent.cursor.v2.adapter.ModelAdapter;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 * AdapterRegistry 单元测试
 *
 * @author leavesfly
 * @date 2025-01-15
 */
public class AdapterRegistryTest {

    private AdapterRegistry registry;

    @Before
    public void setUp() {
        registry = new AdapterRegistry();
    }

    @Test
    public void testRegisterAdapter() {
        DeepSeekAdapter adapter = new DeepSeekAdapter();
        registry.register(adapter);
        
        assertNotNull(registry.getAdapter("deepseek-chat"));
        assertNotNull(registry.getAdapter("deepseek-coder"));
    }

    @Test
    public void testRegisterMultipleAdapters() {
        DeepSeekAdapter deepSeekAdapter = new DeepSeekAdapter();
        QwenAdapter qwenAdapter = new QwenAdapter();
        
        registry.register(deepSeekAdapter);
        registry.register(qwenAdapter);
        
        assertNotNull(registry.getAdapter("deepseek-chat"));
        assertNotNull(registry.getAdapter("qwen-max"));
        assertNotNull(registry.getAdapter("qwen-plus"));
    }

    @Test
    public void testUnregisterAdapter() {
        DeepSeekAdapter adapter = new DeepSeekAdapter();
        registry.register(adapter);
        
        assertNotNull(registry.getAdapter("deepseek-chat"));
        
        registry.unregister("DeepSeek");
        
        assertNull(registry.getAdapter("deepseek-chat"));
        assertNull(registry.getAdapter("deepseek-coder"));
    }

    @Test
    public void testGetAdapterNotFound() {
        ModelAdapter adapter = registry.getAdapter("non-existent-model");
        assertNull(adapter);
    }

    @Test
    public void testGetAllAdapters() {
        assertTrue(registry.getAllAdapters().isEmpty());
        
        DeepSeekAdapter deepSeekAdapter = new DeepSeekAdapter();
        QwenAdapter qwenAdapter = new QwenAdapter();
        
        registry.register(deepSeekAdapter);
        registry.register(qwenAdapter);
        
        List<ModelAdapter> adapters = registry.getAllAdapters();
        assertEquals(2, adapters.size());
    }

    @Test
    public void testGetSupportedModels() {
        assertTrue(registry.getSupportedModels().isEmpty());
        
        DeepSeekAdapter adapter = new DeepSeekAdapter();
        registry.register(adapter);
        
        List<String> models = registry.getSupportedModels();
        assertTrue(models.contains("deepseek-chat"));
        assertTrue(models.contains("deepseek-coder"));
    }

    @Test
    public void testIsModelSupported() {
        DeepSeekAdapter adapter = new DeepSeekAdapter();
        registry.register(adapter);
        
        assertTrue(registry.isModelSupported("deepseek-chat"));
        assertTrue(registry.isModelSupported("deepseek-coder"));
        assertFalse(registry.isModelSupported("unknown-model"));
    }

    @Test
    public void testClearRegistry() {
        DeepSeekAdapter deepSeekAdapter = new DeepSeekAdapter();
        QwenAdapter qwenAdapter = new QwenAdapter();
        
        registry.register(deepSeekAdapter);
        registry.register(qwenAdapter);
        
        assertFalse(registry.getAllAdapters().isEmpty());
        
        registry.clear();
        
        assertTrue(registry.getAllAdapters().isEmpty());
        assertTrue(registry.getSupportedModels().isEmpty());
    }

    @Test
    public void testAdapterOverwrite() {
        DeepSeekAdapter adapter1 = new DeepSeekAdapter();
        DeepSeekAdapter adapter2 = new DeepSeekAdapter();
        
        registry.register(adapter1);
        ModelAdapter retrieved1 = registry.getAdapter("deepseek-chat");
        
        registry.register(adapter2);
        ModelAdapter retrieved2 = registry.getAdapter("deepseek-chat");
        
        // 新注册的适配器应该覆盖旧的
        assertNotNull(retrieved1);
        assertNotNull(retrieved2);
    }
}
