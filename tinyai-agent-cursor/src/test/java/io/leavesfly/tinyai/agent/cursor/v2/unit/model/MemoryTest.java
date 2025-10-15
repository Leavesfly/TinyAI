package io.leavesfly.tinyai.agent.cursor.v2.unit.model;

import io.leavesfly.tinyai.agent.cursor.v2.model.Memory;
import io.leavesfly.tinyai.agent.cursor.v2.model.Memory.MemoryType;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Memory 数据模型单元测试
 *
 * @author leavesfly
 * @date 2025-01-15
 */
public class MemoryTest {

    @Test
    public void testMemoryCreation() {
        Memory memory = new Memory();
        memory.setSessionId("session-001");
        memory.setType(MemoryType.WORKING);
        memory.setContent("Test memory content");
        memory.setImportance(0.8);
        
        assertEquals("session-001", memory.getSessionId());
        assertEquals(MemoryType.WORKING, memory.getType());
        assertEquals("Test memory content", memory.getContent());
        assertEquals(0.8, memory.getImportance(), 0.001);
    }

    @Test
    public void testMemoryTypes() {
        assertEquals(4, MemoryType.values().length);
        
        MemoryType[] types = {
            MemoryType.WORKING,
            MemoryType.SHORT_TERM,
            MemoryType.LONG_TERM,
            MemoryType.SEMANTIC
        };
        
        for (MemoryType type : types) {
            assertNotNull(type);
        }
    }

    @Test
    public void testAccessCount() {
        Memory memory = new Memory();
        assertEquals(0, memory.getAccessCount());
        
        memory.recordAccess();
        assertEquals(1, memory.getAccessCount());
        
        memory.recordAccess();
        memory.recordAccess();
        assertEquals(3, memory.getAccessCount());
    }

    @Test
    public void testUpdateLastAccess() {
        Memory memory = new Memory();
        assertTrue(memory.getLastAccessAt() > 0);
        
        long firstAccess = memory.getLastAccessAt();
        
        try {
            Thread.sleep(10);
        } catch (InterruptedException e) {
            // ignore
        }
        
        memory.recordAccess();
        assertTrue(memory.getLastAccessAt() > firstAccess);
    }

    @Test
    public void testIsExpired() {
        Memory memory = new Memory();
        long currentTime = System.currentTimeMillis();
        
        // 工作记忆 - 30分钟过期
        memory.setType(MemoryType.WORKING);
        memory.setCreatedAt(currentTime);
        assertFalse(memory.isExpired(currentTime, 30 * 60 * 1000));
        
        // 设置为31分钟前
        memory.setCreatedAt(currentTime - (31 * 60 * 1000));
        assertTrue(memory.isExpired(currentTime, 30 * 60 * 1000));
        
        // 短期记忆 - 2小时过期
        memory.setType(MemoryType.SHORT_TERM);
        memory.setCreatedAt(currentTime);
        assertFalse(memory.isExpired(currentTime, 2 * 60 * 60 * 1000));
        
        memory.setCreatedAt(currentTime - (3 * 60 * 60 * 1000));
        assertTrue(memory.isExpired(currentTime, 2 * 60 * 60 * 1000));
    }

    @Test
    public void testVectorEmbedding() {
        Memory memory = new Memory();
        assertNull(memory.getEmbedding());
        
        double[] vector = new double[]{0.1, 0.2, 0.3};
        memory.setEmbedding(vector);
        
        assertArrayEquals(vector, memory.getEmbedding(), 0.001);
    }

    @Test
    public void testMetadataOperations() {
        Memory memory = new Memory();
        memory.setProjectId("project-123");
        memory.putMetadata("source", "code_analysis");
        memory.putMetadata("key1", "value1");
        memory.putMetadata("key2", 123);
        
        assertEquals("project-123", memory.getProjectId());
        assertEquals("code_analysis", memory.getMetadata("source"));
        assertEquals("value1", memory.getMetadata("key1"));
        assertEquals(123, memory.getMetadata("key2"));
    }
}
