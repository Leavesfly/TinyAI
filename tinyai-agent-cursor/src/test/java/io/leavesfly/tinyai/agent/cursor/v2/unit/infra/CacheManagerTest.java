package io.leavesfly.tinyai.agent.cursor.v2.unit.infra;

import io.leavesfly.tinyai.agent.cursor.v2.infra.cache.CacheManager;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * CacheManager 单元测试
 *
 * @author leavesfly
 * @date 2025-01-15
 */
public class CacheManagerTest {

    private CacheManager cacheManager;

    @Before
    public void setUp() {
        cacheManager = new CacheManager();
    }

    @Test
    public void testL1CachePutAndGet() {
        String key = "test-key";
        String value = "test-value";
        
        cacheManager.putL1(key, value);
        Object cached = cacheManager.getL1(key);
        
        assertEquals(value, cached);
    }

    @Test
    public void testL1CacheRemove() {
        String key = "test-key";
        String value = "test-value";
        
        cacheManager.putL1(key, value);
        assertNotNull(cacheManager.getL1(key));
        
        cacheManager.removeL1(key);
        assertNull(cacheManager.getL1(key));
    }

    @Test
    public void testL1CacheClear() {
        cacheManager.putL1("key1", "value1");
        cacheManager.putL1("key2", "value2");
        
        cacheManager.clearL1();
        
        assertNull(cacheManager.getL1("key1"));
        assertNull(cacheManager.getL1("key2"));
    }

    @Test
    public void testL2CachePutAndGet() {
        String sessionId = "session-001";
        String key = "context-key";
        String value = "context-value";
        
        cacheManager.putL2(sessionId, key, value);
        Object cached = cacheManager.getL2(sessionId, key);
        
        assertEquals(value, cached);
    }

    @Test
    public void testL2CacheIsolation() {
        String session1 = "session-001";
        String session2 = "session-002";
        String key = "same-key";
        
        cacheManager.putL2(session1, key, "value1");
        cacheManager.putL2(session2, key, "value2");
        
        assertEquals("value1", cacheManager.getL2(session1, key));
        assertEquals("value2", cacheManager.getL2(session2, key));
    }

    @Test
    public void testL2CacheClearSession() {
        String sessionId = "session-001";
        
        cacheManager.putL2(sessionId, "key1", "value1");
        cacheManager.putL2(sessionId, "key2", "value2");
        
        cacheManager.clearL2(sessionId);
        
        assertNull(cacheManager.getL2(sessionId, "key1"));
        assertNull(cacheManager.getL2(sessionId, "key2"));
    }

    @Test
    public void testL3CachePutAndGet() {
        String projectId = "project-001";
        String key = "rules-key";
        Object value = java.util.Arrays.asList("rule1", "rule2");
        
        cacheManager.putL3(projectId, key, value);
        Object cached = cacheManager.getL3(projectId, key);
        
        assertEquals(value, cached);
    }

    @Test
    public void testL3CacheIsolation() {
        String project1 = "project-001";
        String project2 = "project-002";
        String key = "same-key";
        
        cacheManager.putL3(project1, key, "value1");
        cacheManager.putL3(project2, key, "value2");
        
        assertEquals("value1", cacheManager.getL3(project1, key));
        assertEquals("value2", cacheManager.getL3(project2, key));
    }

    @Test
    public void testL3CacheClearProject() {
        String projectId = "project-001";
        
        cacheManager.putL3(projectId, "key1", "value1");
        cacheManager.putL3(projectId, "key2", "value2");
        
        cacheManager.clearL3(projectId);
        
        assertNull(cacheManager.getL3(projectId, "key1"));
        assertNull(cacheManager.getL3(projectId, "key2"));
    }

    @Test
    public void testCacheStats() {
        cacheManager.putL1("key1", "value1");
        cacheManager.getL1("key1"); // hit
        cacheManager.getL1("key2"); // miss
        
        CacheManager.CacheStats stats = cacheManager.getStats();
        assertNotNull(stats);
        assertTrue(stats.l1Size > 0);
    }

    @Test
    public void testClearAll() {
        // L1 缓存
        cacheManager.putL1("l1-key", "l1-value");
        
        // L2 缓存
        cacheManager.putL2("session-001", "l2-key", "l2-value");
        
        // L3 缓存
        cacheManager.putL3("project-001", "l3-key", "l3-value");
        
        // 清空所有缓存
        cacheManager.clearAll();
        
        assertNull(cacheManager.getL1("l1-key"));
        assertNull(cacheManager.getL2("session-001", "l2-key"));
        assertNull(cacheManager.getL3("project-001", "l3-key"));
    }

    @Test
    public void testCacheOverwrite() {
        String key = "test-key";
        
        cacheManager.putL1(key, "value1");
        assertEquals("value1", cacheManager.getL1(key));
        
        cacheManager.putL1(key, "value2");
        assertEquals("value2", cacheManager.getL1(key));
    }

    @Test
    public void testNullValues() {
        String key = "null-key";
        
        cacheManager.putL1(key, null);
        assertNull(cacheManager.getL1(key));
    }
}
