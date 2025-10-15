package io.leavesfly.tinyai.agent.cursor.v2.infra.cache;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 缓存管理器
 * 实现L1(内存缓存)、L2(会话缓存)、L3(项目缓存)三级缓存架构
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class CacheManager {
    
    /**
     * L1缓存：代码补全结果（LRU缓存，最近100个）
     */
    private final LRUCache<String, CacheEntry> l1Cache;
    
    /**
     * L2缓存：会话级缓存
     */
    private final Map<String, Map<String, CacheEntry>> l2Cache;
    
    /**
     * L3缓存：项目级缓存
     */
    private final Map<String, Map<String, CacheEntry>> l3Cache;
    
    /**
     * L1缓存容量
     */
    private final int l1Capacity;
    
    /**
     * L1缓存过期时间（毫秒，默认5分钟）
     */
    private final long l1Ttl;
    
    public CacheManager() {
        this(100, 5 * 60 * 1000L);
    }
    
    public CacheManager(int l1Capacity, long l1Ttl) {
        this.l1Capacity = l1Capacity;
        this.l1Ttl = l1Ttl;
        this.l1Cache = new LRUCache<>(l1Capacity);
        this.l2Cache = new ConcurrentHashMap<>();
        this.l3Cache = new ConcurrentHashMap<>();
    }
    
    /**
     * L1缓存操作：代码补全结果缓存
     */
    public void putL1(String key, Object value) {
        l1Cache.put(key, new CacheEntry(value, System.currentTimeMillis(), l1Ttl));
    }
    
    public Object getL1(String key) {
        CacheEntry entry = l1Cache.get(key);
        if (entry != null && !entry.isExpired()) {
            entry.recordAccess();
            return entry.getValue();
        }
        // 过期则移除
        if (entry != null) {
            l1Cache.remove(key);
        }
        return null;
    }
    
    public void removeL1(String key) {
        l1Cache.remove(key);
    }
    
    public void clearL1() {
        l1Cache.clear();
    }
    
    /**
     * L2缓存操作：会话级缓存
     */
    public void putL2(String sessionId, String key, Object value) {
        Map<String, CacheEntry> sessionCache = l2Cache.computeIfAbsent(
            sessionId, k -> new ConcurrentHashMap<>()
        );
        sessionCache.put(key, new CacheEntry(value, System.currentTimeMillis(), -1));
    }
    
    public Object getL2(String sessionId, String key) {
        Map<String, CacheEntry> sessionCache = l2Cache.get(sessionId);
        if (sessionCache != null) {
            CacheEntry entry = sessionCache.get(key);
            if (entry != null) {
                entry.recordAccess();
                return entry.getValue();
            }
        }
        return null;
    }
    
    public void removeL2(String sessionId, String key) {
        Map<String, CacheEntry> sessionCache = l2Cache.get(sessionId);
        if (sessionCache != null) {
            sessionCache.remove(key);
        }
    }
    
    public void clearL2(String sessionId) {
        l2Cache.remove(sessionId);
    }
    
    public void clearAllL2() {
        l2Cache.clear();
    }
    
    /**
     * L3缓存操作：项目级缓存
     */
    public void putL3(String projectId, String key, Object value) {
        Map<String, CacheEntry> projectCache = l3Cache.computeIfAbsent(
            projectId, k -> new ConcurrentHashMap<>()
        );
        projectCache.put(key, new CacheEntry(value, System.currentTimeMillis(), -1));
    }
    
    public Object getL3(String projectId, String key) {
        Map<String, CacheEntry> projectCache = l3Cache.get(projectId);
        if (projectCache != null) {
            CacheEntry entry = projectCache.get(key);
            if (entry != null) {
                entry.recordAccess();
                return entry.getValue();
            }
        }
        return null;
    }
    
    public void removeL3(String projectId, String key) {
        Map<String, CacheEntry> projectCache = l3Cache.get(projectId);
        if (projectCache != null) {
            projectCache.remove(key);
        }
    }
    
    public void clearL3(String projectId) {
        l3Cache.remove(projectId);
    }
    
    public void clearAllL3() {
        l3Cache.clear();
    }
    
    /**
     * 清理所有缓存
     */
    public void clearAll() {
        clearL1();
        clearAllL2();
        clearAllL3();
    }
    
    /**
     * 获取缓存统计信息
     */
    public CacheStats getStats() {
        CacheStats stats = new CacheStats();
        stats.l1Size = l1Cache.size();
        stats.l2SessionCount = l2Cache.size();
        stats.l3ProjectCount = l3Cache.size();
        
        // 计算L2总条目数
        for (Map<String, CacheEntry> sessionCache : l2Cache.values()) {
            stats.l2TotalEntries += sessionCache.size();
        }
        
        // 计算L3总条目数
        for (Map<String, CacheEntry> projectCache : l3Cache.values()) {
            stats.l3TotalEntries += projectCache.size();
        }
        
        return stats;
    }
    
    /**
     * LRU缓存实现
     */
    private static class LRUCache<K, V> extends LinkedHashMap<K, V> {
        private final int capacity;
        
        public LRUCache(int capacity) {
            super(capacity, 0.75f, true);
            this.capacity = capacity;
        }
        
        @Override
        protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
            return size() > capacity;
        }
    }
    
    /**
     * 缓存条目
     */
    private static class CacheEntry {
        private final Object value;
        private final long createdAt;
        private final long ttl; // -1表示不过期
        private long lastAccessAt;
        private int accessCount;
        
        public CacheEntry(Object value, long createdAt, long ttl) {
            this.value = value;
            this.createdAt = createdAt;
            this.ttl = ttl;
            this.lastAccessAt = createdAt;
            this.accessCount = 0;
        }
        
        public Object getValue() {
            return value;
        }
        
        public boolean isExpired() {
            if (ttl < 0) {
                return false;
            }
            return (System.currentTimeMillis() - createdAt) > ttl;
        }
        
        public void recordAccess() {
            this.lastAccessAt = System.currentTimeMillis();
            this.accessCount++;
        }
        
        public int getAccessCount() {
            return accessCount;
        }
    }
    
    /**
     * 缓存统计信息
     */
    public static class CacheStats {
        public int l1Size;
        public int l2SessionCount;
        public int l2TotalEntries;
        public int l3ProjectCount;
        public int l3TotalEntries;
        
        @Override
        public String toString() {
            return "CacheStats{" +
                    "L1=" + l1Size +
                    ", L2(sessions=" + l2SessionCount + ", entries=" + l2TotalEntries + ")" +
                    ", L3(projects=" + l3ProjectCount + ", entries=" + l3TotalEntries + ")" +
                    '}';
        }
    }
}
