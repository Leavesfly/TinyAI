package io.leavesfly.tinyai.agent.cursor.v2.component.memory;

import io.leavesfly.tinyai.agent.cursor.v2.model.Memory;
import io.leavesfly.tinyai.agent.cursor.v2.model.Memory.MemoryType;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 记忆管理器
 * 管理工作记忆、短期记忆、长期记忆和语义记忆
 * 
 * 功能：
 * - 添加和检索各类记忆
 * - 自动清理过期记忆
 * - 基于重要性和访问频率的记忆优先级
 * - 记忆向量化和相似度检索
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class MemoryManager {
    
    /**
     * 所有记忆的存储（memoryId -> Memory）
     */
    private final Map<String, Memory> memories;
    
    /**
     * 会话记忆索引（sessionId -> List<memoryId>）
     */
    private final Map<String, List<String>> sessionIndex;
    
    /**
     * 项目记忆索引（projectId -> List<memoryId>）
     */
    private final Map<String, List<String>> projectIndex;
    
    /**
     * 类型索引（memoryType -> List<memoryId>）
     */
    private final Map<MemoryType, List<String>> typeIndex;
    
    /**
     * 工作记忆TTL（毫秒，默认30分钟）
     */
    private long workingMemoryTtl = 30 * 60 * 1000L;
    
    /**
     * 短期记忆TTL（毫秒，默认2小时）
     */
    private long shortTermMemoryTtl = 2 * 60 * 60 * 1000L;
    
    public MemoryManager() {
        this.memories = new ConcurrentHashMap<>();
        this.sessionIndex = new ConcurrentHashMap<>();
        this.projectIndex = new ConcurrentHashMap<>();
        this.typeIndex = new ConcurrentHashMap<>();
    }
    
    /**
     * 添加记忆
     */
    public void addMemory(Memory memory) {
        if (memory == null || memory.getId() == null) {
            throw new IllegalArgumentException("Memory and memory ID cannot be null");
        }
        
        // 存储记忆
        memories.put(memory.getId(), memory);
        
        // 更新索引
        updateIndexes(memory);
    }
    
    /**
     * 批量添加记忆
     */
    public void addMemories(List<Memory> memoryList) {
        for (Memory memory : memoryList) {
            addMemory(memory);
        }
    }
    
    /**
     * 获取记忆
     */
    public Memory getMemory(String memoryId) {
        Memory memory = memories.get(memoryId);
        if (memory != null) {
            memory.recordAccess();
        }
        return memory;
    }
    
    /**
     * 检索会话记忆
     */
    public List<Memory> retrieveSessionMemories(String sessionId, MemoryType type) {
        List<String> memoryIds = sessionIndex.get(sessionId);
        if (memoryIds == null || memoryIds.isEmpty()) {
            return new ArrayList<>();
        }
        
        return memoryIds.stream()
                .map(memories::get)
                .filter(Objects::nonNull)
                .filter(m -> type == null || m.getType() == type)
                .filter(this::isMemoryValid)
                .peek(Memory::recordAccess)
                .sorted(this::compareByImportance)
                .collect(Collectors.toList());
    }
    
    /**
     * 检索项目记忆
     */
    public List<Memory> retrieveProjectMemories(String projectId, MemoryType type) {
        List<String> memoryIds = projectIndex.get(projectId);
        if (memoryIds == null || memoryIds.isEmpty()) {
            return new ArrayList<>();
        }
        
        return memoryIds.stream()
                .map(memories::get)
                .filter(Objects::nonNull)
                .filter(m -> type == null || m.getType() == type)
                .filter(this::isMemoryValid)
                .peek(Memory::recordAccess)
                .sorted(this::compareByImportance)
                .collect(Collectors.toList());
    }
    
    /**
     * 按类型检索记忆
     */
    public List<Memory> retrieveMemoriesByType(MemoryType type) {
        List<String> memoryIds = typeIndex.get(type);
        if (memoryIds == null || memoryIds.isEmpty()) {
            return new ArrayList<>();
        }
        
        return memoryIds.stream()
                .map(memories::get)
                .filter(Objects::nonNull)
                .filter(this::isMemoryValid)
                .peek(Memory::recordAccess)
                .sorted(this::compareByImportance)
                .collect(Collectors.toList());
    }
    
    /**
     * 语义检索记忆（基于向量相似度）
     */
    public List<Memory> retrieveSimilarMemories(double[] queryEmbedding, int topK, double threshold) {
        if (queryEmbedding == null) {
            throw new IllegalArgumentException("Query embedding cannot be null");
        }
        
        // 只在语义记忆中搜索
        List<String> semanticMemoryIds = typeIndex.getOrDefault(MemoryType.SEMANTIC, new ArrayList<>());
        
        return semanticMemoryIds.stream()
                .map(memories::get)
                .filter(Objects::nonNull)
                .filter(this::isMemoryValid)
                .filter(m -> m.getEmbedding() != null)
                .map(m -> new ScoredMemory(m, cosineSimilarity(queryEmbedding, m.getEmbedding())))
                .filter(sm -> sm.score >= threshold)
                .sorted(Comparator.comparingDouble(sm -> -sm.score))
                .limit(topK)
                .map(sm -> {
                    sm.memory.recordAccess();
                    return sm.memory;
                })
                .collect(Collectors.toList());
    }
    
    /**
     * 更新记忆
     */
    public void updateMemory(String memoryId, String newContent) {
        Memory memory = memories.get(memoryId);
        if (memory != null) {
            memory.updateContent(newContent);
        }
    }
    
    /**
     * 删除记忆
     */
    public void deleteMemory(String memoryId) {
        Memory memory = memories.remove(memoryId);
        if (memory != null) {
            removeFromIndexes(memory);
        }
    }
    
    /**
     * 清除会话记忆
     */
    public void clearSessionMemory(String sessionId) {
        List<String> memoryIds = sessionIndex.remove(sessionId);
        if (memoryIds != null) {
            for (String memoryId : memoryIds) {
                Memory memory = memories.remove(memoryId);
                if (memory != null) {
                    removeFromIndexes(memory);
                }
            }
        }
    }
    
    /**
     * 清除项目记忆
     */
    public void clearProjectMemory(String projectId) {
        List<String> memoryIds = projectIndex.remove(projectId);
        if (memoryIds != null) {
            for (String memoryId : memoryIds) {
                Memory memory = memories.remove(memoryId);
                if (memory != null) {
                    removeFromIndexes(memory);
                }
            }
        }
    }
    
    /**
     * 清理过期记忆
     */
    public int cleanupExpiredMemories() {
        long currentTime = System.currentTimeMillis();
        int cleaned = 0;
        
        List<String> expiredIds = new ArrayList<>();
        
        for (Memory memory : memories.values()) {
            boolean expired = false;
            
            switch (memory.getType()) {
                case WORKING:
                    expired = memory.isExpired(currentTime, workingMemoryTtl);
                    break;
                case SHORT_TERM:
                    expired = memory.isExpired(currentTime, shortTermMemoryTtl);
                    break;
                case LONG_TERM:
                case SEMANTIC:
                    // 长期记忆和语义记忆不自动过期
                    break;
            }
            
            if (expired) {
                expiredIds.add(memory.getId());
            }
        }
        
        for (String memoryId : expiredIds) {
            deleteMemory(memoryId);
            cleaned++;
        }
        
        return cleaned;
    }
    
    /**
     * 获取记忆统计信息
     */
    public MemoryStats getStats() {
        MemoryStats stats = new MemoryStats();
        stats.totalMemories = memories.size();
        
        for (MemoryType type : MemoryType.values()) {
            List<String> typeMemories = typeIndex.getOrDefault(type, new ArrayList<>());
            switch (type) {
                case WORKING:
                    stats.workingMemories = typeMemories.size();
                    break;
                case SHORT_TERM:
                    stats.shortTermMemories = typeMemories.size();
                    break;
                case LONG_TERM:
                    stats.longTermMemories = typeMemories.size();
                    break;
                case SEMANTIC:
                    stats.semanticMemories = typeMemories.size();
                    break;
            }
        }
        
        stats.activeSessions = sessionIndex.size();
        stats.activeProjects = projectIndex.size();
        
        return stats;
    }
    
    /**
     * 更新索引
     */
    private void updateIndexes(Memory memory) {
        // 会话索引
        if (memory.getSessionId() != null) {
            sessionIndex.computeIfAbsent(memory.getSessionId(), k -> new ArrayList<>())
                    .add(memory.getId());
        }
        
        // 项目索引
        if (memory.getProjectId() != null) {
            projectIndex.computeIfAbsent(memory.getProjectId(), k -> new ArrayList<>())
                    .add(memory.getId());
        }
        
        // 类型索引
        typeIndex.computeIfAbsent(memory.getType(), k -> new ArrayList<>())
                .add(memory.getId());
    }
    
    /**
     * 从索引中移除
     */
    private void removeFromIndexes(Memory memory) {
        if (memory.getSessionId() != null) {
            List<String> sessionMemories = sessionIndex.get(memory.getSessionId());
            if (sessionMemories != null) {
                sessionMemories.remove(memory.getId());
            }
        }
        
        if (memory.getProjectId() != null) {
            List<String> projectMemories = projectIndex.get(memory.getProjectId());
            if (projectMemories != null) {
                projectMemories.remove(memory.getId());
            }
        }
        
        List<String> typeMemories = typeIndex.get(memory.getType());
        if (typeMemories != null) {
            typeMemories.remove(memory.getId());
        }
    }
    
    /**
     * 检查记忆是否有效
     */
    private boolean isMemoryValid(Memory memory) {
        long currentTime = System.currentTimeMillis();
        
        switch (memory.getType()) {
            case WORKING:
                return !memory.isExpired(currentTime, workingMemoryTtl);
            case SHORT_TERM:
                return !memory.isExpired(currentTime, shortTermMemoryTtl);
            case LONG_TERM:
            case SEMANTIC:
                return true;
            default:
                return false;
        }
    }
    
    /**
     * 按重要性比较
     */
    private int compareByImportance(Memory m1, Memory m2) {
        // 先按重要性降序
        int importanceCompare = Double.compare(m2.getImportance(), m1.getImportance());
        if (importanceCompare != 0) {
            return importanceCompare;
        }
        
        // 重要性相同，按访问次数降序
        int accessCompare = Integer.compare(m2.getAccessCount(), m1.getAccessCount());
        if (accessCompare != 0) {
            return accessCompare;
        }
        
        // 最后按创建时间降序（新的在前）
        return Long.compare(m2.getCreatedAt(), m1.getCreatedAt());
    }
    
    /**
     * 计算余弦相似度
     */
    private double cosineSimilarity(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        if (norm1 == 0 || norm2 == 0) {
            return 0.0;
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * 配置项
     */
    public void setWorkingMemoryTtl(long ttl) {
        this.workingMemoryTtl = ttl;
    }
    
    public void setShortTermMemoryTtl(long ttl) {
        this.shortTermMemoryTtl = ttl;
    }
    
    /**
     * 记忆评分结果（内部使用）
     */
    private static class ScoredMemory {
        final Memory memory;
        final double score;
        
        ScoredMemory(Memory memory, double score) {
            this.memory = memory;
            this.score = score;
        }
    }
    
    /**
     * 记忆统计信息
     */
    public static class MemoryStats {
        public int totalMemories;
        public int workingMemories;
        public int shortTermMemories;
        public int longTermMemories;
        public int semanticMemories;
        public int activeSessions;
        public int activeProjects;
        
        @Override
        public String toString() {
            return "MemoryStats{" +
                    "total=" + totalMemories +
                    ", working=" + workingMemories +
                    ", shortTerm=" + shortTermMemories +
                    ", longTerm=" + longTermMemories +
                    ", semantic=" + semanticMemories +
                    ", sessions=" + activeSessions +
                    ", projects=" + activeProjects +
                    '}';
        }
    }
}
