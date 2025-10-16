package io.leavesfly.tinyai.agent.context;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.*;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 记忆管理系统
 * 负责管理工作记忆、情节记忆和语义记忆
 * 
 * @author 山泽
 */
public class MemoryManager {
    
    private static final int WORKING_MEMORY_MAX_SIZE = 10;  // 工作记忆容量限制
    
    private final String dbPath;
    private Connection connection;
    
    // 不同类型的记忆存储
    private final Deque<Memory> workingMemory;           // 工作记忆（容量受限）
    private final List<Memory> episodicMemory;           // 情节记忆
    private final Map<String, Memory> semanticMemory;    // 语义记忆
    private final Map<String, Memory> memoryIndex;       // 记忆索引
    
    // 构造函数
    public MemoryManager() {
        this(":memory:");
    }
    
    public MemoryManager(String dbPath) {
        this.dbPath = dbPath;
        this.workingMemory = new ConcurrentLinkedDeque<>();
        this.episodicMemory = new ArrayList<>();
        this.semanticMemory = new HashMap<>();
        this.memoryIndex = new HashMap<>();
        
        // 初始化数据库
        initDatabase();
    }
    
    /**
     * 初始化记忆数据库
     */
    private void initDatabase() {
        try {
            connection = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
            
            String createTableSQL = "CREATE TABLE IF NOT EXISTS memories (" +
                    "id TEXT PRIMARY KEY," +
                    "content TEXT NOT NULL," +
                    "memory_type TEXT NOT NULL," +
                    "timestamp REAL NOT NULL," +
                    "importance REAL DEFAULT 0.0," +
                    "access_count INTEGER DEFAULT 0," +
                    "last_accessed REAL," +
                    "embedding TEXT," +
                    "metadata TEXT" +
                    ")";
            
            try (Statement stmt = connection.createStatement()) {
                stmt.execute(createTableSQL);
            }
            
        } catch (SQLException e) {
            System.err.println("初始化数据库失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 添加记忆
     * 
     * @param content 记忆内容
     * @param memoryType 记忆类型
     * @param importance 重要性分数
     * @param metadata 元数据
     * @return 记忆ID
     */
    public String addMemory(String content, String memoryType, double importance, Map<String, Object> metadata) {
        String memoryId = generateMemoryId(content);
        
        Memory memory = new Memory(memoryId, content, memoryType, importance);
        if (metadata != null) {
            memory.setMetadata(metadata);
        }
        
        // 根据记忆类型存储
        switch (memoryType.toLowerCase()) {
            case "working":
                // 工作记忆有容量限制
                if (workingMemory.size() >= WORKING_MEMORY_MAX_SIZE) {
                    workingMemory.removeFirst();  // 移除最旧的记忆
                }
                workingMemory.addLast(memory);
                break;
                
            case "episodic":
                episodicMemory.add(memory);
                break;
                
            case "semantic":
                String semanticKey = extractSemanticKey(content);
                semanticMemory.put(semanticKey, memory);
                break;
                
            default:
                throw new IllegalArgumentException("不支持的记忆类型: " + memoryType);
        }
        
        // 添加到索引
        memoryIndex.put(memoryId, memory);
        
        // 保存到数据库
        saveMemoryToDb(memory);
        
        return memoryId;
    }
    
    /**
     * 添加记忆（重载方法）
     */
    public String addMemory(String content, String memoryType) {
        return addMemory(content, memoryType, 0.0, null);
    }
    
    public String addMemory(String content, String memoryType, double importance) {
        return addMemory(content, memoryType, importance, null);
    }
    
    /**
     * 检索相关记忆
     * 
     * @param query 查询内容
     * @param memoryType 记忆类型过滤（可选）
     * @param limit 返回数量限制
     * @return 相关记忆列表
     */
    public List<Memory> retrieveMemories(String query, String memoryType, int limit) {
        List<Memory> memories = new ArrayList<>();
        
        // 从工作记忆检索
        if (memoryType == null || "working".equals(memoryType)) {
            for (Memory memory : workingMemory) {
                if (isRelevant(query, memory.getContent())) {
                    memories.add(memory);
                }
            }
        }
        
        // 从情节记忆检索
        if (memoryType == null || "episodic".equals(memoryType)) {
            for (Memory memory : episodicMemory) {
                if (isRelevant(query, memory.getContent())) {
                    memories.add(memory);
                }
            }
        }
        
        // 从语义记忆检索
        if (memoryType == null || "semantic".equals(memoryType)) {
            for (Memory memory : semanticMemory.values()) {
                if (isRelevant(query, memory.getContent())) {
                    memories.add(memory);
                }
            }
        }
        
        // 按重要性和访问次数排序
        memories.sort((m1, m2) -> {
            int importanceCompare = Double.compare(m2.getImportance(), m1.getImportance());
            if (importanceCompare != 0) {
                return importanceCompare;
            }
            return Integer.compare(m2.getAccessCount(), m1.getAccessCount());
        });
        
        // 更新访问统计
        List<Memory> result = memories.stream()
                .limit(limit)
                .collect(Collectors.toList());
        
        for (Memory memory : result) {
            memory.incrementAccess();
            saveMemoryToDb(memory);
        }
        
        return result;
    }
    
    /**
     * 检索相关记忆（重载方法）
     */
    public List<Memory> retrieveMemories(String query) {
        return retrieveMemories(query, null, 5);
    }
    
    public List<Memory> retrieveMemories(String query, int limit) {
        return retrieveMemories(query, null, limit);
    }
    
    /**
     * 记忆整合
     * 将重要的工作记忆转移到长期记忆
     */
    public void consolidateMemories() {
        List<Memory> toConsolidate = new ArrayList<>();
        
        for (Memory memory : workingMemory) {
            // 根据访问频率和重要性决定是否转移到长期记忆
            if (memory.getAccessCount() > 2 || memory.getImportance() > 0.7) {
                toConsolidate.add(memory);
            }
        }
        
        for (Memory memory : toConsolidate) {
            // 从工作记忆移除
            workingMemory.remove(memory);
            
            // 转移到情节记忆
            memory.setMemoryType("episodic");
            episodicMemory.add(memory);
            
            // 更新数据库
            saveMemoryToDb(memory);
        }
    }
    
    /**
     * 获取记忆统计信息
     */
    public Map<String, Object> getMemoryStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("working_memory_count", workingMemory.size());
        stats.put("episodic_memory_count", episodicMemory.size());
        stats.put("semantic_memory_count", semanticMemory.size());
        stats.put("total_memories", workingMemory.size() + episodicMemory.size() + semanticMemory.size());
        return stats;
    }
    
    /**
     * 根据ID获取记忆
     */
    public Memory getMemoryById(String id) {
        return memoryIndex.get(id);
    }
    
    /**
     * 删除记忆
     */
    public boolean deleteMemory(String id) {
        Memory memory = memoryIndex.remove(id);
        if (memory == null) {
            return false;
        }
        
        // 从相应的存储中移除
        switch (memory.getMemoryType().toLowerCase()) {
            case "working":
                workingMemory.remove(memory);
                break;
            case "episodic":
                episodicMemory.remove(memory);
                break;
            case "semantic":
                semanticMemory.values().remove(memory);
                break;
        }
        
        // 从数据库删除
        try {
            String sql = "DELETE FROM memories WHERE id = ?";
            try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
                pstmt.setString(1, id);
                pstmt.executeUpdate();
            }
        } catch (SQLException e) {
            System.err.println("删除记忆失败: " + e.getMessage());
            return false;
        }
        
        return true;
    }
    
    /**
     * 生成记忆ID
     */
    private String generateMemoryId(String content) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            String input = content + System.currentTimeMillis();
            byte[] hash = md.digest(input.getBytes());
            
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            
            return hexString.toString().substring(0, 16);
        } catch (NoSuchAlgorithmException e) {
            // 备用方案
            return "mem_" + System.currentTimeMillis() + "_" + Math.abs(content.hashCode());
        }
    }
    
    /**
     * 提取语义记忆的键
     */
    private String extractSemanticKey(String content) {
        // 简单的关键词提取
        Pattern pattern = Pattern.compile("\\w+");
        Matcher matcher = pattern.matcher(content.toLowerCase());
        
        Set<String> words = new LinkedHashSet<>();
        while (matcher.find() && words.size() < 5) {
            String word = matcher.group();
            if (word.length() > 2) {  // 过滤短词
                words.add(word);
            }
        }
        
        return String.join(" ", words);
    }
    
    /**
     * 判断记忆是否与查询相关
     */
    private boolean isRelevant(String query, String content) {
        if (query == null || content == null) {
            return false;
        }
        
        // 直接包含关系检查（对中文更友好）
        if (content.toLowerCase().contains(query.toLowerCase())) {
            return true;
        }
        
        Set<String> queryWords = extractWords(query.toLowerCase());
        Set<String> contentWords = extractWords(content.toLowerCase());
        
        // 计算词汇重叠度
        Set<String> intersection = new HashSet<>(queryWords);
        intersection.retainAll(contentWords);
        
        return !intersection.isEmpty();
    }
    
    /**
     * 提取单词集合
     */
    private Set<String> extractWords(String text) {
        // 支持中英文词汇提取
        Pattern pattern = Pattern.compile("[\\w\\u4e00-\\u9fa5]+");
        Matcher matcher = pattern.matcher(text);
        
        Set<String> words = new HashSet<>();
        while (matcher.find()) {
            String word = matcher.group();
            // 对中文词汇更宽松的长度要求
            if (word.length() >= 1) {
                words.add(word);
            }
        }
        
        return words;
    }
    
    /**
     * 保存记忆到数据库
     */
    private void saveMemoryToDb(Memory memory) {
        try {
            String sql = "INSERT OR REPLACE INTO memories " +
                    "(id, content, memory_type, timestamp, importance, access_count, " +
                    "last_accessed, embedding, metadata) " +
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
            
            try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
                pstmt.setString(1, memory.getId());
                pstmt.setString(2, memory.getContent());
                pstmt.setString(3, memory.getMemoryType());
                pstmt.setDouble(4, memory.getTimestamp().toEpochSecond(ZoneOffset.UTC));
                pstmt.setDouble(5, memory.getImportance());
                pstmt.setInt(6, memory.getAccessCount());
                pstmt.setDouble(7, memory.getLastAccessed().toEpochSecond(ZoneOffset.UTC));
                pstmt.setString(8, null); // embedding暂时为null
                pstmt.setString(9, "{}"); // metadata简化为空JSON
                
                pstmt.executeUpdate();
            }
            
        } catch (SQLException e) {
            System.err.println("保存记忆到数据库失败: " + e.getMessage());
        }
    }
    
    /**
     * 关闭资源
     */
    public void close() {
        try {
            if (connection != null && !connection.isClosed()) {
                connection.close();
            }
        } catch (SQLException e) {
            System.err.println("关闭数据库连接失败: " + e.getMessage());
        }
    }
}