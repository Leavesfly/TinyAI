package io.leavesfly.tinyai.agent.rag;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 向量数据库类
 * 使用内存Map存储文档和向量嵌入信息（避免第三方依赖）
 */
public class VectorDatabase {
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
    
    private final Map<String, Document> documents;  // 内存存储文档映射

    /**
     * 默认构造函数（使用内存存储）
     */
    public VectorDatabase() {
        this.documents = new ConcurrentHashMap<>();
    }

    /**
     * 构造函数（保持兼容性）
     * @param dbPath 数据库路径（忽略，使用内存存储）
     */
    public VectorDatabase(String dbPath) {
        this(); // 使用内存存储
    }

    /**
     * 添加文档
     * @param document 文档对象
     */
    public void addDocument(Document document) {
        documents.put(document.getId(), document);
    }

    /**
     * 批量添加文档
     * @param documentsList 文档列表
     */
    public void addDocuments(List<Document> documentsList) {
        for (Document document : documentsList) {
            documents.put(document.getId(), document);
        }
    }

    /**
     * 获取指定ID的文档
     * @param docId 文档ID
     * @return 文档对象，不存在则返回null
     */
    public Document getDocument(String docId) {
        return documents.get(docId);
    }

    /**
     * 获取所有文档
     * @return 文档列表
     */
    public List<Document> getAllDocuments() {
        return new ArrayList<>(documents.values());
    }

    /**
     * 根据元数据查询文档
     * @param metadataKey 元数据键
     * @param metadataValue 元数据值
     * @return 匹配的文档列表
     */
    public List<Document> getDocumentsByMetadata(String metadataKey, String metadataValue) {
        return documents.values().stream()
                .filter(doc -> {
                    Map<String, Object> metadata = doc.getMetadata();
                    return metadata != null && metadataValue.equals(metadata.get(metadataKey));
                })
                .collect(Collectors.toList());
    }

    /**
     * 删除指定ID的文档
     * @param docId 文档ID
     * @return 是否删除成功
     */
    public boolean deleteDocument(String docId) {
        return documents.remove(docId) != null;
    }

    /**
     * 获取文档总数
     * @return 文档数量
     */
    public int countDocuments() {
        return documents.size();
    }

    /**
     * 清空所有文档
     */
    public void clearAllDocuments() {
        documents.clear();
    }

    /**
     * 关闭数据库连接（内存实现无需关闭）
     */
    public void close() {
        // 内存实现无需关闭
    }

    /**
     * 检查连接是否有效
     * @return 连接是否有效
     */
    public boolean isConnectionValid() {
        return true; // 内存实现始终有效
    }

    /**
     * 获取存储的文档数
     * @return 文档Map的大小
     */
    public int size() {
        return documents.size();
    }
    
    /**
     * 检查是否为空
     * @return 是否为空
     */
    public boolean isEmpty() {
        return documents.isEmpty();
    }
    
    /**
     * 检查是否包含指定ID的文档
     * @param docId 文档ID
     * @return 是否包含
     */
    public boolean containsDocument(String docId) {
        return documents.containsKey(docId);
    }
}