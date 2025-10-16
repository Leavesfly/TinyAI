package io.leavesfly.tinyai.agent.context;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * RAG (检索增强生成) 系统
 * 负责文档存储、检索和上下文生成
 * 
 * @author 山泽
 */
public class RAGSystem {
    
    private final List<Document> documents;           // 文档库
    private final List<List<Double>> documentEmbeddings; // 文档嵌入向量
    private final SimpleEmbedding embeddingModel;     // 嵌入模型
    private final Map<String, Integer> documentIndex; // 文档索引
    
    // 构造函数
    public RAGSystem() {
        this(128);
    }
    
    public RAGSystem(int embeddingDim) {
        this.documents = new ArrayList<>();
        this.documentEmbeddings = new ArrayList<>();
        this.embeddingModel = new SimpleEmbedding(embeddingDim);
        this.documentIndex = new HashMap<>();
    }
    
    /**
     * 添加文档到知识库
     */
    public void addDocument(String docId, String content, Map<String, Object> metadata) {
        if (docId == null || content == null) {
            throw new IllegalArgumentException("文档ID和内容不能为空");
        }
        
        // 检查是否已存在
        if (documentIndex.containsKey(docId)) {
            updateDocument(docId, content, metadata);
            return;
        }
        
        Document document = new Document(docId, content, metadata);
        documents.add(document);
        
        // 更新文档索引
        documentIndex.put(docId, documents.size() - 1);
        
        // 重新训练嵌入模型并计算所有文档的嵌入向量
        rebuildEmbeddings();
    }
    
    public void addDocument(String docId, String content) {
        addDocument(docId, content, null);
    }
    
    /**
     * 更新文档
     */
    public void updateDocument(String docId, String content, Map<String, Object> metadata) {
        Integer index = documentIndex.get(docId);
        if (index == null) {
            throw new IllegalArgumentException("文档不存在: " + docId);
        }
        
        Document document = documents.get(index);
        document.setContent(content);
        document.setTimestamp(LocalDateTime.now());
        
        if (metadata != null) {
            document.setMetadata(metadata);
        }
        
        rebuildEmbeddings();
    }
    
    /**
     * 删除文档
     */
    public boolean deleteDocument(String docId) {
        Integer index = documentIndex.get(docId);
        if (index == null) {
            return false;
        }
        
        documents.remove(index.intValue());
        documentEmbeddings.remove(index.intValue());
        
        // 重建索引
        documentIndex.clear();
        for (int i = 0; i < documents.size(); i++) {
            documentIndex.put(documents.get(i).getId(), i);
        }
        
        return true;
    }
    
    /**
     * 检索相关文档
     */
    public List<RetrievalResult> retrieve(String query, int topK) {
        if (documents.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<Double> queryEmbedding = embeddingModel.encode(query);
        List<RetrievalResult> results = new ArrayList<>();
        
        for (int i = 0; i < documents.size(); i++) {
            Document doc = documents.get(i);
            List<Double> docEmbedding = documentEmbeddings.get(i);
            double similarity = embeddingModel.similarity(queryEmbedding, docEmbedding);
            results.add(new RetrievalResult(doc, similarity));
        }
        
        results.sort((a, b) -> Double.compare(b.getSimilarity(), a.getSimilarity()));
        return results.stream().limit(topK).collect(Collectors.toList());
    }
    
    public List<RetrievalResult> retrieve(String query) {
        return retrieve(query, 3);
    }
    
    /**
     * 获取查询相关的上下文
     */
    public String getContext(String query, int maxLength) {
        List<RetrievalResult> relevantDocs = retrieve(query);
        
        if (relevantDocs.isEmpty()) {
            return "";
        }
        
        List<String> contextParts = new ArrayList<>();
        int currentLength = 0;
        
        for (RetrievalResult result : relevantDocs) {
            String content = result.getDocument().getContent();
            String docPart = String.format("文档 %s: %s", 
                                         result.getDocument().getId(), content);
            
            if (currentLength + docPart.length() <= maxLength) {
                contextParts.add(docPart);
                currentLength += docPart.length();
            } else {
                int remaining = maxLength - currentLength;
                if (remaining > 50) {
                    String truncated = content.substring(0, Math.min(content.length(), remaining - 20)) + "...";
                    contextParts.add(String.format("文档 %s: %s", result.getDocument().getId(), truncated));
                }
                break;
            }
        }
        
        return String.join("\n\n", contextParts);
    }
    
    public String getContext(String query) {
        return getContext(query, 1000);
    }
    
    // 其他实用方法
    public Document getDocument(String docId) {
        Integer index = documentIndex.get(docId);
        return index != null ? documents.get(index) : null;
    }
    
    public List<Document> getAllDocuments() {
        return new ArrayList<>(documents);
    }
    
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("document_count", documents.size());
        stats.put("embedding_dimension", embeddingModel.getDimension());
        stats.put("vocabulary_size", embeddingModel.getVocabularySize());
        stats.put("is_trained", embeddingModel.isTrained());
        return stats;
    }
    
    public void clear() {
        documents.clear();
        documentEmbeddings.clear();
        documentIndex.clear();
    }
    
    /**
     * 重新构建嵌入向量
     */
    private void rebuildEmbeddings() {
        if (documents.isEmpty()) {
            documentEmbeddings.clear();
            return;
        }
        
        List<String> allTexts = documents.stream()
                .map(Document::getContent)
                .collect(Collectors.toList());
        
        embeddingModel.fit(allTexts);
        
        documentEmbeddings.clear();
        for (Document doc : documents) {
            List<Double> embedding = embeddingModel.encode(doc.getContent());
            documentEmbeddings.add(embedding);
        }
    }
}