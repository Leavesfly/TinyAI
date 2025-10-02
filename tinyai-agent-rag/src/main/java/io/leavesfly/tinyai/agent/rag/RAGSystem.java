package io.leavesfly.tinyai.agent.rag;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * RAG检索增强生成系统核心类
 * 整合分词、向量化、检索等功能，提供完整的RAG服务
 */
public class RAGSystem {
    
    // 相似度计算方法枚举
    public enum SimilarityMethod {
        COSINE,         // 余弦相似度
        EUCLIDEAN,      // 欧几里得距离转相似度
        MANHATTAN,      // 曼哈顿距离转相似度
        PEARSON         // 皮尔逊相关系数
    }
    
    private int vectorDim;                      // 向量维度
    private double similarityThreshold;        // 相似度阈值
    
    // 核心组件
    private TFIDFVectorizer vectorizer;         // TF-IDF向量化器
    private VectorDatabase vectorDb;            // 向量数据库
    
    // 状态信息
    private boolean isTrained;                  // 是否已训练
    private int documentsCount;                 // 文档总数

    /**
     * 构造函数
     * @param vectorDim 向量维度
     * @param similarityThreshold 相似度阈值
     */
    public RAGSystem(int vectorDim, double similarityThreshold) {
        this.vectorDim = vectorDim;
        this.similarityThreshold = similarityThreshold;
        
        // 初始化组件
        this.vectorizer = new TFIDFVectorizer(vectorDim);
        this.vectorDb = new VectorDatabase();
        
        // 初始化状态
        this.isTrained = false;
        this.documentsCount = 0;
    }

    /**
     * 默认构造函数
     */
    public RAGSystem() {
        this(512, 0.1);
    }

    /**
     * 批量添加文档
     * @param documentsData 文档数据列表，每个元素包含id、content、metadata
     */
    public void addDocuments(List<Map<String, Object>> documentsData) {
        System.out.println("正在添加 " + documentsData.size() + " 个文档...");
        
        // 准备文档内容用于训练向量化器
        List<String> contents = documentsData.stream()
                .map(doc -> (String) doc.get("content"))
                .collect(Collectors.toList());
        
        // 训练向量化器（如果还未训练）
        if (!isTrained) {
            System.out.println("训练向量化器...");
            vectorizer.fit(contents);
            isTrained = true;
        }
        
        // 向量化文档并存储
        List<Document> documents = new ArrayList<>();
        for (int i = 0; i < documentsData.size(); i++) {
            Map<String, Object> docData = documentsData.get(i);
            
            String docId = (String) docData.getOrDefault("id", "doc_" + System.currentTimeMillis() + "_" + i);
            String content = (String) docData.get("content");
            @SuppressWarnings("unchecked")
            Map<String, Object> metadata = (Map<String, Object>) docData.getOrDefault("metadata", new HashMap<>());
            
            // 计算文档向量
            List<Double> embedding = vectorizer.transform(content);
            
            // 创建文档对象
            Document document = new Document(docId, content, metadata, embedding, LocalDateTime.now());
            documents.add(document);
            
            if ((i + 1) % 10 == 0) { // 每10个文档显示一次进度
                System.out.println("已处理 " + (i + 1) + "/" + documentsData.size() + " 个文档");
            }
        }
        
        // 批量存储到向量数据库
        vectorDb.addDocuments(documents);
        
        documentsCount = vectorDb.countDocuments();
        System.out.println("文档添加完成！当前共有 " + documentsCount + " 个文档");
    }

    /**
     * 添加单个文档
     * @param docId 文档ID
     * @param content 文档内容
     * @param metadata 文档元数据
     */
    public void addDocument(String docId, String content, Map<String, Object> metadata) {
        Map<String, Object> docData = new HashMap<>();
        docData.put("id", docId);
        docData.put("content", content);
        docData.put("metadata", metadata != null ? metadata : new HashMap<>());
        
        addDocuments(Collections.singletonList(docData));
    }

    /**
     * 检索相关文档
     * @param query 查询文本
     * @param topK 返回的文档数量
     * @param similarityMethod 相似度计算方法
     * @return 查询结果列表
     */
    public List<QueryResult> search(String query, int topK, SimilarityMethod similarityMethod) {
        if (!isTrained) {
            System.out.println("RAG系统尚未训练，请先添加文档");
            return new ArrayList<>();
        }
        
        System.out.println("检索查询: '" + query + "'");
        
        // 向量化查询
        List<Double> queryEmbedding = vectorizer.transform(query);
        
        // 获取所有文档
        List<Document> allDocuments = vectorDb.getAllDocuments();
        
        // 计算相似度
        List<DocumentSimilarity> similarities = new ArrayList<>();
        for (Document doc : allDocuments) {
            if (doc.getEmbedding() != null && !doc.getEmbedding().isEmpty()) {
                double similarity = calculateSimilarity(queryEmbedding, doc.getEmbedding(), similarityMethod);
                
                if (similarity >= similarityThreshold) {
                    similarities.add(new DocumentSimilarity(doc, similarity));
                }
            }
        }
        
        // 按相似度排序
        similarities.sort((a, b) -> Double.compare(b.similarity, a.similarity));
        
        // 构建结果
        List<QueryResult> results = new ArrayList<>();
        int count = Math.min(topK, similarities.size());
        for (int i = 0; i < count; i++) {
            DocumentSimilarity docSim = similarities.get(i);
            results.add(new QueryResult(docSim.document, docSim.similarity, i + 1));
        }
        
        System.out.println("找到 " + results.size() + " 个相关文档");
        return results;
    }

    /**
     * 检索相关文档（使用默认余弦相似度）
     * @param query 查询文本
     * @param topK 返回的文档数量
     * @return 查询结果列表
     */
    public List<QueryResult> search(String query, int topK) {
        return search(query, topK, SimilarityMethod.COSINE);
    }

    /**
     * 检索相关文档（使用默认参数）
     * @param query 查询文本
     * @return 查询结果列表
     */
    public List<QueryResult> search(String query) {
        return search(query, 5, SimilarityMethod.COSINE);
    }

    /**
     * 为查询生成上下文
     * @param query 查询文本
     * @param maxContextLength 最大上下文长度
     * @return 生成的上下文文本
     */
    public String generateContext(String query, int maxContextLength) {
        List<QueryResult> searchResults = search(query, 5);
        
        if (searchResults.isEmpty()) {
            return "未找到相关内容。";
        }
        
        List<String> contextParts = new ArrayList<>();
        int currentLength = 0;
        
        for (QueryResult result : searchResults) {
            Document doc = result.getDocument();
            String content = doc.getContent();
            
            // 添加文档信息头
            String docHeader = String.format("[文档 %s, 相似度: %.3f]\n", doc.getId(), result.getSimilarity());
            
            if (currentLength + docHeader.length() + content.length() <= maxContextLength) {
                contextParts.add(docHeader + content);
                currentLength += docHeader.length() + content.length();
            } else {
                // 截断内容以适应长度限制
                int remainingSpace = maxContextLength - currentLength - docHeader.length();
                if (remainingSpace > 50) { // 确保有足够空间显示有意义的内容
                    String truncatedContent = content.substring(0, remainingSpace - 3) + "...";
                    contextParts.add(docHeader + truncatedContent);
                }
                break;
            }
        }
        
        return String.join("\n\n", contextParts);
    }

    /**
     * 为查询生成上下文（使用默认长度）
     * @param query 查询文本
     * @return 生成的上下文文本
     */
    public String generateContext(String query) {
        return generateContext(query, 1000);
    }

    /**
     * 获取系统统计信息
     * @return 统计信息映射
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("documentsCount", documentsCount);
        stats.put("vectorDimension", vectorDim);
        stats.put("isTrained", isTrained);
        stats.put("vocabularySize", isTrained ? vectorizer.getTokenizer().getVocabSize() : 0);
        stats.put("similarityThreshold", similarityThreshold);
        stats.put("actualFeatureDimension", isTrained ? vectorizer.getFeatureDimension() : 0);
        
        return stats;
    }

    /**
     * 根据ID获取文档
     * @param docId 文档ID
     * @return 文档对象
     */
    public Document getDocument(String docId) {
        return vectorDb.getDocument(docId);
    }

    /**
     * 获取所有文档
     * @return 文档列表
     */
    public List<Document> getAllDocuments() {
        return vectorDb.getAllDocuments();
    }

    /**
     * 删除文档
     * @param docId 文档ID
     * @return 是否删除成功
     */
    public boolean deleteDocument(String docId) {
        boolean success = vectorDb.deleteDocument(docId);
        if (success) {
            documentsCount = vectorDb.countDocuments();
        }
        return success;
    }

    /**
     * 清空所有文档
     */
    public void clearAllDocuments() {
        vectorDb.clearAllDocuments();
        documentsCount = 0;
    }

    /**
     * 设置相似度阈值
     * @param threshold 新的相似度阈值
     */
    public void setSimilarityThreshold(double threshold) {
        this.similarityThreshold = threshold;
    }

    /**
     * 获取相似度阈值
     * @return 相似度阈值
     */
    public double getSimilarityThreshold() {
        return similarityThreshold;
    }

    /**
     * 检查系统是否已训练
     * @return 是否已训练
     */
    public boolean isTrained() {
        return isTrained;
    }

    /**
     * 获取文档数量
     * @return 文档数量
     */
    public int getDocumentsCount() {
        return documentsCount;
    }

    /**
     * 关闭系统资源
     */
    public void close() {
        if (vectorDb != null) {
            vectorDb.close();
        }
    }

    /**
     * 计算两个向量的相似度
     */
    private double calculateSimilarity(List<Double> vec1, List<Double> vec2, SimilarityMethod method) {
        switch (method) {
            case COSINE:
                return VectorSimilarity.cosineSimilarity(vec1, vec2);
            case EUCLIDEAN:
                double euclideanDist = VectorSimilarity.euclideanDistance(vec1, vec2);
                return VectorSimilarity.distanceToSimilarity(euclideanDist);
            case MANHATTAN:
                double manhattanDist = VectorSimilarity.manhattanDistance(vec1, vec2);
                return VectorSimilarity.distanceToSimilarity(manhattanDist);
            case PEARSON:
                return Math.abs(VectorSimilarity.pearsonCorrelation(vec1, vec2)); // 使用绝对值
            default:
                return VectorSimilarity.cosineSimilarity(vec1, vec2);
        }
    }

    /**
     * 文档相似度内部类
     */
    private static class DocumentSimilarity {
        final Document document;
        final double similarity;
        
        DocumentSimilarity(Document document, double similarity) {
            this.document = document;
            this.similarity = similarity;
        }
    }

    @Override
    public String toString() {
        return "RAGSystem{" +
                "vectorDim=" + vectorDim +
                ", similarityThreshold=" + similarityThreshold +
                ", documentsCount=" + documentsCount +
                ", isTrained=" + isTrained +
                '}';
    }
}