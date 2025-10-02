package io.leavesfly.tinyai.agent.rag;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.*;

/**
 * RAG系统单元测试
 * 测试RAG系统的各个组件和功能
 */
public class TestRag {
    
    private RAGSystem ragSystem;
    private List<Map<String, Object>> testDocuments;

    @Before
    public void setUp() {
        // 初始化RAG系统
        ragSystem = new RAGSystem(100, 0.05);
        
        // 准备测试文档
        testDocuments = new ArrayList<>();
        
        Map<String, Object> doc1 = new HashMap<>();
        doc1.put("id", "test_doc_1");
        doc1.put("content", "Python是一种高级编程语言，广泛用于数据科学和机器学习。");
        Map<String, Object> metadata1 = new HashMap<>();
        metadata1.put("category", "编程");
        metadata1.put("language", "zh");
        doc1.put("metadata", metadata1);
        testDocuments.add(doc1);
        
        Map<String, Object> doc2 = new HashMap<>();
        doc2.put("id", "test_doc_2");
        doc2.put("content", "机器学习是人工智能的一个分支，使用算法从数据中学习模式。");
        Map<String, Object> metadata2 = new HashMap<>();
        metadata2.put("category", "AI");
        metadata2.put("language", "zh");
        doc2.put("metadata", metadata2);
        testDocuments.add(doc2);
        
        Map<String, Object> doc3 = new HashMap<>();
        doc3.put("id", "test_doc_3");
        doc3.put("content", "深度学习使用神经网络模型处理复杂的数据和任务。");
        Map<String, Object> metadata3 = new HashMap<>();
        metadata3.put("category", "AI");
        metadata3.put("language", "zh");
        doc3.put("metadata", metadata3);
        testDocuments.add(doc3);
    }

    @Test
    public void testSimpleTokenizer() {
        System.out.println("测试SimpleTokenizer...");
        
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        
        // 测试中文分词
        List<String> tokens1 = tokenizer.tokenize("Python是编程语言");
        assertFalse("分词结果不应为空", tokens1.isEmpty());
        assertTrue("应包含'python'", tokens1.contains("python"));
        assertTrue("应包含'是'", tokens1.contains("是"));
        
        // 测试英文分词
        List<String> tokens2 = tokenizer.tokenize("machine learning algorithm");
        assertTrue("应包含'machine'", tokens2.contains("machine"));
        assertTrue("应包含'learning'", tokens2.contains("learning"));
        
        // 测试词汇表构建
        List<String> texts = Arrays.asList("Python编程", "机器学习", "深度学习");
        tokenizer.buildVocab(texts);
        assertTrue("词汇表大小应大于0", tokenizer.getVocabSize() > 0);
        
        System.out.println("SimpleTokenizer测试通过");
    }

    @Test
    public void testTFIDFVectorizer() {
        System.out.println("测试TFIDFVectorizer...");
        
        TFIDFVectorizer vectorizer = new TFIDFVectorizer(50);
        
        List<String> documents = Arrays.asList(
            "Python是编程语言",
            "机器学习很有趣",
            "深度学习使用神经网络"
        );
        
        // 测试训练
        vectorizer.fit(documents);
        assertTrue("模型应已训练", vectorizer.isTrained());
        assertTrue("特征维度应大于0", vectorizer.getFeatureDimension() > 0);
        
        // 测试向量化
        List<Double> vector = vectorizer.transform("Python编程");
        assertNotNull("向量不应为null", vector);
        assertEquals("向量维度应与特征维度一致", vectorizer.getFeatureDimension(), vector.size());
        
        System.out.println("TFIDFVectorizer测试通过");
    }

    @Test
    public void testVectorSimilarity() {
        System.out.println("测试VectorSimilarity...");
        
        List<Double> vec1 = Arrays.asList(1.0, 2.0, 3.0);
        List<Double> vec2 = Arrays.asList(2.0, 4.0, 6.0);
        List<Double> vec3 = Arrays.asList(1.0, 0.0, 0.0);
        
        // 测试余弦相似度
        double cosineSim = VectorSimilarity.cosineSimilarity(vec1, vec2);
        assertTrue("余弦相似度应在[0,1]范围内", cosineSim >= 0.0 && cosineSim <= 1.0);
        assertEquals("平行向量余弦相似度应为1.0", 1.0, cosineSim, 0.001);
        
        // 测试欧几里得距离
        double euclideanDist = VectorSimilarity.euclideanDistance(vec1, vec3);
        assertTrue("欧几里得距离应大于0", euclideanDist > 0);
        
        // 测试曼哈顿距离
        double manhattanDist = VectorSimilarity.manhattanDistance(vec1, vec3);
        assertTrue("曼哈顿距离应大于0", manhattanDist > 0);
        
        // 测试向量标准化
        List<Double> normalized = VectorSimilarity.normalize(vec1);
        double norm = VectorSimilarity.l2Norm(normalized);
        assertEquals("标准化向量的L2范数应为1.0", 1.0, norm, 0.001);
        
        System.out.println("VectorSimilarity测试通过");
    }

    @Test
    public void testVectorDatabase() {
        System.out.println("测试VectorDatabase...");
        
        VectorDatabase db = new VectorDatabase();
        
        // 创建测试文档
        List<Double> embedding = Arrays.asList(0.1, 0.2, 0.3, 0.4);
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("type", "test");
        
        Document doc = new Document("test_id", "测试内容", metadata, embedding, null);
        
        // 测试添加文档
        db.addDocument(doc);
        assertEquals("文档数量应为1", 1, db.countDocuments());
        
        // 测试获取文档
        Document retrieved = db.getDocument("test_id");
        assertNotNull("应能获取到文档", retrieved);
        assertEquals("文档ID应匹配", "test_id", retrieved.getId());
        assertEquals("文档内容应匹配", "测试内容", retrieved.getContent());
        
        // 测试获取所有文档
        List<Document> allDocs = db.getAllDocuments();
        assertEquals("所有文档数量应为1", 1, allDocs.size());
        
        // 测试删除文档
        boolean deleted = db.deleteDocument("test_id");
        assertTrue("应能成功删除文档", deleted);
        assertEquals("删除后文档数量应为0", 0, db.countDocuments());
        
        db.close();
        System.out.println("VectorDatabase测试通过");
    }

    @Test
    public void testRAGSystem() {
        System.out.println("测试RAGSystem...");
        
        // 测试添加文档
        ragSystem.addDocuments(testDocuments);
        assertTrue("系统应已训练", ragSystem.isTrained());
        assertEquals("文档数量应为3", 3, ragSystem.getDocumentsCount());
        
        // 测试检索
        List<QueryResult> results = ragSystem.search("Python编程", 2);
        assertNotNull("检索结果不应为null", results);
        assertFalse("检索结果不应为空", results.isEmpty());
        
        // 验证结果排序
        if (results.size() > 1) {
            assertTrue("结果应按相似度降序排列", 
                results.get(0).getSimilarity() >= results.get(1).getSimilarity());
        }
        
        // 测试上下文生成
        String context = ragSystem.generateContext("机器学习");
        assertNotNull("上下文不应为null", context);
        assertFalse("上下文不应为空", context.trim().isEmpty());
        
        // 测试获取统计信息
        Map<String, Object> stats = ragSystem.getStatistics();
        assertNotNull("统计信息不应为null", stats);
        assertTrue("统计信息应包含文档数量", stats.containsKey("documentsCount"));
        
        // 测试不同相似度方法
        List<QueryResult> euclideanResults = ragSystem.search("深度学习", 2, RAGSystem.SimilarityMethod.EUCLIDEAN);
        assertNotNull("欧几里得检索结果不应为null", euclideanResults);
        
        ragSystem.close();
        System.out.println("RAGSystem测试通过");
    }

    @Test
    public void testDocumentOperations() {
        System.out.println("测试Document操作...");
        
        ragSystem.addDocuments(testDocuments);
        
        // 测试获取单个文档
        Document doc = ragSystem.getDocument("test_doc_1");
        assertNotNull("应能获取到文档", doc);
        assertEquals("文档ID应匹配", "test_doc_1", doc.getId());
        
        // 测试获取所有文档
        List<Document> allDocs = ragSystem.getAllDocuments();
        assertEquals("所有文档数量应为3", 3, allDocs.size());
        
        // 测试删除文档
        boolean deleted = ragSystem.deleteDocument("test_doc_1");
        assertTrue("应能成功删除文档", deleted);
        assertEquals("删除后文档数量应为2", 2, ragSystem.getDocumentsCount());
        
        // 测试清空所有文档
        ragSystem.clearAllDocuments();
        assertEquals("清空后文档数量应为0", 0, ragSystem.getDocumentsCount());
        
        ragSystem.close();
        System.out.println("Document操作测试通过");
    }

    @Test
    public void testQueryResult() {
        System.out.println("测试QueryResult...");
        
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("test", "value");
        Document doc = new Document("test_id", "测试内容", metadata);
        
        QueryResult result = new QueryResult(doc, 0.85, 1);
        
        assertEquals("文档应匹配", doc, result.getDocument());
        assertEquals("相似度应匹配", 0.85, result.getSimilarity(), 0.001);
        assertEquals("排名应匹配", 1, result.getRank());
        
        // 测试toString方法
        String resultStr = result.toString();
        assertNotNull("toString结果不应为null", resultStr);
        assertTrue("应包含文档ID", resultStr.contains("test_id"));
        
        System.out.println("QueryResult测试通过");
    }

    @Test
    public void testEdgeCases() {
        System.out.println("测试边界情况...");
        
        RAGSystem emptyRag = new RAGSystem();
        
        // 测试空系统检索
        List<QueryResult> emptyResults = emptyRag.search("测试查询");
        assertTrue("空系统检索结果应为空", emptyResults.isEmpty());
        
        // 测试空向量相似度
        List<Double> emptyVec1 = new ArrayList<>();
        List<Double> emptyVec2 = new ArrayList<>();
        double emptySim = VectorSimilarity.cosineSimilarity(emptyVec1, emptyVec2);
        assertEquals("空向量相似度应为0", 0.0, emptySim, 0.001);
        
        // 测试不同长度向量
        List<Double> vec1 = Arrays.asList(1.0, 2.0);
        List<Double> vec2 = Arrays.asList(1.0, 2.0, 3.0);
        double diffLenSim = VectorSimilarity.cosineSimilarity(vec1, vec2);
        assertEquals("不同长度向量相似度应为0", 0.0, diffLenSim, 0.001);
        
        emptyRag.close();
        System.out.println("边界情况测试通过");
    }

    @Test
    public void testPerformance() {
        System.out.println("测试性能...");
        
        // 创建大量测试文档
        List<Map<String, Object>> largeDocs = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            Map<String, Object> doc = new HashMap<>();
            doc.put("id", "perf_doc_" + i);
            doc.put("content", "这是第" + i + "个性能测试文档，包含一些随机内容用于测试向量化和检索性能。");
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("index", i);
            metadata.put("category", "performance");
            doc.put("metadata", metadata);
            largeDocs.add(doc);
        }
        
        long startTime = System.currentTimeMillis();
        
        RAGSystem perfRag = new RAGSystem(200, 0.05);
        perfRag.addDocuments(largeDocs);
        
        long addTime = System.currentTimeMillis();
        
        // 测试检索性能
        List<QueryResult> results = perfRag.search("性能测试", 10);
        
        long searchTime = System.currentTimeMillis();
        
        System.out.println("添加50个文档耗时: " + (addTime - startTime) + "ms");
        System.out.println("检索耗时: " + (searchTime - addTime) + "ms");
        System.out.println("检索到" + results.size() + "个结果");
        
        assertTrue("添加文档时间应在合理范围内", (addTime - startTime) < 5000);
        assertTrue("检索时间应在合理范围内", (searchTime - addTime) < 1000);
        
        perfRag.close();
        System.out.println("性能测试通过");
    }
}
