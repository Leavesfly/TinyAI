package io.leavesfly.tinyai.agent.cursor.v2.component.rag;

import io.leavesfly.tinyai.agent.cursor.v2.model.Context.CodeSnippet;

import java.util.*;
import java.util.stream.Collectors;

/**
 * RAG检索引擎
 * 封装tinyai-agent-rag的检索增强生成能力
 * 
 * 功能：
 * - 代码库向量化索引
 * - 语义相似度检索
 * - 混合检索策略（精确匹配+语义检索）
 * - 代码片段重排序
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class RAGEngine {
    
    /**
     * 代码片段索引（filePath -> CodeBlock列表）
     */
    private final Map<String, List<CodeBlock>> codeIndex;
    
    /**
     * 向量索引（用于语义检索）
     */
    private final List<CodeBlock> vectorIndex;
    
    /**
     * 检索策略
     */
    private RetrievalStrategy strategy = RetrievalStrategy.HYBRID;
    
    /**
     * 相似度阈值
     */
    private double similarityThreshold = 0.7;
    
    public RAGEngine() {
        this.codeIndex = new HashMap<>();
        this.vectorIndex = new ArrayList<>();
    }
    
    /**
     * 索引代码库
     * 
     * @param projectId 项目ID
     * @param codeFiles 代码文件列表
     */
    public void indexCodebase(String projectId, List<CodeFile> codeFiles) {
        for (CodeFile file : codeFiles) {
            indexFile(projectId, file);
        }
    }
    
    /**
     * 索引单个文件
     */
    public void indexFile(String projectId, CodeFile file) {
        // 分块
        List<CodeBlock> blocks = chunkCode(file);
        
        // 向量化（TODO: 集成实际的embedding服务）
        for (CodeBlock block : blocks) {
            block.embedding = generateEmbedding(block.content);
            block.projectId = projectId;
        }
        
        // 添加到索引
        codeIndex.put(file.filePath, blocks);
        vectorIndex.addAll(blocks);
    }
    
    /**
     * 语义检索
     * 
     * @param query 查询文本
     * @param topK 返回Top-K结果
     * @return 代码片段列表
     */
    public List<CodeSnippet> semanticSearch(String query, int topK) {
        return semanticSearch(query, topK, null);
    }
    
    /**
     * 语义检索（指定项目）
     */
    public List<CodeSnippet> semanticSearch(String query, int topK, String projectId) {
        // 向量化查询
        double[] queryEmbedding = generateEmbedding(query);
        
        // 计算相似度并排序
        List<ScoredBlock> scoredBlocks = vectorIndex.stream()
                .filter(block -> projectId == null || projectId.equals(block.projectId))
                .map(block -> new ScoredBlock(block, cosineSimilarity(queryEmbedding, block.embedding)))
                .filter(sb -> sb.score >= similarityThreshold)
                .sorted(Comparator.comparingDouble(sb -> -sb.score))
                .limit(topK)
                .collect(Collectors.toList());
        
        // 转换为CodeSnippet
        return scoredBlocks.stream()
                .map(sb -> new CodeSnippet(sb.block.filePath, sb.block.content, sb.score))
                .collect(Collectors.toList());
    }
    
    /**
     * 精确匹配检索
     * 
     * @param keywords 关键词列表
     * @param topK 返回Top-K结果
     * @return 代码片段列表
     */
    public List<CodeSnippet> exactSearch(List<String> keywords, int topK) {
        return exactSearch(keywords, topK, null);
    }
    
    /**
     * 精确匹配检索（指定项目）
     */
    public List<CodeSnippet> exactSearch(List<String> keywords, int topK, String projectId) {
        List<ScoredBlock> scoredBlocks = new ArrayList<>();
        
        for (CodeBlock block : vectorIndex) {
            if (projectId != null && !projectId.equals(block.projectId)) {
                continue;
            }
            
            double score = calculateKeywordScore(block.content, keywords);
            if (score > 0) {
                scoredBlocks.add(new ScoredBlock(block, score));
            }
        }
        
        return scoredBlocks.stream()
                .sorted(Comparator.comparingDouble(sb -> -sb.score))
                .limit(topK)
                .map(sb -> new CodeSnippet(sb.block.filePath, sb.block.content, sb.score))
                .collect(Collectors.toList());
    }
    
    /**
     * 混合检索
     * 结合精确匹配和语义检索
     */
    public List<CodeSnippet> hybridSearch(String query, List<String> keywords, int topK, String projectId) {
        // 语义检索结果
        List<CodeSnippet> semanticResults = semanticSearch(query, topK * 2, projectId);
        
        // 精确匹配结果
        List<CodeSnippet> exactResults = exactSearch(keywords, topK * 2, projectId);
        
        // 合并和去重
        Map<String, CodeSnippet> merged = new HashMap<>();
        
        for (CodeSnippet snippet : semanticResults) {
            String key = snippet.getFilePath() + ":" + snippet.getContent().hashCode();
            merged.put(key, snippet);
        }
        
        for (CodeSnippet snippet : exactResults) {
            String key = snippet.getFilePath() + ":" + snippet.getContent().hashCode();
            CodeSnippet existing = merged.get(key);
            if (existing != null) {
                // 融合分数
                snippet.setScore((existing.getScore() + snippet.getScore()) / 2.0);
            }
            merged.put(key, snippet);
        }
        
        // 重新排序
        return merged.values().stream()
                .sorted(Comparator.comparingDouble(s -> -s.getScore()))
                .limit(topK)
                .collect(Collectors.toList());
    }
    
    /**
     * 查找相关代码片段
     * 根据当前上下文查找相关代码
     */
    public List<CodeSnippet> findRelated(String currentFile, String currentCode, int topK) {
        // 使用当前代码作为查询
        List<String> keywords = extractKeywords(currentCode);
        
        return hybridSearch(currentCode, keywords, topK, null).stream()
                .filter(snippet -> !currentFile.equals(snippet.getFilePath()))
                .collect(Collectors.toList());
    }
    
    /**
     * 代码分块
     */
    private List<CodeBlock> chunkCode(CodeFile file) {
        List<CodeBlock> blocks = new ArrayList<>();
        
        // 简单的基于方法/类的分块策略
        // TODO: 实现更智能的代码分块（识别方法、类等）
        String[] lines = file.content.split("\n");
        StringBuilder currentBlock = new StringBuilder();
        int blockStartLine = 0;
        
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            currentBlock.append(line).append("\n");
            
            // 简单策略：每50行或遇到类/方法结束分块
            if ((i - blockStartLine >= 50) || 
                (line.trim().equals("}") && currentBlock.toString().contains("{"))) {
                
                if (currentBlock.length() > 0) {
                    CodeBlock block = new CodeBlock();
                    block.filePath = file.filePath;
                    block.content = currentBlock.toString();
                    block.startLine = blockStartLine;
                    block.endLine = i;
                    blocks.add(block);
                    
                    currentBlock = new StringBuilder();
                    blockStartLine = i + 1;
                }
            }
        }
        
        // 添加最后一个块
        if (currentBlock.length() > 0) {
            CodeBlock block = new CodeBlock();
            block.filePath = file.filePath;
            block.content = currentBlock.toString();
            block.startLine = blockStartLine;
            block.endLine = lines.length - 1;
            blocks.add(block);
        }
        
        return blocks;
    }
    
    /**
     * 生成向量（模拟实现）
     * TODO: 集成LLMGateway的embed()方法
     */
    private double[] generateEmbedding(String text) {
        // 模拟向量生成（实际应调用LLM的embedding API）
        double[] embedding = new double[768];
        Random random = new Random(text.hashCode());
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] = random.nextGaussian();
        }
        return embedding;
    }
    
    /**
     * 计算余弦相似度
     */
    private double cosineSimilarity(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            return 0.0;
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
     * 计算关键词得分
     */
    private double calculateKeywordScore(String content, List<String> keywords) {
        if (keywords == null || keywords.isEmpty()) {
            return 0.0;
        }
        
        String lowerContent = content.toLowerCase();
        int matchCount = 0;
        
        for (String keyword : keywords) {
            if (lowerContent.contains(keyword.toLowerCase())) {
                matchCount++;
            }
        }
        
        return (double) matchCount / keywords.size();
    }
    
    /**
     * 提取关键词
     */
    private List<String> extractKeywords(String code) {
        // 简单的关键词提取（实际应使用更复杂的NLP技术）
        List<String> keywords = new ArrayList<>();
        
        // 提取类名、方法名等
        String[] words = code.split("[^a-zA-Z0-9_]+");
        for (String word : words) {
            if (word.length() > 3 && Character.isUpperCase(word.charAt(0))) {
                keywords.add(word);
            }
        }
        
        return keywords.stream().distinct().limit(10).collect(Collectors.toList());
    }
    
    /**
     * 配置项
     */
    public void setStrategy(RetrievalStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void setSimilarityThreshold(double threshold) {
        this.similarityThreshold = threshold;
    }
    
    /**
     * 清除索引
     */
    public void clearIndex() {
        codeIndex.clear();
        vectorIndex.clear();
    }
    
    /**
     * 获取统计信息
     */
    public RAGStats getStats() {
        RAGStats stats = new RAGStats();
        stats.totalFiles = codeIndex.size();
        stats.totalBlocks = vectorIndex.size();
        return stats;
    }
    
    /**
     * 代码文件
     */
    public static class CodeFile {
        public String filePath;
        public String content;
        public String language;
        
        public CodeFile(String filePath, String content) {
            this.filePath = filePath;
            this.content = content;
        }
    }
    
    /**
     * 代码块
     */
    private static class CodeBlock {
        String projectId;
        String filePath;
        String content;
        int startLine;
        int endLine;
        double[] embedding;
    }
    
    /**
     * 评分结果
     */
    private static class ScoredBlock {
        final CodeBlock block;
        final double score;
        
        ScoredBlock(CodeBlock block, double score) {
            this.block = block;
            this.score = score;
        }
    }
    
    /**
     * 检索策略
     */
    public enum RetrievalStrategy {
        /** 仅精确匹配 */
        EXACT,
        /** 仅语义检索 */
        SEMANTIC,
        /** 混合检索 */
        HYBRID
    }
    
    /**
     * RAG统计信息
     */
    public static class RAGStats {
        public int totalFiles;
        public int totalBlocks;
        
        @Override
        public String toString() {
            return "RAGStats{files=" + totalFiles + ", blocks=" + totalBlocks + "}";
        }
    }
}
