package io.leavesfly.tinyai.agent.context;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 简单的文本嵌入系统
 * 基于TF-IDF的模拟实现，用于文本向量化和相似度计算
 * 
 * @author 山泽
 */
public class SimpleEmbedding {
    
    private final int dimension;                        // 向量维度
    private Map<String, Integer> vocabulary;            // 词汇表
    private Map<String, Double> idf;                    // 逆文档频率
    private boolean isTrained;                          // 是否已训练
    
    // 构造函数
    public SimpleEmbedding() {
        this(128);
    }
    
    public SimpleEmbedding(int dimension) {
        this.dimension = dimension;
        this.vocabulary = new HashMap<>();
        this.idf = new HashMap<>();
        this.isTrained = false;
    }
    
    /**
     * 训练嵌入模型
     * 基于提供的文本集合构建词汇表和IDF值
     * 
     * @param texts 训练文本集合
     */
    public void fit(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return;
        }
        
        // 统计词频和文档频率
        Map<String, Integer> wordCounts = new HashMap<>();
        Map<String, Set<Integer>> docWordCounts = new HashMap<>();
        
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);
            Set<String> wordsInDoc = extractWords(text);
            
            for (String word : wordsInDoc) {
                // 统计总词频
                wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
                
                // 统计包含该词的文档
                docWordCounts.computeIfAbsent(word, k -> new HashSet<>()).add(i);
            }
        }
        
        // 构建词汇表索引（只包含频率足够的词汇）
        this.vocabulary = new HashMap<>();
        int index = 0;
        for (String word : wordCounts.keySet()) {
            if (wordCounts.get(word) >= 1) { // 至少出现一次
                this.vocabulary.put(word, index++);
            }
        }
        
        // 计算IDF值
        this.idf = new HashMap<>();
        int numDocs = texts.size();
        for (Map.Entry<String, Set<Integer>> entry : docWordCounts.entrySet()) {
            String word = entry.getKey();
            if (vocabulary.containsKey(word)) {
                int docFreq = entry.getValue().size();
                // 使用平滑化的IDF计算，避免IDF为0
                double idfValue = Math.log((double) (numDocs + 1) / (docFreq + 1)) + 1.0;
                this.idf.put(word, idfValue);
            }
        }
        
        this.isTrained = true;
    }
    
    /**
     * 编码文本为向量
     * 
     * @param text 输入文本
     * @return 向量表示
     */
    public List<Double> encode(String text) {
        List<Double> vector = new ArrayList<>(Collections.nCopies(dimension, 0.0));
        
        if (!isTrained || vocabulary.isEmpty()) {
            return vector;
        }
        
        // 提取词汇并统计词频
        Set<String> words = extractWords(text);
        Map<String, Integer> wordFreq = new HashMap<>();
        
        for (String word : words) {
            wordFreq.put(word, wordFreq.getOrDefault(word, 0) + 1);
        }
        
        int totalWords = words.size();
        if (totalWords == 0) {
            return vector;
        }
        
        // 计算TF-IDF向量
        for (Map.Entry<String, Integer> entry : wordFreq.entrySet()) {
            String word = entry.getKey();
            int count = entry.getValue();
            
            if (vocabulary.containsKey(word)) {
                int wordIndex = vocabulary.get(word) % dimension;
                
                // 计算TF-IDF值
                double tf = (double) count / totalWords;
                double idfValue = idf.getOrDefault(word, 1.0);
                double tfidf = tf * idfValue;
                
                // 累加到对应维度
                vector.set(wordIndex, vector.get(wordIndex) + tfidf);
            } else {
                // 处理未知词汇：使用一个固定的缺省值和随机维度
                int unknownIndex = Math.abs(word.hashCode()) % dimension;
                double unknownValue = 0.1; // 给未知词一个小的值
                vector.set(unknownIndex, vector.get(unknownIndex) + unknownValue);
            }
        }
        
        // L2归一化
        return normalize(vector);
    }
    
    /**
     * 计算两个向量的余弦相似度
     * 
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 相似度值 (0-1)
     */
    public double similarity(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.size() != vec2.size()) {
            return 0.0;
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vec1.size(); i++) {
            double v1 = vec1.get(i);
            double v2 = vec2.get(i);
            
            dotProduct += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * 计算文本相似度
     * 
     * @param text1 文本1
     * @param text2 文本2
     * @return 相似度值 (0-1)
     */
    public double textSimilarity(String text1, String text2) {
        List<Double> vec1 = encode(text1);
        List<Double> vec2 = encode(text2);
        return similarity(vec1, vec2);
    }
    
    /**
     * 批量编码文本
     * 
     * @param texts 文本列表
     * @return 向量列表
     */
    public List<List<Double>> batchEncode(List<String> texts) {
        return texts.stream()
                .map(this::encode)
                .collect(Collectors.toList());
    }
    
    /**
     * 在文本集合中查找最相似的文本
     * 
     * @param query 查询文本
     * @param candidates 候选文本集合
     * @param topK 返回前K个最相似的
     * @return 相似度排序的结果
     */
    public List<SimilarityResult> findMostSimilar(String query, List<String> candidates, int topK) {
        List<Double> queryVector = encode(query);
        
        List<SimilarityResult> results = new ArrayList<>();
        
        for (int i = 0; i < candidates.size(); i++) {
            String candidate = candidates.get(i);
            List<Double> candidateVector = encode(candidate);
            double similarity = similarity(queryVector, candidateVector);
            
            results.add(new SimilarityResult(i, candidate, similarity));
        }
        
        // 按相似度降序排序
        results.sort((a, b) -> Double.compare(b.getSimilarity(), a.getSimilarity()));
        
        // 返回前topK个结果
        return results.stream()
                .limit(topK)
                .collect(Collectors.toList());
    }
    
    /**
     * 提取文本中的词汇
     */
    public Set<String> extractWords(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new HashSet<>();
        }
        
        Set<String> words = new HashSet<>();
        
        // 对中文文本，按字符切分（简化处理）
        String processedText = text.toLowerCase().trim();
        
        // 先提取英文单词
        Pattern englishPattern = Pattern.compile("[a-zA-Z]+");
        Matcher englishMatcher = englishPattern.matcher(processedText);
        while (englishMatcher.find()) {
            String word = englishMatcher.group();
            if (word.length() >= 2) {
                words.add(word);
            }
        }
        
        // 对中文，按字符和双字组合切分
        Pattern chinesePattern = Pattern.compile("[\u4e00-\u9fa5]");
        Matcher chineseMatcher = chinesePattern.matcher(processedText);
        
        List<String> chineseChars = new ArrayList<>();
        while (chineseMatcher.find()) {
            chineseChars.add(chineseMatcher.group());
        }
        
        // 添加单个中文字符
        for (String ch : chineseChars) {
            words.add(ch);
        }
        
        // 添加中文双字词组
        for (int i = 0; i < chineseChars.size() - 1; i++) {
            String bigram = chineseChars.get(i) + chineseChars.get(i + 1);
            words.add(bigram);
        }
        
        return words;
    }
    
    /**
     * L2归一化向量
     */
    private List<Double> normalize(List<Double> vector) {
        double normSquared = 0.0;
        for (double value : vector) {
            normSquared += value * value;
        }
        
        if (normSquared == 0.0) {
            return vector;
        }
        
        final double norm = Math.sqrt(normSquared);
        return vector.stream()
                .map(value -> value / norm)
                .collect(Collectors.toList());
    }
    
    // Getter 方法
    public int getDimension() {
        return dimension;
    }
    
    public boolean isTrained() {
        return isTrained;
    }
    
    public int getVocabularySize() {
        return vocabulary.size();
    }
    
    public Set<String> getVocabulary() {
        return vocabulary.keySet();
    }
    
    /**
     * 相似度结果类
     */
    public static class SimilarityResult {
        private final int index;
        private final String text;
        private final double similarity;
        
        public SimilarityResult(int index, String text, double similarity) {
            this.index = index;
            this.text = text;
            this.similarity = similarity;
        }
        
        public int getIndex() {
            return index;
        }
        
        public String getText() {
            return text;
        }
        
        public double getSimilarity() {
            return similarity;
        }
        
        @Override
        public String toString() {
            return String.format("SimilarityResult{index=%d, similarity=%.3f, text='%s'}", 
                               index, similarity, 
                               text.length() > 50 ? text.substring(0, 50) + "..." : text);
        }
    }
}