package io.leavesfly.tinyai.agent.rag;

import java.util.*;

/**
 * TF-IDF向量化器类
 * 将文本转换为TF-IDF特征向量
 */
public class TFIDFVectorizer {
    private int maxFeatures;                        // 最大特征数
    private SimpleTokenizer tokenizer;              // 分词器
    private Map<String, Double> idfScores;          // IDF分数映射
    private List<String> featureNames;              // 特征名称列表
    private boolean isTrained;                      // 是否已训练

    /**
     * 构造函数
     * @param maxFeatures 最大特征数
     */
    public TFIDFVectorizer(int maxFeatures) {
        this.maxFeatures = maxFeatures;
        this.tokenizer = new SimpleTokenizer();
        this.idfScores = new HashMap<>();
        this.featureNames = new ArrayList<>();
        this.isTrained = false;
    }

    /**
     * 训练TF-IDF模型
     * @param documents 文档列表
     */
    public void fit(List<String> documents) {
        System.out.println("开始训练TF-IDF模型...");
        
        // 构建词汇表
        tokenizer.buildVocab(documents);
        
        // 计算文档频率
        Map<String, Integer> docFrequencies = new HashMap<>();
        int totalDocs = documents.size();
        
        for (String doc : documents) {
            List<String> tokens = tokenizer.tokenize(doc);
            Set<String> uniqueTokens = new HashSet<>(tokens); // 去重
            
            for (String token : uniqueTokens) {
                docFrequencies.put(token, docFrequencies.getOrDefault(token, 0) + 1);
            }
        }
        
        // 计算IDF分数
        idfScores.clear();
        for (Map.Entry<String, Integer> entry : docFrequencies.entrySet()) {
            String token = entry.getKey();
            int df = entry.getValue();
            double idf = Math.log((double) totalDocs / df);
            idfScores.put(token, idf);
        }
        
        // 选择前maxFeatures个最重要的特征（按IDF分数排序）
        List<Map.Entry<String, Double>> sortedFeatures = new ArrayList<>(idfScores.entrySet());
        sortedFeatures.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        featureNames.clear();
        for (int i = 0; i < Math.min(maxFeatures, sortedFeatures.size()); i++) {
            featureNames.add(sortedFeatures.get(i).getKey());
        }
        
        isTrained = true;
        System.out.println("TF-IDF模型训练完成，特征维度: " + featureNames.size());
    }

    /**
     * 将单个文本转换为TF-IDF向量
     * @param text 输入文本
     * @return TF-IDF向量
     */
    public List<Double> transform(String text) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        if (featureNames.isEmpty()) {
            return Collections.nCopies(maxFeatures, 0.0);
        }
        
        // 分词并统计词频
        List<String> tokens = tokenizer.tokenize(text);
        Map<String, Integer> tokenCounts = new HashMap<>();
        for (String token : tokens) {
            tokenCounts.put(token, tokenCounts.getOrDefault(token, 0) + 1);
        }
        
        int totalTokens = tokens.size();
        
        // 计算TF-IDF向量
        List<Double> vector = new ArrayList<>();
        for (String feature : featureNames) {
            int count = tokenCounts.getOrDefault(feature, 0);
            double tf = totalTokens > 0 ? (double) count / totalTokens : 0.0;
            double idf = idfScores.getOrDefault(feature, 0.0);
            double tfidf = tf * idf;
            vector.add(tfidf);
        }
        
        return vector;
    }

    /**
     * 训练并转换文档列表
     * @param documents 文档列表
     * @return TF-IDF向量列表
     */
    public List<List<Double>> fitTransform(List<String> documents) {
        fit(documents);
        
        List<List<Double>> vectors = new ArrayList<>();
        for (String doc : documents) {
            vectors.add(transform(doc));
        }
        
        return vectors;
    }

    /**
     * 获取特征名称列表
     * @return 特征名称列表
     */
    public List<String> getFeatureNames() {
        return new ArrayList<>(featureNames);
    }

    /**
     * 获取IDF分数映射
     * @return IDF分数映射
     */
    public Map<String, Double> getIdfScores() {
        return new HashMap<>(idfScores);
    }

    /**
     * 获取分词器
     * @return 分词器对象
     */
    public SimpleTokenizer getTokenizer() {
        return tokenizer;
    }

    /**
     * 获取最大特征数
     * @return 最大特征数
     */
    public int getMaxFeatures() {
        return maxFeatures;
    }

    /**
     * 检查模型是否已训练
     * @return 是否已训练
     */
    public boolean isTrained() {
        return isTrained;
    }

    /**
     * 获取实际特征维度
     * @return 特征维度
     */
    public int getFeatureDimension() {
        return featureNames.size();
    }

    /**
     * 获取指定特征的IDF分数
     * @param feature 特征名称
     * @return IDF分数
     */
    public double getIdfScore(String feature) {
        return idfScores.getOrDefault(feature, 0.0);
    }

    @Override
    public String toString() {
        return "TFIDFVectorizer{" +
                "maxFeatures=" + maxFeatures +
                ", featureDimension=" + featureNames.size() +
                ", isTrained=" + isTrained +
                ", vocabSize=" + tokenizer.getVocabSize() +
                '}';
    }
}