package io.leavesfly.tinyai.agent.context;

/**
 * 检索结果类
 * 用于RAG系统返回检索到的文档及其相似度
 * 
 * @author 山泽
 */
public class RetrievalResult {
    
    private final Document document;    // 检索到的文档
    private final double similarity;    // 相似度分数
    
    // 构造函数
    public RetrievalResult(Document document, double similarity) {
        this.document = document;
        this.similarity = similarity;
    }
    
    // Getter 方法
    public Document getDocument() {
        return document;
    }
    
    public double getSimilarity() {
        return similarity;
    }
    
    @Override
    public String toString() {
        return String.format("RetrievalResult{docId='%s', similarity=%.3f}",
                           document.getId(), similarity);
    }
}