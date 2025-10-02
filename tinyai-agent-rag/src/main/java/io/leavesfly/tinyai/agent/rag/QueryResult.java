package io.leavesfly.tinyai.agent.rag;

/**
 * 查询结果结构类
 * 表示RAG系统检索的结果，包含文档、相似度分数和排名
 */
public class QueryResult {
    private Document document;      // 匹配的文档
    private double similarity;      // 相似度分数
    private int rank;              // 排名

    /**
     * 构造函数
     */
    public QueryResult(Document document, double similarity, int rank) {
        this.document = document;
        this.similarity = similarity;
        this.rank = rank;
    }

    // Getter和Setter方法
    public Document getDocument() {
        return document;
    }

    public void setDocument(Document document) {
        this.document = document;
    }

    public double getSimilarity() {
        return similarity;
    }

    public void setSimilarity(double similarity) {
        this.similarity = similarity;
    }

    public int getRank() {
        return rank;
    }

    public void setRank(int rank) {
        this.rank = rank;
    }

    @Override
    public String toString() {
        return "QueryResult{" +
                "document=" + document.getId() +
                ", similarity=" + String.format("%.4f", similarity) +
                ", rank=" + rank +
                '}';
    }
}