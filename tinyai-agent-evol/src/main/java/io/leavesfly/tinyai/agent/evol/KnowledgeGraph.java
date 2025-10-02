package io.leavesfly.tinyai.agent.evol;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 知识图谱管理类
 * 用于构建和管理Agent的知识网络
 * 
 * @author 山泽
 */
public class KnowledgeGraph {
    
    /**
     * 概念节点类
     */
    public static class ConceptNode {
        private Map<String, Object> properties;
        private long createdAt;
        private int accessCount;
        private double[] embedding;
        
        public ConceptNode(Map<String, Object> properties) {
            this.properties = new HashMap<>(properties);
            this.createdAt = System.currentTimeMillis();
            this.accessCount = 0;
            this.embedding = generateRandomEmbedding(128);
        }
        
        private double[] generateRandomEmbedding(int dimension) {
            double[] embedding = new double[dimension];
            Random random = new Random();
            for (int i = 0; i < dimension; i++) {
                embedding[i] = random.nextGaussian();
            }
            return normalizeVector(embedding);
        }
        
        private double[] normalizeVector(double[] vector) {
            double norm = 0.0;
            for (double v : vector) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);
            
            if (norm > 0) {
                for (int i = 0; i < vector.length; i++) {
                    vector[i] /= norm;
                }
            }
            return vector;
        }
        
        public void incrementAccess() {
            accessCount++;
        }
        
        // Getters and Setters
        public Map<String, Object> getProperties() { return properties; }
        public void setProperties(Map<String, Object> properties) { this.properties = properties; }
        public long getCreatedAt() { return createdAt; }
        public int getAccessCount() { return accessCount; }
        public double[] getEmbedding() { return embedding; }
        public void setEmbedding(double[] embedding) { this.embedding = embedding; }
    }
    
    /**
     * 关系边类
     */
    public static class RelationEdge {
        private double weight;
        private long createdAt;
        private int usageCount;
        
        public RelationEdge(double weight) {
            this.weight = weight;
            this.createdAt = System.currentTimeMillis();
            this.usageCount = 1;
        }
        
        public void updateWeight(double newWeight, double learningRate) {
            this.weight = (1 - learningRate) * this.weight + learningRate * newWeight;
            this.usageCount++;
        }
        
        // Getters and Setters
        public double getWeight() { return weight; }
        public void setWeight(double weight) { this.weight = weight; }
        public long getCreatedAt() { return createdAt; }
        public int getUsageCount() { return usageCount; }
        public void incrementUsage() { usageCount++; }
    }
    
    /** 概念节点存储 */
    private final Map<String, ConceptNode> nodes;
    
    /** 关系边存储: fromConcept -> toConcept -> relation -> edge */
    private final Map<String, Map<String, Map<String, RelationEdge>>> edges;
    
    /** 学习率 */
    private double learningRate;
    
    public KnowledgeGraph() {
        this.nodes = new ConcurrentHashMap<>();
        this.edges = new ConcurrentHashMap<>();
        this.learningRate = 0.1;
    }
    
    /**
     * 添加概念节点
     * 
     * @param concept 概念名称
     * @param properties 概念属性
     */
    public void addConcept(String concept, Map<String, Object> properties) {
        if (concept == null || concept.trim().isEmpty()) {
            return;
        }
        
        if (!nodes.containsKey(concept)) {
            nodes.put(concept, new ConceptNode(properties));
        } else {
            // 更新现有概念的属性
            ConceptNode existingNode = nodes.get(concept);
            existingNode.getProperties().putAll(properties);
        }
    }
    
    /**
     * 添加关系边
     * 
     * @param fromConcept 起始概念
     * @param toConcept 目标概念
     * @param relation 关系类型
     * @param weight 关系权重
     */
    public void addRelation(String fromConcept, String toConcept, String relation, double weight) {
        if (fromConcept == null || toConcept == null || relation == null) {
            return;
        }
        
        // 确保概念节点存在
        if (!nodes.containsKey(fromConcept)) {
            addConcept(fromConcept, new HashMap<>());
        }
        if (!nodes.containsKey(toConcept)) {
            addConcept(toConcept, new HashMap<>());
        }
        
        // 添加或更新关系边
        edges.computeIfAbsent(fromConcept, k -> new ConcurrentHashMap<>())
              .computeIfAbsent(toConcept, k -> new ConcurrentHashMap<>())
              .compute(relation, (k, v) -> {
                  if (v == null) {
                      return new RelationEdge(weight);
                  } else {
                      v.updateWeight(weight, learningRate);
                      return v;
                  }
              });
    }
    
    /**
     * 查找相关概念
     * 
     * @param concept 起始概念
     * @param maxDistance 最大搜索距离
     * @return 相关概念列表
     */
    public List<String> findRelatedConcepts(String concept, int maxDistance) {
        if (!nodes.containsKey(concept)) {
            return new ArrayList<>();
        }
        
        Set<String> visited = new HashSet<>();
        Queue<ConceptDistance> queue = new LinkedList<>();
        List<String> related = new ArrayList<>();
        
        queue.offer(new ConceptDistance(concept, 0));
        
        while (!queue.isEmpty()) {
            ConceptDistance current = queue.poll();
            String currentConcept = current.concept;
            int currentDistance = current.distance;
            
            if (visited.contains(currentConcept) || currentDistance > maxDistance) {
                continue;
            }
            
            visited.add(currentConcept);
            nodes.get(currentConcept).incrementAccess(); // 记录访问
            
            if (currentDistance > 0) {
                related.add(currentConcept);
            }
            
            // 添加邻接节点
            if (edges.containsKey(currentConcept)) {
                for (String neighbor : edges.get(currentConcept).keySet()) {
                    if (!visited.contains(neighbor)) {
                        queue.offer(new ConceptDistance(neighbor, currentDistance + 1));
                    }
                }
            }
        }
        
        return related;
    }
    
    /**
     * 计算概念相似度
     * 
     * @param concept1 概念1
     * @param concept2 概念2
     * @return 相似度值 (0-1)
     */
    public double getConceptSimilarity(String concept1, String concept2) {
        if (!nodes.containsKey(concept1) || !nodes.containsKey(concept2)) {
            return 0.0;
        }
        
        if (concept1.equals(concept2)) {
            return 1.0;
        }
        
        double[] embedding1 = nodes.get(concept1).getEmbedding();
        double[] embedding2 = nodes.get(concept2).getEmbedding();
        
        return calculateCosineSimilarity(embedding1, embedding2);
    }
    
    /**
     * 计算余弦相似度
     * 
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 余弦相似度
     */
    private double calculateCosineSimilarity(double[] vec1, double[] vec2) {
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
        
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        
        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (norm1 * norm2);
    }
    
    /**
     * 获取概念的直接关系
     * 
     * @param concept 概念名称
     * @return 关系映射 (目标概念 -> 关系类型 -> 边)
     */
    public Map<String, Map<String, RelationEdge>> getDirectRelations(String concept) {
        return edges.getOrDefault(concept, new HashMap<>());
    }
    
    /**
     * 获取最强关系的概念
     * 
     * @param concept 起始概念
     * @param relationTypes 关系类型过滤器（null表示所有类型）
     * @return 最强关系的概念，如果没有则返回null
     */
    public String getStrongestRelatedConcept(String concept, Set<String> relationTypes) {
        if (!edges.containsKey(concept)) {
            return null;
        }
        
        String strongestConcept = null;
        double maxWeight = 0.0;
        
        for (Map.Entry<String, Map<String, RelationEdge>> targetEntry : edges.get(concept).entrySet()) {
            String targetConcept = targetEntry.getKey();
            Map<String, RelationEdge> relations = targetEntry.getValue();
            
            for (Map.Entry<String, RelationEdge> relationEntry : relations.entrySet()) {
                String relationType = relationEntry.getKey();
                RelationEdge edge = relationEntry.getValue();
                
                if (relationTypes == null || relationTypes.contains(relationType)) {
                    if (edge.getWeight() > maxWeight) {
                        maxWeight = edge.getWeight();
                        strongestConcept = targetConcept;
                    }
                }
            }
        }
        
        return strongestConcept;
    }
    
    /**
     * 清理低权重或过期的关系
     * 
     * @param weightThreshold 权重阈值
     * @param ageThreshold 年龄阈值（毫秒）
     */
    public void cleanupWeakRelations(double weightThreshold, long ageThreshold) {
        long currentTime = System.currentTimeMillis();
        
        for (String fromConcept : new HashSet<>(edges.keySet())) {
            Map<String, Map<String, RelationEdge>> conceptEdges = edges.get(fromConcept);
            
            for (String toConcept : new HashSet<>(conceptEdges.keySet())) {
                Map<String, RelationEdge> relations = conceptEdges.get(toConcept);
                
                relations.entrySet().removeIf(entry -> {
                    RelationEdge edge = entry.getValue();
                    return edge.getWeight() < weightThreshold || 
                           (currentTime - edge.getCreatedAt()) > ageThreshold;
                });
                
                if (relations.isEmpty()) {
                    conceptEdges.remove(toConcept);
                }
            }
            
            if (conceptEdges.isEmpty()) {
                edges.remove(fromConcept);
            }
        }
    }
    
    /**
     * 获取知识图谱统计信息
     * 
     * @return 统计信息
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("conceptCount", nodes.size());
        
        int totalEdges = 0;
        for (Map<String, Map<String, RelationEdge>> conceptEdges : edges.values()) {
            for (Map<String, RelationEdge> relations : conceptEdges.values()) {
                totalEdges += relations.size();
            }
        }
        stats.put("relationCount", totalEdges);
        
        // 最活跃的概念
        String mostActiveConcept = null;
        int maxAccess = 0;
        for (Map.Entry<String, ConceptNode> entry : nodes.entrySet()) {
            int accessCount = entry.getValue().getAccessCount();
            if (accessCount > maxAccess) {
                maxAccess = accessCount;
                mostActiveConcept = entry.getKey();
            }
        }
        stats.put("mostActiveConcept", mostActiveConcept);
        stats.put("maxAccessCount", maxAccess);
        
        return stats;
    }
    
    /**
     * 概念距离辅助类
     */
    private static class ConceptDistance {
        final String concept;
        final int distance;
        
        ConceptDistance(String concept, int distance) {
            this.concept = concept;
            this.distance = distance;
        }
    }
    
    // Getters and Setters
    
    public Map<String, ConceptNode> getNodes() {
        return nodes;
    }
    
    public Map<String, Map<String, Map<String, RelationEdge>>> getEdges() {
        return edges;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}