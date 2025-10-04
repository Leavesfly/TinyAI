package io.leavesfly.tinyai.agent.research;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 动态知识图谱
 * 管理研究过程中的知识节点和连接关系
 * 
 * @author 山泽
 */
public class KnowledgeGraph {
    
    /** 知识节点存储 */
    private final Map<String, KnowledgeNode> nodes;
    
    /** 连接权重映射 (nodeId1 -> nodeId2 -> weight) */
    private final Map<String, Map<String, Double>> connections;
    
    /** 领域节点映射 */
    private final Map<String, Set<String>> domains;
    
    /** 更新历史记录 */
    private final List<Map<String, Object>> updateHistory;
    
    /** 相似度计算缓存 */
    private final Map<String, Double> similarityCache;
    
    /**
     * 构造函数
     */
    public KnowledgeGraph() {
        this.nodes = new HashMap<>();
        this.connections = new HashMap<>();
        this.domains = new HashMap<>();
        this.updateHistory = new ArrayList<>();
        this.similarityCache = new HashMap<>();
    }
    
    /**
     * 添加知识节点
     */
    public String addNode(KnowledgeNode node) {
        if (node == null || node.getId() == null) {
            throw new IllegalArgumentException("节点和节点ID不能为空");
        }
        
        String nodeId = node.getId();
        nodes.put(nodeId, node);
        
        // 更新领域映射
        domains.computeIfAbsent(node.getDomain(), k -> new HashSet<>()).add(nodeId);
        
        // 自动发现连接
        discoverConnections(node);
        
        // 记录更新历史
        Map<String, Object> updateRecord = new HashMap<>();
        updateRecord.put("action", "add_node");
        updateRecord.put("nodeId", nodeId);
        updateRecord.put("timestamp", LocalDateTime.now());
        updateHistory.add(updateRecord);
        
        return nodeId;
    }
    
    /**
     * 添加节点连接
     */
    public void addConnection(String nodeId1, String nodeId2, double weight, String relationType) {
        if (!nodes.containsKey(nodeId1) || !nodes.containsKey(nodeId2)) {
            throw new IllegalArgumentException("连接的节点必须存在");
        }
        
        if (weight < 0.0 || weight > 1.0) {
            throw new IllegalArgumentException("连接权重必须在0.0-1.0范围内");
        }
        
        // 双向连接
        connections.computeIfAbsent(nodeId1, k -> new HashMap<>()).put(nodeId2, weight);
        connections.computeIfAbsent(nodeId2, k -> new HashMap<>()).put(nodeId1, weight);
        
        // 更新节点连接集合
        nodes.get(nodeId1).addConnection(nodeId2);
        nodes.get(nodeId2).addConnection(nodeId1);
        
        // 记录更新历史
        Map<String, Object> updateRecord = new HashMap<>();
        updateRecord.put("action", "add_connection");
        updateRecord.put("nodeId1", nodeId1);
        updateRecord.put("nodeId2", nodeId2);
        updateRecord.put("weight", weight);
        updateRecord.put("relationType", relationType);
        updateRecord.put("timestamp", LocalDateTime.now());
        updateHistory.add(updateRecord);
    }
    
    /**
     * 添加节点连接（简化版本）
     */
    public void addConnection(String nodeId1, String nodeId2, double weight) {
        addConnection(nodeId1, nodeId2, weight, "related");
    }
    
    /**
     * 自动发现节点连接
     */
    private void discoverConnections(KnowledgeNode newNode) {
        String newNodeId = newNode.getId();
        
        for (Map.Entry<String, KnowledgeNode> entry : nodes.entrySet()) {
            String existingId = entry.getKey();
            KnowledgeNode existingNode = entry.getValue();
            
            if (!existingId.equals(newNodeId)) {
                // 计算相似度
                double similarity = calculateSimilarity(newNode.getContent(), existingNode.getContent());
                
                // 如果相似度超过阈值，建立连接
                if (similarity > 0.3) {
                    addConnection(newNodeId, existingId, similarity, "auto_discovered");
                }
            }
        }
    }
    
    /**
     * 计算文本相似度
     */
    private double calculateSimilarity(String text1, String text2) {
        if (text1 == null || text2 == null || text1.trim().isEmpty() || text2.trim().isEmpty()) {
            return 0.0;
        }
        
        // 使用缓存避免重复计算
        String cacheKey = generateCacheKey(text1, text2);
        if (similarityCache.containsKey(cacheKey)) {
            return similarityCache.get(cacheKey);
        }
        
        // 简单的基于词汇重叠的相似度计算
        Set<String> words1 = extractWords(text1.toLowerCase());
        Set<String> words2 = extractWords(text2.toLowerCase());
        
        if (words1.isEmpty() || words2.isEmpty()) {
            return 0.0;
        }
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        double similarity = union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
        
        // 缓存结果
        similarityCache.put(cacheKey, similarity);
        
        return similarity;
    }
    
    /**
     * 提取文本中的单词
     */
    private Set<String> extractWords(String text) {
        return Arrays.stream(text.split("[\\s\\p{Punct}]+"))
                .filter(word -> word.length() > 2)  // 过滤太短的词
                .collect(Collectors.toSet());
    }
    
    /**
     * 生成缓存键
     */
    private String generateCacheKey(String text1, String text2) {
        // 确保缓存键的一致性（顺序无关）
        String[] texts = {text1, text2};
        Arrays.sort(texts);
        return String.join("|||", texts);
    }
    
    /**
     * 获取相关节点
     */
    public List<KnowledgeNode> getRelatedNodes(String nodeId, int maxDistance) {
        if (!nodes.containsKey(nodeId)) {
            return new ArrayList<>();
        }
        
        List<KnowledgeNode> relatedNodes = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Queue<NodeDistance> queue = new LinkedList<>();
        
        queue.offer(new NodeDistance(nodeId, 0));
        
        while (!queue.isEmpty()) {
            NodeDistance current = queue.poll();
            String currentId = current.nodeId;
            int distance = current.distance;
            
            if (visited.contains(currentId) || distance > maxDistance) {
                continue;
            }
            
            visited.add(currentId);
            
            // 不包括起始节点本身
            if (distance > 0) {
                relatedNodes.add(nodes.get(currentId));
            }
            
            // 添加邻居节点到队列
            KnowledgeNode currentNode = nodes.get(currentId);
            for (String neighborId : currentNode.getConnections()) {
                if (!visited.contains(neighborId)) {
                    queue.offer(new NodeDistance(neighborId, distance + 1));
                }
            }
        }
        
        // 按连接权重排序
        relatedNodes.sort((node1, node2) -> {
            double weight1 = getConnectionWeight(nodeId, node1.getId());
            double weight2 = getConnectionWeight(nodeId, node2.getId());
            return Double.compare(weight2, weight1);  // 降序排序
        });
        
        return relatedNodes;
    }
    
    /**
     * 获取连接权重
     */
    private double getConnectionWeight(String nodeId1, String nodeId2) {
        return connections.getOrDefault(nodeId1, new HashMap<>()).getOrDefault(nodeId2, 0.0);
    }
    
    /**
     * 搜索相关节点
     */
    public List<KnowledgeNode> searchNodes(String query, String domain) {
        List<NodeSimilarity> results = new ArrayList<>();
        
        for (KnowledgeNode node : nodes.values()) {
            // 领域过滤
            if (domain != null && !domain.equals(node.getDomain())) {
                continue;
            }
            
            // 计算相似度
            double similarity = calculateSimilarity(query, node.getContent());
            
            if (similarity > 0.05) {  // 降低最低相似度阈值
                results.add(new NodeSimilarity(node, similarity));
            }
        }
        
        // 按相似度降序排序
        results.sort((a, b) -> Double.compare(b.similarity, a.similarity));
        
        // 返回前10个结果
        return results.stream()
                .limit(10)
                .map(ns -> ns.node)
                .collect(Collectors.toList());
    }
    
    /**
     * 搜索相关节点（不指定领域）
     */
    public List<KnowledgeNode> searchNodes(String query) {
        return searchNodes(query, null);
    }
    
    /**
     * 获取领域概览
     */
    public Map<String, Object> getDomainOverview(String domain) {
        if (!domains.containsKey(domain)) {
            return new HashMap<>();
        }
        
        Set<String> nodeIds = domains.get(domain);
        List<KnowledgeNode> domainNodes = nodeIds.stream()
                .map(nodes::get)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        
        // 统计信息
        Map<String, Integer> nodeTypes = new HashMap<>();
        double avgConfidence = 0.0;
        
        for (KnowledgeNode node : domainNodes) {
            nodeTypes.merge(node.getNodeType(), 1, Integer::sum);
            avgConfidence += node.getConfidence();
        }
        
        if (!domainNodes.isEmpty()) {
            avgConfidence /= domainNodes.size();
        }
        
        // 找到中心节点（连接最多的节点）
        List<KnowledgeNode> centralNodes = domainNodes.stream()
                .sorted((a, b) -> Integer.compare(b.getConnectionCount(), a.getConnectionCount()))
                .limit(3)
                .collect(Collectors.toList());
        
        // 构建概览信息
        Map<String, Object> overview = new HashMap<>();
        overview.put("domain", domain);
        overview.put("totalNodes", domainNodes.size());
        overview.put("nodeTypes", nodeTypes);
        overview.put("averageConfidence", avgConfidence);
        overview.put("centralNodes", centralNodes.stream()
                .map(node -> Map.of("id", node.getId(), "content", 
                    node.getContent().length() > 100 ? 
                    node.getContent().substring(0, 100) + "..." : 
                    node.getContent()))
                .collect(Collectors.toList()));
        
        // 最后更新时间
        if (!domainNodes.isEmpty()) {
            LocalDateTime lastUpdated = domainNodes.stream()
                    .map(KnowledgeNode::getTimestamp)
                    .max(LocalDateTime::compareTo)
                    .orElse(null);
            overview.put("lastUpdated", lastUpdated);
        }
        
        return overview;
    }
    
    /**
     * 获取知识图谱统计信息
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("totalNodes", nodes.size());
        stats.put("totalConnections", connections.values().stream()
                .mapToInt(Map::size)
                .sum() / 2);  // 除以2因为是双向连接
        stats.put("domains", domains.keySet());
        stats.put("domainCount", domains.size());
        stats.put("updateHistorySize", updateHistory.size());
        stats.put("cacheSize", similarityCache.size());
        
        return stats;
    }
    
    /**
     * 节点与距离的内部类
     */
    private static class NodeDistance {
        final String nodeId;
        final int distance;
        
        NodeDistance(String nodeId, int distance) {
            this.nodeId = nodeId;
            this.distance = distance;
        }
    }
    
    /**
     * 节点与相似度的内部类
     */
    private static class NodeSimilarity {
        final KnowledgeNode node;
        final double similarity;
        
        NodeSimilarity(KnowledgeNode node, double similarity) {
            this.node = node;
            this.similarity = similarity;
        }
    }
    
    // Getter方法
    public Map<String, KnowledgeNode> getNodes() {
        return new HashMap<>(nodes);
    }
    
    public Map<String, Map<String, Double>> getConnections() {
        return connections;
    }
    
    public Map<String, Set<String>> getDomains() {
        return domains;
    }
    
    public List<Map<String, Object>> getUpdateHistory() {
        return updateHistory;
    }
    
    /**
     * 检查节点是否存在
     */
    public boolean hasNode(String nodeId) {
        return nodes.containsKey(nodeId);
    }
    
    /**
     * 获取指定节点
     */
    public KnowledgeNode getNode(String nodeId) {
        return nodes.get(nodeId);
    }
    
    /**
     * 生成节点ID（基于内容的哈希）
     */
    public static String generateNodeId(String content) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(content.getBytes());
            StringBuilder hexString = new StringBuilder();
            
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            
            return hexString.toString().substring(0, 16);
        } catch (NoSuchAlgorithmException e) {
            return "node_" + Math.abs(content.hashCode());
        }
    }
}