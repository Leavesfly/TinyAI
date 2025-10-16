package io.leavesfly.tinyai.agent.research.v2.model;

import java.util.*;

/**
 * 研究计划
 * 包含研究问题的DAG结构和执行策略
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ResearchPlan {
    
    /**
     * 计划ID
     */
    private String planId;
    
    /**
     * 研究问题列表
     */
    private List<ResearchQuestion> questions;
    
    /**
     * 问题依赖关系图（邻接表表示）
     */
    private Map<String, List<String>> dependencyGraph;
    
    /**
     * 预估研究深度（问题层级数）
     */
    private int estimatedDepth;
    
    /**
     * 规划策略
     */
    private PlanningStrategy strategy;
    
    /**
     * 预估总耗时（秒）
     */
    private long estimatedDurationSeconds;
    
    public ResearchPlan() {
        this.planId = UUID.randomUUID().toString();
        this.questions = new ArrayList<>();
        this.dependencyGraph = new HashMap<>();
        this.strategy = PlanningStrategy.HYBRID;
    }
    
    /**
     * 添加研究问题
     */
    public void addQuestion(ResearchQuestion question) {
        this.questions.add(question);
        this.dependencyGraph.putIfAbsent(question.getQuestionId(), new ArrayList<>());
    }
    
    /**
     * 添加依赖关系
     * @param fromId 依赖问题ID
     * @param toId 被依赖问题ID
     */
    public void addDependency(String fromId, String toId) {
        this.dependencyGraph.computeIfAbsent(fromId, k -> new ArrayList<>()).add(toId);
    }
    
    /**
     * 获取无依赖的问题（可以立即执行）
     */
    public List<ResearchQuestion> getRootQuestions() {
        List<ResearchQuestion> roots = new ArrayList<>();
        Set<String> dependentIds = new HashSet<>();
        
        // 收集所有被依赖的问题ID
        for (List<String> deps : dependencyGraph.values()) {
            dependentIds.addAll(deps);
        }
        
        // 找出没有被任何问题依赖的问题
        for (ResearchQuestion q : questions) {
            if (!dependentIds.contains(q.getQuestionId())) {
                roots.add(q);
            }
        }
        
        return roots;
    }
    
    /**
     * 拓扑排序获取执行顺序
     */
    public List<List<ResearchQuestion>> getExecutionLevels() {
        List<List<ResearchQuestion>> levels = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Map<String, ResearchQuestion> questionMap = new HashMap<>();
        
        for (ResearchQuestion q : questions) {
            questionMap.put(q.getQuestionId(), q);
        }
        
        // 构建入度表
        Map<String, Integer> inDegree = new HashMap<>();
        for (ResearchQuestion q : questions) {
            inDegree.put(q.getQuestionId(), 0);
        }
        
        for (Map.Entry<String, List<String>> entry : dependencyGraph.entrySet()) {
            for (String toId : entry.getValue()) {
                inDegree.put(toId, inDegree.get(toId) + 1);
            }
        }
        
        // BFS分层
        Queue<String> queue = new LinkedList<>();
        for (Map.Entry<String, Integer> entry : inDegree.entrySet()) {
            if (entry.getValue() == 0) {
                queue.offer(entry.getKey());
            }
        }
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<ResearchQuestion> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                String qId = queue.poll();
                currentLevel.add(questionMap.get(qId));
                visited.add(qId);
                
                // 更新依赖此问题的其他问题的入度
                if (dependencyGraph.containsKey(qId)) {
                    for (String nextId : dependencyGraph.get(qId)) {
                        int degree = inDegree.get(nextId) - 1;
                        inDegree.put(nextId, degree);
                        if (degree == 0) {
                            queue.offer(nextId);
                        }
                    }
                }
            }
            
            if (!currentLevel.isEmpty()) {
                levels.add(currentLevel);
            }
        }
        
        return levels;
    }
    
    // Getters and Setters
    
    public String getPlanId() {
        return planId;
    }
    
    public void setPlanId(String planId) {
        this.planId = planId;
    }
    
    public List<ResearchQuestion> getQuestions() {
        return questions;
    }
    
    public void setQuestions(List<ResearchQuestion> questions) {
        this.questions = questions;
    }
    
    public Map<String, List<String>> getDependencyGraph() {
        return dependencyGraph;
    }
    
    public void setDependencyGraph(Map<String, List<String>> dependencyGraph) {
        this.dependencyGraph = dependencyGraph;
    }
    
    public int getEstimatedDepth() {
        return estimatedDepth;
    }
    
    public void setEstimatedDepth(int estimatedDepth) {
        this.estimatedDepth = estimatedDepth;
    }
    
    public PlanningStrategy getStrategy() {
        return strategy;
    }
    
    public void setStrategy(PlanningStrategy strategy) {
        this.strategy = strategy;
    }
    
    public long getEstimatedDurationSeconds() {
        return estimatedDurationSeconds;
    }
    
    public void setEstimatedDurationSeconds(long estimatedDurationSeconds) {
        this.estimatedDurationSeconds = estimatedDurationSeconds;
    }
}
