package io.leavesfly.tinyai.agent.pattern;

import java.util.*;

/**
 * 协作式Agent：多个Agent协同工作
 * 
 * 通过多个专家Agent的协同工作来解决复杂问题
 * 
 * @author 山泽
 */
public class CollaborativeAgent extends BaseAgent {
    
    /** 专家Agent注册表 */
    private final Map<String, BaseAgent> specialists;
    
    /** 协调历史记录 */
    private final List<String> coordinationHistory;
    
    public CollaborativeAgent() {
        this("Collaborative Agent");
    }
    
    public CollaborativeAgent(String name) {
        super(name, 20); // 协作Agent可能需要更多步骤
        this.specialists = new HashMap<>();
        this.coordinationHistory = new ArrayList<>();
    }
    
    /**
     * 添加专家Agent
     * @param name 专家名称
     * @param agent 专家Agent实例
     */
    public void addSpecialist(String name, BaseAgent agent) {
        specialists.put(name, agent);
    }
    
    /**
     * 移除专家Agent
     * @param name 专家名称
     */
    public void removeSpecialist(String name) {
        specialists.remove(name);
    }
    
    /**
     * 获取所有专家名称
     */
    public Set<String> getSpecialistNames() {
        return new HashSet<>(specialists.keySet());
    }
    
    /**
     * 路由查询到合适的专家
     * @param query 查询内容
     * @return 专家类型名称
     */
    private String routeQuery(String query) {
        String queryLower = query.toLowerCase();
        
        if (containsAny(queryLower, new String[]{"计算", "数学", "算"})) {
            return "calculator_expert";
        } else if (containsAny(queryLower, new String[]{"分析", "研究", "深入"})) {
            return "analysis_expert";
        } else if (containsAny(queryLower, new String[]{"计划", "规划", "步骤"})) {
            return "planning_expert";
        } else if (containsAny(queryLower, new String[]{"反思", "评估", "改进"})) {
            return "reflect_expert";
        } else {
            return "general_expert";
        }
    }
    
    /**
     * 检查字符串是否包含任何指定的子字符串
     */
    private boolean containsAny(String text, String[] substrings) {
        if (text == null) return false;
        for (String substring : substrings) {
            if (text.contains(substring)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 选择验证专家
     * @param primaryExpertType 主要专家类型
     * @return 验证专家名称，如果没有其他专家返回null
     */
    private String selectValidator(String primaryExpertType) {
        List<String> availableValidators = new ArrayList<>();
        
        for (String expertName : specialists.keySet()) {
            if (!expertName.equals(primaryExpertType)) {
                availableValidators.add(expertName);
            }
        }
        
        if (availableValidators.isEmpty()) {
            return null;
        }
        
        // 简单选择第一个可用的验证专家
        return availableValidators.get(0);
    }
    
    /**
     * 执行专家咨询
     * @param expertName 专家名称
     * @param query 查询内容
     * @return 咨询结果
     */
    private String consultExpert(String expertName, String query) {
        BaseAgent expert = specialists.get(expertName);
        if (expert == null) {
            return "专家 " + expertName + " 不存在";
        }
        
        try {
            String result = expert.process(query);
            coordinationHistory.add("咨询了专家 " + expertName + ": " + query);
            return result;
        } catch (Exception e) {
            return "专家咨询失败: " + e.getMessage();
        }
    }
    
    /**
     * 整合多个专家的结果
     * @param results 专家结果列表
     * @return 整合后的结果
     */
    private String integrateResults(List<String> results) {
        if (results.isEmpty()) {
            return "没有获得任何专家意见";
        }
        
        if (results.size() == 1) {
            return results.get(0);
        }
        
        StringBuilder integrated = new StringBuilder();
        integrated.append("基于多位专家的协作分析：\n\n");
        
        for (int i = 0; i < results.size(); i++) {
            integrated.append("专家观点 ").append(i + 1).append(":\n");
            integrated.append(results.get(i)).append("\n\n");
        }
        
        integrated.append("综合结论: 通过多专家协作，提供了全面而准确的解决方案。");
        
        return integrated.toString();
    }
    
    @Override
    public String process(String query) {
        clearSteps();
        
        setState(AgentState.THINKING);
        
        // 第一步：分析查询并路由
        String expertType = routeQuery(query);
        addStep("routing", "将查询路由到: " + expertType);
        
        List<String> expertResults = new ArrayList<>();
        
        setState(AgentState.ACTING);
        
        // 第二步：调用主要专家
        if (specialists.containsKey(expertType)) {
            BaseAgent expert = specialists.get(expertType);
            String expertResult = consultExpert(expertType, query);
            addStep("expert_consultation", expertType + "处理结果: " + expertResult);
            expertResults.add(expertResult);
            
            setState(AgentState.REFLECTING);
            
            // 第三步：验证和整合（可选择另一个专家验证）
            if (specialists.size() > 1) {
                String validatorName = selectValidator(expertType);
                if (validatorName != null) {
                    String validationQuery = "请验证这个回答的质量: " + expertResult;
                    String validationResult = consultExpert(validatorName, validationQuery);
                    addStep("validation", validatorName + "验证结果: " + validationResult);
                    expertResults.add("验证意见: " + validationResult);
                }
            }
            
            setState(AgentState.THINKING);
            
            // 第四步：整合结果
            String finalResult = integrateResults(expertResults);
            addStep("integration", "整合多专家意见完成");
            
            setState(AgentState.DONE);
            return finalResult;
            
        } else {
            // 如果没有找到对应专家，尝试使用任何可用的专家
            if (!specialists.isEmpty()) {
                String availableExpert = specialists.keySet().iterator().next();
                String result = consultExpert(availableExpert, query);
                addStep("fallback_consultation", "使用备选专家 " + availableExpert + ": " + result);
                setState(AgentState.DONE);
                return result;
            } else {
                setState(AgentState.DONE);
                return "抱歉，我没有找到处理'" + query + "'的专家";
            }
        }
    }
    
    /**
     * 获取协调历史
     */
    public List<String> getCoordinationHistory() {
        return new ArrayList<>(coordinationHistory);
    }
    
    /**
     * 清空协调历史
     */
    public void clearCoordinationHistory() {
        coordinationHistory.clear();
    }
    
    /**
     * 获取专家状态摘要
     */
    public String getSpecialistSummary() {
        if (specialists.isEmpty()) {
            return "当前没有注册的专家";
        }
        
        StringBuilder summary = new StringBuilder();
        summary.append("注册的专家 (").append(specialists.size()).append("个):\n");
        
        for (Map.Entry<String, BaseAgent> entry : specialists.entrySet()) {
            BaseAgent agent = entry.getValue();
            summary.append("- ").append(entry.getKey())
                   .append(" (").append(agent.getClass().getSimpleName()).append(")")
                   .append(" - 状态: ").append(agent.getState())
                   .append("\n");
        }
        
        return summary.toString().trim();
    }
    
    /**
     * 重置所有专家状态
     */
    public void resetAllSpecialists() {
        for (BaseAgent agent : specialists.values()) {
            agent.reset();
        }
    }
    
    @Override
    public void reset() {
        super.reset();
        clearCoordinationHistory();
        resetAllSpecialists();
    }
}