package io.leavesfly.tinyai.agent.pattern;

import java.util.*;

/**
 * Reflect模式Agent: 具有自我反思能力
 * 
 * 在执行任务后进行反思，评估执行效果并改进策略
 * 
 * @author 山泽
 */
public class ReflectAgent extends BaseAgent {
    
    /** 反思记录 */
    private final List<String> reflections;
    private final Random random;
    
    public ReflectAgent() {
        this("Reflect Agent", 10);
    }
    
    public ReflectAgent(String name) {
        this(name, 10);
    }
    
    public ReflectAgent(String name, int maxSteps) {
        super(name, maxSteps);
        this.reflections = new ArrayList<>();
        this.random = new Random();
        registerDefaultTools();
    }
    
    /**
     * 注册默认工具
     */
    private void registerDefaultTools() {
        // 分析工具
        addTool("analyze", this::analyzeTool, "分析工具");
        
        // 评估工具
        addTool("evaluate", this::evaluateTool, "评估工具");
    }
    
    /**
     * 分析工具实现
     */
    private Object analyzeTool(Map<String, Object> args) {
        String text = (String) args.get("text");
        if (text == null) {
            return "分析结果: 文本为空";
        }
        
        List<String> analysis = new ArrayList<>();
        analysis.add("文本长度: " + text.length() + " 字符");
        analysis.add("词汇数量: " + text.split("\\s+").length + " 个");
        analysis.add("包含问号: " + (text.contains("?") || text.contains("？") ? "是" : "否"));
        
        // 简单的情感分析
        String sentiment = "中性";
        if (containsAny(text, new String[]{"好", "棒", "优秀", "太好了", "很棒"})) {
            sentiment = "积极";
        } else if (containsAny(text, new String[]{"差", "糟糕", "不好", "失败"})) {
            sentiment = "消极";
        }
        analysis.add("情感倾向: " + sentiment);
        
        return "分析结果: " + String.join("; ", analysis);
    }
    
    /**
     * 评估工具实现
     */
    private Object evaluateTool(Map<String, Object> args) {
        String criteria = (String) args.get("criteria");
        if (criteria == null) {
            criteria = "通用质量";
        }
        
        Map<String, Integer> scores = new HashMap<>();
        scores.put("准确性", 7 + random.nextInt(4)); // 7-10
        scores.put("完整性", 6 + random.nextInt(4)); // 6-9
        scores.put("清晰度", 8 + random.nextInt(3)); // 8-10
        
        List<String> scoreStrings = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : scores.entrySet()) {
            scoreStrings.add(entry.getKey() + ":" + entry.getValue() + "/10");
        }
        
        return "评估结果(" + criteria + "): " + String.join("; ", scoreStrings);
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
     * 初始尝试
     */
    private String initialAttempt(String query) {
        addStep("initial_attempt", "开始处理查询: " + query);
        
        if (query.contains("分析")) {
            // 使用分析工具
            Map<String, Object> args = new HashMap<>();
            args.put("text", query);
            Action action = new Action("analyze", args);
            Object result = callTool(action);
            
            addStep("action", "使用分析工具: " + action.getArguments());
            addStep("observation", String.valueOf(result));
            return String.valueOf(result);
        } else {
            String response = "对于问题'" + query + "'，我的初始回答是：这是一个需要仔细思考的问题。";
            addStep("initial_response", response);
            return response;
        }
    }
    
    /**
     * 反思过程
     */
    private String reflect(String initialResponse, String query) {
        String reflectionPrompt = String.format(
                "原始问题: %s\n" +
                "初始回答: %s\n" +
                "\n" +
                "反思要点:\n" +
                "1. 我的回答是否完整地解决了问题？\n" +
                "2. 是否有遗漏的重要信息？\n" +
                "3. 回答的质量如何？\n" +
                "4. 如何改进？\n", query, initialResponse);
        
        List<String> currentReflections = new ArrayList<>();
        
        // 完整性反思
        if (initialResponse.length() < 50) {
            currentReflections.add("回答过于简短，可能不够完整");
        } else {
            currentReflections.add("回答长度适中");
        }
        
        // 相关性反思
        Set<String> queryKeywords = new HashSet<>(Arrays.asList(query.toLowerCase().split("\\s+")));
        Set<String> responseKeywords = new HashSet<>(Arrays.asList(initialResponse.toLowerCase().split("\\s+")));
        
        // 计算关键词重叠
        queryKeywords.retainAll(responseKeywords);
        int overlap = queryKeywords.size();
        
        if (overlap < 2) {
            currentReflections.add("回答与问题的相关性可能不足");
        } else {
            currentReflections.add("回答与问题相关性良好");
        }
        
        // 工具使用反思
        boolean usedTools = steps.stream().anyMatch(step -> "action".equals(step.getStepType()));
        if (!usedTools) {
            currentReflections.add("可能需要使用工具来提供更准确的信息");
        } else {
            currentReflections.add("适当使用了工具");
        }
        
        String reflection = "反思结果: " + String.join("; ", currentReflections);
        addStep("reflection", reflection);
        reflections.add(reflection);
        
        return reflection;
    }
    
    /**
     * 基于反思改进回答
     */
    private String improve(String initialResponse, String reflection, String query) {
        List<String> improvements = new ArrayList<>();
        
        if (reflection.contains("简短")) {
            improvements.add("提供更详细的解释");
        }
        
        if (reflection.contains("相关性不足")) {
            improvements.add("更直接地回答问题");
        }
        
        if (reflection.contains("需要使用工具")) {
            // 使用评估工具
            Map<String, Object> args = new HashMap<>();
            args.put("criteria", "回答质量");
            Action action = new Action("evaluate", args);
            Object result = callTool(action);
            
            addStep("action", "使用评估工具: " + action.getArguments());
            addStep("observation", String.valueOf(result));
            improvements.add("工具评估: " + result);
        }
        
        if (improvements.isEmpty()) {
            improvements.add("回答已经比较完善");
        }
        
        String improvedResponse = initialResponse + "\n\n改进补充: " + String.join("; ", improvements);
        addStep("improvement", improvedResponse);
        
        return improvedResponse;
    }
    
    @Override
    public String process(String query) {
        clearSteps();
        
        setState(AgentState.THINKING);
        
        // 第一步：初始尝试
        String initialResponse = initialAttempt(query);
        
        setState(AgentState.REFLECTING);
        
        // 第二步：反思
        String reflection = reflect(initialResponse, query);
        
        setState(AgentState.THINKING);
        
        // 第三步：改进
        String improvedResponse = improve(initialResponse, reflection, query);
        
        // 第四步：最终反思（可选）
        String finalReflection = "最终反思：通过反思和改进，我提供了更好的回答";
        addStep("final_reflection", finalReflection);
        
        setState(AgentState.DONE);
        return improvedResponse;
    }
    
    /**
     * 获取反思记录
     */
    public List<String> getReflections() {
        return new ArrayList<>(reflections);
    }
    
    /**
     * 清空反思记录
     */
    public void clearReflections() {
        reflections.clear();
    }
    
    @Override
    public void reset() {
        super.reset();
        clearReflections();
    }
}