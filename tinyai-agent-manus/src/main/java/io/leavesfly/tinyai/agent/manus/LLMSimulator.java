package io.leavesfly.tinyai.agent.manus;

import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * LLM模拟器 - 专为Manus智能体系统设计
 * 模拟真实的LLM API调用，为手稿智能体的推理-行动循环提供智能回复
 * 
 * @author 山泽
 */
public class LLMSimulator {
    
    private final String modelName;
    private final double temperature;
    private final int maxTokens;
    private final Random random;
    
    // 针对Manus智能体系统的专用回复模板
    private final Map<String, List<String>> responseTemplates;
    private final Map<String, List<String>> reasoningTemplates;
    private final Map<String, List<String>> actionTemplates;
    
    public LLMSimulator() {
        this("manus-gpt-turbo", 0.8, 2048);
    }
    
    public LLMSimulator(String modelName, double temperature, int maxTokens) {
        this.modelName = modelName;
        this.temperature = temperature;
        this.maxTokens = maxTokens;
        this.random = new Random();
        this.responseTemplates = initializeResponseTemplates();
        this.reasoningTemplates = initializeReasoningTemplates();
        this.actionTemplates = initializeActionTemplates();
    }
    
    /**
     * 异步生成智能体回复
     */
    public CompletableFuture<String> generateResponseAsync(String prompt, String agentType, String context) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 模拟LLM处理延迟
                Thread.sleep(300 + prompt.length() / 10);
                
                return generateSmartResponse(prompt, agentType, context);
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return "抱歉，我在思考过程中遇到了问题。";
            }
        });
    }
    
    /**
     * 同步生成智能体回复
     */
    public String generateResponse(String prompt, String agentType, String context) {
        try {
            return generateResponseAsync(prompt, agentType, context).get();
        } catch (Exception e) {
            return "遇到了一些技术问题，让我重新组织思路：" + e.getMessage();
        }
    }
    
    /**
     * 生成ReAct循环中的思考内容
     */
    public String generateThought(String query, String context, int iteration) {
        List<String> templates = reasoningTemplates.getOrDefault("thinking", Arrays.asList(
            "我需要仔细分析这个问题：{}。基于当前信息{}，我认为{}。",
            "让我思考一下：{}。从{}角度来看，{}。",
            "分析问题：{}。考虑到{}，我的判断是{}。"
        ));
        
        String template = templates.get(random.nextInt(templates.size()));
        List<String> keywords = extractContextualKeywords(query, context, "thinking");
        
        return fillTemplate(template, keywords);
    }
    
    /**
     * 生成ReAct循环中的行动决策
     */
    public String generateAction(String thought, String availableTools) {
        List<String> templates = actionTemplates.getOrDefault("action_planning", Arrays.asList(
            "基于思考结果{}，我需要使用{}工具来{}。",
            "为了解决这个问题，我将{}，具体是使用{}来{}。",
            "根据分析，最佳行动是{}。我会调用{}工具{}。"
        ));
        
        String template = templates.get(random.nextInt(templates.size()));
        List<String> keywords = extractActionKeywords(thought, availableTools);
        
        return fillTemplate(template, keywords);
    }
    
    /**
     * 生成观察结果的分析
     */
    public String generateObservation(String actionResult, String originalQuery) {
        List<String> templates = reasoningTemplates.getOrDefault("observation", Arrays.asList(
            "执行结果是{}。这个结果{}，说明{}。",
            "观察到{}。从这个结果可以看出{}，因此{}。",
            "行动的结果显示{}。这表明{}，接下来{}。"
        ));
        
        String template = templates.get(random.nextInt(templates.size()));
        List<String> keywords = extractObservationKeywords(actionResult, originalQuery);
        
        return fillTemplate(template, keywords);
    }
    
    /**
     * 生成Flow编排的决策
     */
    public String generateFlowDecision(String query, Map<String, String> availableFlows) {
        List<String> flowNames = new ArrayList<>(availableFlows.keySet());
        if (flowNames.isEmpty()) {
            return "当前没有可用的Flow，将使用默认处理方式。";
        }
        
        // 基于查询内容智能选择Flow
        String selectedFlow = selectBestFlow(query, availableFlows);
        
        return String.format("分析查询'%s'，我认为最适合的处理流程是'%s'，因为%s。", 
            query.length() > 30 ? query.substring(0, 30) + "..." : query,
            selectedFlow,
            generateFlowSelectionReason(query, selectedFlow));
    }
    
    /**
     * 生成计划分解结果
     */
    public String generatePlanDecomposition(String goal, List<String> tasks) {
        String template = "为了实现目标'%s'，我将其分解为%d个子任务：\n%s\n这样的分解能够确保目标的逐步实现。";
        
        StringBuilder taskList = new StringBuilder();
        for (int i = 0; i < tasks.size(); i++) {
            taskList.append(String.format("%d. %s\n", i + 1, tasks.get(i)));
        }
        
        return String.format(template, goal, tasks.size(), taskList.toString());
    }
    
    /**
     * 生成智能回复
     */
    private String generateSmartResponse(String prompt, String agentType, String context) {
        List<String> templates = responseTemplates.getOrDefault(agentType, 
            responseTemplates.get("base_agent"));
        
        String template = templates.get(random.nextInt(templates.size()));
        List<String> keywords = extractContextualKeywords(prompt, context, agentType);
        
        return fillTemplate(template, keywords);
    }
    
    /**
     * 提取上下文关键词
     */
    private List<String> extractContextualKeywords(String text, String context, String agentType) {
        List<String> keywords = new ArrayList<>();
        
        // 基于文本内容提取关键词
        if (text.contains("计算") || text.contains("数学") || text.contains("+") || text.contains("-")) {
            keywords.addAll(Arrays.asList("数学计算", "精确计算", "计算器工具"));
        } else if (text.contains("时间") || text.contains("现在") || text.contains("几点")) {
            keywords.addAll(Arrays.asList("当前时间", "时间查询", "时间工具"));
        } else if (text.contains("分析") || text.contains("解析") || text.contains("研究")) {
            keywords.addAll(Arrays.asList("深入分析", "详细研究", "分析工具"));
        } else if (text.contains("计划") || text.contains("任务") || text.contains("步骤")) {
            keywords.addAll(Arrays.asList("任务规划", "逐步执行", "计划制定"));
        } else {
            keywords.addAll(Arrays.asList("问题理解", "智能处理", "合适的方法"));
        }
        
        // 基于智能体类型调整关键词
        switch (agentType) {
            case "thinking":
                keywords.add("仔细考虑");
                keywords.add("深入思考");
                break;
            case "action_planning":
                keywords.add("立即执行");
                keywords.add("使用工具");
                break;
            case "observation":
                keywords.add("结果分析");
                keywords.add("效果评估");
                break;
        }
        
        return keywords;
    }
    
    /**
     * 提取行动关键词
     */
    private List<String> extractActionKeywords(String thought, String availableTools) {
        List<String> keywords = new ArrayList<>();
        
        if (thought.contains("计算") || thought.contains("数学")) {
            keywords.addAll(Arrays.asList("执行计算", "calculator", "获得结果"));
        } else if (thought.contains("时间")) {
            keywords.addAll(Arrays.asList("查询时间", "get_time", "获取信息"));
        } else if (thought.contains("分析")) {
            keywords.addAll(Arrays.asList("进行分析", "text_analyzer", "处理数据"));
        } else {
            keywords.addAll(Arrays.asList("采取行动", "合适工具", "解决问题"));
        }
        
        return keywords;
    }
    
    /**
     * 提取观察关键词
     */
    private List<String> extractObservationKeywords(String actionResult, String originalQuery) {
        List<String> keywords = new ArrayList<>();
        
        if (actionResult.contains("成功") || actionResult.contains("结果")) {
            keywords.addAll(Arrays.asList("成功获得", "满足了需求", "可以回答用户"));
        } else if (actionResult.contains("错误") || actionResult.contains("失败")) {
            keywords.addAll(Arrays.asList("遇到了问题", "需要调整方法", "重新尝试"));
        } else {
            keywords.addAll(Arrays.asList("获得了信息", "有助于解决", "继续处理"));
        }
        
        return keywords;
    }
    
    /**
     * 选择最佳Flow
     */
    private String selectBestFlow(String query, Map<String, String> availableFlows) {
        for (Map.Entry<String, String> entry : availableFlows.entrySet()) {
            String flowId = entry.getKey();
            String flowDescription = entry.getValue();
            
            if ((query.contains("计算") || query.contains("数学")) && flowId.contains("calculation")) {
                return flowId;
            } else if ((query.contains("时间") || query.contains("几点")) && flowId.contains("time")) {
                return flowId;
            } else if ((query.contains("分析") || query.contains("解析")) && flowId.contains("analysis")) {
                return flowId;
            }
        }
        
        // 默认返回第一个可用Flow
        return availableFlows.keySet().iterator().next();
    }
    
    /**
     * 生成Flow选择理由
     */
    private String generateFlowSelectionReason(String query, String selectedFlow) {
        if (selectedFlow.contains("calculation")) {
            return "这是一个数学计算问题，需要专门的计算处理流程";
        } else if (selectedFlow.contains("time")) {
            return "这是一个时间查询请求，时间流程最为合适";
        } else if (selectedFlow.contains("analysis")) {
            return "这需要进行深入分析，分析流程能提供最好的结果";
        } else {
            return "这个流程最能满足当前需求";
        }
    }
    
    /**
     * 填充模板
     */
    private String fillTemplate(String template, List<String> keywords) {
        try {
            // 统计占位符数量
            int placeholderCount = 0;
            String temp = template;
            while (temp.contains("{}")) {
                placeholderCount++;
                temp = temp.replaceFirst("\\{\\}", "PLACEHOLDER");
            }
            
            // 确保有足够的关键词
            while (keywords.size() < placeholderCount) {
                keywords.add("相关内容");
            }
            
            // 替换占位符
            String result = template;
            for (int i = 0; i < placeholderCount && i < keywords.size(); i++) {
                result = result.replaceFirst("\\{\\}", keywords.get(i));
            }
            
            return result;
        } catch (Exception e) {
            return template.replace("{}", "相关内容");
        }
    }
    
    /**
     * 初始化回复模板
     */
    private Map<String, List<String>> initializeResponseTemplates() {
        Map<String, List<String>> templates = new HashMap<>();
        
        // 基础智能体模板
        templates.put("base_agent", Arrays.asList(
            "我理解您的请求：{}。我将使用{}来{}，为您提供准确的结果。",
            "关于{}的问题，我认为最好的处理方式是{}，这样可以确保{}。",
            "让我来处理{}。通过{}，我能够{}。",
            "针对{}，我的方案是{}，预期能够{}。"
        ));
        
        // ReAct智能体模板
        templates.put("react_agent", Arrays.asList(
            "让我思考一下{}。我需要{}来{}，这样能得到最佳结果。",
            "分析您的请求{}，我将通过{}步骤来{}。",
            "理解了{}的需求。我会{}，然后{}。",
            "对于{}，我的推理是{}，因此我将{}。"
        ));
        
        // 工具调用智能体模板
        templates.put("tool_call_agent", Arrays.asList(
            "我将使用{}工具来处理{}，这能够{}。",
            "为了解决{}，我推荐使用{}，因为它能{}。",
            "基于{}的需求，最适合的工具是{}，它将{}。",
            "处理{}问题，我选择{}工具，预期{}。"
        ));
        
        // Manus核心系统模板
        templates.put("manus_core", Arrays.asList(
            "【Manus系统】正在处理{}。我将采用{}模式，通过{}来实现目标。",
            "【智能编排】分析了{}，决定使用{}策略，{}。",
            "【计划驱动】为了{}，我制定了{}计划，将{}。",
            "【Flow编排】检测到{}需求，选择{}流程，确保{}。"
        ));
        
        return templates;
    }
    
    /**
     * 初始化推理模板
     */
    private Map<String, List<String>> initializeReasoningTemplates() {
        Map<String, List<String>> templates = new HashMap<>();
        
        templates.put("thinking", Arrays.asList(
            "让我仔细分析{}。从{}的角度来看，{}是关键因素。",
            "思考这个问题：{}。基于{}，我认为{}。",
            "分析{}的需求。考虑到{}，最佳方案是{}。",
            "理解{}。从{}方面考虑，{}最为重要。"
        ));
        
        templates.put("observation", Arrays.asList(
            "观察到{}。这个结果{}，表明{}。",
            "执行后得到{}。从这可以看出{}，因此{}。",
            "结果显示{}。这意味着{}，接下来应该{}。",
            "行动产生了{}。这证明了{}，所以{}。"
        ));
        
        return templates;
    }
    
    /**
     * 初始化行动模板
     */
    private Map<String, List<String>> initializeActionTemplates() {
        Map<String, List<String>> templates = new HashMap<>();
        
        templates.put("action_planning", Arrays.asList(
            "基于分析{}，我将使用{}工具来{}。",
            "为了{}，我需要调用{}，具体是{}。",
            "根据思考，我的行动是{}。使用{}来{}。",
            "实施计划：{}。通过{}工具{}。"
        ));
        
        return templates;
    }
    
    // Getter方法
    public String getModelName() {
        return modelName;
    }
    
    public double getTemperature() {
        return temperature;
    }
    
    public int getMaxTokens() {
        return maxTokens;
    }
}