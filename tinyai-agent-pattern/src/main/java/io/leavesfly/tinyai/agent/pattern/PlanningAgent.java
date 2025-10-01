package io.leavesfly.tinyai.agent.pattern;

import java.util.*;

/**
 * Planning模式Agent: 先制定计划再执行
 * 
 * 将复杂任务分解为子任务，制定执行计划，然后按计划执行
 * 
 * @author 山泽
 */
public class PlanningAgent extends BaseAgent {
    
    /** 执行计划 */
    private final List<PlanTask> plan;
    
    /** 当前任务索引 */
    private int currentTaskIndex; 
    
    private final Random random;
    
    /**
     * 计划任务内部类
     */
    public static class PlanTask {
        private final String task;
        private final String description;
        private final Map<String, Object> args;
        
        public PlanTask(String task, String description, Map<String, Object> args) {
            this.task = task;
            this.description = description;
            this.args = new HashMap<>(args);
        }
        
        public String getTask() {
            return task;
        }
        
        public String getDescription() {
            return description;
        }
        
        public Map<String, Object> getArgs() {
            return new HashMap<>(args);
        }
        
        public void updateArg(String key, Object value) {
            args.put(key, value);
        }
        
        @Override
        public String toString() {
            return description;
        }
    }
    
    public PlanningAgent() {
        this("Planning Agent", 15);
    }
    
    public PlanningAgent(String name) {
        this(name, 15);
    }
    
    public PlanningAgent(String name, int maxSteps) {
        super(name, maxSteps);
        this.plan = new ArrayList<>();
        this.currentTaskIndex = 0;
        this.random = new Random();
        registerDefaultTools();
    }
    
    /**
     * 注册默认工具
     */
    private void registerDefaultTools() {
        // 研究工具
        addTool("research", this::researchTool, "研究工具");
        
        // 分析工具
        addTool("analyze", this::analyzeTool, "分析工具");
        
        // 综合工具
        addTool("synthesize", this::synthesizeTool, "综合工具");
        
        // 验证工具
        addTool("validate", this::validateTool, "验证工具");
    }
    
    /**
     * 研究工具实现
     */
    private Object researchTool(Map<String, Object> args) {
        String topic = (String) args.get("topic");
        if (topic == null) {
            return "研究结果: 主题为空";
        }
        
        Map<String, String> researchDb = new HashMap<>();
        researchDb.put("python", "Python是一种解释型、面向对象的编程语言，具有简洁的语法");
        researchDb.put("机器学习", "机器学习是让计算机从数据中学习模式的技术");
        researchDb.put("深度学习", "深度学习使用神经网络来模拟人脑的学习过程");
        researchDb.put("ai", "人工智能是使机器能够模拟人类智能的技术");
        researchDb.put("java", "Java是一种面向对象的编程语言，具有跨平台特性");
        researchDb.put("算法", "算法是解决问题的明确指令序列");
        
        String topicLower = topic.toLowerCase();
        for (Map.Entry<String, String> entry : researchDb.entrySet()) {
            if (topicLower.contains(entry.getKey().toLowerCase())) {
                return "研究结果: " + entry.getValue();
            }
        }
        
        return "研究结果: 关于'" + topic + "'的基础信息已收集";
    }
    
    /**
     * 分析工具实现
     */
    private Object analyzeTool(Map<String, Object> args) {
        String data = (String) args.get("data");
        if (data == null) {
            return "分析结果: 数据为空";
        }
        
        String preview = data.length() > 50 ? data.substring(0, 50) + "..." : data;
        return "分析结果: 对'" + preview + "'进行了深入分析，发现了关键模式和趋势";
    }
    
    /**
     * 综合工具实现
     */
    private Object synthesizeTool(Map<String, Object> args) {
        String components = (String) args.get("components");
        return "综合结果: 将多个组件整合形成完整的解决方案";
    }
    
    /**
     * 验证工具实现
     */
    private Object validateTool(Map<String, Object> args) {
        String solution = (String) args.get("solution");
        int score = 8 + random.nextInt(3); // 8-10分
        return "验证结果: 解决方案经过验证，质量评分: " + score + "/10";
    }
    
    /**
     * 制定计划
     */
    private List<PlanTask> createPlan(String query) {
        addStep("planning", "开始为查询制定计划: " + query);
        
        List<PlanTask> newPlan = new ArrayList<>();
        
        // 根据查询类型制定不同的计划
        if (containsAny(query, new String[]{"分析", "研究"})) {
            newPlan.add(new PlanTask("research", "收集相关信息", 
                Map.of("topic", query)));
            newPlan.add(new PlanTask("analyze", "分析收集的信息", 
                Map.of("data", "收集的信息")));
            newPlan.add(new PlanTask("synthesize", "综合分析结果", 
                Map.of("components", "分析结果")));
            newPlan.add(new PlanTask("validate", "验证最终结果", 
                Map.of("solution", "最终结果")));
                
        } else if (containsAny(query, new String[]{"学习", "教"})) {
            newPlan.add(new PlanTask("research", "研究学习主题", 
                Map.of("topic", query)));
            newPlan.add(new PlanTask("analyze", "分析学习要点", 
                Map.of("data", "学习材料")));
            newPlan.add(new PlanTask("synthesize", "整理学习大纲", 
                Map.of("components", "学习要点")));
                
        } else {
            // 通用计划
            newPlan.add(new PlanTask("research", "收集基础信息", 
                Map.of("topic", query)));
            newPlan.add(new PlanTask("analyze", "分析问题", 
                Map.of("data", query)));
            newPlan.add(new PlanTask("synthesize", "形成解决方案", 
                Map.of("components", "问题分析")));
        }
        
        // 构建计划描述
        StringBuilder planDescription = new StringBuilder("制定的执行计划:\n");
        for (int i = 0; i < newPlan.size(); i++) {
            planDescription.append(i + 1).append(". ").append(newPlan.get(i).getDescription()).append("\n");
        }
        
        addStep("plan", planDescription.toString().trim());
        
        return newPlan;
    }
    
    /**
     * 执行计划
     */
    private String executePlan(List<PlanTask> planToExecute) {
        List<String> results = new ArrayList<>();
        
        for (int i = 0; i < planToExecute.size(); i++) {
            PlanTask task = planToExecute.get(i);
            addStep("executing", "执行步骤 " + (i + 1) + ": " + task.getDescription());
            
            // 执行任务
            Action action = new Action(task.getTask(), task.getArgs());
            Object result = callTool(action);
            
            addStep("action", action.toString());
            addStep("observation", String.valueOf(result));
            
            results.add(String.valueOf(result));
            
            // 更新下一个任务的参数（简单实现）
            if (i + 1 < planToExecute.size()) {
                PlanTask nextTask = planToExecute.get(i + 1);
                String resultStr = String.valueOf(result);
                
                if (nextTask.getArgs().containsKey("data")) {
                    nextTask.updateArg("data", resultStr);
                } else if (nextTask.getArgs().containsKey("components")) {
                    nextTask.updateArg("components", resultStr);
                } else if (nextTask.getArgs().containsKey("solution")) {
                    nextTask.updateArg("solution", resultStr);
                }
            }
        }
        
        return String.join("\n", results);
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
    
    @Override
    public String process(String query) {
        clearSteps();
        plan.clear();
        currentTaskIndex = 0;
        
        setState(AgentState.PLANNING);
        
        // 第一阶段：制定计划
        List<PlanTask> newPlan = createPlan(query);
        plan.addAll(newPlan);
        
        setState(AgentState.ACTING);
        
        // 第二阶段：执行计划
        String executionResults = executePlan(plan);
        
        setState(AgentState.THINKING);
        
        // 第三阶段：总结结果
        String summary = "计划执行完成。基于" + plan.size() + "个步骤的执行，我为您的问题'" + query + "'提供了全面的解决方案。";
        addStep("summary", summary);
        
        String finalAnswer = summary + "\n\n执行结果:\n" + executionResults;
        
        setState(AgentState.DONE);
        return finalAnswer;
    }
    
    /**
     * 获取执行计划
     */
    public List<PlanTask> getPlan() {
        return new ArrayList<>(plan);
    }
    
    /**
     * 获取当前任务索引
     */
    public int getCurrentTaskIndex() {
        return currentTaskIndex;
    }
    
    /**
     * 清空计划
     */
    public void clearPlan() {
        plan.clear();
        currentTaskIndex = 0;
    }
    
    @Override
    public void reset() {
        super.reset();
        clearPlan();
    }
}