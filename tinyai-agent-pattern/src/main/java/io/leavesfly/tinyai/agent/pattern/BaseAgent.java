package io.leavesfly.tinyai.agent.pattern;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Agent抽象基类
 * 定义了所有Agent的通用接口和基础功能
 * @author 山泽
 */
public abstract class BaseAgent {
    /** Agent名称 */
    protected final String name;
    
    /** 最大执行步骤数 */
    protected final int maxSteps;
    
    /** 执行步骤记录 */
    protected final List<Step> steps;
    
    /** 当前状态 */
    protected AgentState state;
    
    /** 工具注册表 */
    protected final Map<String, Tool> tools;
    
    /** 记忆列表 */
    protected final List<String> memory;
    
    /**
     * 工具内部类
     */
    public static class Tool {
        private final Function<Map<String, Object>, Object> function;
        private final String description;
        
        public Tool(Function<Map<String, Object>, Object> function, String description) {
            this.function = function;
            this.description = description;
        }
        
        public Object execute(Map<String, Object> arguments) {
            return function.apply(arguments);
        }
        
        public String getDescription() {
            return description;
        }
    }
    
    /**
     * 构造函数
     * @param name Agent名称
     * @param maxSteps 最大执行步骤数
     */
    public BaseAgent(String name, int maxSteps) {
        this.name = name;
        this.maxSteps = maxSteps;
        this.steps = new ArrayList<>();
        this.state = AgentState.THINKING;
        this.tools = new HashMap<>();
        this.memory = new ArrayList<>();
    }
    
    /**
     * 添加工具
     * @param name 工具名称
     * @param function 工具函数
     * @param description 工具描述
     */
    public void addTool(String name, Function<Map<String, Object>, Object> function, String description) {
        tools.put(name, new Tool(function, description));
    }
    
    /**
     * 调用工具
     * @param action 要执行的动作
     * @return 执行结果
     */
    public Object callTool(Action action) {
        String toolName = action.getName();
        
        if (!tools.containsKey(toolName)) {
            action.setError("工具 " + toolName + " 不存在");
            return null;
        }
        
        try {
            Tool tool = tools.get(toolName);
            Object result = tool.execute(action.getArguments());
            action.setResult(result);
            return result;
        } catch (Exception e) {
            action.setError(e.getMessage());
            return null;
        }
    }
    
    /**
     * 添加步骤记录
     * @param stepType 步骤类型
     * @param content 步骤内容
     * @return 创建的步骤对象
     */
    public Step addStep(String stepType, String content) {
        return addStep(stepType, content, new HashMap<>());
    }
    
    /**
     * 添加步骤记录
     * @param stepType 步骤类型
     * @param content 步骤内容
     * @param metadata 元数据
     * @return 创建的步骤对象
     */
    public Step addStep(String stepType, String content, Map<String, Object> metadata) {
        Step step = new Step(stepType, content, metadata);
        steps.add(step);
        return step;
    }
    
    /**
     * 处理查询（抽象方法，需要子类实现）
     * @param query 查询内容
     * @return 处理结果
     */
    public abstract String process(String query);
    
    /**
     * 获取步骤摘要
     * @return 步骤摘要字符串
     */
    public String getStepsSummary() {
        StringBuilder summary = new StringBuilder();
        for (int i = 0; i < steps.size(); i++) {
            Step step = steps.get(i);
            summary.append(i + 1).append(". ").append(step.toString());
            if (i < steps.size() - 1) {
                summary.append("\n");
            }
        }
        return summary.toString();
    }
    
    // Getter methods
    public String getName() {
        return name;
    }
    
    public int getMaxSteps() {
        return maxSteps;
    }
    
    public List<Step> getSteps() {
        return new ArrayList<>(steps);
    }
    
    public AgentState getState() {
        return state;
    }
    
    public void setState(AgentState state) {
        this.state = state;
    }
    
    public Map<String, Tool> getTools() {
        return new HashMap<>(tools);
    }
    
    public List<String> getMemory() {
        return new ArrayList<>(memory);
    }
    
    public void addToMemory(String memory) {
        this.memory.add(memory);
    }
    
    /**
     * 清空步骤记录
     */
    public void clearSteps() {
        steps.clear();
    }
    
    /**
     * 重置Agent状态
     */
    public void reset() {
        clearSteps();
        setState(AgentState.THINKING);
    }
}