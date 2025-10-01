package io.leavesfly.tinyai.agent.pattern;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * ReAct模式Agent: Reasoning + Acting
 * 
 * 交替进行推理(Reasoning)和行动(Acting)，通过观察结果来指导下一步行动
 * 格式: Thought -> Action -> Observation -> Thought -> Action -> ...
 * 
 * @author 山泽
 */
public class ReActAgent extends BaseAgent {
    
    private String currentQuery;
    private final Random random;
    
    public ReActAgent() {
        this("ReAct Agent", 10);
    }
    
    public ReActAgent(String name) {
        this(name, 10);
    }
    
    public ReActAgent(String name, int maxSteps) {
        super(name, maxSteps);
        this.random = new Random();
        registerDefaultTools();
    }
    
    /**
     * 注册默认工具
     */
    private void registerDefaultTools() {
        // 计算器工具
        addTool("calculator", this::calculatorTool, "数学计算工具");
        
        // 搜索工具
        addTool("search", this::searchTool, "搜索工具");
        
        // 记忆查找工具
        addTool("memory", this::memoryLookupTool, "记忆查找工具");
    }
    
    /**
     * 计算器工具实现
     */
    private Object calculatorTool(Map<String, Object> args) {
        String expression = (String) args.get("expression");
        if (expression == null || expression.trim().isEmpty()) {
            return "错误：表达式为空";
        }
        
        try {
            // 简单的数学表达式计算（仅支持基本运算）
            String cleanExpression = expression.replaceAll("[^0-9+\\-*/().,\\s]", "");
            if (cleanExpression.isEmpty()) {
                return "错误：包含非法字符";
            }
            
            // 使用简单的表达式解析（这里使用JavaScript引擎作为示例）
            // 在实际应用中可以使用更安全的数学表达式解析库
            double result = evaluateExpression(cleanExpression);
            return "计算结果: " + result;
        } catch (Exception e) {
            return "计算错误: " + e.getMessage();
        }
    }
    
    /**
     * 简单的数学表达式求值（仅支持基本四则运算）
     */
    private double evaluateExpression(String expression) {
        // 这里实现一个简单的四则运算计算器
        // 去除空格
        expression = expression.replaceAll("\\s", "");
        
        // 简单实现：只处理两个数字和一个运算符的情况
        Pattern pattern = Pattern.compile("([0-9.]+)\\s*([+\\-*/])\\s*([0-9.]+)");
        Matcher matcher = pattern.matcher(expression);
        
        if (matcher.find()) {
            double num1 = Double.parseDouble(matcher.group(1));
            String operator = matcher.group(2);
            double num2 = Double.parseDouble(matcher.group(3));
            
            switch (operator) {
                case "+": return num1 + num2;
                case "-": return num1 - num2;
                case "*": return num1 * num2;
                case "/": 
                    if (num2 == 0) throw new ArithmeticException("除零错误");
                    return num1 / num2;
                default: throw new IllegalArgumentException("不支持的运算符: " + operator);
            }
        }
        
        // 如果是单个数字
        try {
            return Double.parseDouble(expression);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("无效的数学表达式: " + expression);
        }
    }
    
    /**
     * 搜索工具实现
     */
    private Object searchTool(Map<String, Object> args) {
        String query = (String) args.get("query");
        if (query == null) {
            return "搜索结果: 查询为空";
        }
        
        // 模拟搜索结果
        Map<String, String> searchResults = new HashMap<>();
        searchResults.put("天气", "今天天气晴朗，温度25度");
        searchResults.put("新闻", "今日科技新闻：AI技术取得新突破");
        searchResults.put("python", "Python是一种高级编程语言，简单易学");
        searchResults.put("机器学习", "机器学习是人工智能的核心技术");
        searchResults.put("java", "Java是一种面向对象的编程语言，具有跨平台特性");
        
        String queryLower = query.toLowerCase();
        for (Map.Entry<String, String> entry : searchResults.entrySet()) {
            if (queryLower.contains(entry.getKey().toLowerCase())) {
                return "搜索结果: " + entry.getValue();
            }
        }
        
        return "搜索结果: 未找到相关信息";
    }
    
    /**
     * 记忆查找工具实现
     */
    private Object memoryLookupTool(Map<String, Object> args) {
        String keyword = (String) args.get("keyword");
        if (keyword == null) {
            return "未找到相关记忆";
        }
        
        String keywordLower = keyword.toLowerCase();
        StringBuilder relevantMemories = new StringBuilder();
        
        for (String mem : memory) {
            if (mem.toLowerCase().contains(keywordLower)) {
                if (relevantMemories.length() > 0) {
                    relevantMemories.append("; ");
                }
                relevantMemories.append(mem);
            }
        }
        
        if (relevantMemories.length() > 0) {
            return "相关记忆: " + relevantMemories.toString();
        }
        
        return "未找到相关记忆";
    }
    
    /**
     * 思考步骤
     */
    private String think(String query, String context) {
        String prompt = "问题: " + query;
        if (context != null && !context.trim().isEmpty()) {
            prompt += "\n上下文: " + context;
        }
        
        // 根据问题类型进行思考
        if (containsAny(query, new String[]{"+", "-", "*", "/", "计算", "算"})) {
            return "这是一个数学问题，我需要使用计算器工具";
        } else if (containsAny(query, new String[]{"搜索", "查找", "什么是", "天气", "新闻"})) {
            return "这需要搜索信息，我应该使用搜索工具";
        } else if (containsAny(query, new String[]{"记忆", "之前"})) {
            return "这需要查找记忆，我应该使用记忆查找工具";
        } else {
            // 随机选择一个通用思考
            String[] thoughts = {
                "我需要分析这个问题：" + query,
                "让我思考一下需要什么信息来回答这个问题",
                "我应该使用什么工具来获取所需信息？",
                "基于当前信息，我的下一步行动是什么？"
            };
            return thoughts[random.nextInt(thoughts.length)];
        }
    }
    
    /**
     * 检查字符串是否包含任何指定的子字符串
     */
    private boolean containsAny(String text, String[] substrings) {
        for (String substring : substrings) {
            if (text.contains(substring)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 从思考中解析出行动
     */
    private Action parseAction(String thought) {
        if (thought.contains("计算器")) {
            // 尝试从原始查询中提取数学表达式
            Pattern mathPattern = Pattern.compile("[\\d+\\-*/().,\\s]+");
            Matcher matcher = mathPattern.matcher(currentQuery);
            if (matcher.find()) {
                String expression = matcher.group().trim();
                Map<String, Object> args = new HashMap<>();
                args.put("expression", expression);
                return new Action("calculator", args);
            }
        } else if (thought.contains("搜索")) {
            Map<String, Object> args = new HashMap<>();
            args.put("query", currentQuery);
            return new Action("search", args);
        } else if (thought.contains("记忆")) {
            // 提取关键词
            String[] keywords = currentQuery.split("\\s+");
            if (keywords.length > 0) {
                Map<String, Object> args = new HashMap<>();
                args.put("keyword", keywords[0]);
                return new Action("memory", args);
            }
        }
        
        return null;
    }
    
    @Override
    public String process(String query) {
        this.currentQuery = query;
        clearSteps();
        
        // 添加到记忆
        addToMemory("用户询问: " + query);
        
        String context = "";
        
        for (int stepNum = 0; stepNum < maxSteps; stepNum++) {
            setState(AgentState.THINKING);
            
            // Step 1: Think (思考)
            String thought = think(query, context);
            addStep("thought", thought);
            
            setState(AgentState.ACTING);
            
            // Step 2: Act (行动)
            Action action = parseAction(thought);
            if (action != null) {
                addStep("action", action.toString());
                
                // 执行动作
                Object result = callTool(action);
                
                setState(AgentState.OBSERVING);
                
                // Step 3: Observe (观察)
                if (action.hasResult()) {
                    String observation = String.valueOf(action.getResult());
                    addStep("observation", observation);
                    context += "\n" + observation;
                    
                    // 判断是否完成
                    if (observation.contains("计算结果") || observation.contains("搜索结果")) {
                        // 生成最终答案
                        setState(AgentState.THINKING);
                        String finalThought = "基于观察结果，我可以回答用户的问题了";
                        addStep("thought", finalThought);
                        
                        String answer = "根据我的分析和工具使用，" + observation;
                        addStep("answer", answer);
                        setState(AgentState.DONE);
                        return answer;
                    }
                } else if (action.hasError()) {
                    String errorObs = "工具执行失败: " + action.getError();
                    addStep("observation", errorObs);
                    context += "\n" + errorObs;
                }
            } else {
                // 如果没有解析出行动，直接给出答案
                String answer = "基于我的思考：" + thought;
                addStep("answer", answer);
                setState(AgentState.DONE);
                return answer;
            }
        }
        
        setState(AgentState.DONE);
        return "很抱歉，我无法在限定步骤内完成这个任务";
    }
}