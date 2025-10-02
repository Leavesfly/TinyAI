package io.leavesfly.tinyai.agent.evol;

import java.util.*;
import java.util.logging.Logger;

/**
 * 自进化Agent演示程序
 * 展示Agent的自学习和自进化能力
 * 
 * @author 山泽
 */
public class EvolDemo {
    
    private static final Logger logger = Logger.getLogger(EvolDemo.class.getName());
    
    public static void main(String[] args) {
        System.out.println("=== 自进化自学习Agent演示 ===");
        
        // 创建Agent
        SelfEvolvingAgent agent = new SelfEvolvingAgent("学习型AI助手");
        
        // 模拟任务序列
        List<TaskContext> tasks = createTaskSequence();
        
        System.out.println("\n准备处理 " + tasks.size() + " 个任务...\n");
        
        // 处理任务并观察学习过程
        for (int i = 0; i < tasks.size(); i++) {
            TaskContext taskContext = tasks.get(i);
            System.out.println("--- 任务 " + (i + 1) + ": " + taskContext.task + " ---");
            
            SelfEvolvingAgent.TaskResult result = agent.processTask(taskContext.task, taskContext.context);
            
            System.out.println("选择的行动: " + result.getAction());
            System.out.println("执行结果: " + (result.isSuccess() ? "成功" : "失败"));
            System.out.println("奖励值: " + String.format("%.2f", result.getReward()));
            System.out.println("学习洞察: " + result.getLearningInsights());
            
            // 每3个任务显示一次性能摘要
            if ((i + 1) % 3 == 0) {
                displayPerformanceSummary(agent);
            }
            
            System.out.println();
        }
        
        // 最终性能报告
        displayFinalReport(agent);
        
        // 显示学到的策略
        displayLearnedStrategies(agent);
        
        // 显示知识图谱统计
        displayKnowledgeGraphStats(agent);
    }
    
    /**
     * 创建任务序列
     */
    private static List<TaskContext> createTaskSequence() {
        List<TaskContext> tasks = new ArrayList<>();
        
        // 任务1: 搜索
        Map<String, Object> context1 = new HashMap<>();
        context1.put("query", "Python基础教程");
        context1.put("difficulty", "beginner");
        tasks.add(new TaskContext("搜索Python教程", context1));
        
        // 任务2: 计算
        Map<String, Object> context2 = new HashMap<>();
        context2.put("expression", "1000 * 1.05");
        context2.put("context", "finance");
        tasks.add(new TaskContext("计算复合利率", context2));
        
        // 任务3: 分析
        Map<String, Object> userData = new HashMap<>();
        userData.put("users", 100);
        userData.put("active", 80);
        userData.put("retention", 0.75);
        Map<String, Object> context3 = new HashMap<>();
        context3.put("data", userData);
        tasks.add(new TaskContext("分析用户数据", context3));
        
        // 任务4: 规划
        Map<String, Object> context4 = new HashMap<>();
        context4.put("goal", "掌握机器学习");
        context4.put("timeframe", "3个月");
        tasks.add(new TaskContext("制定学习计划", context4));
        
        // 任务5: 再次搜索
        Map<String, Object> context5 = new HashMap<>();
        context5.put("query", "机器学习入门");
        context5.put("difficulty", "intermediate");
        tasks.add(new TaskContext("搜索机器学习资源", context5));
        
        // 任务6: 分析学习进度
        Map<String, Object> progressData = new HashMap<>();
        progressData.put("completed", 5);
        progressData.put("total", 20);
        progressData.put("avg_score", 85);
        Map<String, Object> context6 = new HashMap<>();
        context6.put("data", progressData);
        tasks.add(new TaskContext("分析学习进度", context6));
        
        // 任务7: 计算学习效率
        Map<String, Object> context7 = new HashMap<>();
        context7.put("expression", "85 * 0.8");
        context7.put("context", "learning");
        tasks.add(new TaskContext("计算学习效率", context7));
        
        // 任务8: 优化策略
        Map<String, Object> context8 = new HashMap<>();
        context8.put("goal", "提高学习效率");
        context8.put("current_rate", 0.68);
        tasks.add(new TaskContext("优化学习策略", context8));
        
        return tasks;
    }
    
    /**
     * 显示性能摘要
     */
    private static void displayPerformanceSummary(SelfEvolvingAgent agent) {
        Map<String, Object> performance = agent.getPerformanceSummary();
        System.out.println("\n📊 当前性能摘要:");
        System.out.println("  总任务数: " + performance.get("total_tasks"));
        
        Object successRateObj = performance.get("current_success_rate");
        if (successRateObj instanceof Number) {
            double successRate = ((Number) successRateObj).doubleValue();
            System.out.println("  成功率: " + String.format("%.1f%%", successRate * 100));
        }
        
        System.out.println("  策略数量: " + performance.get("strategies_count"));
        System.out.println("  知识概念: " + performance.get("knowledge_concepts"));
        
        Object explorationRateObj = performance.get("exploration_rate");
        if (explorationRateObj instanceof Number) {
            double explorationRate = ((Number) explorationRateObj).doubleValue();
            System.out.println("  探索率: " + String.format("%.2f", explorationRate));
        }
    }
    
    /**
     * 显示最终报告
     */
    private static void displayFinalReport(SelfEvolvingAgent agent) {
        System.out.println("\n=== 最终学习报告 ===");
        Map<String, Object> finalPerformance = agent.getPerformanceSummary();
        
        System.out.println("总处理任务: " + finalPerformance.get("total_tasks"));
        
        Object successRateObj = finalPerformance.get("current_success_rate");
        if (successRateObj instanceof Number) {
            double successRate = ((Number) successRateObj).doubleValue();
            System.out.println("最终成功率: " + String.format("%.1f%%", successRate * 100));
        }
        
        System.out.println("性能趋势: " + finalPerformance.get("trend"));
        System.out.println("学习策略数: " + finalPerformance.get("strategies_count"));
        System.out.println("知识概念数: " + finalPerformance.get("knowledge_concepts"));
    }
    
    /**
     * 显示学到的策略
     */
    private static void displayLearnedStrategies(SelfEvolvingAgent agent) {
        System.out.println("\n🧠 学到的策略:");
        
        Map<String, Strategy> strategies = agent.getStrategies();
        strategies.values().stream()
                .filter(strategy -> strategy.getUsageCount() > 0)
                .sorted((s1, s2) -> Double.compare(s2.getSuccessRate(), s1.getSuccessRate()))
                .forEach(strategy -> {
                    System.out.println(String.format("  %s: 成功率 %.1f%%, 使用 %d 次", 
                                      strategy.getName(), 
                                      strategy.getSuccessRate() * 100, 
                                      strategy.getUsageCount()));
                });
    }
    
    /**
     * 显示知识图谱统计
     */
    private static void displayKnowledgeGraphStats(SelfEvolvingAgent agent) {
        System.out.println("\n📈 知识图谱统计:");
        
        Map<String, Object> stats = agent.getKnowledgeGraph().getStatistics();
        System.out.println("  概念节点数: " + stats.get("conceptCount"));
        System.out.println("  关系边数: " + stats.get("relationCount"));
        
        Object mostActiveObj = stats.get("mostActiveConcept");
        Object maxAccessObj = stats.get("maxAccessCount");
        if (mostActiveObj != null && maxAccessObj != null) {
            System.out.println("  最活跃概念: " + mostActiveObj + " (访问 " + maxAccessObj + " 次)");
        }
    }
    
    /**
     * 任务上下文类
     */
    private static class TaskContext {
        final String task;
        final Map<String, Object> context;
        
        TaskContext(String task, Map<String, Object> context) {
            this.task = task;
            this.context = context;
        }
    }
    
    /**
     * 快速演示
     */
    public static void quickDemo() {
        System.out.println("=== 快速演示 ===");
        
        SelfEvolvingAgent agent = new SelfEvolvingAgent("快速演示Agent");
        
        // 简单任务序列
        String[] tasks = {"搜索信息", "分析数据", "制定计划"};
        
        for (String task : tasks) {
            SelfEvolvingAgent.TaskResult result = agent.processTask(task, null);
            System.out.println(String.format("任务: %s -> 动作: %s -> 结果: %s", 
                              task, result.getAction(), result.isSuccess() ? "成功" : "失败"));
        }
        
        System.out.println("\n演示完成！");
    }
}
