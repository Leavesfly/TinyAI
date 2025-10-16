package io.leavesfly.tinyai.agent.manus;

import java.util.Map;

import io.leavesfly.tinyai.agent.context.Message;

import java.util.HashMap;

/**
 * LLM模拟版本的Manus智能体演示
 * 展示基于LLM模拟的手稿智能体系统的强大功能
 * 
 * @author 山泽
 */
public class LLMManusDemo {
    
    public static void main(String[] args) {
        System.out.println("=== LLM模拟版本的Manus智能体演示 ===\n");
        
        // 创建启用LLM模拟的Manus实例
        Manus manus = new Manus("LLM-Manus", true, true);
        
        // 展示系统状态
        demonstrateSystemStatus(manus);
        
        // 演示1：基础工具调用（直接Agent模式）
        demonstrateBasicToolCalls(manus);
        
        // 演示2：LLM驱动的计划分解
        demonstrateLLMPlanning(manus);
        
        // 演示3：Flow编排模式
        demonstrateFlowOrchestration(manus);
        
        // 演示4：执行模式智能切换
        demonstrateExecutionModeSwitch(manus);
        
        // 演示5：复杂任务处理
        demonstrateComplexTaskHandling(manus);
        
        // 最终系统状态
        showFinalSystemStatus(manus);
    }
    
    /**
     * 展示系统状态
     */
    private static void demonstrateSystemStatus(Manus manus) {
        System.out.println("【系统初始化状态】");
        System.out.println(formatSystemStatus(manus.getSystemStatus()));
        System.out.println();
    }
    
    /**
     * 演示基础工具调用
     */
    private static void demonstrateBasicToolCalls(Manus manus) {
        System.out.println("=== 演示1：基础工具调用（LLM增强） ===");
        
        // 设置为直接Agent模式
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        
        // 数学计算
        System.out.println("【数学计算】");
        String mathQuery = "帮我计算 25 * 4 + 15 的结果";
        Message mathResponse = manus.processMessage(new Message("user", mathQuery));
        System.out.println("用户：" + mathQuery);
        System.out.println("Manus：" + mathResponse.getContent());
        System.out.println();
        
        // 时间查询
        System.out.println("【时间查询】");
        String timeQuery = "现在几点了？";
        Message timeResponse = manus.processMessage(new Message("user", timeQuery));
        System.out.println("用户：" + timeQuery);
        System.out.println("Manus：" + timeResponse.getContent());
        System.out.println();
        
        // 文本分析
        System.out.println("【文本分析】");
        String textQuery = "请分析这段文本：'人工智能正在改变世界，特别是在医疗、教育和交通领域'";
        Message textResponse = manus.processMessage(new Message("user", textQuery));
        System.out.println("用户：" + textQuery);
        System.out.println("Manus：" + textResponse.getContent());
        System.out.println();
    }
    
    /**
     * 演示LLM驱动的计划分解
     */
    private static void demonstrateLLMPlanning(Manus manus) {
        System.out.println("=== 演示2：LLM驱动的计划分解 ===");
        
        // 启用计划模式
        manus.setPlanningEnabled(true);
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        
        String complexQuery = "请帮我详细分析今天的工作计划，包括时间安排和数据统计";
        Message planResponse = manus.processMessage(new Message("user", complexQuery));
        System.out.println("用户：" + complexQuery);
        System.out.println("Manus：" + planResponse.getContent());
        System.out.println();
        
        // 禁用计划模式以便后续演示
        manus.setPlanningEnabled(false);
    }
    
    /**
     * 演示Flow编排模式
     */
    private static void demonstrateFlowOrchestration(Manus manus) {
        System.out.println("=== 演示3：Flow编排模式 ===");
        
        // 设置为Flow编排模式
        manus.setExecutionMode(ExecutionMode.FLOW_ORCHESTRATION);
        
        // 注册自定义Flow
        FlowDefinition customFlow = new FlowDefinition("数据处理流程", "专门处理数据计算和分析的复合流程");
        Map<String, Object> customNodes = new HashMap<>();
        customNodes.put("type", "tool");
        customNodes.put("name", "calculator");
        customNodes.put("description", "执行数据计算");
        customFlow.setNodes(customNodes);
        manus.registerFlow("data_processing_flow", customFlow);
        
        // 测试Flow编排
        String flowQuery = "请使用数据处理流程计算销售数据：上月销售额 1200，本月销售额 1500，计算增长率";
        Message flowResponse = manus.processMessage(new Message("user", flowQuery));
        System.out.println("用户：" + flowQuery);
        System.out.println("Manus：" + flowResponse.getContent());
        System.out.println();
    }
    
    /**
     * 演示执行模式智能切换
     */
    private static void demonstrateExecutionModeSwitch(Manus manus) {
        System.out.println("=== 演示4：执行模式智能切换 ===");
        
        // 启用LLM决策模式
        manus.setEnableLLMDecisionMaking(true);
        
        System.out.println("【简单查询 - 期望切换到直接模式】");
        String simpleQuery = "3 + 5 = ?";
        Message simpleResponse = manus.processMessage(new Message("user", simpleQuery));
        System.out.println("用户：" + simpleQuery);
        System.out.println("当前模式：" + manus.getExecutionMode().getDescription());
        System.out.println("Manus：" + simpleResponse.getContent());
        System.out.println();
        
        System.out.println("【复杂查询 - 期望切换到Flow编排模式】");
        String complexQuery = "请帮我制定一个完整的项目管理计划，包括时间节点分析和资源配置统计";
        Message complexResponse = manus.processMessage(new Message("user", complexQuery));
        System.out.println("用户：" + complexQuery);
        System.out.println("当前模式：" + manus.getExecutionMode().getDescription());
        System.out.println("Manus：" + complexResponse.getContent());
        System.out.println();
    }
    
    /**
     * 演示复杂任务处理
     */
    private static void demonstrateComplexTaskHandling(Manus manus) {
        System.out.println("=== 演示5：复杂任务处理 ===");
        
        // 综合性任务
        String comprehensiveQuery = "请帮我处理以下任务：" +
            "1. 计算 (100 + 200) * 0.15 的结果 " +
            "2. 获取当前时间 " +
            "3. 分析文本'市场表现良好，用户满意度提升' " +
            "4. 提供一个综合性的总结报告";
        
        Message comprehensiveResponse = manus.processMessage(new Message("user", comprehensiveQuery));
        System.out.println("用户：" + comprehensiveQuery);
        System.out.println("Manus：" + comprehensiveResponse.getContent());
        System.out.println();
    }
    
    /**
     * 显示最终系统状态
     */
    private static void showFinalSystemStatus(Manus manus) {
        System.out.println("=== 最终系统状态 ===");
        System.out.println(formatSystemStatus(manus.getSystemStatus()));
        
        // 显示工具使用统计
        System.out.println("\n【工具使用统计】");
        Map<String, Integer> toolStats = manus.getToolUsageCount();
        toolStats.forEach((tool, count) -> 
            System.out.println(tool + ": " + count + " 次"));
        
        System.out.println("\n=== 演示完成 ===");
    }
    
    /**
     * 格式化系统状态显示
     */
    private static String formatSystemStatus(Map<String, Object> status) {
        StringBuilder sb = new StringBuilder();
        sb.append("系统名称: ").append(status.get("name")).append("\n");
        sb.append("执行模式: ").append(status.get("execution_mode")).append("\n");
        sb.append("计划模式: ").append(status.get("planning_enabled")).append("\n");
        sb.append("LLM决策: ").append(status.get("llm_decision_making")).append("\n");
        sb.append("当前状态: ").append(status.get("current_state")).append("\n");
        sb.append("处理消息数: ").append(status.get("total_messages")).append("\n");
        sb.append("执行计划数: ").append(status.get("total_plans")).append("\n");
        sb.append("Flow执行数: ").append(status.get("total_flows")).append("\n");
        sb.append("注册Flow数: ").append(status.get("registered_flows")).append("\n");
        sb.append("注册工具数: ").append(status.get("registered_tools")).append("\n");
        sb.append("运行时长: ").append(status.get("uptime_hours")).append(" 小时");
        return sb.toString();
    }
    
    /**
     * ReAct模式演示
     */
    private static void demonstrateReActMode() {
        System.out.println("\n=== ReAct模式独立演示 ===");
        
        // 创建启用详细模式的ReAct智能体
        ReActAgent reactAgent = new ReActAgent("LLM-ReAct", true, true);
        
        System.out.println("【ReAct推理-行动循环演示】");
        String reactQuery = "帮我分析：如果我每天学习2小时，30天后总共学习多少小时？";
        Message reactResponse = reactAgent.processMessage(new Message("user", reactQuery));
        System.out.println("用户：" + reactQuery);
        System.out.println("ReAct Agent：" + reactResponse.getContent());
    }
    
    /**
     * 工具调用模式演示
     */
    private static void demonstrateToolCallMode() {
        System.out.println("\n=== 工具调用模式独立演示 ===");
        
        // 创建启用LLM工具选择的ToolCall智能体
        ToolCallAgent toolAgent = new ToolCallAgent("LLM-ToolCall", true, true);
        
        System.out.println("【智能工具选择演示】");
        String toolQuery = "我需要分析这段代码的性能：'for(int i=0; i<1000; i++) { System.out.println(i); }'";
        Message toolResponse = toolAgent.processMessage(new Message("user", toolQuery));
        System.out.println("用户：" + toolQuery);
        System.out.println("ToolCall Agent：" + toolResponse.getContent());
    }
    
    /**
     * 扩展演示：包含所有模式
     */
    public static void runExtendedDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("扩展演示：LLM模拟版本的完整功能展示");
        System.out.println("=".repeat(60));
        
        // ReAct模式演示
        demonstrateReActMode();
        
        // 工具调用模式演示
        demonstrateToolCallMode();
        
        // 性能对比演示
        demonstratePerformanceComparison();
    }
    
    /**
     * 性能对比演示
     */
    private static void demonstratePerformanceComparison() {
        System.out.println("\n=== 性能对比演示 ===");
        
        // 传统模式
        System.out.println("【传统模式】");
        Manus traditionalManus = new Manus("Traditional-Manus", false, false);
        traditionalManus.setLLMEnabled(false);
        
        long startTime = System.currentTimeMillis();
        Message traditionalResponse = traditionalManus.processMessage(
            new Message("user", "计算 15 * 8 + 20"));
        long traditionalTime = System.currentTimeMillis() - startTime;
        
        System.out.println("响应：" + traditionalResponse.getContent());
        System.out.println("耗时：" + traditionalTime + "ms");
        
        // LLM模式
        System.out.println("\n【LLM增强模式】");
        Manus llmManus = new Manus("LLM-Manus", false, true);
        
        startTime = System.currentTimeMillis();
        Message llmResponse = llmManus.processMessage(
            new Message("user", "计算 15 * 8 + 20"));
        long llmTime = System.currentTimeMillis() - startTime;
        
        System.out.println("响应：" + llmResponse.getContent());
        System.out.println("耗时：" + llmTime + "ms");
        
        System.out.println("\n性能对比结果：");
        System.out.println("- 传统模式更快速，适合简单任务");
        System.out.println("- LLM模式更智能，适合复杂任务");
        System.out.println("- 系统可根据任务复杂度智能选择模式");
    }
}