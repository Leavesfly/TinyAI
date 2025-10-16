package io.leavesfly.tinyai.agent.manus;

import java.util.Map;

import io.leavesfly.tinyai.agent.context.Message;

import java.util.HashMap;

/**
 * LLM模拟版本的Manus智能体验证器
 * 验证基于LLM模拟的改造效果
 * 
 * @author 山泽
 */
public class LLMManusValidator {
    
    /**
     * 运行所有验证测试
     */
    public static void main(String[] args) {
        System.out.println("=== LLM模拟版本Manus智能体验证开始 ===\n");
        
        LLMManusValidator validator = new LLMManusValidator();
        
        try {
            validator.validateLLMSimulatorInitialization();
            validator.validateBasicToolCalls();
            validator.validateExecutionModeSwitch();
            validator.validatePlanningMode();
            validator.validateFlowManagement();
            validator.validateReActAgentEnhancement();
            validator.validateToolCallAgentEnhancement();
            validator.validateSystemStatusMonitoring();
            validator.validateLLMDecisionMaking();
            validator.validateErrorHandling();
            validator.validatePerformance();
            validator.validateCompleteWorkflow();
            
            System.out.println("=== 所有验证测试通过 ===");
            
        } catch (Exception e) {
            System.err.println("验证过程中出现错误：" + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 验证LLM模拟器初始化
     */
    public void validateLLMSimulatorInitialization() {
        System.out.println("【验证1】LLM模拟器初始化");
        
        Manus manus = new Manus("Test-Manus", true, true);
        
        assert manus.getLLMSimulator() != null : "LLM模拟器应该正确初始化";
        assert manus.isLLMEnabled() : "LLM功能应该启用";
        assert manus.getSystemPrompt() != null : "系统提示应该存在";
        assert manus.getSystemPrompt().contains("Manus") : "系统提示应该包含Manus标识";
        
        System.out.println("✓ LLM模拟器初始化验证通过");
        System.out.println("  - LLM模拟器: " + manus.getLLMSimulator().getModelName());
        System.out.println("  - 系统提示长度: " + manus.getSystemPrompt().length() + " 字符");
        System.out.println();
    }
    
    /**
     * 验证基础工具调用能力
     */
    public void validateBasicToolCalls() {
        System.out.println("【验证2】基础工具调用能力");
        
        Manus manus = new Manus("Test-Manus");
        
        // 测试计算器工具
        Message mathQuery = new Message("user", "计算 10 + 5");
        Message mathResponse = manus.processMessage(mathQuery);
        
        assert mathResponse != null : "应该返回响应";
        assert mathResponse.getContent() != null : "响应内容不应为空";
        assert mathResponse.getContent().length() > 0 : "响应应该有实际内容";
        
        // 测试时间工具
        Message timeQuery = new Message("user", "现在几点？");
        Message timeResponse = manus.processMessage(timeQuery);
        
        assert timeResponse != null : "时间查询应该返回响应";
        assert timeResponse.getContent() != null : "时间响应内容不应为空";
        
        System.out.println("✓ 基础工具调用验证通过");
        System.out.println("  - 数学计算响应长度: " + mathResponse.getContent().length() + " 字符");
        System.out.println("  - 时间查询响应长度: " + timeResponse.getContent().length() + " 字符");
        System.out.println();
    }
    
    /**
     * 验证执行模式切换
     */
    public void validateExecutionModeSwitch() {
        System.out.println("【验证3】执行模式切换");
        
        Manus manus = new Manus("Test-Manus");
        
        // 测试直接Agent模式
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        assert manus.getExecutionMode() == ExecutionMode.DIRECT_AGENT : "应该设置为直接Agent模式";
        
        // 测试Flow编排模式
        manus.setExecutionMode(ExecutionMode.FLOW_ORCHESTRATION);
        assert manus.getExecutionMode() == ExecutionMode.FLOW_ORCHESTRATION : "应该设置为Flow编排模式";
        
        System.out.println("✓ 执行模式切换验证通过");
        System.out.println("  - 当前模式: " + manus.getExecutionMode().getDescription());
        System.out.println();
    }
    
    /**
     * 验证计划模式功能
     */
    public void validatePlanningMode() {
        System.out.println("【验证4】计划模式功能");
        
        Manus manus = new Manus("Test-Manus", true, true);
        
        assert manus.isPlanningEnabled() : "计划模式应该启用";
        
        Message planQuery = new Message("user", "请制定一个详细的学习计划");
        Message planResponse = manus.processMessage(planQuery);
        
        assert planResponse != null : "计划查询应该返回响应";
        assert planResponse.getContent().contains("计划") || 
               planResponse.getContent().contains("任务") : "响应应该包含计划相关内容";
        
        System.out.println("✓ 计划模式功能验证通过");
        System.out.println("  - 计划响应包含关键词: " + (planResponse.getContent().contains("计划") ? "计划" : "任务"));
        System.out.println();
    }
    
    /**
     * 验证Flow管理
     */
    public void validateFlowManagement() {
        System.out.println("【验证5】Flow管理功能");
        
        Manus manus = new Manus("Test-Manus");
        
        // 注册自定义Flow
        FlowDefinition testFlow = new FlowDefinition("测试流程", "用于测试的流程");
        Map<String, Object> nodes = new HashMap<>();
        nodes.put("type", "tool");
        nodes.put("name", "calculator");
        testFlow.setNodes(nodes);
        
        manus.registerFlow("test_flow", testFlow);
        
        // 验证Flow注册
        assert manus.getRegisteredFlows().containsKey("test_flow") : "Flow应该注册成功";
        
        // 测试Flow移除
        assert manus.removeFlow("test_flow") : "Flow应该移除成功";
        assert !manus.getRegisteredFlows().containsKey("test_flow") : "Flow应该已被移除";
        
        System.out.println("✓ Flow管理功能验证通过");
        System.out.println("  - 当前注册Flow数量: " + manus.getRegisteredFlows().size());
        System.out.println();
    }
    
    /**
     * 验证ReAct智能体LLM增强
     */
    public void validateReActAgentEnhancement() {
        System.out.println("【验证6】ReAct智能体LLM增强");
        
        ReActAgent reactAgent = new ReActAgent("Test-ReAct", false, true);
        
        Message reactQuery = new Message("user", "分析：2 * 3 + 4 等于多少？");
        Message reactResponse = reactAgent.processMessage(reactQuery);
        
        assert reactResponse != null : "ReAct智能体应该返回响应";
        assert reactResponse.getContent() != null : "ReAct响应内容不应为空";
        assert reactAgent.isLLMEnabled() : "ReAct智能体的LLM功能应该启用";
        
        System.out.println("✓ ReAct智能体LLM增强验证通过");
        System.out.println("  - LLM模式: " + reactAgent.isLLMEnabled());
        System.out.println("  - 响应长度: " + reactResponse.getContent().length() + " 字符");
        System.out.println();
    }
    
    /**
     * 验证工具调用智能体LLM增强
     */
    public void validateToolCallAgentEnhancement() {
        System.out.println("【验证7】工具调用智能体LLM增强");
        
        ToolCallAgent toolAgent = new ToolCallAgent("Test-ToolCall", true, true);
        
        Message toolQuery = new Message("user", "我需要分析文本：'这是一个测试'");
        Message toolResponse = toolAgent.processMessage(toolQuery);
        
        assert toolResponse != null : "工具调用智能体应该返回响应";
        assert toolResponse.getContent() != null : "工具调用响应内容不应为空";
        assert toolAgent.isEnableLLMToolSelection() : "应该启用LLM工具选择";
        assert toolAgent.isAutoToolSelection() : "应该启用自动工具选择";
        
        System.out.println("✓ 工具调用智能体LLM增强验证通过");
        System.out.println("  - LLM工具选择: " + toolAgent.isEnableLLMToolSelection());
        System.out.println("  - 自动工具选择: " + toolAgent.isAutoToolSelection());
        System.out.println();
    }
    
    /**
     * 验证系统状态监控
     */
    public void validateSystemStatusMonitoring() {
        System.out.println("【验证8】系统状态监控");
        
        Manus manus = new Manus("Test-Manus", true, true);
        Map<String, Object> status = manus.getSystemStatus();
        
        assert status != null : "系统状态不应为空";
        assert status.containsKey("name") : "状态应该包含名称";
        assert status.containsKey("execution_mode") : "状态应该包含执行模式";
        assert status.containsKey("llm_decision_making") : "状态应该包含LLM决策标识";
        assert status.containsKey("total_messages") : "状态应该包含消息统计";
        
        // 验证初始值
        assert "Test-Manus".equals(status.get("name")) : "名称应该正确";
        assert Boolean.TRUE.equals(status.get("llm_decision_making")) : "LLM决策应该启用";
        
        System.out.println("✓ 系统状态监控验证通过");
        System.out.println("  - 监控项目数: " + status.size());
        System.out.println("  - LLM决策状态: " + status.get("llm_decision_making"));
        System.out.println();
    }
    
    /**
     * 验证LLM决策功能
     */
    public void validateLLMDecisionMaking() {
        System.out.println("【验证9】LLM决策功能");
        
        Manus manus = new Manus("Test-Manus", false, true);
        
        assert manus.isEnableLLMDecisionMaking() : "LLM决策功能应该启用";
        
        // 测试简单查询
        Message simpleQuery = new Message("user", "1 + 1");
        manus.processMessage(simpleQuery);
        
        // 测试复杂查询
        Message complexQuery = new Message("user", "请制定一个包含时间分析和数据计算的综合方案");
        manus.processMessage(complexQuery);
        
        // 验证系统仍然可以正常工作
        Map<String, Object> status = manus.getSystemStatus();
        assert status.containsKey("total_messages") : "系统应该正常记录消息";
        assert ((Integer) status.get("total_messages")) >= 2 : "应该处理了至少2条消息";
        
        System.out.println("✓ LLM决策功能验证通过");
        System.out.println("  - 处理消息数: " + status.get("total_messages"));
        System.out.println();
    }
    
    /**
     * 验证错误处理能力
     */
    public void validateErrorHandling() {
        System.out.println("【验证10】错误处理能力");
        
        Manus manus = new Manus("Test-Manus");
        
        // 测试空输入
        Message emptyQuery = new Message("user", "");
        Message emptyResponse = manus.processMessage(emptyQuery);
        
        assert emptyResponse != null : "即使输入为空也应该返回响应";
        assert emptyResponse.getContent() != null : "错误响应内容不应为空";
        
        // 测试null输入处理（通过捕获异常）
        try {
            Message nullResponse = manus.processMessage(new Message("user", null));
            // 如果没有抛出异常，验证响应
            assert nullResponse != null : "null输入应该有处理机制";
        } catch (Exception e) {
            // 如果抛出异常也是可以接受的
            System.out.println("  - null输入正确抛出异常: " + e.getClass().getSimpleName());
        }
        
        // 验证系统状态仍然正常
        assert manus.getState() == AgentState.DONE || manus.getState() == AgentState.IDLE : 
            "错误处理后系统状态应该正常";
        
        System.out.println("✓ 错误处理能力验证通过");
        System.out.println("  - 系统状态: " + manus.getState());
        System.out.println();
    }
    
    /**
     * 验证性能表现
     */
    public void validatePerformance() {
        System.out.println("【验证11】性能表现");
        
        Manus manus = new Manus("Test-Manus");
        
        long startTime = System.currentTimeMillis();
        
        // 执行一系列操作
        for (int i = 0; i < 5; i++) {
            manus.processMessage(new Message("user", "计算 " + i + " + " + (i + 1)));
        }
        
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        
        // 验证性能在合理范围内
        assert totalTime < 10000 : "性能应该在合理范围内，实际耗时：" + totalTime + "ms";
        
        System.out.println("✓ 性能表现验证通过");
        System.out.println("  - 处理5个查询耗时: " + totalTime + "ms");
        System.out.println("  - 平均每个查询: " + (totalTime / 5) + "ms");
        System.out.println();
    }
    
    /**
     * 验证完整工作流
     */
    public void validateCompleteWorkflow() {
        System.out.println("【验证12】完整工作流");
        
        Manus manus = new Manus("Test-Manus", true, true);
        
        // 执行一个综合性任务
        String complexTask = "请帮我：1) 计算 100 * 1.2 的结果，2) 获取当前时间，3) 分析文本'项目进展顺利'";
        Message response = manus.processMessage(new Message("user", complexTask));
        
        // 验证响应
        assert response != null : "综合任务应该返回响应";
        assert response.getContent() != null : "响应内容不应为空";
        assert response.getContent().length() > 50 : "综合任务的响应应该比较详细";
        
        // 验证系统状态
        Map<String, Object> finalStatus = manus.getSystemStatus();
        assert ((Integer) finalStatus.get("total_messages")) > 0 : "应该处理了消息";
        
        System.out.println("✓ 完整工作流验证通过");
        System.out.println("  - 综合任务响应长度: " + response.getContent().length() + " 字符");
        System.out.println("  - 最终系统状态: " + finalStatus.get("current_state"));
        System.out.println();
    }
}