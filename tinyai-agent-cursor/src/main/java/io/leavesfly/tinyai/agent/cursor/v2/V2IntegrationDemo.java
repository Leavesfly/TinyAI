package io.leavesfly.tinyai.agent.cursor.v2;

import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.Context;
import io.leavesfly.tinyai.agent.cursor.v2.model.Memory;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;
import io.leavesfly.tinyai.agent.cursor.v2.tool.Tool;
import io.leavesfly.tinyai.agent.cursor.v2.tool.ToolOrchestrator;
import io.leavesfly.tinyai.agent.cursor.v2.tool.ToolRegistry;
import io.leavesfly.tinyai.agent.cursor.v2.tool.builtin.CodeAnalyzerTool;
import io.leavesfly.tinyai.agent.cursor.v2.tool.builtin.FileReaderTool;
import io.leavesfly.tinyai.agent.cursor.v2.tool.builtin.RAGSearchTool;

import java.util.HashMap;
import java.util.Map;

/**
 * V2功能集成测试示例
 * 展示如何使用阶段3和阶段4实现的功能
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class V2IntegrationDemo {
    
    public static void main(String[] args) {
        System.out.println("=== TinyAI-Cursor V2 集成测试 ===\n");
        
        // 1. 测试记忆管理器
        testMemoryManager();
        
        // 2. 测试上下文引擎
        testContextEngine();
        
        // 3. 测试工具系统
        testToolSystem();
        
        System.out.println("\n=== 所有测试完成 ===");
    }
    
    /**
     * 测试记忆管理器
     */
    private static void testMemoryManager() {
        System.out.println("### 1. 测试记忆管理器 ###\n");
        
        MemoryManager memoryManager = new MemoryManager();
        
        // 添加工作记忆
        Memory workingMemory = Memory.working("wm1", "session123", "当前正在分析User类");
        workingMemory.setImportance(0.8);
        memoryManager.addMemory(workingMemory);
        
        // 添加短期记忆
        Memory shortTermMemory = Memory.shortTerm("stm1", "session123", "用户想要重构authenticate方法");
        shortTermMemory.setImportance(0.7);
        memoryManager.addMemory(shortTermMemory);
        
        // 添加长期记忆
        Memory longTermMemory = Memory.longTerm("ltm1", "project456", "项目使用Spring Boot框架");
        longTermMemory.setImportance(0.9);
        longTermMemory.putMetadata("type", "rule");
        memoryManager.addMemory(longTermMemory);
        
        // 添加语义记忆（带向量）
        double[] embedding = new double[]{0.1, 0.2, 0.3, 0.4, 0.5};
        Memory semanticMemory = Memory.semantic("sm1", "project456", 
            "UserService负责用户认证和授权", embedding);
        semanticMemory.setImportance(0.85);
        memoryManager.addMemory(semanticMemory);
        
        // 检索会话记忆
        System.out.println("会话记忆:");
        var sessionMemories = memoryManager.retrieveSessionMemories("session123", null);
        sessionMemories.forEach(m -> System.out.println("  - " + m.getContent()));
        
        // 检索项目记忆
        System.out.println("\n项目记忆:");
        var projectMemories = memoryManager.retrieveProjectMemories("project456", null);
        projectMemories.forEach(m -> System.out.println("  - " + m.getContent()));
        
        // 统计信息
        MemoryManager.MemoryStats stats = memoryManager.getStats();
        System.out.println("\n记忆统计: " + stats);
        
        System.out.println("\n✅ 记忆管理器测试完成\n");
    }
    
    /**
     * 测试上下文引擎
     */
    private static void testContextEngine() {
        System.out.println("### 2. 测试上下文引擎 ###\n");
        
        MemoryManager memoryManager = new MemoryManager();
        RAGEngine ragEngine = new RAGEngine();
        ContextEngine contextEngine = new ContextEngine(memoryManager, ragEngine);
        
        // 添加项目规则
        contextEngine.addProjectRule("project123", "遵循阿里巴巴Java开发规范");
        contextEngine.addProjectRule("project123", "所有public方法必须添加JavaDoc注释");
        
        // 构建聊天上下文
        String userQuery = "如何优化这段代码的性能？";
        String currentCode = "public void processData(List<String> data) {\n" +
                            "    for (String item : data) {\n" +
                            "        // 处理逻辑\n" +
                            "    }\n" +
                            "}";
        
        Context chatContext = contextEngine.buildChatContext(
            "session123", 
            "project123", 
            userQuery,
            "UserService.java",
            currentCode
        );
        
        System.out.println("构建的聊天上下文:");
        System.out.println("  - 当前文件: " + chatContext.getCurrentFilePath());
        System.out.println("  - 项目规则数: " + chatContext.getProjectRules().size());
        System.out.println("  - 相关代码片段数: " + chatContext.getRelatedCodeSnippets().size());
        
        // 构建代码补全上下文
        Context completionContext = contextEngine.buildCompletionContext(
            "project123",
            "UserController.java",
            "public User getUser(Long id) {\n    ",
            "\n}",
            new Context.CursorPosition(2, 4)
        );
        
        System.out.println("\n构建的补全上下文:");
        System.out.println("  - 当前文件: " + completionContext.getCurrentFilePath());
        System.out.println("  - 光标位置: Line " + completionContext.getCursorPosition().getLine());
        
        // 统计信息
        ContextEngine.ContextStats contextStats = contextEngine.getStats();
        System.out.println("\n上下文统计: " + contextStats);
        
        System.out.println("\n✅ 上下文引擎测试完成\n");
    }
    
    /**
     * 测试工具系统
     */
    private static void testToolSystem() {
        System.out.println("### 3. 测试工具系统 ###\n");
        
        // 创建工具注册表
        ToolRegistry registry = new ToolRegistry();
        
        // 注册内置工具
        registry.register(new CodeAnalyzerTool());
        registry.register(new RAGSearchTool(new RAGEngine()));
        registry.register(new FileReaderTool());
        
        System.out.println("已注册工具: " + registry.getAllToolNames());
        
        // 创建工具编排器
        ToolOrchestrator orchestrator = new ToolOrchestrator(registry);
        
        // 测试代码分析工具
        System.out.println("\n执行代码分析工具:");
        Map<String, Object> analyzerParams = new HashMap<>();
        analyzerParams.put("code", "public class User {\n" +
                                   "    private String name;\n" +
                                   "    public String getName() { return name; }\n" +
                                   "}");
        analyzerParams.put("analysisType", "all");
        
        ToolResult analyzerResult = orchestrator.executeTool("code_analyzer", analyzerParams);
        System.out.println("  成功: " + analyzerResult.isSuccess());
        System.out.println("  耗时: " + analyzerResult.getExecutionTime() + "ms");
        if (analyzerResult.isSuccess()) {
            System.out.println("  结果:\n" + analyzerResult.getResult());
        }
        
        // 测试RAG检索工具
        System.out.println("\n执行RAG检索工具:");
        Map<String, Object> ragParams = new HashMap<>();
        ragParams.put("query", "用户认证相关代码");
        ragParams.put("topK", 3);
        
        ToolResult ragResult = orchestrator.executeTool("rag_search", ragParams);
        System.out.println("  成功: " + ragResult.isSuccess());
        System.out.println("  耗时: " + ragResult.getExecutionTime() + "ms");
        
        // 查看统计信息
        ToolOrchestrator.OrchestratorStats stats = orchestrator.getStats();
        System.out.println("\n工具执行统计: " + stats);
        
        // 查看执行历史
        System.out.println("\n最近执行历史:");
        var history = orchestrator.getExecutionHistory(5);
        history.forEach(record -> System.out.println("  - " + record));
        
        // 按类别获取工具
        System.out.println("\n代码分析类工具:");
        var analysisTools = registry.getToolsByCategory(Tool.ToolCategory.CODE_ANALYSIS);
        analysisTools.forEach(tool -> System.out.println("  - " + tool.getName()));
        
        System.out.println("\n✅ 工具系统测试完成\n");
        
        // 清理资源
        orchestrator.shutdown();
    }
}
