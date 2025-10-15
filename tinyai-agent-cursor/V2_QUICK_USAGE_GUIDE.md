# TinyAI-Cursor V2 快速使用指南

## 🚀 概述

TinyAI-Cursor V2 阶段3-4已完成，现在可以使用：
- ✅ 记忆管理系统
- ✅ RAG检索引擎  
- ✅ 上下文智能编排
- ✅ 工具调用框架

本指南将帮助您快速上手这些新功能。

---

## 📦 核心组件

### 1. 记忆管理器（MemoryManager）

管理AI助手的各类记忆，支持4种记忆类型。

```java
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.model.Memory;

// 创建记忆管理器
MemoryManager memoryManager = new MemoryManager();

// 添加工作记忆（当前会话的临时状态）
Memory workingMemory = Memory.working(
    "wm1", 
    "session123", 
    "正在分析UserService类的性能问题"
);
workingMemory.setImportance(0.8);
memoryManager.addMemory(workingMemory);

// 添加短期记忆（单次会话的重要信息）
Memory shortTermMemory = Memory.shortTerm(
    "stm1",
    "session123",
    "用户想要优化数据库查询性能"
);
memoryManager.addMemory(shortTermMemory);

// 添加长期记忆（项目规则、用户偏好）
Memory longTermMemory = Memory.longTerm(
    "ltm1",
    "project456",
    "项目使用Spring Boot + MyBatis技术栈"
);
longTermMemory.putMetadata("type", "tech_stack");
memoryManager.addMemory(longTermMemory);

// 添加语义记忆（代码知识库）
double[] embedding = getEmbedding("UserService负责用户认证");
Memory semanticMemory = Memory.semantic(
    "sm1",
    "project456",
    "UserService负责用户认证和授权",
    embedding
);
memoryManager.addMemory(semanticMemory);

// 检索会话记忆
List<Memory> sessionMemories = memoryManager.retrieveSessionMemories(
    "session123", 
    null  // null表示所有类型
);

// 语义检索
double[] queryEmbedding = getEmbedding("如何实现用户登录");
List<Memory> similarMemories = memoryManager.retrieveSimilarMemories(
    queryEmbedding,
    5,      // Top-5
    0.7     // 相似度阈值
);

// 清理过期记忆
int cleaned = memoryManager.cleanupExpiredMemories();

// 查看统计
MemoryManager.MemoryStats stats = memoryManager.getStats();
System.out.println("记忆统计: " + stats);
```

---

### 2. RAG检索引擎（RAGEngine）

从代码库中检索相关代码片段。

```java
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine.CodeFile;
import io.leavesfly.tinyai.agent.cursor.v2.model.Context.CodeSnippet;

// 创建RAG引擎
RAGEngine ragEngine = new RAGEngine();

// 索引代码库
List<CodeFile> codeFiles = new ArrayList<>();
codeFiles.add(new CodeFile(
    "src/UserService.java",
    "public class UserService { ... }"
));
ragEngine.indexCodebase("project123", codeFiles);

// 语义检索
List<CodeSnippet> snippets = ragEngine.semanticSearch(
    "用户认证相关代码",  // 查询
    5,                  // Top-5
    "project123"        // 项目ID
);

for (CodeSnippet snippet : snippets) {
    System.out.println("文件: " + snippet.getFilePath());
    System.out.println("相似度: " + snippet.getScore());
    System.out.println("内容: " + snippet.getContent());
}

// 精确匹配
List<String> keywords = Arrays.asList("authenticate", "login", "password");
List<CodeSnippet> exactMatches = ragEngine.exactSearch(
    keywords,
    5,
    "project123"
);

// 混合检索（推荐）
List<CodeSnippet> hybridResults = ragEngine.hybridSearch(
    "用户登录功能",      // 查询
    keywords,           // 关键词
    5,                  // Top-5
    "project123"        // 项目ID
);

// 查找相关代码
List<CodeSnippet> relatedCode = ragEngine.findRelated(
    "UserController.java",
    currentCodeContent,
    3  // Top-3
);

// 统计信息
RAGEngine.RAGStats stats = ragEngine.getStats();
System.out.println("RAG统计: " + stats);
```

---

### 3. 上下文引擎（ContextEngine）

智能编排上下文，为LLM提供最优输入。

```java
import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.Context;

// 创建上下文引擎
ContextEngine contextEngine = new ContextEngine(
    memoryManager,  // 记忆管理器
    ragEngine       // RAG引擎
);

// 场景1: 构建聊天上下文
Context chatContext = contextEngine.buildChatContext(
    "session123",               // 会话ID
    "project456",               // 项目ID
    "如何优化这段代码？",       // 用户查询
    "UserService.java",         // 当前文件
    currentFileContent          // 当前代码
);

// 场景2: 构建代码补全上下文
Context completionContext = contextEngine.buildCompletionContext(
    "project456",                   // 项目ID
    "UserController.java",          // 文件路径
    "public User getUser(Long id) {\n    ",  // 光标前代码
    "\n}",                          // 光标后代码
    new Context.CursorPosition(2, 4)  // 光标位置
);

// 场景3: 构建代码分析上下文
Context analysisContext = contextEngine.buildAnalysisContext(
    "project456",       // 项目ID
    targetCode,         // 目标代码
    "review"            // 分析类型: review/refactor/debug
);

// 场景4: 构建语义检索上下文
Context searchContext = contextEngine.buildSemanticSearchContext(
    "project456",
    "用户认证流程",
    queryEmbedding  // 可选
);

// 添加项目规则
contextEngine.addProjectRule("project456", "遵循阿里巴巴Java开发规范");
contextEngine.addProjectRule("project456", "所有public方法必须有JavaDoc");

// 动态更新上下文
List<CodeSnippet> newSnippets = ragEngine.semanticSearch("...", 3, "project456");
List<Message> newMessages = Arrays.asList(Message.user("补充问题"));
contextEngine.updateContext(chatContext, newSnippets, newMessages);

// 保存消息到记忆
Message userMessage = Message.user("请帮我重构这段代码");
contextEngine.saveMessageToMemory("session123", userMessage, 0.8);

// 配置
contextEngine.setMaxContextTokens(8000);  // 最大Token数
contextEngine.setAvgTokensPerSnippet(200);  // 每个代码片段平均Token数

// 统计信息
ContextEngine.ContextStats stats = contextEngine.getStats();
System.out.println("上下文统计: " + stats);
```

---

### 4. 工具调用系统

扩展AI助手的能力，支持代码分析、文件读取、RAG检索等。

```java
import io.leavesfly.tinyai.agent.cursor.v2.tool.*;
import io.leavesfly.tinyai.agent.cursor.v2.tool.builtin.*;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;

// 创建工具注册表
ToolRegistry registry = new ToolRegistry();

// 注册内置工具
registry.register(new CodeAnalyzerTool());
registry.register(new RAGSearchTool(ragEngine));
registry.register(new FileReaderTool());

// 创建工具编排器
ToolOrchestrator orchestrator = new ToolOrchestrator(registry);

// 执行代码分析工具
Map<String, Object> analyzerParams = new HashMap<>();
analyzerParams.put("code", sourceCode);
analyzerParams.put("analysisType", "all");  // structure/complexity/quality/all

ToolResult result = orchestrator.executeTool("code_analyzer", analyzerParams);
if (result.isSuccess()) {
    System.out.println("分析结果:\n" + result.getResult());
} else {
    System.err.println("错误: " + result.getError());
}

// 执行RAG检索工具
Map<String, Object> ragParams = new HashMap<>();
ragParams.put("query", "用户认证相关代码");
ragParams.put("topK", 5);
ragParams.put("projectId", "project456");

ToolResult ragResult = orchestrator.executeTool("rag_search", ragParams);

// 执行文件读取工具
Map<String, Object> fileParams = new HashMap<>();
fileParams.put("filePath", "/path/to/User.java");
fileParams.put("maxLines", 100);  // 可选

ToolResult fileResult = orchestrator.executeTool("file_reader", fileParams);

// 批量执行工具（自动并行）
List<ToolCall> toolCalls = Arrays.asList(
    createToolCall("code_analyzer", analyzerParams),
    createToolCall("rag_search", ragParams)
);
List<ToolResult> results = orchestrator.executeTools(toolCalls);

// 将结果转换为消息（供LLM使用）
List<Message> toolMessages = orchestrator.resultsToMessages(results);

// 获取可用工具定义（传递给LLM）
List<ToolDefinition> toolDefinitions = orchestrator.getAvailableTools();

// 按类别获取工具
List<ToolDefinition> analysisTools = orchestrator.getAvailableTools(
    Tool.ToolCategory.CODE_ANALYSIS
);

// 查看执行历史
List<ToolOrchestrator.ToolExecutionRecord> history = 
    orchestrator.getExecutionHistory(10);
for (var record : history) {
    System.out.println(record);
}

// 统计信息
ToolOrchestrator.OrchestratorStats stats = orchestrator.getStats();
System.out.println("工具执行统计: " + stats);

// 配置
orchestrator.setToolExecutionTimeout(30);  // 超时时间（秒）
orchestrator.setEnableParallelExecution(true);  // 启用并行执行

// 清理资源
orchestrator.shutdown();
```

---

## 🛠️ 实现自定义工具

```java
import io.leavesfly.tinyai.agent.cursor.v2.tool.Tool;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;

public class MyCustomTool implements Tool {
    
    @Override
    public String getName() {
        return "my_custom_tool";
    }
    
    @Override
    public String getDescription() {
        return "My custom tool description";
    }
    
    @Override
    public ToolDefinition getDefinition() {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("type", "object");
        
        Map<String, Object> properties = new HashMap<>();
        Map<String, Object> inputParam = new HashMap<>();
        inputParam.put("type", "string");
        inputParam.put("description", "Input parameter");
        properties.put("input", inputParam);
        
        parameters.put("properties", properties);
        parameters.put("required", new String[]{"input"});
        
        return ToolDefinition.create(getName(), getDescription(), parameters);
    }
    
    @Override
    public ToolResult execute(Map<String, Object> parameters) {
        long startTime = System.currentTimeMillis();
        
        try {
            String input = (String) parameters.get("input");
            
            // 执行自定义逻辑
            String output = processInput(input);
            
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.success(getName(), output);
            result.setExecutionTime(executionTime);
            return result;
            
        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.failure(getName(), e.getMessage());
            result.setExecutionTime(executionTime);
            return result;
        }
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        return parameters != null && parameters.containsKey("input");
    }
    
    @Override
    public ToolCategory getCategory() {
        return ToolCategory.GENERAL;
    }
    
    private String processInput(String input) {
        // 自定义处理逻辑
        return "Processed: " + input;
    }
}

// 注册自定义工具
registry.register(new MyCustomTool());
```

---

## 🔗 完整工作流示例

```java
public class CursorV2Workflow {
    
    public static void main(String[] args) {
        // 1. 初始化组件
        MemoryManager memoryManager = new MemoryManager();
        RAGEngine ragEngine = new RAGEngine();
        ContextEngine contextEngine = new ContextEngine(memoryManager, ragEngine);
        
        ToolRegistry toolRegistry = new ToolRegistry();
        toolRegistry.register(new CodeAnalyzerTool());
        toolRegistry.register(new RAGSearchTool(ragEngine));
        ToolOrchestrator toolOrchestrator = new ToolOrchestrator(toolRegistry);
        
        // 2. 索引代码库
        List<RAGEngine.CodeFile> codeFiles = loadProjectFiles();
        ragEngine.indexCodebase("myproject", codeFiles);
        
        // 3. 用户发起请求
        String userQuery = "帮我分析这段代码的性能问题";
        String currentFile = "UserService.java";
        String currentCode = loadFileContent(currentFile);
        
        // 4. 构建上下文
        Context context = contextEngine.buildChatContext(
            "session001",
            "myproject",
            userQuery,
            currentFile,
            currentCode
        );
        
        // 5. 准备LLM请求
        ChatRequest request = ChatRequest.builder()
            .model("deepseek-chat")
            .addSystemMessage(context.buildSystemPrompt())
            .addUserMessage(userQuery)
            .tools(toolOrchestrator.getAvailableTools())
            .build();
        
        // 6. 调用LLM
        ChatResponse response = llmGateway.chat(request);
        
        // 7. 处理工具调用（如果LLM请求使用工具）
        if (response.hasToolCalls()) {
            List<ToolResult> toolResults = toolOrchestrator.executeTools(
                response.getToolCalls()
            );
            
            // 将工具结果转换为消息
            List<Message> toolMessages = toolOrchestrator.resultsToMessages(toolResults);
            
            // 再次调用LLM，提供工具结果
            ChatRequest followUpRequest = ChatRequest.builder()
                .model("deepseek-chat")
                .messages(request.getMessages())
                .addMessage(response.toAssistantMessage())
                .addMessages(toolMessages)
                .build();
            
            response = llmGateway.chat(followUpRequest);
        }
        
        // 8. 保存会话到记忆
        contextEngine.saveMessageToMemory("session001", 
            Message.user(userQuery), 0.8);
        contextEngine.saveMessageToMemory("session001", 
            Message.assistant(response.getContent()), 0.7);
        
        // 9. 返回结果
        System.out.println("AI响应: " + response.getContent());
        
        // 10. 清理资源
        toolOrchestrator.shutdown();
    }
}
```

---

## 🧪 运行测试

```bash
# 编译项目
cd tinyai-agent-cursor
mvn clean compile

# 运行集成测试
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.cursor.v2.V2IntegrationDemo"
```

---

## 📚 相关文档

- [V2实施进度](V2_IMPLEMENTATION_PROGRESS.md) - 完整进度报告
- [阶段3-4总结](V2_STAGE3_4_SUMMARY.md) - 详细实施总结
- [API参考文档](doc/V2-API-Reference.md) - 完整API文档

---

## ❓ 常见问题

### Q1: 如何控制上下文长度？
```java
contextEngine.setMaxContextTokens(8000);  // 设置最大Token数
```

### Q2: 如何自定义记忆过期时间？
```java
memoryManager.setWorkingMemoryTtl(60 * 60 * 1000L);  // 1小时
memoryManager.setShortTermMemoryTtl(4 * 60 * 60 * 1000L);  // 4小时
```

### Q3: 如何禁用工具并行执行？
```java
orchestrator.setEnableParallelExecution(false);
```

### Q4: 如何调整工具执行超时？
```java
orchestrator.setToolExecutionTimeout(60);  // 60秒
```

### Q5: 如何清除所有记忆？
```java
memoryManager.clearSessionMemory("session123");  // 清除会话记忆
memoryManager.clearProjectMemory("project456");  // 清除项目记忆
```

---

## 🎯 最佳实践

1. **记忆管理**:
   - 工作记忆用于当前对话轮次的临时状态
   - 短期记忆用于单次会话的重要信息
   - 长期记忆用于项目规则和用户偏好
   - 语义记忆用于代码知识库

2. **上下文优化**:
   - 根据场景选择合适的上下文构建方法
   - 定期清理过期记忆
   - 合理设置最大Token数

3. **工具使用**:
   - 优先使用混合检索策略
   - 启用工具并行执行提升性能
   - 定期查看执行历史和统计

4. **性能优化**:
   - 缓存项目规则
   - 批量添加记忆
   - 合理使用语义检索

---

**祝您使用愉快！** 🚀

如有问题，请参考详细文档或提交Issue。
