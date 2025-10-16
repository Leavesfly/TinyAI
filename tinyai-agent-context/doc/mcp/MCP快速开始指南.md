# MCP（Model Context Protocol）快速开始指南

## 📖 什么是 MCP？

**Model Context Protocol (MCP)** 是 Anthropic 提出的开放标准协议，用于在 AI 应用与外部数据源、工具之间建立标准化连接。

### 核心概念

1. **Resource（资源）**：只读的数据源，如文件、数据库、API
2. **Tool（工具）**：可执行的功能，如计算器、搜索引擎
3. **Prompt（提示词）**：可复用的提示词模板

## 🚀 5 分钟快速入门

### 步骤 1: 编译项目

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
cd /Users/yefei.yf/Qoder/TinyAI/tinyai-agent-context
mvn clean compile
```

### 步骤 2: 运行演示程序

```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.mcp.MCPDemo"
```

选择演示模式：
- `1` - 基础功能演示
- `2` - 数据分析演示
- `3` - Agent 使用演示
- `4` - 交互式演示

### 步骤 3: 理解核心概念

#### Resource（资源）- 数据提供者

```java
Resource resource = new Resource(
    "file:///docs/readme.md",
    "README",
    ResourceType.FILE
);
resource.setDescription("项目说明文档");
server.registerResource(resource);
server.setResourceContent("file:///docs/readme.md", "# Hello MCP");
```

#### Tool（工具）- 可执行功能

```java
Tool tool = new Tool(
    "search_files",
    "搜索文件内容",
    ToolCategory.SEARCH,
    MCPUtils.createJsonSchema(...),
    args -> searchFiles((String) args.get("keyword"))
);
server.registerTool(tool);
```

#### Prompt（提示词）- 模板复用

```java
Prompt prompt = new Prompt(
    "code_review",
    "代码审查提示词",
    "请审查以下代码：\n{code}\n..."
);
server.registerPrompt(prompt);
```

## 💻 创建第一个 MCP Server

```java
package io.leavesfly.tinyai.agent.mcp;

import java.util.*;

public class MyFirstServer extends MCPServer {
    
    public MyFirstServer() {
        super("My First Server", "1.0.0");
        setupResources();
        setupTools();
    }
    
    private void setupResources() {
        // 注册资源
        Resource resource = new Resource(
            "hello://world",
            "Hello World",
            ResourceType.MEMORY
        );
        registerResource(resource);
        setResourceContent("hello://world", "Hello, MCP!");
    }
    
    private void setupTools() {
        // 注册工具
        registerTool(new Tool(
            "greet",
            "问候工具",
            ToolCategory.CUSTOM,
            MCPUtils.createJsonSchema(
                new HashMap<String, Map<String, Object>>() {{
                    put("name", MCPUtils.createProperty("string", "姓名"));
                }},
                Arrays.asList("name")
            ),
            args -> "你好，" + args.get("name") + "！欢迎使用 MCP！"
        ));
    }
}
```

## 🔌 使用 MCP Client

```java
// 1. 创建服务器和客户端
MyFirstServer server = new MyFirstServer();
MCPClient client = new MCPClient();

// 2. 连接到服务器
client.connect("myserver", server);

// 3. 读取资源
Map<String, Object> content = client.readResource("myserver", "hello://world");
System.out.println(content.get("content"));  // 输出: Hello, MCP!

// 4. 调用工具
Map<String, Object> args = new HashMap<>();
args.put("name", "张三");
Map<String, Object> result = client.callTool("myserver", "greet", args);
System.out.println(result.get("content"));  // 输出: 你好，张三！欢迎使用 MCP！
```

## 🤖 集成到 Agent

```java
// 创建 Agent
MCPEnabledAgent agent = new MCPEnabledAgent("智能助手");

// 连接多个服务器
agent.connectToServer("filesystem", new FileSystemMCPServer());
agent.connectToServer("dataanalysis", new DataAnalysisMCPServer());

// Agent 自动选择合适的工具和资源
String response = agent.processQuery("搜索包含 API 的文档");
System.out.println(response);
```

## 📊 实际应用示例

### 示例 1: 文件系统助手

```java
FileSystemMCPServer fsServer = new FileSystemMCPServer();
MCPClient client = new MCPClient();
client.connect("fs", fsServer);

// 搜索文件
Map<String, Object> args = new HashMap<>();
args.put("keyword", "MCP");
Map<String, Object> result = client.callTool("fs", "search_files", args);

// 列出目录
args.clear();
args.put("path", "/docs");
result = client.callTool("fs", "list_directory", args);
```

### 示例 2: 数据分析助手

```java
DataAnalysisMCPServer daServer = new DataAnalysisMCPServer();
MCPClient client = new MCPClient();
client.connect("data", daServer);

// 查询数据
Map<String, Object> args = new HashMap<>();
args.put("data_uri", "db://users");
args.put("filter_field", "city");
args.put("filter_value", "北京");
Map<String, Object> users = client.callTool("data", "query_data", args);

// 统计分析
args.clear();
args.put("data_uri", "db://sales");
args.put("field", "amount");
Map<String, Object> stats = client.callTool("data", "calculate_statistics", args);
```

## 🧪 运行测试

```bash
# 运行所有 MCP 测试
mvn test -Dtest=MCPServerTest,MCPClientTest

# 运行特定测试
mvn test -Dtest=MCPServerTest#testCallTool
```

## 🎯 常见场景

### 场景 1: 知识库问答

```java
class KnowledgeBaseMCPServer extends MCPServer {
    public KnowledgeBaseMCPServer() {
        super("Knowledge Base", "1.0.0");
        
        // 注册文档资源
        for (Document doc : documents) {
            Resource resource = new Resource(
                "kb://doc/" + doc.getId(),
                doc.getTitle(),
                ResourceType.DOCUMENT
            );
            registerResource(resource);
            setResourceContent(resource.getUri(), doc.getContent());
        }
        
        // 注册语义搜索工具
        registerTool(new Tool(
            "semantic_search",
            "语义搜索文档",
            ToolCategory.SEARCH,
            createSearchSchema(),
            args -> semanticSearch((String) args.get("query"))
        ));
    }
}
```

### 场景 2: 代码助手

```java
class CodeAssistantMCPServer extends MCPServer {
    public CodeAssistantMCPServer() {
        super("Code Assistant", "1.0.0");
        
        // 代码文件资源
        registerCodeFiles();
        
        // 代码分析工具
        registerTool(new Tool(
            "analyze_code",
            "分析代码质量",
            ToolCategory.CUSTOM,
            createCodeAnalysisSchema(),
            args -> analyzeCode((String) args.get("code"))
        ));
    }
}
```

## 📖 深入学习

### 推荐阅读顺序

1. **基础理解**
   - 运行演示程序，体验 MCP 功能
   - 理解三大核心组件（Resource, Tool, Prompt）

2. **核心实现**
   - 阅读 `MCPServer.java` 和 `MCPClient.java`
   - 研究示例实现（FileSystemMCPServer, DataAnalysisMCPServer）

3. **实践应用**
   - 修改示例，添加自定义资源和工具
   - 创建自己的 MCP Server

4. **进阶内容**
   - 阅读 `MCP实现总结文档.md`
   - 探索性能优化和安全控制

## ⚠️ 常见问题

### Q1: 如何添加自定义工具？

```java
registerTool(new Tool(
    "my_tool",
    "自定义工具",
    ToolCategory.CUSTOM,
    MCPUtils.createJsonSchema(
        new HashMap<String, Map<String, Object>>() {{
            put("arg1", MCPUtils.createProperty("string", "参数1"));
        }},
        Arrays.asList("arg1")
    ),
    args -> {
        // 工具实现逻辑
        return "result";
    }
));
```

### Q2: 如何处理大文件资源？

```java
@Override
protected Object loadResourceContent(String uri) {
    // 分块读取或使用流式传输
    if (uri.startsWith("file://")) {
        String path = uri.substring(7);
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            return reader.lines()
                        .limit(1000)  // 限制行数
                        .collect(Collectors.joining("\n"));
        } catch (IOException e) {
            return "Error: " + e.getMessage();
        }
    }
    return super.loadResourceContent(uri);
}
```

### Q3: 如何实现权限控制？

```java
class SecureMCPServer extends MCPServer {
    private Map<String, Set<String>> permissions = new HashMap<>();
    
    @Override
    public ToolResult callTool(ToolCall toolCall) {
        if (!hasPermission(currentUser(), toolCall.getName())) {
            return new ToolResult(
                toolCall.getId(),
                null,
                true,
                "权限不足",
                0.0
            );
        }
        return super.callTool(toolCall);
    }
}
```

## 🎉 恭喜！

你已经掌握了 MCP 的基础知识！现在可以：

✅ 理解 MCP 的核心概念  
✅ 创建自己的 MCP Server  
✅ 使用 MCP Client 访问资源和工具  
✅ 将 MCP 集成到 AI Agent  

**继续探索**：
- 运行 `MCPDemo` 体验完整功能
- 查看 `MCP实现总结文档.md` 深入学习
- 实现你自己的 MCP 应用场景

祝你在 MCP 的世界里玩得开心！🚀

---

**相关文档**：
- `MCP实现总结文档.md` - 完整技术文档
- `../src/main/java/io/leavesfly/tinyai/agent/mcp/` - 源代码
- `../src/test/java/io/leavesfly/tinyai/agent/mcp/` - 测试代码
