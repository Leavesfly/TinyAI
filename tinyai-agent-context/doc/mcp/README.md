# MCP (Model Context Protocol) Java 实现

## 📌 简介

本目录包含 MCP（Model Context Protocol）协议的完整 Java 实现。MCP 是 Anthropic 提出的开放标准协议，用于在 AI 应用与外部数据源、工具之间建立标准化连接。

## 🏗️ 架构概览

```
mcp/
├── 核心数据结构
│   ├── ResourceType.java           # 资源类型枚举
│   ├── ToolCategory.java           # 工具类别枚举
│   ├── Resource.java               # 资源定义
│   ├── ResourceContent.java        # 资源内容
│   ├── Tool.java                   # 工具定义
│   ├── ToolCall.java               # 工具调用请求
│   ├── ToolResult.java             # 工具调用结果
│   ├── Prompt.java                 # 提示词模板
│   ├── MCPRequest.java             # MCP 请求
│   └── MCPResponse.java            # MCP 响应
│
├── 核心服务
│   ├── MCPServer.java              # MCP 服务器
│   ├── MCPClient.java              # MCP 客户端
│   └── MCPUtils.java               # 工具辅助类
│
├── 示例实现
│   ├── FileSystemMCPServer.java    # 文件系统示例
│   ├── DataAnalysisMCPServer.java  # 数据分析示例
│   ├── MCPEnabledAgent.java        # MCP Agent
│   └── MCPDemo.java                # 完整演示程序
│
└── README.md                       # 本文件
```

## ✨ 核心特性

- ✅ **完整的 MCP 协议实现**：Resource、Tool、Prompt 三大组件
- ✅ **零第三方依赖**：仅使用 Java 标准库
- ✅ **JSON-RPC 2.0 支持**：标准化的请求/响应协议
- ✅ **示例丰富**：文件系统、数据分析等实用示例
- ✅ **Agent 集成**：展示如何集成到 AI Agent 系统
- ✅ **完整测试**：28 个单元测试全部通过

## 🚀 快速开始

### 1. 创建 MCP Server

```java
public class MyMCPServer extends MCPServer {
    public MyMCPServer() {
        super("My Server", "1.0.0");
        
        // 注册资源
        Resource resource = new Resource("test://data", "Test Data", ResourceType.MEMORY);
        registerResource(resource);
        setResourceContent("test://data", "Hello MCP!");
        
        // 注册工具
        registerTool(new Tool(
            "echo",
            "回显工具",
            ToolCategory.CUSTOM,
            MCPUtils.createJsonSchema(...),
            args -> args.get("message")
        ));
    }
}
```

### 2. 使用 MCP Client

```java
// 创建服务器和客户端
MyMCPServer server = new MyMCPServer();
MCPClient client = new MCPClient();

// 连接
client.connect("myserver", server);

// 调用工具
Map<String, Object> args = new HashMap<>();
args.put("message", "Hello");
Map<String, Object> result = client.callTool("myserver", "echo", args);
```

### 3. 集成到 Agent

```java
MCPEnabledAgent agent = new MCPEnabledAgent("智能助手");
agent.connectToServer("myserver", server);

String response = agent.processQuery("你好");
```

## 📊 组件说明

### Resource（资源）

资源提供只读的上下文数据：

```java
Resource resource = new Resource(
    "file:///docs/readme.md",  // URI
    "README",                   // 名称
    ResourceType.FILE           // 类型
);
resource.setMimeType("text/markdown");
```

支持的资源类型：
- `FILE` - 文件资源
- `DATABASE` - 数据库资源
- `API` - API 资源
- `MEMORY` - 内存资源
- `DOCUMENT` - 文档资源

### Tool（工具）

工具提供可执行的功能：

```java
Tool tool = new Tool(
    "calculate",                    // 名称
    "计算工具",                      // 描述
    ToolCategory.COMPUTATION,       // 类别
    inputSchema,                    // JSON Schema
    args -> performCalculation(args) // 执行函数
);
```

支持的工具类别：
- `COMPUTATION` - 计算类工具
- `SEARCH` - 搜索类工具
- `DATA_ACCESS` - 数据访问工具
- `SYSTEM` - 系统工具
- `CUSTOM` - 自定义工具

### Prompt（提示词）

提示词提供可复用的模板：

```java
Prompt prompt = new Prompt(
    "code_review",
    "代码审查提示词",
    "请审查以下 {language} 代码：\n{code}\n..."
);

// 渲染
Map<String, Object> params = new HashMap<>();
params.put("language", "Java");
params.put("code", "public class ...");
String rendered = prompt.render(params);
```

## 🎯 示例场景

### 文件系统管理

`FileSystemMCPServer` 提供：
- 文件资源访问
- 文件内容搜索
- 目录列表查询
- 文件分析提示词

### 数据分析

`DataAnalysisMCPServer` 提供：
- 数据库资源访问
- 统计计算工具
- 数据查询工具
- 分析报告模板

## 📖 API 参考

### MCPServer

主要方法：
- `registerResource(Resource)` - 注册资源
- `registerTool(Tool)` - 注册工具
- `registerPrompt(Prompt)` - 注册提示词
- `handleRequest(MCPRequest)` - 处理请求

### MCPClient

主要方法：
- `connect(String, MCPServer)` - 连接服务器
- `listResources(String)` - 列出资源
- `readResource(String, String)` - 读取资源
- `listTools(String)` - 列出工具
- `callTool(String, String, Map)` - 调用工具
- `getPrompt(String, String, Map)` - 获取提示词

## 🧪 测试

运行测试：

```bash
mvn test -Dtest=MCPServerTest,MCPClientTest
```

测试覆盖：
- 服务器初始化
- 资源管理
- 工具调用
- 提示词渲染
- 客户端连接
- 错误处理

## 📚 相关文档

- [`../../doc/MCP实现总结文档.md`](../../doc/MCP实现总结文档.md) - 完整技术文档
- [`../../doc/MCP快速开始指南.md`](../../doc/MCP快速开始指南.md) - 快速入门指南
- [`../../doc/mcp/`](../../doc/mcp/) - Python 参考实现和文档

## 🔗 外部资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP GitHub](https://github.com/modelcontextprotocol)

## 👨‍💻 作者

山泽 - TinyAI 项目

## 📄 许可

本项目采用 TinyAI 项目相同的许可证。

---

**Note**: 本实现基于 Python 版本移植，遵循 MCP 协议规范，适用于 Java 生态系统。
