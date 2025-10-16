# MCP（Model Context Protocol）Java 实现总结文档

## 📌 项目概述

本项目是基于 Anthropic 提出的 Model Context Protocol（MCP）协议的 Java 语言实现，参考了 Python 版本的设计和实现。MCP 是一个开放标准协议，旨在解决 AI 应用与外部数据源、工具之间的集成问题，提供统一的接口规范。

## 🎯 实现目标

1. **完整性**：实现 MCP 协议的三大核心组件（Resource、Tool、Prompt）
2. **简洁性**：减少第三方依赖，优先复用 TinyAI 工程现有组件
3. **可扩展性**：提供清晰的接口和示例，易于扩展
4. **实用性**：包含完整的演示程序和测试用例

## 🏗️ 技术架构

### 核心组件

```
io.leavesfly.tinyai.agent.mcp/
├── 核心数据结构
│   ├── ResourceType.java           # 资源类型枚举
│   ├── ToolCategory.java           # 工具类别枚举
│   ├── Resource.java               # 资源定义
│   ├── ResourceContent.java        # 资源内容
│   ├── Tool.java                   # 工具定义
│   ├── ToolCall.java               # 工具调用
│   ├── ToolResult.java             # 工具结果
│   ├── Prompt.java                 # 提示词模板
│   ├── MCPRequest.java             # MCP 请求（JSON-RPC 2.0）
│   └── MCPResponse.java            # MCP 响应（JSON-RPC 2.0）
│
├── 核心服务
│   ├── MCPServer.java              # MCP 服务器核心类
│   ├── MCPClient.java              # MCP 客户端核心类
│   └── MCPUtils.java               # 工具辅助类
│
├── 示例实现
│   ├── FileSystemMCPServer.java    # 文件系统示例
│   ├── DataAnalysisMCPServer.java  # 数据分析示例
│   ├── MCPEnabledAgent.java        # MCP Agent 示例
│   └── MCPDemo.java                # 完整演示程序
│
└── 测试
    ├── MCPServerTest.java          # 服务器测试
    └── MCPClientTest.java          # 客户端测试
```

### 设计原则

1. **标准化协议**：严格遵循 JSON-RPC 2.0 规范
2. **面向对象设计**：清晰的类层次和职责划分
3. **函数式编程**：工具函数使用 `Function<Map, Object>` 接口
4. **最小依赖**：仅依赖 Java 标准库，无额外第三方库
5. **类型安全**：使用泛型和枚举提高类型安全性

## 🔑 关键设计与实现

### 1. 资源管理（Resource）

**设计思路**：
- 使用 URI 作为资源唯一标识
- 支持多种资源类型（FILE, DATABASE, API, MEMORY, DOCUMENT）
- 资源内容与元数据分离
- 内置缓存机制提升性能

**关键代码**：
```java
public class Resource {
    private String uri;                    // 资源标识
    private String name;                   // 资源名称
    private ResourceType resourceType;     // 资源类型
    private String mimeType;               // MIME 类型
    private Map<String, Object> metadata;  // 元数据
}
```

### 2. 工具调用（Tool）

**设计思路**：
- 使用 `Function<Map<String, Object>, Object>` 作为工具执行函数
- JSON Schema 描述输入参数
- 支持工具分类管理
- 完整的错误处理和执行时间统计

**关键代码**：
```java
public class Tool {
    private String name;                              // 工具名称
    private ToolCategory category;                    // 工具类别
    private Map<String, Object> inputSchema;          // JSON Schema
    private Function<Map<String, Object>, Object> function;  // 执行函数
}
```

### 3. 提示词模板（Prompt）

**设计思路**：
- 简单的模板占位符替换 `{key}`
- 参数定义描述
- 支持模板复用

**关键代码**：
```java
public class Prompt {
    private String template;                      // 模板内容
    private List<Map<String, Object>> arguments;  // 参数定义
    
    public String render(Map<String, Object> params) {
        String result = template;
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            result = result.replace("{" + entry.getKey() + "}", 
                                   entry.getValue().toString());
        }
        return result;
    }
}
```

### 4. JSON-RPC 2.0 协议

**设计思路**：
- 标准的请求/响应结构
- 支持同步调用模式
- 完整的错误码定义（-32601: 方法不存在, -32603: 内部错误）

**关键代码**：
```java
public class MCPRequest {
    private String jsonrpc = "2.0";
    private String id;
    private String method;
    private Map<String, Object> params;
}

public class MCPResponse {
    private String jsonrpc = "2.0";
    private String id;
    private Object result;
    private Map<String, Object> error;
}
```

### 5. 服务器-客户端架构

**MCPServer 核心职责**：
- 注册和管理资源、工具、提示词
- 处理 MCP 请求
- 资源内容缓存
- 动态资源加载（可重写）

**MCPClient 核心职责**：
- 管理服务器连接
- 发送请求并处理响应
- 提供便捷的 API（listResources, callTool, getPrompt 等）

## 💡 实现亮点

### 1. 最小化第三方依赖

本实现完全基于 Java 标准库，没有引入任何第三方依赖（除了单元测试的 JUnit），这使得：
- 部署简单，无需管理复杂依赖
- 与现有 TinyAI 工程无缝集成
- 减少潜在的依赖冲突

### 2. 灵活的工具注册机制

```java
// 使用 Lambda 表达式简化工具注册
server.registerTool(new Tool(
    "search_files",
    "搜索文件",
    ToolCategory.SEARCH,
    schema,
    args -> searchFiles((String) args.get("keyword"))
));
```

### 3. 完整的示例实现

提供了两个完整的示例服务器：

**FileSystemMCPServer**：
- 模拟文件资源管理
- 文件搜索工具
- 目录列表工具
- 文件分析提示词

**DataAnalysisMCPServer**：
- 模拟数据库资源
- 统计计算工具
- 数据查询工具
- 数据分析报告提示词

### 4. Agent 集成示例

`MCPEnabledAgent` 展示了如何将 MCP 集成到 AI Agent 系统中：
- 多服务器连接管理
- 能力发现机制
- 智能意图识别
- 自动工具选择

## 📊 测试覆盖

### 单元测试统计

- **MCPServerTest**: 14 个测试用例
  - 服务器初始化
  - 资源管理（注册、列表、读取）
  - 工具管理（注册、列表、调用）
  - 提示词管理（注册、列表、渲染）
  - RPC 请求处理
  - 错误处理

- **MCPClientTest**: 14 个测试用例
  - 客户端初始化
  - 连接管理
  - 资源操作
  - 工具调用
  - 提示词获取
  - 错误场景

**测试结果**：28/28 通过（100% 通过率）

## 🔧 关键技术细节

### 1. 资源内容缓存

```java
protected Map<String, Object> resourceContentCache;

protected Object loadResourceContent(String uri) {
    // 子类可重写此方法实现动态加载
    return "资源 " + uri + " 的内容";
}
```

### 2. 工具执行时间统计

```java
public ToolResult callTool(ToolCall toolCall) {
    long startTime = System.currentTimeMillis();
    try {
        Object result = tool.getFunction().apply(toolCall.getArguments());
        double executionTime = (System.currentTimeMillis() - startTime) / 1000.0;
        return new ToolResult(toolCall.getId(), result, false, null, executionTime);
    } catch (Exception e) {
        // 错误处理...
    }
}
```

### 3. JSON Schema 辅助方法

```java
public static Map<String, Object> createJsonSchema(
    Map<String, Map<String, Object>> properties, 
    List<String> required) {
    Map<String, Object> schema = new HashMap<>();
    schema.put("type", "object");
    schema.put("properties", properties);
    schema.put("required", required);
    return schema;
}
```

## ⚠️ 注意事项与限制

### 1. 当前限制

1. **协议实现**：
   - 仅实现了同步调用模式，未实现异步/WebSocket 通信
   - 未实现完整的 JSON-RPC 2.0 批量请求

2. **功能限制**：
   - 资源加载为内存模式，未实现实际文件/数据库访问
   - 工具参数验证仅通过 JSON Schema 描述，未实现实际验证逻辑

3. **性能优化**：
   - 资源缓存为简单 HashMap，未实现 LRU/LFU 策略
   - 未实现连接池和请求批处理

### 2. 安全考虑

在实际生产环境中，需要添加：
- 权限验证机制
- 输入参数验证
- 资源访问控制
- 审计日志
- 速率限制

### 3. 扩展建议

1. **协议扩展**：
   - 实现 WebSocket 传输层
   - 支持流式响应
   - 添加心跳机制

2. **功能增强**：
   - 实现真实的文件系统访问
   - 集成数据库连接
   - 添加更多内置工具

3. **性能优化**：
   - 实现智能缓存策略
   - 添加连接池
   - 支持并发请求

## 📝 使用示例

### 基础使用

```java
// 1. 创建服务器
FileSystemMCPServer server = new FileSystemMCPServer();

// 2. 创建客户端
MCPClient client = new MCPClient();
client.connect("filesystem", server);

// 3. 列出资源
List<Map<String, Object>> resources = client.listResources("filesystem");

// 4. 调用工具
Map<String, Object> args = new HashMap<>();
args.put("keyword", "API");
Map<String, Object> result = client.callTool("filesystem", "search_files", args);

// 5. 获取提示词
Map<String, Object> promptArgs = new HashMap<>();
promptArgs.put("filename", "test.md");
promptArgs.put("content", "示例内容");
String prompt = client.getPrompt("filesystem", "analyze_file", promptArgs);
```

### Agent 集成

```java
// 创建支持 MCP 的 Agent
MCPEnabledAgent agent = new MCPEnabledAgent("智能助手");

// 连接多个 MCP Server
agent.connectToServer("filesystem", new FileSystemMCPServer());
agent.connectToServer("dataanalysis", new DataAnalysisMCPServer());

// 处理用户查询（Agent 自动选择合适的工具）
String response = agent.processQuery("搜索包含 API 的文档");
```

## 🚀 运行演示

### 编译项目

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
cd /Users/yefei.yf/Qoder/TinyAI/tinyai-agent-context
mvn clean compile
```

### 运行测试

```bash
mvn test -Dtest=MCPServerTest,MCPClientTest
```

### 运行演示程序

```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.mcp.MCPDemo"
```

演示程序提供 4 种模式：
1. 基础功能演示
2. 数据分析演示
3. Agent 使用演示
4. 交互式演示

## 📚 技术文档参考

本实现参考了以下文档：
- `doc/mcp/26_README_MCP.md` - MCP 介绍
- `doc/mcp/26_architecture.md` - 架构设计
- `doc/mcp/26_mcp_core.py` - Python 核心实现
- `doc/mcp/26_mcp_demo.py` - Python 演示代码
- `doc/mcp/26_QUICKSTART.md` - 快速开始指南
- `doc/mcp/26_MCP_SUMMARY.md` - 完整总结

## 🎓 学习价值

通过本实现，可以学习到：

1. **协议设计**：如何设计标准化的协议接口
2. **架构设计**：服务器-客户端架构的实践
3. **函数式编程**：Java 函数式接口的应用
4. **设计模式**：策略模式、工厂模式的使用
5. **测试驱动**：完整的单元测试实践

## 🌟 总结

本项目成功将 Python 版本的 MCP 协议移植到 Java，实现了：

✅ **完整的协议实现**：三大核心组件（Resource、Tool、Prompt）  
✅ **零第三方依赖**：仅使用 Java 标准库  
✅ **可扩展架构**：清晰的接口和示例  
✅ **完整测试覆盖**：28 个测试用例全部通过  
✅ **实用演示程序**：4 种演示模式  
✅ **详细文档**：API 文档和使用指南  

MCP 协议为 AI Agent 与外部世界的交互提供了标准化、可扩展的解决方案。本 Java 实现为 TinyAI 项目提供了强大的上下文管理能力，可以轻松集成文件系统、数据库、API 等各种外部资源和工具。

---

**作者**：山泽  
**日期**：2025-10-16  
**版本**：1.0.0  
**项目**：TinyAI - tinyai-agent-context 模块
