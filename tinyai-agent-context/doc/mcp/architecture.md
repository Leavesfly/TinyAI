# MCP 架构可视化

## 🏗️ 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Application                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Application Layer                         │ │
│  │  - 用户交互界面                                             │ │
│  │  - 业务逻辑处理                                             │ │
│  │  - 响应生成                                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     Agent Layer                             │ │
│  │  ┌──────────────┬──────────────┬──────────────────────┐    │ │
│  │  │  ReAct Agent │ Planning     │ Multi-Agent System   │    │ │
│  │  │              │ Agent        │                      │    │ │
│  │  └──────────────┴──────────────┴──────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   MCP Client SDK                            │ │
│  │  - Resource 访问接口                                        │ │
│  │  - Tool 调用接口                                            │ │
│  │  - Prompt 获取接口                                          │ │
│  │  - 连接管理                                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ JSON-RPC 2.0 Protocol
                             │ (HTTP / WebSocket / Stdio)
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      MCP Server Framework                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              MCP Protocol Handler                           │ │
│  │  - 请求解析                                                 │ │
│  │  - 方法路由                                                 │ │
│  │  - 响应封装                                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌──────────────┬─────────────────┬─────────────────────────┐  │
│  │   Resource   │      Tool       │       Prompt            │  │
│  │   Manager    │    Registry     │      Manager            │  │
│  │              │                 │                         │  │
│  │  - 资源注册  │  - 工具注册     │  - 提示词注册           │  │
│  │  - 内容缓存  │  - 参数验证     │  - 模板渲染             │  │
│  │  - 访问控制  │  - 执行管理     │  - 参数注入             │  │
│  └──────────────┴─────────────────┴─────────────────────────┘  │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              External Data & Services Layer                 │ │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │ │
│  │  │ File     │ Database │ API      │ Search   │ Custom   │  │ │
│  │  │ System   │          │ Services │ Engine   │ Services │  │ │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 数据流程图

### 1. Resource 读取流程

```
User Query
    ↓
Agent 分析请求
    ↓
MCP Client.read_resource(uri)
    ↓
JSON-RPC Request
    {
      "method": "resources/read",
      "params": {"uri": "file:///doc.md"}
    }
    ↓
MCP Server 接收请求
    ↓
Resource Manager
    ├─ 检查权限
    ├─ 查找资源
    ├─ 读取内容
    └─ 返回数据
    ↓
JSON-RPC Response
    {
      "result": {
        "uri": "file:///doc.md",
        "content": "...",
        "mimeType": "text/markdown"
      }
    }
    ↓
MCP Client 接收响应
    ↓
Agent 处理内容
    ↓
返回给用户
```

### 2. Tool 调用流程

```
User Request: "计算 10 + 5"
    ↓
Agent 识别需要计算
    ↓
MCP Client.call_tool("calculator", {
    "operation": "add",
    "a": 10,
    "b": 5
})
    ↓
JSON-RPC Request
    {
      "method": "tools/call",
      "params": {
        "name": "calculator",
        "arguments": {"operation": "add", "a": 10, "b": 5}
      }
    }
    ↓
MCP Server 接收请求
    ↓
Tool Registry
    ├─ 查找工具
    ├─ 验证参数 (JSON Schema)
    ├─ 执行函数
    │   calculator(operation="add", a=10, b=5)
    │       ↓
    │   return {"result": 15}
    └─ 封装结果
    ↓
JSON-RPC Response
    {
      "result": {
        "callId": "...",
        "content": {"result": 15},
        "isError": false,
        "executionTime": 0.001
      }
    }
    ↓
MCP Client 接收响应
    ↓
Agent 整合结果
    ↓
返回给用户: "计算结果是 15"
```

### 3. Prompt 使用流程

```
Agent 需要生成代码审查提示
    ↓
MCP Client.get_prompt("code_review", 
    language="python",
    code="def hello(): print('hi')"
)
    ↓
JSON-RPC Request
    {
      "method": "prompts/get",
      "params": {
        "name": "code_review",
        "arguments": {
          "language": "python",
          "code": "def hello(): print('hi')"
        }
      }
    }
    ↓
MCP Server 接收请求
    ↓
Prompt Manager
    ├─ 查找模板
    ├─ 渲染模板
    │   template.format(
    │       language="python",
    │       code="def hello(): print('hi')"
    │   )
    └─ 返回结果
    ↓
JSON-RPC Response
    {
      "result": {
        "prompt": "请审查以下 python 代码：\n\n```python\ndef hello(): print('hi')\n```\n..."
      }
    }
    ↓
MCP Client 接收响应
    ↓
Agent 使用提示词发送给 LLM
    ↓
返回审查结果
```

## 🎯 组件交互图

### Resource 组件

```
┌──────────────────────────────────────┐
│        Resource Component             │
├──────────────────────────────────────┤
│  Resource Definition                  │
│  ┌──────────────────────────────┐    │
│  │ uri: "file:///path/to/file"  │    │
│  │ name: "document.pdf"         │    │
│  │ type: FILE                   │    │
│  │ description: "..."           │    │
│  │ mimeType: "application/pdf"  │    │
│  │ metadata: {...}              │    │
│  └──────────────────────────────┘    │
│               ↓                       │
│  Resource Content                     │
│  ┌──────────────────────────────┐    │
│  │ uri: "file:///path/to/file"  │    │
│  │ content: <binary or text>    │    │
│  │ mimeType: "application/pdf"  │    │
│  └──────────────────────────────┘    │
│               ↓                       │
│  Storage Layer                        │
│  ┌───────┬──────────┬──────────┐     │
│  │ Cache │ File I/O │ Database │     │
│  └───────┴──────────┴──────────┘     │
└──────────────────────────────────────┘
```

### Tool 组件

```
┌──────────────────────────────────────┐
│         Tool Component                │
├──────────────────────────────────────┤
│  Tool Definition                      │
│  ┌──────────────────────────────┐    │
│  │ name: "calculator"           │    │
│  │ description: "..."           │    │
│  │ category: COMPUTATION        │    │
│  │ inputSchema: {               │    │
│  │   type: "object",            │    │
│  │   properties: {...},         │    │
│  │   required: [...]            │    │
│  │ }                            │    │
│  │ function: calc_func          │    │
│  └──────────────────────────────┘    │
│               ↓                       │
│  Tool Call                            │
│  ┌──────────────────────────────┐    │
│  │ id: "call-123"               │    │
│  │ name: "calculator"           │    │
│  │ arguments: {                 │    │
│  │   operation: "add",          │    │
│  │   a: 10, b: 5                │    │
│  │ }                            │    │
│  └──────────────────────────────┘    │
│               ↓                       │
│  Validation & Execution               │
│  ┌───────────────────────────────┐   │
│  │ 1. Validate with JSON Schema  │   │
│  │ 2. Execute function           │   │
│  │ 3. Capture result/error       │   │
│  │ 4. Measure execution time     │   │
│  └───────────────────────────────┘   │
│               ↓                       │
│  Tool Result                          │
│  ┌──────────────────────────────┐    │
│  │ callId: "call-123"           │    │
│  │ content: {result: 15}        │    │
│  │ isError: false               │    │
│  │ executionTime: 0.001         │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
```

### Prompt 组件

```
┌──────────────────────────────────────┐
│        Prompt Component               │
├──────────────────────────────────────┤
│  Prompt Definition                    │
│  ┌──────────────────────────────┐    │
│  │ name: "code_review"          │    │
│  │ description: "..."           │    │
│  │ template: """                │    │
│  │   Review {language} code:    │    │
│  │   ```{language}              │    │
│  │   {code}                     │    │
│  │   ```                        │    │
│  │   ...                        │    │
│  │ """                          │    │
│  │ arguments: [                 │    │
│  │   {name: "language", ...},   │    │
│  │   {name: "code", ...}        │    │
│  │ ]                            │    │
│  └──────────────────────────────┘    │
│               ↓                       │
│  Rendering                            │
│  ┌───────────────────────────────┐   │
│  │ 1. Validate arguments         │   │
│  │ 2. Substitute placeholders    │   │
│  │ 3. Apply formatting           │   │
│  │ 4. Return rendered text       │   │
│  └───────────────────────────────┘   │
│               ↓                       │
│  Rendered Prompt                      │
│  ┌──────────────────────────────┐    │
│  │ "Review python code:         │    │
│  │  ```python                   │    │
│  │  def hello(): ...            │    │
│  │  ```                         │    │
│  │  ..."                        │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
```

## 🔐 安全架构

```
┌─────────────────────────────────────────┐
│         Security Layer                   │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────────────────────────┐     │
│  │     Authentication             │     │
│  │  - API Key / Token             │     │
│  │  - OAuth 2.0                   │     │
│  │  - Certificate-based           │     │
│  └────────────────────────────────┘     │
│                ↓                         │
│  ┌────────────────────────────────┐     │
│  │     Authorization              │     │
│  │  - Role-based (RBAC)           │     │
│  │  - Resource-level permissions  │     │
│  │  - Tool execution permissions  │     │
│  └────────────────────────────────┘     │
│                ↓                         │
│  ┌────────────────────────────────┐     │
│  │     Input Validation           │     │
│  │  - JSON Schema validation      │     │
│  │  - SQL injection prevention    │     │
│  │  - Path traversal protection   │     │
│  │  - XSS prevention              │     │
│  └────────────────────────────────┘     │
│                ↓                         │
│  ┌────────────────────────────────┐     │
│  │     Audit & Logging            │     │
│  │  - Request logging             │     │
│  │  - Sensitive operation tracking│     │
│  │  - Error logging               │     │
│  │  - Performance metrics         │     │
│  └────────────────────────────────┘     │
│                                          │
└─────────────────────────────────────────┘
```

## ⚡ 性能优化架构

```
┌─────────────────────────────────────────┐
│      Performance Layer                  │
├─────────────────────────────────────────┤
│                                         │
│  ┌────────────────────────────────┐     │
│  │     Caching Strategy           │     │
│  │  ┌──────────┬──────────────┐   │     │
│  │  │ L1 Cache │ Memory       │   │     │
│  │  │          │ (Fast)       │   │     │
│  │  ├──────────┼──────────────┤   │     │
│  │  │ L2 Cache │ Redis        │   │     │
│  │  │          │ (Shared)     │   │     │
│  │  ├──────────┼──────────────┤   │     │
│  │  │ L3 Cache │ Disk         │   │     │
│  │  │          │ (Persistent) │   │     │
│  │  └──────────┴──────────────┘   │     │
│  └────────────────────────────────┘     │
│                ↓                        │
│  ┌────────────────────────────────┐     │
│  │     Connection Pooling         │     │
│  │  - Reuse MCP connections       │     │
│  │  - Connection lifecycle mgmt   │     │
│  │  - Health checks               │     │
│  └────────────────────────────────┘     │
│                ↓                        │
│  ┌────────────────────────────────┐     │
│  │     Async Processing           │     │
│  │  - Non-blocking I/O            │     │
│  │  - Concurrent tool calls       │     │
│  │  - Background tasks            │     │
│  └────────────────────────────────┘     │
│                ↓                        │
│  ┌────────────────────────────────┐     │
│  │     Request Batching           │     │
│  │  - Batch resource reads        │     │
│  │  - Bulk tool calls             │     │
│  │  - Reduce round trips          │     │
│  └────────────────────────────────┘     │
│                                         │
└─────────────────────────────────────────┘
```

## 🌐 分布式架构（高级）

```
┌──────────────────────────────────────────────────────────┐
│              Load Balancer                                │
│          (HAProxy / Nginx)                                │
└─────────────────┬────────────────────────────────────────┘
                  │
      ┌───────────┴───────────┬───────────┐
      ↓                       ↓           ↓
┌──────────┐           ┌──────────┐  ┌──────────┐
│ MCP      │           │ MCP      │  │ MCP      │
│ Server 1 │           │ Server 2 │  │ Server N │
│          │           │          │  │          │
│ ┌──────┐ │           │ ┌──────┐ │  │ ┌──────┐ │
│ │Cache │ │           │ │Cache │ │  │ │Cache │ │
│ └──────┘ │           │ └──────┘ │  │ └──────┘ │
└────┬─────┘           └────┬─────┘  └────┬─────┘
     │                      │             │
     └──────────┬───────────┴─────────────┘
                ↓
      ┌─────────────────┐
      │  Shared Cache   │
      │  (Redis Cluster)│
      └─────────────────┘
                ↓
      ┌─────────────────┐
      │   Data Layer    │
      │  - Databases    │
      │  - File Systems │
      │  - APIs         │
      └─────────────────┘
```

---


