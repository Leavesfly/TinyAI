# TinyAI-Cursor V2 快速开始指南

## 概述

TinyAI-Cursor V2 已完成基础设施层（阶段1）和LLM统一网关层（阶段2）的开发，提供了企业级AI编程助手的核心能力。

## 核心特性

### ✅ 已实现
- **多模型支持**: DeepSeek、Qwen等模型统一接口
- **智能路由**: 自动选择最佳模型适配器
- **降级机制**: 模型不可用时自动切换
- **多级缓存**: L1/L2/L3三级缓存优化性能
- **流式响应**: 支持实时Token流输出
- **工具调用**: 完整的Tool Calling数据结构
- **会话管理**: 完整的会话生命周期管理
- **记忆系统**: 4种记忆类型（工作/短期/长期/语义）

## 快速开始

### 1. 基本使用

```java
import io.leavesfly.tinyai.agent.cursor.v2.adapter.*;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.*;

// 1. 创建适配器注册表
AdapterRegistry registry = new AdapterRegistry();

// 2. 注册模型适配器
DeepSeekAdapter deepSeekAdapter = new DeepSeekAdapter();
deepSeekAdapter.setApiKey("sk-your-api-key");
registry.register(deepSeekAdapter);

// 3. 创建LLM网关
LLMGateway gateway = new LLMGatewayImpl(registry);
gateway.setPreferredModel("deepseek-chat");

// 4. 发送请求
ChatRequest request = ChatRequest.builder()
    .addUserMessage("请解释Java的Stream API")
    .temperature(0.7)
    .maxTokens(500)
    .build();

ChatResponse response = gateway.chat(request);
System.out.println(response.getContent());
```

### 2. 流式对话

```java
gateway.chatStream(request, new StreamCallback() {
    @Override
    public void onToken(String token) {
        System.out.print(token); // 实时输出
    }
    
    @Override
    public void onComplete(ChatResponse response) {
        System.out.println("\n完成");
    }
    
    @Override
    public void onError(Throwable error) {
        error.printStackTrace();
    }
});
```

### 3. 代码补全

```java
String prefix = "public class Calculator {\n    public int add(int a, int b) {";
String completion = gateway.complete(prefix, null, "java", 100);
System.out.println(completion);
```

### 4. 使用工具调用

```java
// 定义工具
Map<String, Object> params = new HashMap<>();
params.put("type", "object");
params.put("properties", Map.of(
    "code", Map.of("type", "string", "description", "要分析的代码")
));
params.put("required", List.of("code"));

ToolDefinition analyzerTool = ToolDefinition.create(
    "code_analyzer",
    "分析代码质量",
    params
);

// 添加到请求
ChatRequest request = ChatRequest.builder()
    .addSystemMessage("你是一个代码助手，可以调用工具分析代码")
    .addUserMessage("分析这段代码: public void test() {}")
    .addTool(analyzerTool)
    .toolChoice("auto")
    .build();

ChatResponse response = gateway.chat(request);

// 检查是否有工具调用
if (response.hasToolCalls()) {
    for (ToolCall toolCall : response.getToolCalls()) {
        System.out.println("调用工具: " + toolCall.getFunction().getName());
        System.out.println("参数: " + toolCall.getFunction().getArguments());
    }
}
```

### 5. 会话管理

```java
import io.leavesfly.tinyai.agent.cursor.v2.infra.storage.SessionStore;

SessionStore sessionStore = new SessionStore();

// 创建会话
Session session = new Session("project-123", "user-456");
session.addMessage(Message.user("你好"));
session.addMessage(Message.assistant("你好！有什么可以帮助你的？"));

// 保存会话
sessionStore.save(session);

// 获取会话
Session retrieved = sessionStore.get(session.getSessionId());

// 获取最近消息
List<Message> recentMessages = session.getRecentMessages(10);

// 清理过期会话
int cleaned = sessionStore.cleanupExpiredSessions();
System.out.println("清理了 " + cleaned + " 个过期会话");
```

### 6. 缓存使用

```java
import io.leavesfly.tinyai.agent.cursor.v2.infra.cache.CacheManager;

CacheManager cache = new CacheManager();

// L1缓存：代码补全结果（5分钟TTL）
String cacheKey = "completion:" + prefix.hashCode();
cache.putL1(cacheKey, completionResult);

// 从缓存获取
Object cached = cache.getL1(cacheKey);
if (cached != null) {
    return (String) cached; // 缓存命中
}

// L2缓存：会话级数据
cache.putL2(sessionId, "context", contextData);

// L3缓存：项目级配置
cache.putL3(projectId, "coding-rules", projectRules);

// 查看缓存统计
System.out.println(cache.getStats());
```

## 架构说明

### 目录结构

```
v2/
├── model/              # 数据模型层
│   ├── Message.java
│   ├── ChatRequest.java
│   ├── ChatResponse.java
│   ├── Context.java
│   ├── Session.java
│   ├── Memory.java
│   └── ...
├── adapter/            # 模型适配器层
│   ├── ModelAdapter.java
│   ├── BaseModelAdapter.java
│   ├── AdapterRegistry.java
│   ├── DeepSeekAdapter.java
│   └── QwenAdapter.java
├── service/            # 服务层
│   ├── LLMGateway.java
│   ├── LLMGatewayImpl.java
│   └── StreamCallback.java
├── infra/              # 基础设施层
│   ├── cache/
│   │   └── CacheManager.java
│   └── storage/
│       └── SessionStore.java
└── component/          # 组件层（待实现）
    ├── memory/
    ├── rag/
    └── ...
```

### 设计模式

1. **适配器模式**: `ModelAdapter` 统一不同模型的API
2. **注册表模式**: `AdapterRegistry` 管理适配器
3. **策略模式**: `LLMGateway` 支持多种路由策略
4. **建造者模式**: `ChatRequest.Builder` 构建请求
5. **回调模式**: `StreamCallback` 处理流式响应

## 扩展指南

### 添加新模型适配器

```java
public class CustomAdapter extends BaseModelAdapter {
    
    public CustomAdapter() {
        super("CustomModel");
    }
    
    @Override
    public boolean supports(String modelName) {
        return modelName.startsWith("custom-");
    }
    
    @Override
    public String[] getSupportedModels() {
        return new String[]{"custom-v1", "custom-v2"};
    }
    
    @Override
    public ChatResponse chat(ChatRequest request) {
        // 实现API调用逻辑
        return executeWithRetry(() -> {
            // 构建请求
            // 调用API
            // 解析响应
        });
    }
    
    // 实现其他必需方法...
}

// 注册适配器
CustomAdapter customAdapter = new CustomAdapter();
customAdapter.setApiKey("your-api-key");
registry.register(customAdapter);
```

## 配置建议

### API密钥管理

```java
// 推荐从环境变量读取
String deepSeekKey = System.getenv("DEEPSEEK_API_KEY");
String qwenKey = System.getenv("QWEN_API_KEY");

deepSeekAdapter.setApiKey(deepSeekKey);
qwenAdapter.setApiKey(qwenKey);
```

### 缓存配置

```java
// 自定义缓存容量和TTL
CacheManager cache = new CacheManager(
    200,              // L1容量
    10 * 60 * 1000L   // L1 TTL（10分钟）
);
```

### 重试配置

```java
deepSeekAdapter.setMaxRetries(5);           // 最大重试5次
deepSeekAdapter.setRetryDelay(2000L);       // 初始延迟2秒
```

## 注意事项

### ⚠️ 当前限制

1. **HTTP客户端**: 适配器中的API调用为模拟实现，需要集成真实HTTP客户端
2. **JSON序列化**: 需要引入JSON库（Gson或Jackson）
3. **SSE流式**: 流式响应的SSE解析需要实现
4. **错误处理**: 部分边界情况的错误处理待完善

### 🔧 待完成功能

- 上下文引擎（ContextEngine）
- 记忆管理器（MemoryManager）
- RAG检索引擎（RAGEngine）
- 工具调用系统（ToolOrchestrator）
- 代码智能服务（CodeIntelligenceService）
- API控制器层（CursorV2Controller）

## 性能优化建议

1. **启用缓存**: 对频繁请求启用L1缓存
2. **批量向量化**: 使用 `embed(List<String>)` 而非多次调用 `embedSingle()`
3. **流式响应**: 对长文本生成使用 `chatStream()` 提升体验
4. **降级配置**: 配置备用模型提高可用性
5. **会话清理**: 定期调用 `sessionStore.cleanupExpiredSessions()`

## 故障排查

### 问题：适配器不可用
```java
if (!adapter.isAvailable()) {
    System.err.println("API密钥未配置");
}
```

### 问题：模型不支持
```java
if (!registry.isModelSupported("model-name")) {
    List<String> supported = registry.getSupportedModels();
    System.out.println("支持的模型: " + supported);
}
```

### 问题：缓存未命中
```java
CacheManager.CacheStats stats = cache.getStats();
System.out.println(stats); // 查看缓存统计
```

## 下一步

查阅完整设计文档：
- `V2_IMPLEMENTATION_PROGRESS.md` - 实施进度报告
- `doc/TinyAI-Cursor技术架构文档.md` - V2设计文档

---

**最后更新**: 2025年  
**状态**: 阶段1-2已完成，可投入使用（需集成真实API）
