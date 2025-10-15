# TinyAI-Cursor V2 API参考文档

## 目录
- [LLMGateway API](#llmgateway-api)
- [ContextEngine API](#contextengine-api)
- [MemoryManager API](#memorymanager-api)
- [RAGEngine API](#ragengine-api)
- [ToolRegistry API](#toolregistry-api)
- [数据模型](#数据模型)

---

## LLMGateway API

大语言模型统一网关接口

### 方法列表

#### chat()
同步对话请求

```java
ChatResponse chat(ChatRequest request)
```

**参数**：
- `request`: `ChatRequest` - 聊天请求对象

**返回**：
- `ChatResponse` - 聊天响应对象

**示例**：
```java
ChatRequest request = ChatRequest.builder()
    .model("deepseek-chat")
    .addUserMessage("你好")
    .temperature(0.7)
    .build();
    
ChatResponse response = gateway.chat(request);
String content = response.getContent();
```

#### chatStream()
流式对话请求

```java
void chatStream(ChatRequest request, StreamCallback callback)
```

**参数**：
- `request`: `ChatRequest` - 聊天请求对象
- `callback`: `StreamCallback` - 流式响应回调

**示例**：
```java
gateway.chatStream(request, new StreamCallback() {
    @Override
    public void onToken(String token) {
        System.out.print(token);
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

#### complete()
代码补全

```java
String complete(String prefix, String suffix, String language, int maxTokens)
```

**参数**：
- `prefix`: `String` - 光标前的代码
- `suffix`: `String` - 光标后的代码（可选）
- `language`: `String` - 编程语言
- `maxTokens`: `int` - 最大生成token数

**返回**：
- `String` - 补全建议

#### embed()
文本向量化

```java
List<double[]> embed(List<String> texts)
```

---

## ContextEngine API

上下文引擎，整合多源上下文

### 方法列表

#### buildContext()
构建完整上下文

```java
Context buildContext(ContextRequest request)
```

**参数**：
- `request`: `ContextRequest` - 上下文请求

**返回**：
- `Context` - 完整上下文对象

**示例**：
```java
ContextRequest request = new ContextRequest();
request.query = "如何实现登录";
request.sessionId = "session-123";
request.projectId = "project-456";
request.enableRAG = true;

Context context = contextEngine.buildContext(request);
```

#### buildCompletionContext()
快速构建代码补全上下文

```java
Context buildCompletionContext(String filePath, String fileContent, 
                               CursorPosition cursorPosition, String projectId)
```

#### buildChatContext()
快速构建对话上下文

```java
Context buildChatContext(String query, String sessionId, String projectId)
```

---

## MemoryManager API

记忆管理器，管理4种记忆类型

### 方法列表

#### addMemory()
添加记忆

```java
void addMemory(Memory memory)
```

**示例**：
```java
Memory memory = Memory.shortTerm("mem-001", sessionId, "用户询问了登录功能");
memoryManager.addMemory(memory);
```

#### retrieveSessionMemories()
检索会话记忆

```java
List<Memory> retrieveSessionMemories(String sessionId, MemoryType type)
```

**参数**：
- `sessionId`: `String` - 会话ID
- `type`: `MemoryType` - 记忆类型（null表示所有类型）

**返回**：
- `List<Memory>` - 记忆列表

#### retrieveSimilarMemories()
语义检索记忆

```java
List<Memory> retrieveSimilarMemories(double[] queryEmbedding, int topK, double threshold)
```

**参数**：
- `queryEmbedding`: `double[]` - 查询向量
- `topK`: `int` - 返回Top-K结果
- `threshold`: `double` - 相似度阈值（0-1）

#### clearSessionMemory()
清除会话记忆

```java
void clearSessionMemory(String sessionId)
```

#### cleanupExpiredMemories()
清理过期记忆

```java
int cleanupExpiredMemories()
```

**返回**：清理的记忆数量

---

## RAGEngine API

检索增强生成引擎

### 方法列表

#### indexCodebase()
索引代码库

```java
void indexCodebase(String projectId, List<CodeFile> codeFiles)
```

**示例**：
```java
List<RAGEngine.CodeFile> files = new ArrayList<>();
files.add(new RAGEngine.CodeFile("/path/to/Main.java", javaCode));

ragEngine.indexCodebase("project-123", files);
```

#### semanticSearch()
语义检索

```java
List<CodeSnippet> semanticSearch(String query, int topK, String projectId)
```

**参数**：
- `query`: `String` - 查询文本
- `topK`: `int` - 返回Top-K结果
- `projectId`: `String` - 项目ID（可选，null表示全局检索）

**返回**：
- `List<CodeSnippet>` - 代码片段列表

**示例**：
```java
List<CodeSnippet> results = ragEngine.semanticSearch(
    "如何实现用户认证", 5, "project-123"
);

for (CodeSnippet snippet : results) {
    System.out.println(snippet.getFilePath());
    System.out.println(snippet.getContent());
    System.out.println("Score: " + snippet.getScore());
}
```

#### exactSearch()
精确匹配检索

```java
List<CodeSnippet> exactSearch(List<String> keywords, int topK, String projectId)
```

#### hybridSearch()
混合检索

```java
List<CodeSnippet> hybridSearch(String query, List<String> keywords, int topK, String projectId)
```

---

## ToolRegistry API

工具注册表

### 方法列表

#### registerTool()
注册工具

```java
void registerTool(Tool tool)
```

**示例**：
```java
Tool myTool = new MyCustomTool();
toolRegistry.registerTool(myTool);
```

#### getTool()
获取工具

```java
Tool getTool(String toolName)
```

#### getAllToolDefinitions()
获取所有工具定义

```java
List<ToolDefinition> getAllToolDefinitions()
```

**用于构建ChatRequest的tools参数**：
```java
List<ToolDefinition> tools = toolRegistry.getAllToolDefinitions();

ChatRequest request = ChatRequest.builder()
    .addUserMessage("分析这段代码")
    .tools(tools)
    .toolChoice("auto")
    .build();
```

---

## 数据模型

### ChatRequest

聊天请求构建器

```java
ChatRequest request = ChatRequest.builder()
    .model("deepseek-chat")
    .addSystemMessage("你是AI助手")
    .addUserMessage("问题内容")
    .temperature(0.7)
    .topP(0.9)
    .maxTokens(500)
    .stream(false)
    .build();
```

### ChatResponse

聊天响应对象

```java
// 获取内容
String content = response.getContent();

// 检查工具调用
if (response.hasToolCalls()) {
    List<ToolCall> toolCalls = response.getToolCalls();
}

// 获取使用量
ChatResponse.Usage usage = response.getUsage();
int totalTokens = usage.getTotalTokens();

// 检查错误
if (!response.isSuccess()) {
    String errorMsg = response.getError().getMessage();
}
```

### Memory

记忆对象

```java
// 创建不同类型的记忆
Memory working = Memory.working("id1", sessionId, "临时数据");
Memory shortTerm = Memory.shortTerm("id2", sessionId, "会话数据");
Memory longTerm = Memory.longTerm("id3", projectId, "项目规则");
Memory semantic = Memory.semantic("id4", projectId, "代码片段", embedding);

// 访问记忆
memory.recordAccess();
int count = memory.getAccessCount();

// 更新记忆
memory.updateContent("新内容");
```

### Session

会话对象

```java
// 创建会话
Session session = new Session("project-123", "user-456");

// 添加消息
session.addMessage(Message.user("你好"));
session.addMessage(Message.assistant("你好！"));

// 获取最近消息
List<Message> recent = session.getRecentMessages(10);

// 工作记忆
session.putWorkingMemory("key", value);
Object data = session.getWorkingMemory("key");

// 检查状态
boolean active = session.isActive();
boolean expired = session.isExpired();
```

### Context

上下文对象

```java
// 获取系统提示词
String systemPrompt = context.buildSystemPrompt();

// 访问组件
String filePath = context.getCurrentFilePath();
List<CodeSnippet> snippets = context.getRelatedCodeSnippets();
List<Message> history = context.getConversationHistory();
List<String> rules = context.getProjectRules();

// 扩展上下文
context.putExtra("key", value);
Object extra = context.getExtra("key");
```

---

## 完整使用示例

### 智能对话

```java
// 1. 初始化组件
AdapterRegistry registry = new AdapterRegistry();
registry.register(new DeepSeekAdapter(apiKey));

LLMGateway gateway = new LLMGatewayImpl(registry);
MemoryManager memoryManager = new MemoryManager();
RAGEngine ragEngine = new RAGEngine();
ContextEngine contextEngine = new ContextEngine(memoryManager, ragEngine);

// 2. 索引代码库
ragEngine.indexCodebase(projectId, codeFiles);

// 3. 构建上下文
Context context = contextEngine.buildChatContext(
    "如何实现登录功能", sessionId, projectId
);

// 4. 构建请求
ChatRequest request = ChatRequest.builder()
    .model("deepseek-chat")
    .addSystemMessage(context.buildSystemPrompt())
    .addUserMessage("如何实现登录功能")
    .build();

// 5. 发送请求
ChatResponse response = gateway.chat(request);

// 6. 保存记忆
Memory memory = Memory.shortTerm(
    UUID.randomUUID().toString(),
    sessionId,
    "讨论了登录功能实现"
);
memoryManager.addMemory(memory);
```

### 代码补全

```java
// 1. 构建上下文
Context context = contextEngine.buildCompletionContext(
    filePath, fileContent, cursorPosition, projectId
);

// 2. 调用补全
String completion = gateway.complete(
    prefix, suffix, "java", 100
);
```

---

## 配置项

### LLMGateway

```java
gateway.setPreferredModel("deepseek-chat");
gateway.addFallbackModel("qwen-max");
gateway.setEnableFallback(true);
```

### ContextEngine

```java
contextEngine.setMaxContextLength(4000);
contextEngine.setRagTopK(5);
contextEngine.setMaxHistoryMessages(10);
```

### MemoryManager

```java
memoryManager.setWorkingMemoryTtl(30 * 60 * 1000L); // 30分钟
memoryManager.setShortTermMemoryTtl(2 * 60 * 60 * 1000L); // 2小时
```

### RAGEngine

```java
ragEngine.setStrategy(RAGEngine.RetrievalStrategy.HYBRID);
ragEngine.setSimilarityThreshold(0.7);
```

---

**版本**: v2.0.0  
**最后更新**: 2025年
