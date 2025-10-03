# TinyAI-Cursor 使用指南

> **快速上手AI智能编程助手**

## 🚀 快速开始

### 1. 基本使用

```java
import io.leavesfly.tinyai.agent.cursor.AICodingCursor;

// 创建AI编程助手
AICodingCursor cursor = new AICodingCursor("我的助手");

// 分析代码
String code = """
    public class Calculator {
        public int add(int a, int b) {
            return a + b;
        }
    }
    """;

Map<String, Object> analysis = cursor.analyzeCode(code);
System.out.println("语法有效: " + analysis.get("syntax_valid"));
System.out.println("复杂度: " + analysis.get("complexity"));
```

### 2. 核心功能

#### 📊 代码分析
```java
// 获取详细分析结果
Map<String, Object> result = cursor.analyzeCode(javaCode);
Boolean syntaxValid = (Boolean) result.get("syntax_valid");
Integer complexity = (Integer) result.get("complexity");
List<CodeIssue> issues = (List<CodeIssue>) result.get("issues");
```

#### 🤖 代码生成
```java
// 生成不同类型的代码
String method = cursor.generateCode("method validateEmail");
String cls = cursor.generateCode("class UserManager");
String test = cursor.generateCode("test calculateSum");
```

#### 🔧 重构建议
```java
// 获取重构建议
List<RefactorSuggestion> suggestions = cursor.suggestRefactor(complexCode);
for (RefactorSuggestion suggestion : suggestions) {
    System.out.println("建议: " + suggestion.getDescription());
    System.out.println("优先级: " + suggestion.getPriority());
}
```

#### 🐛 错误调试
```java
// 诊断代码错误
Map<String, Object> diagnosis = cursor.debugCode(buggyCode);
Boolean errorFound = (Boolean) diagnosis.get("error_found");
String errorType = (String) diagnosis.get("error_type");
List<String> suggestions = (List<String>) diagnosis.get("suggestions");
```

#### 📋 综合审查
```java
// 执行全面代码审查
Map<String, Object> review = cursor.reviewCode(sourceCode);
Double score = (Double) review.get("overall_score");
List<String> recommendations = (List<String>) review.get("recommendations");

System.out.println("代码质量评分: " + score + "/100");
recommendations.forEach(System.out::println);
```

#### 💬 AI对话
```java
// 智能编程咨询
String response = cursor.chat("什么是单例模式？如何实现？");
System.out.println("AI回复: " + response);

String help = cursor.chat("如何优化这段代码的性能？");
System.out.println("优化建议: " + help);
```

---

## ⚙️ 配置管理

### 系统配置
```java
// 设置偏好
Map<String, Object> preferences = Map.of(
    "language", "java",
    "style", "standard",
    "max_suggestions", 5,
    "debug_level", "detailed",
    "enable_ai_chat", true
);
cursor.updatePreferences(preferences);

// 获取当前配置
Map<String, Object> currentPrefs = cursor.getPreferences();
```

### 系统管理
```java
// 查看系统状态
Map<String, Object> status = cursor.getSystemStatus();
System.out.println("运行时长: " + status.get("uptime_minutes") + " 分钟");

// 获取操作统计
Map<String, Integer> stats = cursor.getOperationStats();
System.out.println("分析次数: " + stats.getOrDefault("analyze", 0));

// 清理会话历史
cursor.clearSessionHistory();
```

---

## 🎯 实用场景

### 1. 日常开发助手
```java
public class DevelopmentHelper {
    private AICodingCursor cursor = new AICodingCursor("开发助手");
    
    // 快速质量检查
    public void quickQualityCheck(String code) {
        Map<String, Object> analysis = cursor.analyzeCode(code);
        
        if (!(Boolean) analysis.get("syntax_valid")) {
            System.out.println("⚠️ 代码存在语法错误");
        }
        
        Integer complexity = (Integer) analysis.get("complexity");
        if (complexity > 10) {
            System.out.println("⚠️ 代码复杂度过高: " + complexity);
        }
        
        List<CodeIssue> issues = (List<CodeIssue>) analysis.get("issues");
        if (!issues.isEmpty()) {
            System.out.println("🔍 发现 " + issues.size() + " 个问题");
            issues.forEach(issue -> 
                System.out.println("  • " + issue.getMessage()));
        }
    }
    
    // 智能代码补全
    public String smartComplete(String request) {
        return cursor.generateCode(request);
    }
}
```

### 2. 代码审查自动化
```java
public class AutoReviewer {
    private AICodingCursor cursor = new AICodingCursor("审查员");
    
    public ReviewReport reviewFile(String filePath) throws IOException {
        String code = Files.readString(Paths.get(filePath));
        
        // 执行综合审查
        Map<String, Object> review = cursor.reviewCode(code);
        
        double score = (Double) review.get("overall_score");
        List<String> recommendations = (List<String>) review.get("recommendations");
        List<RefactorSuggestion> refactors = (List<RefactorSuggestion>) review.get("refactor_suggestions");
        
        return new ReviewReport(filePath, score, recommendations, refactors);
    }
}
```

### 3. 学习编程助手
```java
public class LearningAssistant {
    private AICodingCursor cursor = new AICodingCursor("学习助手");
    
    public void explainCode(String studentCode) {
        // 分析学生代码
        Map<String, Object> analysis = cursor.analyzeCode(studentCode);
        
        // 提供学习建议
        String explanation = cursor.chat(
            "请分析这段代码的优点和缺点，并给出学习建议：\n" + studentCode);
        
        System.out.println("📚 代码分析:");
        System.out.println("复杂度: " + analysis.get("complexity"));
        
        System.out.println("\n💡 学习建议:");
        System.out.println(explanation);
        
        // 生成改进版本
        String improved = cursor.generateCode("improved version of this code");
        System.out.println("\n✨ 改进示例:");
        System.out.println(improved);
    }
}
```

---

## 🎮 交互式使用

### 运行演示程序
```bash
# 使用Maven运行演示
cd tinyai-agent-cursor
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.cursor.CursorDemo"
```

### 支持的命令
- `analyze:<code>` - 分析代码
- `generate:<request>` - 生成代码
- `refactor:<code>` - 重构建议
- `debug:<code>` - 调试代码
- `review:<code>` - 代码审查
- `chat:<message>` - AI对话
- `status` - 系统状态
- `help` - 显示帮助
- `quit` - 退出程序

---

## 💡 最佳实践

### 1. 代码质量控制
```java
// 设置质量阈值
public boolean isCodeQualityAcceptable(String code) {
    Map<String, Object> review = cursor.reviewCode(code);
    double score = (Double) review.get("overall_score");
    
    // 设置最低质量标准
    return score >= 80.0;
}

// 强制重构检查
public List<RefactorSuggestion> getMandatoryRefactors(String code) {
    return cursor.suggestRefactor(code).stream()
            .filter(RefactorSuggestion::isHighPriority)
            .collect(Collectors.toList());
}
```

### 2. 性能优化
```java
// 批量分析
public Map<String, Map<String, Object>> batchAnalyze(List<String> codeList) {
    Map<String, Map<String, Object>> results = new HashMap<>();
    
    for (int i = 0; i < codeList.size(); i++) {
        String code = codeList.get(i);
        String key = "code_" + i;
        results.put(key, cursor.analyzeCode(code));
        
        // 定期清理缓存避免内存压力
        if (i % 100 == 0) {
            cursor.getAnalyzer().clearCache();
        }
    }
    
    return results;
}
```

### 3. 错误处理
```java
// 安全的代码分析
public Optional<Map<String, Object>> safeAnalyze(String code) {
    try {
        if (code == null || code.trim().isEmpty()) {
            return Optional.empty();
        }
        
        if (code.length() > 50000) { // 限制代码长度
            System.out.println("⚠️ 代码过长，可能影响性能");
        }
        
        return Optional.of(cursor.analyzeCode(code));
        
    } catch (Exception e) {
        System.err.println("分析失败: " + e.getMessage());
        return Optional.empty();
    }
}
```

---

## 🔧 故障排除

### 常见问题

#### 1. 内存不足
```java
// 监控内存使用
Runtime runtime = Runtime.getRuntime();
long freeMemory = runtime.freeMemory();
long totalMemory = runtime.totalMemory();
double memoryUsage = (double)(totalMemory - freeMemory) / totalMemory;

if (memoryUsage > 0.8) {
    cursor.getAnalyzer().clearCache();
    System.gc();
}
```

#### 2. 分析超时
```java
// 添加超时控制
public Map<String, Object> analyzeWithTimeout(String code, long timeoutMs) {
    ExecutorService executor = Executors.newSingleThreadExecutor();
    
    try {
        Future<Map<String, Object>> future = executor.submit(() -> cursor.analyzeCode(code));
        return future.get(timeoutMs, TimeUnit.MILLISECONDS);
    } catch (TimeoutException e) {
        System.err.println("分析超时");
        return Collections.emptyMap();
    } catch (Exception e) {
        System.err.println("分析异常: " + e.getMessage());
        return Collections.emptyMap();
    } finally {
        executor.shutdown();
    }
}
```

#### 3. 缓存清理
```java
// 定期维护
public void performMaintenance() {
    // 清理过期缓存
    cursor.getAnalyzer().clearCache();
    
    // 清理会话历史
    if (cursor.getSessionHistory().size() > 500) {
        cursor.clearSessionHistory();
    }
    
    // 重置统计
    cursor.getOperationStats().clear();
    
    System.out.println("✅ 系统维护完成");
}
```

---

## 📈 性能监控

### 性能指标
```java
public class PerformanceTracker {
    private long startTime;
    private Map<String, Long> operationTimes = new HashMap<>();
    
    public void trackOperation(String operation, Runnable task) {
        long start = System.currentTimeMillis();
        task.run();
        long duration = System.currentTimeMillis() - start;
        
        operationTimes.put(operation, duration);
        
        if (duration > 1000) { // 超过1秒的操作
            System.out.println("⚠️ 慢操作: " + operation + " 耗时 " + duration + "ms");
        }
    }
    
    public void printReport() {
        System.out.println("=== 性能报告 ===");
        operationTimes.forEach((op, time) -> 
            System.out.println(op + ": " + time + "ms"));
    }
}
```

---

## 📚 扩展示例

### 自定义分析器
```java
public class ProjectSpecificAnalyzer {
    private AICodingCursor cursor = new AICodingCursor("项目分析器");
    
    public void addProjectRules() {
        // 这里可以扩展自定义规则
        cursor.updatePreferences(Map.of(
            "check_logging", true,
            "check_security", true,
            "enforce_patterns", true
        ));
    }
    
    public List<String> checkProjectStandards(String code) {
        List<String> violations = new ArrayList<>();
        
        // 检查日志规范
        if (!code.contains("logger") && code.contains("System.out")) {
            violations.add("应使用logger而不是System.out");
        }
        
        // 检查异常处理
        if (code.contains("catch") && !code.contains("log")) {
            violations.add("异常处理应包含日志记录");
        }
        
        return violations;
    }
}
```

---

## 📝 总结

TinyAI-Cursor提供了强大的AI编程辅助能力，通过简单的API调用即可实现：

- 🔍 **智能代码分析**: 深度理解代码结构和质量
- 🤖 **智能代码生成**: 根据需求自动生成代码
- 🔧 **智能重构建议**: 识别改进机会并提供具体方案
- 🐛 **智能错误调试**: 快速定位和修复代码问题
- 💬 **AI编程咨询**: 获得专业的编程建议和帮助

通过合理配置和使用最佳实践，可以显著提升开发效率和代码质量。

---

**快速链接**:
- [技术架构文档](./TinyAI-Cursor技术架构文档.md)
- [API参考文档](./TinyAI-Cursor-API参考文档.md)
- [项目主页](../README.md)

**作者**: 山泽  
**版本**: v1.0.0  
**更新时间**: 2025-10-03