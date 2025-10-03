# TinyAI-Cursor API参考文档

> **智能编程助手API接口详细说明**

## 📋 文档概述

本文档提供TinyAI-Cursor模块的详细API参考、使用示例和开发指南，帮助开发者快速集成和使用智能编程助手功能。

---

## 🚀 核心API接口

### 1. AICodingCursor 主控制器API

#### 构造函数
```java
// 默认构造函数
public AICodingCursor()

// 指定名称的构造函数
public AICodingCursor(String name)
```

#### 代码分析API
```java
/**
 * 分析Java代码
 * @param code 待分析的代码字符串
 * @return 包含分析结果的Map对象
 */
public Map<String, Object> analyzeCode(String code)
```

**返回结果结构**:
```java
{
    "syntax_valid": Boolean,           // 语法是否有效
    "syntax_issues": List<CodeIssue>,  // 语法问题列表
    "imports": List<String>,           // 导入语句列表
    "classes": List<Map>,              // 类定义信息
    "methods": List<Map>,              // 方法定义信息
    "variables": List<Map>,            // 变量定义信息
    "metrics": {                       // 代码度量指标
        "total_lines": Integer,        // 总行数
        "code_lines": Integer,         // 代码行数
        "comment_lines": Integer,      // 注释行数
        "blank_lines": Integer,        // 空行数
        "comment_ratio": Double        // 注释比例
    },
    "complexity": Integer,             // 圈复杂度
    "issues": List<CodeIssue>          // 代码质量问题
}
```

**使用示例**:
```java
AICodingCursor cursor = new AICodingCursor();
String code = "public class Test { public void method() {} }";
Map<String, Object> result = cursor.analyzeCode(code);

// 检查语法有效性
Boolean isValid = (Boolean) result.get("syntax_valid");
System.out.println("语法有效: " + isValid);

// 获取复杂度
Integer complexity = (Integer) result.get("complexity");
System.out.println("圈复杂度: " + complexity);
```

#### 代码生成API
```java
/**
 * 根据请求生成代码
 * @param request 代码生成请求字符串
 * @return 生成的代码字符串
 */
public String generateCode(String request)
```

**支持的请求类型**:
- `"method methodName"` - 生成方法
- `"class ClassName"` - 生成类
- `"test methodName"` - 生成测试代码
- `"interface InterfaceName"` - 生成接口

**使用示例**:
```java
// 生成方法
String method = cursor.generateCode("method calculateSum");
System.out.println(method);

// 生成类
String cls = cursor.generateCode("class UserManager");
System.out.println(cls);

// 生成测试
String test = cursor.generateCode("test validateEmail");
System.out.println(test);
```

#### 重构建议API
```java
/**
 * 获取代码重构建议
 * @param code 待重构的代码
 * @return 重构建议列表
 */
public List<RefactorSuggestion> suggestRefactor(String code)
```

**RefactorSuggestion对象结构**:
```java
public class RefactorSuggestion {
    private String suggestionType;      // 建议类型
    private String description;         // 描述
    private String beforeCode;          // 重构前代码
    private String afterCode;           // 重构后代码
    private List<String> benefitsSummary; // 收益总结
    private String estimatedImpact;     // 影响评估
    private int priority;               // 优先级
}
```

**使用示例**:
```java
String complexCode = "public void longMethod() { /* 长方法 */ }";
List<RefactorSuggestion> suggestions = cursor.suggestRefactor(complexCode);

for (RefactorSuggestion suggestion : suggestions) {
    System.out.println("建议: " + suggestion.getDescription());
    System.out.println("影响: " + suggestion.getEstimatedImpact());
    System.out.println("优先级: " + suggestion.getPriority());
}
```

#### 调试诊断API
```java
/**
 * 诊断代码错误
 * @param code 待诊断的代码
 * @return 诊断结果
 */
public Map<String, Object> debugCode(String code)

/**
 * 诊断代码错误（带错误消息）
 * @param code 待诊断的代码
 * @param errorMessage 错误消息
 * @return 诊断结果
 */
public Map<String, Object> debugCode(String code, String errorMessage)
```

**诊断结果结构**:
```java
{
    "error_found": Boolean,           // 是否发现错误
    "error_type": String,             // 错误类型
    "error_line": Integer,            // 错误行号
    "diagnosis": String,              // 诊断描述
    "suggestions": List<String>,      // 修复建议
    "fixed_code": String,             // 修复后的代码
    "confidence": Double              // 诊断置信度
}
```

#### 综合代码审查API
```java
/**
 * 综合代码审查
 * @param code 待审查的代码
 * @return 审查报告
 */
public Map<String, Object> reviewCode(String code)
```

**审查报告结构**:
```java
{
    "overall_score": Double,                    // 总体评分(0-100)
    "analysis": Map<String, Object>,           // 分析结果
    "refactor_suggestions": List<RefactorSuggestion>, // 重构建议
    "debug_info": Map<String, Object>,         // 调试信息
    "recommendations": List<String>,           // 改进建议
    "review_time": String                      // 审查时间
}
```

#### AI对话API
```java
/**
 * 智能对话功能
 * @param userInput 用户输入
 * @return AI回复
 */
public String chat(String userInput)
```

**使用示例**:
```java
String response = cursor.chat("什么是单例模式？");
System.out.println("AI回复: " + response);

String codeHelp = cursor.chat("如何优化这段代码的性能？");
System.out.println("优化建议: " + codeHelp);
```

#### 系统管理API
```java
/**
 * 获取系统状态
 * @return 系统状态信息
 */
public Map<String, Object> getSystemStatus()

/**
 * 获取操作统计
 * @return 操作统计信息
 */
public Map<String, Integer> getOperationStats()

/**
 * 获取会话历史
 * @return 会话历史记录
 */
public List<Message> getSessionHistory()

/**
 * 清空会话历史
 */
public void clearSessionHistory()

/**
 * 更新偏好设置
 * @param newPreferences 新的偏好设置
 */
public void updatePreferences(Map<String, Object> newPreferences)

/**
 * 获取偏好设置
 * @return 当前偏好设置
 */
public Map<String, Object> getPreferences()
```

---

## 🔧 配置选项详解

### 偏好设置参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `language` | String | "java" | 编程语言 |
| `style` | String | "standard" | 代码风格 |
| `auto_refactor` | Boolean | true | 自动重构建议 |
| `debug_level` | String | "detailed" | 调试详细级别 |
| `max_suggestions` | Integer | 10 | 最大建议数量 |
| `enable_ai_chat` | Boolean | true | 启用AI对话 |

### 配置示例
```java
Map<String, Object> preferences = new HashMap<>();
preferences.put("language", "java");
preferences.put("style", "google");
preferences.put("max_suggestions", 5);
preferences.put("debug_level", "basic");

cursor.updatePreferences(preferences);
```

---

## 📚 数据模型详解

### 1. CodeIssue 代码问题模型
```java
public class CodeIssue {
    private String issueType;        // 问题类型
    private String severity;         // 严重程度: critical, high, medium, low
    private String message;          // 问题描述
    private int lineNumber;          // 行号
    private String suggestion;       // 修复建议
    
    // 构造函数和getter/setter方法
}
```

### 2. RefactorSuggestion 重构建议模型
```java
public class RefactorSuggestion {
    private String suggestionType;      // 建议类型
    private String description;         // 描述
    private String beforeCode;          // 重构前代码
    private String afterCode;           // 重构后代码
    private List<String> benefitsSummary; // 收益列表
    private String estimatedImpact;     // 影响评估
    private int priority;               // 优先级分数
    
    // 判断是否为高优先级
    public boolean isHighPriority() {
        return priority > 50;
    }
}
```

---

## 🎯 使用场景与最佳实践

### 1. 日常开发辅助

#### 代码质量检查
```java
public class CodeQualityChecker {
    private AICodingCursor cursor = new AICodingCursor("质量检查器");
    
    public QualityReport checkCodeQuality(String sourceCode) {
        // 执行全面代码审查
        Map<String, Object> review = cursor.reviewCode(sourceCode);
        
        double score = (Double) review.get("overall_score");
        List<String> recommendations = (List<String>) review.get("recommendations");
        
        return new QualityReport(score, recommendations);
    }
}
```

#### 智能代码生成
```java
public class CodeAssistant {
    private AICodingCursor cursor = new AICodingCursor("代码助手");
    
    public String generateServiceClass(String serviceName) {
        String request = "class " + serviceName + "Service";
        return cursor.generateCode(request);
    }
    
    public String generateTestMethod(String methodName, String className) {
        String request = "test " + methodName + " for " + className;
        return cursor.generateCode(request);
    }
}
```

### 2. 代码审查自动化

#### 自动化审查流程
```java
public class AutoCodeReview {
    private AICodingCursor cursor = new AICodingCursor("自动审查");
    
    public ReviewResult performReview(String codeFile) {
        try {
            String code = Files.readString(Paths.get(codeFile));
            
            // 执行分析
            Map<String, Object> analysis = cursor.analyzeCode(code);
            
            // 获取重构建议
            List<RefactorSuggestion> suggestions = cursor.suggestRefactor(code);
            
            // 检查错误
            Map<String, Object> debugInfo = cursor.debugCode(code);
            
            return new ReviewResult(analysis, suggestions, debugInfo);
            
        } catch (IOException e) {
            throw new RuntimeException("文件读取失败", e);
        }
    }
}
```

### 3. 教学和学习辅助

#### 代码学习助手
```java
public class LearningAssistant {
    private AICodingCursor cursor = new AICodingCursor("学习助手");
    
    public LearningReport analyzeStudentCode(String studentCode) {
        // 分析代码结构
        Map<String, Object> analysis = cursor.analyzeCode(studentCode);
        
        // 提供改进建议
        List<RefactorSuggestion> improvements = cursor.suggestRefactor(studentCode);
        
        // 解释代码问题
        String explanation = cursor.chat("这段代码有什么问题，如何改进？");
        
        return new LearningReport(analysis, improvements, explanation);
    }
}
```

---

## 🚨 错误处理和异常管理

### 1. 常见异常类型

#### 输入验证异常
```java
// 空代码输入
if (code == null || code.trim().isEmpty()) {
    throw new IllegalArgumentException("代码不能为空");
}

// 代码长度限制
if (code.length() > MAX_CODE_LENGTH) {
    throw new IllegalArgumentException("代码长度超过限制");
}
```

#### 分析异常处理
```java
try {
    Map<String, Object> result = cursor.analyzeCode(code);
    // 处理正常结果
} catch (OutOfMemoryError e) {
    // 内存不足处理
    System.err.println("代码过大，分析失败");
} catch (Exception e) {
    // 其他异常处理
    System.err.println("分析异常: " + e.getMessage());
}
```

### 2. 错误恢复策略

#### 分析失败恢复
```java
public Map<String, Object> safeAnalyzeCode(String code) {
    try {
        return cursor.analyzeCode(code);
    } catch (Exception e) {
        // 返回基础分析结果
        Map<String, Object> fallback = new HashMap<>();
        fallback.put("syntax_valid", false);
        fallback.put("error", e.getMessage());
        fallback.put("fallback", true);
        return fallback;
    }
}
```

---

## 📊 性能监控和优化

### 1. 性能指标监控

#### 操作统计监控
```java
public class PerformanceMonitor {
    private AICodingCursor cursor;
    
    public void printPerformanceReport() {
        Map<String, Integer> stats = cursor.getOperationStats();
        Map<String, Object> status = cursor.getSystemStatus();
        
        System.out.println("=== 性能报告 ===");
        System.out.println("运行时长: " + status.get("uptime_minutes") + " 分钟");
        System.out.println("分析操作: " + stats.getOrDefault("analyze", 0) + " 次");
        System.out.println("生成操作: " + stats.getOrDefault("generate", 0) + " 次");
        System.out.println("缓存命中: " + status.get("cache_size"));
    }
}
```

### 2. 缓存优化

#### 缓存管理策略
```java
// 定期清理缓存
public void optimizeCache() {
    if (cursor.getAnalyzer().getCacheSize() > 1000) {
        cursor.getAnalyzer().clearCache();
        System.out.println("缓存已清理");
    }
}

// 预热常用分析
public void warmupCache() {
    String[] commonPatterns = {
        "public class Example {}",
        "public void method() {}",
        "private String field;"
    };
    
    for (String pattern : commonPatterns) {
        cursor.analyzeCode(pattern);
    }
}
```

---

## 🔮 扩展开发指南

### 1. 自定义分析规则

#### 扩展CodeAnalyzer
```java
public class CustomCodeAnalyzer extends CodeAnalyzer {
    @Override
    protected List<CodeIssue> findCodeIssues(String code) {
        List<CodeIssue> issues = super.findCodeIssues(code);
        
        // 添加自定义检查
        if (code.contains("System.out.println")) {
            issues.add(new CodeIssue("debug_print", "low", 
                "发现调试打印语句", 1, "移除调试代码"));
        }
        
        // 检查TODO注释
        if (code.contains("TODO")) {
            issues.add(new CodeIssue("todo_comment", "medium",
                "发现TODO注释", 1, "完成待办事项"));
        }
        
        return issues;
    }
}
```

### 2. 自定义代码模板

#### 扩展CodeGenerator
```java
public class CustomCodeGenerator extends CodeGenerator {
    public CustomCodeGenerator() {
        super();
        // 添加自定义模板
        addTemplate("spring_controller", createSpringControllerTemplate());
        addTemplate("jpa_entity", createJpaEntityTemplate());
    }
    
    private String createSpringControllerTemplate() {
        return """
            @RestController
            @RequestMapping("/{endpoint}")
            public class {name}Controller {
                
                @Autowired
                private {name}Service service;
                
                @GetMapping
                public ResponseEntity<List<{name}>> getAll() {
                    return ResponseEntity.ok(service.findAll());
                }
            }
            """;
    }
}
```

### 3. 自定义重构模式

#### 扩展RefactorAgent
```java
public class CustomRefactorAgent extends RefactorAgent {
    public CustomRefactorAgent(CodeAnalyzer analyzer) {
        super(analyzer);
        addCustomPatterns();
    }
    
    private void addCustomPatterns() {
        // 添加Spring相关重构模式
        addRefactorPattern("spring_autowired", 
            new RefactorPattern("使用构造函数注入替代@Autowired", 1, 
                "constructor_injection", "中等"));
        
        // 添加微服务相关模式
        addRefactorPattern("large_service", 
            new RefactorPattern("拆分大型服务类", 300, 
                "service_split", "高"));
    }
}
```

---

## 📖 示例项目

### 完整使用示例
```java
public class ComprehensiveExample {
    public static void main(String[] args) {
        // 创建AI编程助手
        AICodingCursor cursor = new AICodingCursor("示例助手");
        
        // 配置系统
        Map<String, Object> config = Map.of(
            "language", "java",
            "style", "standard",
            "max_suggestions", 5
        );
        cursor.updatePreferences(config);
        
        // 示例代码
        String sampleCode = """
            public class UserService {
                private UserRepository repository;
                
                public User createUser(String name, String email, String phone, 
                                     String address, String city, String country) {
                    if (name != null && email != null && phone != null && 
                        address != null && city != null && country != null) {
                        if (name.length() > 0 && email.contains("@")) {
                            if (phone.length() > 10) {
                                User user = new User();
                                user.setName(name);
                                user.setEmail(email);
                                user.setPhone(phone);
                                user.setAddress(address);
                                user.setCity(city);
                                user.setCountry(country);
                                return repository.save(user);
                            }
                        }
                    }
                    return null;
                }
            }
            """;
        
        // 执行全面分析
        System.out.println("=== 代码分析 ===");
        Map<String, Object> analysis = cursor.analyzeCode(sampleCode);
        System.out.println("语法有效: " + analysis.get("syntax_valid"));
        System.out.println("复杂度: " + analysis.get("complexity"));
        
        // 获取重构建议
        System.out.println("\n=== 重构建议 ===");
        List<RefactorSuggestion> suggestions = cursor.suggestRefactor(sampleCode);
        suggestions.forEach(s -> 
            System.out.println("• " + s.getDescription() + " [" + s.getEstimatedImpact() + "]"));
        
        // 执行代码审查
        System.out.println("\n=== 代码审查 ===");
        Map<String, Object> review = cursor.reviewCode(sampleCode);
        Double score = (Double) review.get("overall_score");
        System.out.println("质量评分: " + String.format("%.1f", score) + "/100");
        
        @SuppressWarnings("unchecked")
        List<String> recommendations = (List<String>) review.get("recommendations");
        System.out.println("改进建议:");
        recommendations.forEach(r -> System.out.println("• " + r));
        
        // AI对话咨询
        System.out.println("\n=== AI对话 ===");
        String response = cursor.chat("如何优化这个方法的参数过多问题？");
        System.out.println("AI建议: " + response);
        
        // 生成改进版本
        System.out.println("\n=== 代码生成 ===");
        String improvedCode = cursor.generateCode("class ImprovedUserService with builder pattern");
        System.out.println("生成的改进代码:\n" + improvedCode.substring(0, Math.min(300, improvedCode.length())) + "...");
        
        // 显示系统状态
        System.out.println("\n=== 系统状态 ===");
        Map<String, Object> status = cursor.getSystemStatus();
        System.out.println("运行时长: " + status.get("uptime_minutes") + " 分钟");
        System.out.println("操作次数: " + status.get("session_operations"));
        System.out.println("缓存大小: " + status.get("cache_size"));
    }
}
```

---

## 📝 更新日志

### v1.0.0 (2025-10-03)
- ✅ 初始版本发布
- ✅ 实现核心代码分析功能
- ✅ 实现智能代码生成功能
- ✅ 实现重构建议系统
- ✅ 实现错误调试功能
- ✅ 集成AI对话能力
- ✅ 完善系统管理功能
- ✅ 编写全面的API文档

---

**作者**: 山泽  
**文档版本**: v1.0.0  
**最后更新**: 2025-10-03