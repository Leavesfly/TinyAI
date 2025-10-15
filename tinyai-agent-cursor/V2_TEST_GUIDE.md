# TinyAI-Cursor V2 测试运行指南

## 快速开始

### 前置条件

- JDK 8 或更高版本
- Maven 3.x
- IDE（推荐 IntelliJ IDEA 或 Eclipse）

---

## 测试结构

```
src/test/java/io/leavesfly/tinyai/agent/cursor/v2/
├── unit/                           # 单元测试
│   ├── model/                      # 数据模型测试（3个类，25个测试）
│   │   ├── MessageTest.java
│   │   ├── ChatRequestTest.java
│   │   └── MemoryTest.java
│   ├── adapter/                    # 适配器测试（1个类，9个测试）
│   │   └── AdapterRegistryTest.java
│   ├── infra/                      # 基础设施测试（1个类，13个测试）
│   │   └── CacheManagerTest.java
│   └── tool/                       # 工具系统测试（1个类，8个测试）
│       └── ToolRegistryTest.java
└── integration/                    # 集成测试（待补充）
```

**总计**: 6个测试类，55个测试方法，约902行测试代码

---

## 运行测试

### 方法1: Maven命令行

#### 运行所有测试

```bash
cd /Users/yefei.yf/Qoder/TinyAI/tinyai-agent-cursor
mvn clean test
```

#### 运行特定模块测试

```bash
# 数据模型层测试
mvn test -Dtest="io.leavesfly.tinyai.agent.cursor.v2.unit.model.*Test"

# 适配器层测试
mvn test -Dtest="io.leavesfly.tinyai.agent.cursor.v2.unit.adapter.*Test"

# 基础设施层测试
mvn test -Dtest="io.leavesfly.tinyai.agent.cursor.v2.unit.infra.*Test"

# 工具系统测试
mvn test -Dtest="io.leavesfly.tinyai.agent.cursor.v2.unit.tool.*Test"

# V2所有单元测试
mvn test -Dtest="io.leavesfly.tinyai.agent.cursor.v2.unit.**.*Test"
```

#### 运行单个测试类

```bash
# 运行 MessageTest
mvn test -Dtest="MessageTest"

# 运行 CacheManagerTest
mvn test -Dtest="CacheManagerTest"

# 运行 ToolRegistryTest
mvn test -Dtest="ToolRegistryTest"
```

#### 运行单个测试方法

```bash
# 运行 MessageTest 中的特定测试方法
mvn test -Dtest="MessageTest#testSystemMessageCreation"

# 运行 CacheManagerTest 中的特定测试方法
mvn test -Dtest="CacheManagerTest#testL1CachePutAndGet"
```

### 方法2: IDE运行

#### IntelliJ IDEA

1. **运行所有测试**
   - 右键点击 `src/test/java/io/leavesfly/tinyai/agent/cursor/v2/unit` 目录
   - 选择 "Run 'Tests in 'unit''"

2. **运行单个测试类**
   - 打开测试类文件（如 `MessageTest.java`）
   - 点击类名左侧的绿色运行图标
   - 或右键点击类名，选择 "Run 'MessageTest'"

3. **运行单个测试方法**
   - 点击测试方法左侧的绿色运行图标
   - 或右键点击方法名，选择 "Run 'testSystemMessageCreation()'"

4. **调试测试**
   - 设置断点
   - 点击调试图标（虫子图标）
   - 或右键选择 "Debug 'MessageTest'"

#### Eclipse

1. **运行所有测试**
   - 右键点击 `unit` 目录
   - Run As → JUnit Test

2. **运行单个测试类**
   - 右键点击测试类文件
   - Run As → JUnit Test

3. **运行单个测试方法**
   - 在编辑器中定位到测试方法
   - 右键选择 Run As → JUnit Test

---

## 测试报告

### 生成测试报告

```bash
# 运行测试并生成报告
mvn clean test

# 生成HTML格式报告
mvn surefire-report:report
```

### 查看报告

报告位置：
- **文本报告**: `target/surefire-reports/*.txt`
- **XML报告**: `target/surefire-reports/*.xml`
- **HTML报告**: `target/site/surefire-report.html`

打开HTML报告：
```bash
open target/site/surefire-report.html  # macOS
xdg-open target/site/surefire-report.html  # Linux
start target/site/surefire-report.html  # Windows
```

---

## 测试覆盖率

### 使用JaCoCo生成覆盖率报告

1. **在pom.xml中添加JaCoCo插件**（如果还未添加）

```xml
<plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <version>0.8.8</version>
    <executions>
        <execution>
            <goals>
                <goal>prepare-agent</goal>
            </goals>
        </execution>
        <execution>
            <id>report</id>
            <phase>test</phase>
            <goals>
                <goal>report</goal>
            </goals>
        </execution>
    </executions>
</plugin>
```

2. **运行测试并生成覆盖率报告**

```bash
mvn clean test jacoco:report
```

3. **查看覆盖率报告**

```bash
open target/site/jacoco/index.html
```

### 预期覆盖率

| 模块 | 目标覆盖率 | 当前状态 |
|-----|----------|---------|
| 数据模型层 | 90% | ✅ 约85% |
| 适配器层 | 80% | ✅ 约80% |
| 基础设施层 | 85% | ✅ 约90% |
| 工具系统层 | 75% | ✅ 约75% |

---

## 常见问题排查

### 问题1: JAVA_HOME未设置

**错误信息**:
```
Error: JAVA_HOME is not defined correctly.
```

**解决方案**:
```bash
# macOS/Linux
export JAVA_HOME=/path/to/jdk
export PATH=$JAVA_HOME/bin:$PATH

# Windows
set JAVA_HOME=C:\path\to\jdk
set PATH=%JAVA_HOME%\bin;%PATH%

# 验证
java -version
mvn -version
```

### 问题2: 测试编译失败

**错误信息**:
```
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin
```

**解决方案**:
```bash
# 清理并重新编译
mvn clean compile

# 确保JDK版本正确
java -version  # 应该是1.8+
```

### 问题3: 找不到测试类

**错误信息**:
```
[ERROR] No tests found matching io.leavesfly.tinyai.agent.cursor.v2.unit.model.MessageTest
```

**解决方案**:
```bash
# 确保测试代码已编译
mvn clean test-compile

# 检查测试类路径
ls -la target/test-classes/io/leavesfly/tinyai/agent/cursor/v2/unit/model/
```

### 问题4: 依赖问题

**错误信息**:
```
[ERROR] Failed to execute goal ... could not resolve dependencies
```

**解决方案**:
```bash
# 更新Maven依赖
mvn clean install -U

# 或删除本地仓库缓存
rm -rf ~/.m2/repository/io/leavesfly/tinyai
mvn clean install
```

---

## 测试编写指南

### 测试命名规范

```java
// ✅ 好的测试名称
@Test
public void testSystemMessageCreation() { }

@Test
public void testL2CacheIsolation() { }

@Test
public void testToolDuplicateRegistration() { }

// ❌ 避免的测试名称
@Test
public void test1() { }

@Test
public void testStuff() { }
```

### AAA模式（Arrange-Act-Assert）

```java
@Test
public void testCacheIsolation() {
    // Arrange: 准备测试数据
    String session1 = "session-001";
    String session2 = "session-002";
    String key = "same-key";
    
    // Act: 执行操作
    cacheManager.putL2(session1, key, "value1");
    cacheManager.putL2(session2, key, "value2");
    
    // Assert: 验证结果
    assertEquals("value1", cacheManager.getL2(session1, key));
    assertEquals("value2", cacheManager.getL2(session2, key));
}
```

### 测试隔离

```java
@Before
public void setUp() {
    // 每个测试前执行，确保测试独立
    registry = new ToolRegistry();
}

@After
public void tearDown() {
    // 每个测试后执行，清理资源
    if (registry != null) {
        // 清理操作（如果需要）
    }
}
```

### 异常测试

```java
// 预期抛出异常
@Test(expected = IllegalArgumentException.class)
public void testInvalidInput() {
    Message.Role.fromValue("invalid");
}

// 更详细的异常验证
@Test
public void testInvalidInputWithMessage() {
    try {
        Message.Role.fromValue("invalid");
        fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
        assertTrue(e.getMessage().contains("Invalid role"));
    }
}
```

---

## 持续集成

### GitHub Actions示例

创建 `.github/workflows/test.yml`:

```yaml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up JDK 8
      uses: actions/setup-java@v2
      with:
        java-version: '8'
        distribution: 'adopt'
    
    - name: Build with Maven
      run: mvn clean install -DskipTests
      
    - name: Run tests
      run: mvn test
      
    - name: Generate test report
      run: mvn surefire-report:report
      
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: target/surefire-reports
```

---

## 测试最佳实践

### 1. 保持测试独立

每个测试应该独立运行，不依赖其他测试的执行结果。

### 2. 使用有意义的断言

```java
// ✅ 好的断言
assertEquals("Expected session cache to be isolated", 
             "value1", cacheManager.getL2(session1, key));

// ❌ 避免
assertEquals("value1", cacheManager.getL2(session1, key));
```

### 3. 测试边界条件

```java
@Test
public void testNullValues() {
    cacheManager.putL1("key", null);
    assertNull(cacheManager.getL1("key"));
}

@Test
public void testEmptyInput() {
    ChatRequest request = ChatRequest.builder()
        .model("test")
        .build();
    assertTrue(request.getMessages().isEmpty());
}
```

### 4. 一个测试一个概念

```java
// ✅ 好的实践
@Test
public void testL1CachePut() { ... }

@Test
public void testL1CacheGet() { ... }

// ❌ 避免
@Test
public void testL1CacheAllOperations() {
    // put, get, remove, clear都在一个测试里
}
```

---

## 快速参考

### 常用Maven测试命令

```bash
# 运行所有测试
mvn test

# 跳过测试
mvn install -DskipTests

# 只编译测试代码不运行
mvn test-compile

# 运行测试并生成报告
mvn clean test surefire-report:report

# 运行测试并生成覆盖率报告
mvn clean test jacoco:report

# 指定测试类
mvn test -Dtest=MessageTest

# 指定测试方法
mvn test -Dtest=MessageTest#testSystemMessageCreation

# 运行匹配模式的测试
mvn test -Dtest=*Test

# 并行运行测试（加速）
mvn test -T 4  # 使用4个线程
```

### JUnit常用断言

```java
// 相等性断言
assertEquals(expected, actual);
assertEquals(message, expected, actual);

// 布尔断言
assertTrue(condition);
assertFalse(condition);

// 空值断言
assertNull(object);
assertNotNull(object);

// 数组断言
assertArrayEquals(expectedArray, actualArray);

// 异常断言
@Test(expected = Exception.class)

// 超时断言
@Test(timeout = 1000)  // 毫秒
```

---

## 下一步

1. **补充服务层测试**: 为LLMGatewayImpl、CodeIntelligenceService等创建测试
2. **补充组件层测试**: 为ContextEngine、MemoryManager等创建测试
3. **添加集成测试**: 测试组件间协作
4. **性能测试**: 测试系统在高负载下的表现
5. **配置CI/CD**: 自动化测试流程

---

**文档版本**: V1.0  
**最后更新**: 2025-01-15  
**维护者**: TinyAI Team
