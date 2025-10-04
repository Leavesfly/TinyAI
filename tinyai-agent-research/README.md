# 🔬 TinyAI 深度研究智能体模块 (tinyai-agent-research)

## 📖 项目概述

TinyAI 深度研究智能体模块是一个基于 Java 实现的高级研究 AI 系统，提供多阶段推理、知识图谱构建、自适应学习等先进功能。该模块专门用于执行深度、系统性的研究任务，为用户提供全面、准确的研究结果。

## 🌟 核心特性

### 1. 🧠 多阶段研究管道
- **问题分析阶段**: 深入分析问题复杂度和关键概念
- **信息收集阶段**: 多源信息汇集和知识检索
- **深度分析阶段**: 智能推理和模式识别
- **综合处理阶段**: 信息整合和洞察生成
- **验证检查阶段**: 逻辑一致性检查和证据评估
- **结论生成阶段**: 最终答案生成和后续建议

### 2. 🗺️ 动态知识图谱
- **自动节点发现**: 智能识别和连接相关知识点
- **相似度计算**: 基于内容的知识关联度分析
- **多领域支持**: 跨领域知识整合和管理
- **实时更新**: 研究过程中动态扩展知识库

### 3. 🎯 自适应推理机制
- **推理模式自选**: 根据问题特征智能选择推理策略
  - **快速模式** (Quick): 紧急简单问题的快速响应
  - **彻底模式** (Thorough): 复杂问题的全面深入分析
  - **创意模式** (Creative): 创新性问题的发散思维
  - **分析模式** (Analytical): 数据驱动的系统分析
  - **系统模式** (Systematic): 结构化的逐步推理

### 4. 🔧 多工具集成
- **网络搜索**: 最新信息和观点获取
- **文献检索**: 学术资源和研究报告
- **数据分析**: 统计分析和趋势识别
- **专家知识**: 领域专家见解整合
- **趋势分析**: 发展趋势和未来预测

### 5. 📊 持续学习能力
- **性能跟踪**: 研究质量和效率监控
- **自我改进**: 基于历史表现的策略优化
- **领域专业化**: 特定领域的深度知识积累

## 🏗️ 系统架构

```
DeepResearch Agent
├── 🧠 IntelligentReasoner      # 智能推理器
│   ├── 推理模式选择
│   ├── 多策略推理引擎
│   └── 性能历史跟踪
├── 🗺️ KnowledgeGraph           # 知识图谱
│   ├── 动态节点管理
│   ├── 智能连接发现
│   ├── 相似度计算
│   └── 领域概览生成
├── ⚙️ ResearchPipeline         # 研究管道
│   ├── 六阶段研究流程
│   ├── 工具调用管理
│   ├── 结果整合处理
│   └── 质量评估体系
└── 🎯 DeepResearchAgent        # 核心Agent
    ├── 研究任务协调
    ├── 学习指标更新
    ├── 多模式研究支持
    └── 性能报告生成
```

## 🚀 快速开始

### 基础使用

```java
import io.leavesfly.tinyai.agent.research.DeepResearchAgent;
import java.util.Map;

// 创建研究Agent
DeepResearchAgent agent = new DeepResearchAgent("我的研究助手", "人工智能");

// 添加领域知识
agent.addDomainKnowledge("深度学习是机器学习的一个分支", "人工智能", "concept");

// 执行研究
Map<String, Object> result = agent.research(
    "深度学习在计算机视觉中的应用",
    3,      // 复杂度 (1-5)
    4,      // 深度要求 (1-5)
    2       // 紧急程度 (1-5)
);

// 查看结果
System.out.println("研究答案: " + result.get("finalAnswer"));
System.out.println("置信度: " + result.get("totalConfidence"));
System.out.println("关键洞察: " + result.get("keyInsights"));
```

### 高级功能

#### 1. 探索性研究
```java
// 深入探索某个主题
Map<String, Object> explorationResult = agent.exploreResearchTopic("量子计算", 3);
System.out.println("探索了 " + explorationResult.get("totalQuestionsExplored") + " 个相关问题");
```

#### 2. 协作式研究
```java
// 多视角分析
List<String> perspectives = Arrays.asList("技术", "法律", "社会", "哲学");
Map<String, Object> collaborationResult = agent.collaborativeResearch(
    "人工智能的伦理问题", perspectives);
```

#### 3. 性能监控
```java
// 获取Agent性能报告
Map<String, Object> performance = agent.getPerformanceReport();
System.out.println("研究次数: " + performance.get("researchHistoryCount"));
```

## 📋 研究配置参数

### 复杂度等级 (complexity)
- **1**: 简单问题，基础概念
- **2**: 普通问题，需要一些分析
- **3**: 中等复杂问题，需要多角度思考
- **4**: 复杂问题，需要深入研究
- **5**: 极复杂问题，需要专家级分析

### 深度要求 (depth_required)
- **1**: 浅层回答，基本信息
- **2**: 一般深度，包含一些细节
- **3**: 中等深度，较全面分析
- **4**: 深入分析，多层次解构
- **5**: 极深入，专业级研究

### 紧急程度 (urgency)
- **1**: 不紧急，可以长时间研究
- **2**: 一般紧急，正常时间要求
- **3**: 较紧急，需要适度加速
- **4**: 紧急，需要快速响应
- **5**: 极紧急，优先快速模式

## 🔧 推理模式详解

### Quick Mode (快速模式)
**适用场景**: 简单问题、紧急响应  
**特点**: 快速关键词匹配、直接调用已有知识、生成初步答案

### Thorough Mode (彻底模式)
**适用场景**: 复杂问题、深度要求高  
**特点**: 问题多维度分解、系统性信息收集、多角度分析验证

### Creative Mode (创意模式)  
**适用场景**: 创新问题、发散思维  
**特点**: 非传统思考角度、跨领域知识联想、假设性方案生成

### Analytical Mode (分析模式)
**适用场景**: 数据分析、比较研究  
**特点**: 变量因素识别、因果关系建模、量化权重分析

### Systematic Mode (系统模式)
**适用场景**: 结构化研究、系统梳理  
**特点**: 概念框架构建、逻辑顺序收集、知识结构图建立

## 📊 质量评估体系

研究质量通过以下维度评估：

- **完整性** (20%): 研究步骤的全面程度
- **深度** (25%): 思考和分析的深入程度  
- **多样性** (20%): 工具和方法的多样化使用
- **洞察力** (20%): 发现关键洞察的能力
- **置信度** (15%): 整体结果的可信程度

最终质量评分范围：0.0 - 1.0

## 🧪 运行演示

### 方法1: 运行演示程序
```bash
cd tinyai-agent-research
mvn compile exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.research.ResearchDemo"
```

### 方法2: 简单演示
```java
public static void main(String[] args) {
    DeepResearchAgent agent = new DeepResearchAgent();
    agent.addDomainKnowledge("Java是一种面向对象的编程语言", "编程");
    
    Map<String, Object> result = agent.research("Java的特点是什么？");
    System.out.println("研究结果: " + result.get("finalAnswer"));
}
```

## 📈 示例输出

```
🔍 开始深度研究: 什么是深度学习？
🎨 研究配置: 复杂度=3, 深度=3, 紧急度=2
🧠 选择推理模式: systematic

🎯 研究结果:
  ✅ 置信度: 0.75
  📋 研究步骤数: 18
  🔧 使用工具数: 3
  💡 关键洞察数: 4
  🏆 质量评分: 0.901
  🧠 推理模式: systematic

📖 最终答案:
基于深度研究分析，对问题'什么是深度学习？'的研究发现了 4 个关键洞察，
整体置信度为 0.75。研究表明该问题具有多维度特征，需要综合考虑多个因素。

💡 关键洞察:
  1. 发现关键模式: 技术发展呈指数增长趋势，跨领域应用越来越普遍
  2. 识别知识缺口: 缺乏最新的实证研究数据，理论与实践之间的桥梁待建立
  3. 研究过程揭示了问题的多层次结构
  4. 多工具融合提供了更全面的视角
```

## 🔮 高级特性

### 知识图谱可视化
```java
// 获取领域概览
Map<String, Object> overview = agent.getKnowledgeOverview("人工智能");
System.out.println("节点总数: " + overview.get("totalNodes"));
System.out.println("中心节点: " + overview.get("centralNodes"));
```

### 批量研究
```java
String[] queries = {
    "机器学习算法比较",
    "深度学习发展趋势", 
    "AI伦理考量"
};

for (String query : queries) {
    Map<String, Object> result = agent.research(query, 2, 3, 2);
    // 处理结果...
}
```

## 🛠️ 技术架构

### 依赖组件
- **Java 8+**: 基础运行环境
- **JUnit 4**: 单元测试框架
- **TinyAI Agent Base**: 基础智能体框架
- **TinyAI Agent Pattern**: 智能体模式库
- **TinyAI Agent Multi**: 多智能体系统

### 模块结构
- `DeepResearchAgent`: 主要接口类
- `IntelligentReasoner`: 推理引擎核心
- `KnowledgeGraph`: 知识图谱管理
- `ResearchPipeline`: 研究流程控制
- `ResearchPhase`: 研究阶段枚举
- `ReasoningMode`: 推理模式枚举

### 数据存储
- **内存存储**: 运行时知识图谱和会话数据
- **对象序列化**: 配置和结果序列化
- **历史记录**: 研究历史和性能指标

## 🎯 使用场景

### 学术研究
- 文献综述和研究现状分析
- 跨学科知识整合
- 研究方向探索

### 商业分析  
- 市场趋势研究
- 竞争对手分析
- 技术可行性评估

### 技术调研
- 新技术评估
- 解决方案对比
- 最佳实践研究

### 教育培训
- 知识点深入解析
- 概念关系梳理
- 学习路径规划

## 📦 构建和测试

### 编译项目
```bash
mvn clean compile
```

### 运行测试
```bash
mvn test
```

### 打包项目
```bash
mvn package
```

### 运行演示
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.research.ResearchDemo"
```

## ⚠️ 注意事项

1. **Java版本**: 需要Java 8或更高版本
2. **内存使用**: 复杂研究任务可能需要较多内存
3. **处理时间**: 深度研究可能需要较长处理时间
4. **知识质量**: 输出质量依赖于输入知识库的质量
5. **模拟环境**: 当前版本使用模拟的工具响应，实际部署可接入真实API

## 🔄 版本信息

### v1.0.0 (当前版本)
- ✅ 完整的六阶段研究管道
- ✅ 五种智能推理模式
- ✅ 动态知识图谱构建
- ✅ 多工具集成支持
- ✅ 自适应学习机制
- ✅ 协作式和探索性研究
- ✅ 完整的质量评估体系
- ✅ 全面的单元测试覆盖

### 未来规划
- 🔄 真实LLM API集成
- 🔄 多模态信息处理
- 🔄 分布式研究能力
- 🔄 可视化界面开发
- 🔄 持久化存储支持

## 📞 支持与反馈

如有问题或建议，欢迎提出：
- 📧 技术问题：通过代码注释查看实现细节
- 🐛 Bug报告：检查错误日志和参数配置
- 💡 功能建议：参考架构扩展开发

## 🤝 贡献指南

欢迎为项目贡献代码：
1. Fork 项目
2. 创建特性分支
3. 提交代码变更
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

---

**TinyAI 深度研究智能体 - 让AI驱动的深度研究触手可及！** 🚀

*构建时间: 2025年10月4日*  
*版本: v1.0.0*  
*作者: 山泽*