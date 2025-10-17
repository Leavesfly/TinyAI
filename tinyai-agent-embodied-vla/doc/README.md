# TinyAI VLA 训练示例集

本目录包含TinyAI VLA模块的完整训练示例，展示如何在不同任务场景下训练视觉-语言-动作具身智能体。

## 📚 示例列表

### 1. PickAndPlace训练示例 (`PickAndPlaceTrainingExample.java`)

**难度**: ⭐⭐ 入门级

**学习目标**:
- 掌握VLA智能体的基本使用流程
- 理解环境初始化和配置
- 学习行为克隆训练方法
- 掌握模型评估与保存

**训练内容**:
```java
// 完整训练流程
1. 创建PickAndPlace任务环境
2. 初始化VLA智能体
3. 收集专家演示数据（可选）
4. 行为克隆训练
5. 模型性能评估
6. 训练过程可视化
7. 模型保存与加载
```

**运行方式**:
```bash
cd /Users/yefei.yf/Qoder/TinyAI
javac -cp "tinyai-agent-embodied-vla/target/classes:..." \
  examples/PickAndPlaceTrainingExample.java
java -cp "..." PickAndPlaceTrainingExample
```

**预期结果**:
- 训练100个回合后，平均奖励 > 80
- 成功率 > 85%
- 单个回合平均步数 < 50

---

### 2. StackBlocks训练示例 (`StackBlocksTrainingExample.java`)

**难度**: ⭐⭐⭐ 进阶级

**学习目标**:
- 掌握课程学习（Curriculum Learning）策略
- 理解复杂任务的分阶段训练
- 学习如何处理序列化决策问题
- 掌握训练曲线分析

**训练策略**:
```
阶段1: 堆叠2个方块 (30回合)
   ↓ 难度递增
阶段2: 堆叠3个方块 (50回合)
   ↓ 难度递增
阶段3: 堆叠4个方块 (70回合)
   ↓ 最终评估
综合测试: 所有难度下的性能
```

**运行方式**:
```bash
java -cp "..." StackBlocksTrainingExample
```

**预期结果**:
- 2块方块：成功率 > 90%
- 3块方块：成功率 > 70%
- 4块方块：成功率 > 50%

**关键技术**:
- **课程学习**: 从简单到复杂逐步训练
- **奖励塑形**: 根据堆叠高度给予递增奖励
- **稳定性检测**: 判断方块是否稳定堆叠

---

### 3. 模型微调示例 (`ModelFineTuningExample.java`)

**难度**: ⭐⭐⭐⭐ 高级

**学习目标**:
- 掌握迁移学习（Transfer Learning）方法
- 理解预训练与微调的区别
- 学习层冻结（Layer Freezing）技术
- 掌握学习率调优策略

**训练流程**:
```
步骤1: 源任务预训练 (PickAndPlace, 50回合)
   ↓
步骤2: 保存预训练模型
   ↓
步骤3: 冻结编码器层
   ↓
步骤4: 目标任务微调 (OpenDrawer, 20回合)
   ↓
步骤5: 性能对比分析
```

**运行方式**:
```bash
java -cp "..." ModelFineTuningExample
```

**微调策略**:
| 层类型 | 预训练 | 微调 |
|--------|--------|------|
| Vision Encoder | ✓ 训练 | ✗ 冻结 |
| Language Encoder | ✓ 训练 | ✗ 冻结 |
| Proprioception Encoder | ✓ 训练 | ✗ 冻结 |
| Transformer Core | ✓ 训练 | ✓ 训练 |
| Action Decoder | ✓ 训练 | ✓ 训练 |

**学习率配置**:
- 预训练: 0.001
- 微调: 0.0001 (10倍降低)

**预期优势**:
- 训练时间减少 60%
- 所需数据减少 70%
- 性能提升 15-30%

---

## 🚀 快速开始

### 前置条件

1. **环境要求**:
   - JDK 17 或更高
   - Maven 3.6+
   - 至少 4GB 可用内存

2. **依赖安装**:
```bash
cd /Users/yefei.yf/Qoder/TinyAI
mvn clean install -pl tinyai-agent-embodied-vla
```

### 运行示例

**方法1: 使用Maven**
```bash
# 运行PickAndPlace示例
mvn exec:java -pl tinyai-agent-embodied-vla \
  -Dexec.mainClass="io.leavesfly.tinyai.agent.vla.examples.PickAndPlaceTrainingExample"

# 运行StackBlocks示例
mvn exec:java -pl tinyai-agent-embodied-vla \
  -Dexec.mainClass="io.leavesfly.tinyai.agent.vla.examples.StackBlocksTrainingExample"

# 运行微调示例
mvn exec:java -pl tinyai-agent-embodied-vla \
  -Dexec.mainClass="io.leavesfly.tinyai.agent.vla.examples.ModelFineTuningExample"
```

**方法2: 直接编译运行**
```bash
# 编译
javac -d build -cp "..." examples/*.java

# 运行
java -cp "build:..." PickAndPlaceTrainingExample
```

---

## 📊 训练技巧

### 1. 超参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **learning_rate** | 0.001 | 初始学习率，根据损失曲线调整 |
| **hidden_dim** | 768 | 隐藏层维度，影响模型容量 |
| **num_heads** | 8 | 注意力头数，通常为8或16 |
| **num_layers** | 6 | Transformer层数，6-12层为佳 |
| **batch_size** | 32 | 批次大小，受内存限制 |

### 2. 训练策略

**策略1: 从简单到复杂**
```
简单任务 → 中等任务 → 复杂任务
（课程学习）
```

**策略2: 预训练+微调**
```
大规模预训练 → 特定任务微调
（迁移学习）
```

**策略3: 多任务学习**
```
同时训练多个相关任务
（共享表示学习）
```

### 3. 常见问题解决

**问题1: 训练不收敛**
- ✓ 降低学习率（除以10）
- ✓ 增加训练回合数
- ✓ 检查奖励函数设计
- ✓ 使用学习率调度器

**问题2: 过拟合**
- ✓ 增加训练数据多样性
- ✓ 使用Dropout正则化
- ✓ 提前停止（Early Stopping）
- ✓ 数据增强

**问题3: 内存不足**
- ✓ 减小batch_size
- ✓ 使用梯度累积
- ✓ 降低模型维度
- ✓ 使用梯度检查点

---

## 📈 性能基准

### PickAndPlace任务

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 平均奖励 | > 80 | 100回合后 |
| 成功率 | > 85% | 完成任务比例 |
| 平均步数 | < 50 | 每个回合 |
| 训练时间 | < 30分钟 | CPU训练 |

### StackBlocks任务

| 方块数 | 成功率目标 | 平均奖励目标 |
|--------|-----------|------------|
| 2块 | > 90% | > 180 |
| 3块 | > 70% | > 250 |
| 4块 | > 50% | > 300 |

### 微调效果对比

| 方法 | 训练时间 | 数据需求 | 最终性能 |
|------|---------|---------|---------|
| 从头训练 | 100% | 100% | 基准 |
| 微调 | 40% | 30% | +20% |

---

## 🔬 进阶实验

### 实验1: 不同编码器对比

测试不同视觉编码器架构的性能：
- ResNet-18
- ResNet-50
- Vision Transformer (ViT)

### 实验2: 注意力机制分析

可视化跨模态注意力权重：
- 语言指令如何影响视觉注意力
- 不同任务阶段的注意力模式变化

### 实验3: 多任务学习

同时训练多个任务：
```java
Task[] tasks = {
    PICK_AND_PLACE,
    STACK_BLOCKS,
    OPEN_DRAWER
};
// 共享编码器，独立解码器
```

---

## 📚 相关文档

- [技术架构文档](../doc/技术架构文档.md)
- [使用指南](../doc/使用指南.md)
- [最佳实践指南](../doc/最佳实践指南.md)
- [故障排查手册](../doc/故障排查手册.md)

---

## 🤝 贡献

欢迎贡献更多训练示例！

**示例需求**:
- [ ] PourWater任务训练示例
- [ ] AssembleParts任务训练示例
- [ ] 多智能体协作示例
- [ ] 真实机器人部署示例

**贡献指南**:
1. Fork本项目
2. 创建特性分支
3. 添加示例代码和文档
4. 提交Pull Request

---

**TinyAI VLA Team**  
*Vision-Language-Action Intelligence for Everyone*
