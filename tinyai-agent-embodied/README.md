# TinyAI 具身智能模块（Embodied Intelligence）

## 📖 模块简介

`tinyai-agent-embodied` 是 TinyAI 智能体系统层的重要组成部分，专注于**具身智能**（Embodied Intelligence）技术的实现。本模块以**自动驾驶场景**为典型应用案例，展现了智能体通过与物理环境的直接交互来感知、学习和决策的完整能力。

### 核心特性

- 🚗 **完整的自动驾驶模拟环境**：基于简化自行车模型的车辆动力学
- 🧠 **感知-决策-执行闭环**：模拟真实智能体的完整工作流程
- 🔄 **端到端学习支持**：支持强化学习、端到端学习等多种学习策略
- 🎯 **场景化设计**：内置多种驾驶场景（高速公路、城市道路、测试场等）
- 📊 **可扩展架构**：模块化设计，便于添加新传感器、新场景、新算法

## 🏗️ 系统架构

```
具身智能体架构（✅ 已完成）
├── 环境仿真层
│   ├── SimpleDrivingEnv - 驾驶环境 ✅
│   ├── VehicleDynamics - 车辆动力学 ✅
│   └── ScenarioLoader - 场景加载器 ✅
│
├── 感知层 ✅
│   ├── SensorSuite - 传感器组件 ✅
│   │   ├── CameraSensor - 相机传感器 ✅
│   │   ├── LidarSensor - 激光雷达 ✅
│   │   ├── IMUSensor - 惯性测量单元 ✅
│   │   ├── GPSSensor - GPS定位 ✅
│   │   └── SpeedometerSensor - 速度计 ✅
│   ├── PerceptionModule - 感知处理 ✅
│   └── FeatureExtractor - 特征提取 ✅
│
├── 决策层 ✅
│   ├── DecisionModule - 决策模块 ✅
│   ├── SafetyConstraint - 安全约束 ✅
│   └── PolicyNetwork - 策略网络 ✅
│
├── 执行层 ✅
│   └── ExecutionModule - 执行模块 ✅
│
└── 学习层 ✅
    ├── LearningEngine - 学习引擎 ✅
    ├── DQNLearner - DQN学习器 ✅
    ├── EndToEndLearner - 端到端学习器 ✅
    └── EpisodicMemory - 情景记忆 ✅
```

## 🚀 快速开始

### 1. 环境要求

- JDK 17 或更高版本
- Maven 3.6+

### 2. 编译模块

```bash
cd /path/to/TinyAI
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
mvn clean compile -pl tinyai-agent-embodied -am
```

### 3. 运行演示程序

#### 简单环境演示
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.embodied.SimpleDemo" \
              -pl tinyai-agent-embodied
```

#### 完整智能体演示
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.embodied.AgentDemo" \
              -pl tinyai-agent-embodied
```

### 4. 基础使用示例

#### 示例1：简单环境交互

```java
// 1. 创建环境配置
EnvironmentConfig config = EnvironmentConfig.createTestConfig();

// 2. 初始化驾驶环境
DrivingEnvironment env = new SimpleDrivingEnv(config);

// 3. 重置环境
PerceptionState state = env.reset();

// 4. 交互循环
for (int step = 0; step < 100; step++) {
    // 生成动作（简单策略：直行）
    DrivingAction action = new DrivingAction(0.0, 0.3, 0.0);
    
    // 执行动作
    StepResult result = env.step(action);
    
    // 处理结果
    if (result.isDone()) {
        break;
    }
}

// 5. 清理资源
env.close();
```

#### 示例2：完整智能体使用

```java
// 1. 创建智能体配置
EnvironmentConfig config = EnvironmentConfig.createHighwayConfig();

// 2. 创建具身智能体
EmbodiedAgent agent = new EmbodiedAgent(config);

// 3. 单步运行模式
agent.reset();
for (int step = 0; step < 100; step++) {
    StepResult result = agent.step();
    System.out.println("步骤: " + step + ", 奖励: " + result.getReward());
    
    if (result.isDone()) {
        break;
    }
}

// 4. 完整情景运行
Episode episode = agent.runEpisode(200);
System.out.println("情景长度: " + episode.getLength());
System.out.println("总奖励: " + episode.getTotalReward());
System.out.println("平均奖励: " + episode.getAverageReward());

// 5. 清理资源
agent.close();
```

#### 示例3：使用学习引擎

```java
// 1. 创建学习引擎
LearningEngine learningEngine = new LearningEngine();
learningEngine.setStrategy(LearningStrategy.DQN);

// 2. 创建智能体并运行训练情景
EmbodiedAgent agent = new EmbodiedAgent(config);
for (int i = 0; i < 10; i++) {
    Episode episode = agent.runEpisode(200);
    
    // 从情景中学习
    learningEngine.learnFromEpisode(episode);
    
    System.out.println("Episode " + i + ": Reward = " + episode.getTotalReward());
}

// 3. 保存训练好的策略
learningEngine.savePolicy("highway_policy.model");
```

## 📊 核心组件说明

### 1. 数据模型

| 类名 | 说明 | 主要字段 |
|------|------|---------|
| `VehicleState` | 车辆状态 | position, speed, heading |
| `PerceptionState` | 感知状态 | vehicleState, obstacles, laneInfo |
| `DrivingAction` | 驾驶动作 | steering, throttle, brake |
| `StepResult` | 步进结果 | observation, reward, done |
| `ObstacleInfo` | 障碍物信息 | type, position, velocity |
| `LaneGeometry` | 车道信息 | laneId, width, deviation |

### 2. 车辆动力学

采用**简化自行车模型**（Bicycle Model）：

```
状态更新方程：
x(t+Δt) = x(t) + v·cos(θ)·Δt
y(t+Δt) = y(t) + v·sin(θ)·Δt
θ(t+Δt) = θ(t) + (v/L)·tan(δ)·Δt
v(t+Δt) = v(t) + a·Δt
```

其中：
- `L`: 车辆轴距（2.7米）
- `δ`: 转向角
- `a`: 加速度

### 3. 场景配置

内置场景类型：

| 场景 | 车道数 | 限速 | 车辆密度 | 复杂度 |
|------|-------|------|---------|--------|
| TEST | 2 | 60 km/h | 5 辆/km | ★☆☆☆☆ |
| HIGHWAY | 3 | 120 km/h | 20 辆/km | ★★☆☆☆ |
| URBAN | 2 | 60 km/h | 40 辆/km | ★★★★☆ |
| RURAL | 2 | 80 km/h | 10 辆/km | ★★☆☆☆ |
| PARKING_LOT | 1 | 20 km/h | 50 辆/km | ★★★☆☆ |
| INTERSECTION | 3 | 50 km/h | 30 辆/km | ★★★★★ |

### 4. 奖励函数

组合奖励设计：

```
R_total = w1·R_speed + w2·R_lane + w3·R_collision + w4·R_comfort

其中：
R_speed = 1 - |v - v_target| / v_max           (速度奖励)
R_lane = exp(-lateral_deviation²)              (车道保持)
R_collision = -100 (发生碰撞) / -10·d (距离过近) (碰撞惩罚)
R_comfort = -|a| - |δ|                          (舒适性)

权重配置：w1=0.3, w2=0.4, w3=1.0, w4=0.1
```

## 🎯 开发进度

### ✅ 已完成（全部8个阶段）

**阶段一：基础架构搭建**
- [x] 项目基础架构与Maven配置
- [x] 核心数据模型（16个类）
- [x] 枚举类型定义（5个）
- [x] 核心接口规范（2个）

**阶段二：环境模拟实现**
- [x] 车辆动力学模型（VehicleDynamics）
- [x] 简单驾驶环境（SimpleDrivingEnv）
- [x] 场景加载器（ScenarioLoader）
- [x] 6种内置场景配置

**阶段三：感知模块开发**
- [x] 传感器系统（5种传感器实现）
- [x] 传感器组件集合（SensorSuite）
- [x] 感知处理模块（PerceptionModule）
- [x] 特征提取器（FeatureExtractor）

**阶段四：决策执行模块**
- [x] 决策模块（DecisionModule）
- [x] 策略网络（PolicyNetwork）
- [x] 安全约束（SafetyConstraint）
- [x] 执行模块（ExecutionModule）

**阶段五：学习引擎集成**
- [x] 学习引擎核心（LearningEngine）
- [x] DQN强化学习器（DQNLearner）
- [x] 端到端学习器（EndToEndLearner）
- [x] 情景记忆管理（EpisodicMemory）

**阶段六：智能体核心**
- [x] 完整具身智能体（EmbodiedAgent）
- [x] 感知-决策-执行-学习闭环
- [x] 单步与情景运行支持

**阶段七：测试与验证**
- [x] 完整演示程序（AgentDemo）
- [x] 简单验证程序（SimpleDemo）
- [x] 编译验证通过
- [x] 运行验证通过

**阶段八：文档编写**
- [x] README文档（255行）
- [x] 技术架构文档（485行）
- [x] 实施总结文档（471行）

### 📊 完成统计

| 类别 | 数量 | 状态 |
|-----|------|------|
| Java类文件 | 35个 | ✅ 完成 |
| 数据模型 | 16个 | ✅ 完成 |
| 环境实现 | 4个 | ✅ 完成 |
| 传感器 | 6个 | ✅ 完成 |
| 感知模块 | 2个 | ✅ 完成 |
| 决策模块 | 3个 | ✅ 完成 |
| 执行模块 | 1个 | ✅ 完成 |
| 学习引擎 | 4个 | ✅ 完成 |
| 演示程序 | 2个 | ✅ 完成 |
| 文档资料 | 3个 | ✅ 完成 |

## 📚 依赖关系

本模块依赖以下TinyAI核心模块：

```xml
<dependencies>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-ndarr</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-func</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-nnet</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-ml</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-rl</artifactId>
    </dependency>
</dependencies>
```

## 🔬 技术细节

### 车辆物理参数

```java
wheelbase = 2.7m          // 轴距
maxSteeringAngle = 0.6rad  // 最大转向角（约34度）
maxAcceleration = 3.0m/s²  // 最大加速度
maxDeceleration = 8.0m/s²  // 最大减速度
frictionCoeff = 0.8        // 路面摩擦系数
```

### 仿真参数

```java
timeStep = 0.05s          // 时间步长（20Hz控制频率）
maxSteps = 2000           // 最大步数
roadLength = 1000m        // 道路长度
```

## 📖 相关文档

- [**技术架构文档**](doc/技术架构文档.md) - 详细的系统设计文档（485行）
- [**实施总结文档**](doc/实施总结.md) - 完整的开发进度总结（471行）
- [**TinyAI 主文档**](../../README.md) - 项目总体介绍

## 🎓 学习路径

建议按照以下顺序学习本模块：

1. **基础概念** - 阅读技术架构文档，理解具身智能的基本原理
2. **环境交互** - 运行 SimpleDemo，熟悉环境与动作空间
3. **智能体使用** - 运行 AgentDemo，体验完整闭环
4. **深入学习** - 阅读源代码，理解各模块实现
5. **自定义扩展** - 尝试添加新场景、新传感器或新策略

## 💡 核心亮点

### 1. 完整的具身智能架构
- **感知层**：5种传感器（相机、雷达、IMU、GPS、速度计）
- **决策层**：策略网络 + 安全约束
- **执行层**：动作执行 + 反馈处理
- **学习层**：DQN + 端到端学习 + 情景记忆

### 2. 高保真物理仿真
- 基于**自行车模型**的车辆动力学
- 考虑摩擦系数、空气阻力等物理因素
- 支持自定义物理参数

### 3. 多样化场景支持
- 内置6种典型驾驶场景
- 从简单测试到复杂路口，难度逐渐递增
- 支持自定义场景创建

### 4. 灵活的学习策略
- **DQN**：强化学习，适合无监督学习
- **端到端**：直接从感知到动作的映射
- **情景记忆**：支持经验回放，提高学习效率

### 5. 纯 Java 实现
- 完全基于 Java 17 实现
- 零外部依赖（除 JDK 和 TinyAI 模块）
- 充分复用 TinyAI 深度学习核心组件

## 🚀 高级功能

### 自定义场景

```java
// 创建自定义场景
ScenarioLoader loader = new ScenarioLoader();

Map<String, Object> customParams = new HashMap<>();
customParams.put("laneCount", 4);
customParams.put("speedLimit", 140.0);
customParams.put("vehicleDensity", 25);
customParams.put("weatherVisibility", 0.8);

EnvironmentConfig custom = loader.createCustomScenario(
    ScenarioType.HIGHWAY, customParams);
```

### 调整奖励权重

```java
EnvironmentConfig config = EnvironmentConfig.createTestConfig();

// 调整奖励权重：更重视安全
config.setSpeedRewardWeight(0.2);
config.setLaneKeepingWeight(0.3);
config.setCollisionPenaltyWeight(1.5);
config.setComfortWeight(0.05);
```

### 传感器数据访问

```java
DrivingEnvironment env = new SimpleDrivingEnv(config);

// 获取特定传感器数据
NdArray cameraData = env.getSensorData(SensorType.CAMERA);
NdArray lidarData = env.getSensorData(SensorType.LIDAR);
NdArray imuData = env.getSensorData(SensorType.IMU);
```

### 情景记忆管理

```java
EpisodicMemory memory = new EpisodicMemory(10000);

// 存储情景
Episode episode = agent.runEpisode(200);
memory.storeEpisode(episode);

// 批量采样
List<Transition> batch = memory.sampleBatch(32);

// 场景筛选
List<Episode> highwayEpisodes = memory.filterByScenario(ScenarioType.HIGHWAY);
```

## 🤝 贡献指南

欢迎贡献代码！虽然核心功能已完成，但仍有许多可以改进的地方：

### 功能增强
1. **更高保真的传感器模拟**
   - 实现真实的图像生成（基于渲染）
   - 实现点云数据生成
   - 添加传感器噪声模型

2. **新场景类型**
   - 复杂路口场景（信号灯、行人）
   - 恶劣天气场景（雨雪、雾）
   - 夜间驾驶场景

3. **优化车辆动力学模型**
   - 引入轮胎滑移模型
   - 考虑悬架系统
   - 支持不同车型

4. **增强学习算法**
   - 实现 PPO 算法
   - 实现 SAC 算法
   - 添加模仿学习支持
   - 实现逆强化学习

### 测试与文档
5. **单元测试**
   - 编写各模块的单元测试
   - 增加集成测试
   - 添加性能基准测试

6. **完善文档**
   - 添加更多使用示例
   - 编写详细的 API 文档
   - 制作视频教程

### 性能优化
7. **计算效率**
   - 优化障碍物管理算法
   - 实现批量环境并行执行
   - 添加 GPU 加速支持

8. **内存优化**
   - 优化对象复用机制
   - 实现更高效的缓存策略

## 🧪 单元测试

本模块提供了全面的单元测试覆盖，确保核心功能的正确性和稳定性。

### 测试统计

- **总测试数**: 116个
- **通过率**: 100%
- **覆盖率**: 核心类100%覆盖

### 测试文件

| 测试类 | 测试数 | 说明 |
|-------|--------|------|
| DrivingActionTest | 12 | 驾驶动作测试 |
| Vector3DTest | 11 | 三维向量测试 |
| VehicleStateTest | 9 | 车辆状态测试 |
| PerceptionStateTest | 10 | 感知状态测试 |
| EpisodeTest | 14 | 情景记录测试 |
| EnvironmentConfigTest | 8 | 环境配置测试 |
| EmbodiedAgentTest | 15 | 智能体集成测试 |
| PerceptionModuleTest | 8 | 感知模块测试 |
| DecisionModuleTest | 7 | 决策模块测试 |
| ExecutionModuleTest | 10 | 执行模块测试 |
| SensorSuiteTest | 12 | 传感器套件测试 |

### 运行测试

```bash
# 运行所有测试
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
cd tinyai-agent-embodied
mvn test

# 运行特定测试类
mvn test -Dtest=DrivingActionTest

# 运行特定测试方法
mvn test -Dtest=DrivingActionTest#testClip
```

### 测试结果示例

```
[INFO] Running io.leavesfly.tinyai.agent.embodied.EmbodiedAgentTest
[INFO] Tests run: 15, Failures: 0, Errors: 0, Skipped: 0
[INFO] Running io.leavesfly.tinyai.agent.embodied.model.DrivingActionTest
[INFO] Tests run: 12, Failures: 0, Errors: 0, Skipped: 0
...
[INFO] Tests run: 116, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
```

详细测试报告请参阅：[单元测试报告](doc/单元测试报告.md)

## 📄 许可证

本项目遵循与 TinyAI 主项目相同的许可证。

## 🙏 致谢

感谢 TinyAI 项目团队提供的深度学习和强化学习基础组件支持。

## 📊 项目统计

| 类别 | 数量 | 说明 |
|-----|------|------|
| Java 源文件 | 35+ | 包括所有模块实现 |
| 代码行数 | 3000+ | 不包含注释和空行 |
| 文档页面 | 1200+ | README + 技术文档 + 总结 |
| 支持场景 | 6种 | 从测试到复杂路口 |
| 传感器类型 | 5种 | 相机、雷达、IMU等 |
| 学习策略 | 2种 | DQN + 端到端 |

## 🛠️ 技术栈

| 项目 | 版本/配置 | 说明 |
|-----|----------|------|
| Java | JDK 17+ | 核心语言 |
| Maven | 3.6+ | 构建工具 |
| TinyAI NdArray | 1.0.0 | 多维数组库 |
| TinyAI AutoGrad | 1.0.0 | 自动微分 |
| TinyAI NeuralNet | 1.0.0 | 神经网络 |
| TinyAI RL | 1.0.0 | 强化学习 |

## ❓ 常见问题

### Q1: 如何调整仿真速度?

```java
EnvironmentConfig config = EnvironmentConfig.createTestConfig();
config.setTimeStep(0.1);  // 设置为 0.1秒(10Hz)
```

### Q2: 如何设置更长的情景?

```java
config.setMaxSteps(5000);  // 设置最大步数
```

### Q3: 如何保存和加载训练好的策略?

```java
// 保存
learningEngine.savePolicy("my_policy.model");

// 加载
learningEngine.loadPolicy("my_policy.model");
```

### Q4: 如何自定义车辆参数?

```java
VehicleDynamics dynamics = new VehicleDynamics();
dynamics.setWheelbase(3.0);  // 设置轴距为 3米
dynamics.setMaxSteeringAngle(0.7);  // 增加最大转向角
```

### Q5: 如何启用调试模式?

```java
config.setDebugMode(true);  // 启用详细日志
```

## 📝 更新日志

### v1.0.0 (2025-10-17)
- ✅ 完成所有8个开发阶段
- ✅ 实现完整的感知-决策-执行-学习闭环
- ✅ 35+ Java 类文件,3000+ 行代码
- ✅ 编译和运行验证通过
- ✅ 完整技术文档(1200+ 行)

## 🔗 相关链接

- [TinyAI 主项目](../../README.md)
- [TinyAI 深度学习框架](../../tinyai-deeplearning-ml/README.md)
- [TinyAI 强化学习模块](../../tinyai-deeplearning-rl/README.md)
- [TinyAI 智能体框架](../../tinyai-agent-context/README.md)

---

**TinyAI 具身智能模块** - 让Java也能玩转自动驾驶! 🚗💨
