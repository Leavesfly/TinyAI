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
具身智能体架构
├── 环境仿真层
│   ├── SimpleDrivingEnv - 驾驶环境
│   ├── VehicleDynamics - 车辆动力学
│   └── ScenarioLoader - 场景加载器
│
├── 感知层（待实现）
│   ├── SensorSuite - 传感器组件
│   ├── PerceptionModule - 感知处理
│   └── FeatureExtractor - 特征提取
│
├── 决策层（待实现）
│   ├── DecisionModule - 决策模块
│   ├── SafetyConstraint - 安全约束
│   └── PolicyNetwork - 策略网络
│
├── 执行层（待实现）
│   ├── ExecutionModule - 执行模块
│   └── ActionConverter - 动作转换
│
└── 学习层（待实现）
    ├── LearningEngine - 学习引擎
    ├── DQNLearner - DQN学习器
    ├── E2ELearner - 端到端学习器
    └── EpisodicMemory - 情景记忆
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

```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.embodied.SimpleDemo" \
              -pl tinyai-agent-embodied
```

### 4. 基础使用示例

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

### ✅ 已完成（阶段一、二）

- [x] 项目基础架构
- [x] 核心数据模型（11个类）
- [x] 枚举类型定义（5个）
- [x] 核心接口规范
- [x] 车辆动力学模型
- [x] 简单驾驶环境
- [x] 场景加载器

### 🚧 待实现（阶段三-八）

- [ ] 感知模块（传感器、特征提取）
- [ ] 决策模块（策略网络、安全约束）
- [ ] 执行模块（动作转换、反馈）
- [ ] 学习引擎（DQN、端到端、模仿学习）
- [ ] 情景记忆管理
- [ ] 完整智能体集成
- [ ] 单元测试和集成测试
- [ ] 完整文档和示例

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

- [技术架构文档](doc/技术架构文档.md) - 详细的系统设计文档
- [设计文档](../../design_doc.md) - 完整的模块设计规范
- [TinyAI 主文档](../../README.md) - 项目总体介绍

## 🤝 贡献指南

欢迎贡献代码！可以从以下方面参与：

1. 实现待完成的模块（感知、决策、学习等）
2. 添加新的场景类型
3. 优化车辆动力学模型
4. 编写单元测试
5. 完善文档

## 📄 许可证

本项目遵循与 TinyAI 主项目相同的许可证。

## 🙏 致谢

感谢 TinyAI 项目团队提供的深度学习和强化学习基础组件支持。

---

**TinyAI 具身智能模块** - 让Java也能玩转自动驾驶！ 🚗💨
