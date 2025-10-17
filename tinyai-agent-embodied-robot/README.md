# TinyAI 具身智能扫地机器人模块

## 项目简介

`tinyai-agent-embodied-robot` 是 TinyAI 项目中的具身智能扫地机器人模块，展示了端到端学习在家庭清扫机器人场景中的实际应用。本模块实现了完整的智能体工作流程：环境感知、智能决策、动作执行和强化学习。

## 核心特性

- **完整的清扫环境仿真**：支持多种场景类型（简单房间、客厅、卧室、厨房等）
- **多传感器融合**：集成摄像头、激光雷达、碰撞传感器等多种传感器
- **智能路径规划**：基于神经网络的决策模块，实现高效路径覆盖
- **强化学习支持**：支持DQN、DDPG、PPO等多种学习策略
- **端到端学习**：直接从原始感知到动作的映射
- **纯Java实现**：零外部依赖，完全基于TinyAI核心模块

## 模块架构

项目采用分层架构设计：

```
智能体核心层 (RobotAgent)
    ↓
功能模块层
    ├── 感知模块 (PerceptionModule)
    ├── 决策模块 (DecisionModule)
    ├── 执行模块 (ExecutionModule)
    └── 学习引擎 (LearningEngine)
    ↓
环境仿真层
    ├── 清扫环境 (CleaningEnvironment)
    ├── 机器人动力学 (RobotDynamics)
    ├── 传感器系统 (SensorSuite)
    └── 场景加载器 (ScenarioLoader)
    ↓
TinyAI核心层
    ├── NdArray (多维数组)
    ├── AutoGrad (自动微分)
    ├── NeuralNet (神经网络)
    └── RL (强化学习)
```

## 数据模型

### 核心类

已完成实现的核心数据模型：

- **Vector2D**: 二维向量，表示位置和方向
- **BoundingBox**: 包围盒，用于碰撞检测
- **RobotState**: 机器人状态（位置、朝向、电量、尘盒等）
- **CleaningAction**: 清扫动作（速度控制、刷子功率、吸力等）
- **CleaningState**: 完整观测状态（传感器数据、地图信息）
- **ObstacleInfo**: 障碍物信息
- **ChargingStationInfo**: 充电站信息
- **FloorMap**: 地面清扫地图（覆盖率、灰尘分布）
- **StepResult**: 环境步进结果
- **ExecutionFeedback**: 动作执行反馈
- **Transition**: 状态转移（强化学习）
- **Episode**: 完整情景

### 枚举类型

- **ScenarioType**: 场景类型（简单房间、客厅、卧室等）
- **SensorType**: 传感器类型（摄像头、雷达、碰撞传感器等）
- **ObstacleType**: 障碍物类型（墙壁、家具、楼梯等）
- **FloorType**: 地面类型（瓷砖、木地板、地毯等）
- **ActionType**: 动作类型（前进、转向、清扫等）
- **LearningStrategy**: 学习策略（DQN、DDPG、PPO等）

## 环境要求

- Java 17 或更高版本
- Maven 3.6+
- TinyAI 核心模块（ndarr, func, nnet, ml, rl）

## 快速开始

### 编译项目

```bash
# 设置Java环境
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home

# 编译模块
cd /path/to/TinyAI
mvn clean compile -pl tinyai-agent-embodied-robot -am
```

### 运行演示

```bash
# 简单演示
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.robot.SimpleDemo" \
    -pl tinyai-agent-embodied-robot
```

## 开发进度

### 已完成

- ✅ Maven模块结构和基础配置
- ✅ 核心数据模型（18个类和6个枚举）
- ✅ 基础几何类（Vector2D、BoundingBox）
- ✅ 机器人状态管理
- ✅ 动作定义和转换
- ✅ 地图和环境对象
- ✅ 环境仿真层（RobotDynamics、CleaningEnvironment）
- ✅ 传感器系统（Sensor、SensorSuite）
- ✅ 感知模块（PerceptionModule）
- ✅ 决策模块（DecisionModule）
- ✅ 执行模块（ExecutionModule）
- ✅ 学习引擎（LearningEngine、EpisodicMemory）
- ✅ 简单演示程序
- ✅ 单元测试（3个测试类）

### 进行中

无，所有核心功能已完成

### 待开发

- ⏳ 高级传感器实现（可选）
- ⏳ 神经网络策略（可选）
- ⏳ DQN/PPO学习算法（可选）
- ⏳ 可视化界面（可选）

## 项目结构

```
tinyai-agent-embodied-robot/
├── src/
│   ├── main/java/io/leavesfly/tinyai/agent/robot/
│   │   ├── model/           # 数据模型（已完成）
│   │   ├── env/             # 环境仿真（待实现）
│   │   ├── dynamics/        # 机器人动力学（待实现）
│   │   ├── sensor/          # 传感器系统（待实现）
│   │   ├── perception/      # 感知模块（待实现）
│   │   ├── decision/        # 决策模块（待实现）
│   │   ├── execution/       # 执行模块（待实现）
│   │   ├── learning/        # 学习引擎（待实现）
│   │   ├── RobotAgent.java  # 智能体核心（待实现）
│   │   └── *Demo.java       # 演示程序（待实现）
│   └── test/                # 测试代码（待实现）
├── doc/                     # 文档目录
├── README.md               # 本文件
└── pom.xml                 # Maven配置
```

## 技术细节

### 机器人物理参数

- 轮距：0.25米
- 最大线速度：0.5 m/s
- 最大角速度：π/2 rad/s
- 机器人半径：0.175米
- 电池容量：3000 mAh
- 尘盒容量：0.6升

### 奖励函数设计

采用多维度组合奖励：

- 覆盖奖励：新清扫面积
- 效率奖励：单位时间清扫面积
- 能量惩罚：能量消耗
- 碰撞惩罚：安全性考虑
- 灰尘清除奖励：清洁效果

### 场景难度

| 场景类型 | 房间大小 | 障碍物数量 | 难度等级 |
|---------|---------|----------|----------|
| SIMPLE_ROOM | 5m × 5m | 2-3 | ★☆☆☆☆ |
| LIVING_ROOM | 8m × 6m | 8-10 | ★★☆☆☆ |
| BEDROOM | 6m × 5m | 12-15 | ★★★☆☆ |
| KITCHEN | 5m × 4m | 15-18 | ★★★☆☆ |
| MULTI_ROOM | 12m × 10m | 20-25 | ★★★★☆ |
| COMPLEX_LAYOUT | 15m × 12m | 30+ | ★★★★★ |

## 依赖模块

本模块依赖以下TinyAI核心模块：

- `tinyai-deeplearning-ndarr`: 多维数组库
- `tinyai-deeplearning-func`: 自动微分引擎
- `tinyai-deeplearning-nnet`: 神经网络层
- `tinyai-deeplearning-ml`: 机器学习核心
- `tinyai-deeplearning-rl`: 强化学习模块

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目是 TinyAI 的一部分，遵循 TinyAI 的许可证。

## 联系方式

- 项目主页：https://github.com/yourusername/TinyAI
- 问题反馈：https://github.com/yourusername/TinyAI/issues

## 致谢

感谢 TinyAI 团队的支持和贡献！

---

**注意**: 本模块当前处于开发阶段，部分功能尚未实现。请参考"开发进度"章节了解最新状态。
