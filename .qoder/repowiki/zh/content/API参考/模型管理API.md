# 模型管理API

<cite>
**本文档中引用的文件**
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java)
- [ModelInfo.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfo.java)
- [ModelSerializer.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java)
- [ParameterManager.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ParameterManager.java)
- [ModelInfoExporter.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfoExporter.java)
- [ModelSerializationExample.java](file://tinyai-dl-case/src/main/java/io/leavesfly/tinyai/example/ModelSerializationExample.java)
</cite>

## 目录
1. [简介](#简介)
2. [模型生命周期管理](#模型生命周期管理)
3. [模型序列化机制](#模型序列化机制)
4. [参数管理功能](#参数管理功能)
5. [模型信息管理](#模型信息管理)
6. [运行时操作方法](#运行时操作方法)
7. [异常处理与最佳实践](#异常处理与最佳实践)
8. [完整示例](#完整示例)

## 简介
模型管理API为TinyAI框架提供了完整的模型创建、保存、加载和参数管理功能。该API以Model类为核心，结合ModelInfo、ModelSerializer、ParameterManager等辅助类，实现了模型的全生命周期管理。API设计注重类型安全、性能优化和易用性，支持多种序列化格式和灵活的参数操作。

**Section sources**
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L1-L50)

## 模型生命周期管理

### 模型创建与初始化
模型通过Model类的构造函数创建，需要提供模型名称和神经网络结构（Block）。构造函数会自动初始化模型信息，包括输入输出形状、参数数量和架构类型。

```mermaid
classDiagram
class Model {
+String name
+Block block
+ModelInfo modelInfo
+Model(_name, _block)
+initializeModelInfo()
+saveModel(filePath)
+loadModel(filePath)
+saveParameters(filePath)
+loadParameters(filePath)
+resetState()
+forward(inputs)
+clearGrads()
+getModelInfo()
+updateTrainingInfo(epochs, finalLoss, optimizer, learningRate)
+addMetric(metricName, value)
+printModelInfo()
}
class ModelInfo {
+String modelName
+String modelVersion
+Date createdTime
+Shape inputShape
+Shape outputShape
+long totalParameters
+int trainedEpochs
+double finalLoss
+String optimizerType
+double learningRate
+Map~String, Double~ metrics
}
Model --> ModelInfo : "包含"
Model --> Block : "使用"
```

**Diagram sources**
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L1-L50)
- [ModelInfo.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfo.java#L1-L50)

**Section sources**
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L50-L100)

## 模型序列化机制

### 序列化格式与推荐实践
模型序列化支持多种格式，包括完整模型、压缩模型、仅参数文件和检查点。推荐使用ModelSerializer类而非传统的Java序列化，以获得更好的性能和功能。

```mermaid
flowchart TD
Start([创建模型]) --> SaveComplete["保存完整模型<br>saveModel(filePath)"]
Start --> SaveCompressed["保存压缩模型<br>saveModelCompressed(filePath)"]
Start --> SaveParams["仅保存参数<br>saveParameters(filePath)"]
Start --> SaveCheckpoint["保存检查点<br>saveCheckpoint(filePath, epoch, loss)"]
LoadComplete["加载完整模型<br>loadModel(filePath)"] --> End([模型使用])
LoadCompressed["加载压缩模型<br>自动检测格式"] --> End
LoadParams["加载参数<br>loadParameters(filePath)"] --> End
LoadCheckpoint["从检查点恢复<br>resumeFromCheckpoint(filePath)"] --> End
SaveComplete --> Validate["验证模型<br>validateModel(filePath)"]
SaveCompressed --> Validate
SaveParams --> Validate
SaveCheckpoint --> Validate
```

**Diagram sources**
- [ModelSerializer.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java#L1-L100)
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L150-L250)

**Section sources**
- [ModelSerializer.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java#L1-L200)
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L150-L250)

## 参数管理功能

### 参数操作与管理
ParameterManager类提供了专门的参数操作功能，包括参数保存、加载、复制、比较和统计。这些功能支持在不同模型间灵活地管理参数。

```mermaid
classDiagram
class ParameterManager {
+saveParameters(parameters, filePath)
+loadParameters(filePath)
+copyParameters(sourceModel, targetModel, strict)
+compareParameters(model1, model2, tolerance)
+getParameterStats(parameters)
+deepCopyParameters(parameters)
+filterParameters(parameters, pattern)
}
class ModelSerializer {
+saveParameters(model, filePath)
+loadParameters(model, filePath)
+compareModelParameters(model1, model2)
}
ParameterManager --> Parameter : "操作"
ModelSerializer --> ParameterManager : "委托"
Model --> ModelSerializer : "使用"
```

**Diagram sources**
- [ParameterManager.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ParameterManager.java#L1-L50)
- [ModelSerializer.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java#L200-L300)

**Section sources**
- [ParameterManager.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ParameterManager.java#L1-L300)
- [ModelSerializer.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java#L200-L300)

## 模型信息管理

### ModelInfo类结构与功能
ModelInfo类负责收集和存储模型的元数据，包括基本信息、架构信息、训练信息和性能指标。这些信息可用于模型的监控、比较和文档化。

```mermaid
erDiagram
MODEL_INFO {
string modelName PK
string modelVersion
string frameworkVersion
datetime createdTime
datetime lastModifiedTime
string description
string architectureType
string inputShape
string outputShape
int totalLayers
long totalParameters
int trainedEpochs
double finalLoss
double bestLoss
string optimizerType
double learningRate
int batchSize
string lossFunction
long trainingTimeMs
string hardwareInfo
}
METRICS {
string metricName PK
double value
string modelName FK
}
LAYER_COUNTS {
string layerType PK
int count
string modelName FK
}
CUSTOM_PROPERTIES {
string propertyName PK
string propertyValue
string modelName FK
}
MODEL_INFO ||--o{ METRICS : "包含"
MODEL_INFO ||--o{ LAYER_COUNTS : "包含"
MODEL_INFO ||--o{ CUSTOM_PROPERTIES : "包含"
```

**Diagram sources**
- [ModelInfo.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfo.java#L1-L100)
- [ModelInfoExporter.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfoExporter.java#L1-L50)

**Section sources**
- [ModelInfo.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfo.java#L1-L200)
- [ModelInfoExporter.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfoExporter.java#L1-L100)

## 运行时操作方法

### 核心运行时方法
模型提供了多种运行时操作方法，用于模型的前向传播、状态管理和梯度处理。这些方法是模型训练和推理过程中的关键操作。

```mermaid
sequenceDiagram
participant User as "用户代码"
participant Model as "Model"
participant Block as "Block"
User->>Model : forward(inputs)
Model->>Block : layerForward(inputs)
Block-->>Model : output
Model-->>User : output
User->>Model : clearGrads()
Model->>Block : clearGrads()
Block-->>Model : 完成
Model-->>User : 完成
User->>Model : resetState()
Model->>Block : resetState()
Block-->>Model : 完成
Model-->>User : 完成
User->>Model : printModelInfo()
Model->>Model : getModelDetailedInfo()
Model-->>User : 打印信息
```

**Diagram sources**
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L250-L350)
- [ModelInfo.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfo.java#L200-L300)

**Section sources**
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L250-L350)

## 异常处理与最佳实践

### 异常处理策略
API提供了全面的异常处理机制，包括详细的错误消息、渐进式错误恢复和用户友好的警告提示。在参数加载时，会进行形状匹配检查，并自动跳过不匹配的参数。

```mermaid
flowchart TD
LoadParams["加载参数<br>loadParameters(filePath)"] --> CheckFile["检查文件是否存在"]
CheckFile --> |不存在| Error1["抛出RuntimeException"]
CheckFile --> |存在| ReadParams["读取参数映射"]
ReadParams --> ProcessParams["处理每个参数"]
ProcessParams --> CheckName["检查参数名是否存在"]
CheckName --> |不存在| Warn1["打印警告，跳过"]
CheckName --> |存在| CheckShape["检查形状是否匹配"]
CheckShape --> |不匹配| Warn2["打印警告，跳过"]
CheckShape --> |匹配| CopyValue["复制参数值"]
CopyValue --> UpdateCount["更新加载计数"]
UpdateCount --> NextParam["处理下一个参数"]
NextParam --> ProcessParams
ProcessParams --> |完成| Finish["完成加载"]
```

**Diagram sources**
- [ModelSerializer.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java#L250-L350)
- [ParameterManager.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ParameterManager.java#L300-L400)

**Section sources**
- [ModelSerializer.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java#L250-L350)
- [ParameterManager.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/ParameterManager.java#L300-L400)

## 完整示例

### 模型生命周期完整示例
以下示例展示了模型从创建到持久化的完整生命周期管理，包括模型创建、信息设置、保存、加载和验证。

```mermaid
flowchart TD
CreateModel["创建模型<br>new Model(name, block)"] --> SetupInfo["设置模型信息<br>updateTrainingInfo(), addMetric()"]
SetupInfo --> PrintInfo["打印模型信息<br>printModelInfo()"]
PrintInfo --> SaveFull["保存完整模型<br>saveModel(filePath)"]
SaveFull --> SaveCompressed["保存压缩模型<br>saveModelCompressed(filePath)"]
SaveCompressed --> SaveParams["仅保存参数<br>saveParameters(filePath)"]
SaveParams --> SaveCheckpoint["保存检查点<br>saveCheckpoint(filePath, epoch, loss)"]
SaveCheckpoint --> ExportJson["导出JSON报告<br>exportToJson(model, filePath)"]
ExportJson --> LoadFull["加载完整模型<br>loadModel(filePath)"]
LoadFull --> ValidateModel["验证模型<br>validateModel(filePath)"]
ValidateModel --> CompareParams["比较参数<br>compareModelParameters()"]
CompareParams --> UseModel["使用模型进行推理"]
```

**Diagram sources**
- [ModelSerializationExample.java](file://tinyai-dl-case/src/main/java/io/leavesfly/tinyai/example/ModelSerializationExample.java#L1-L200)
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L1-L360)

**Section sources**
- [ModelSerializationExample.java](file://tinyai-dl-case/src/main/java/io/leavesfly/tinyai/example/ModelSerializationExample.java#L1-L300)
- [Model.java](file://tinyai-dl-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L1-L360)