package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Tanh;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.LayerAble;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.func.math.ReLu;
import io.leavesfly.tinyai.func.math.GELU;

import java.util.*;

/**
 * DeepSeek V3 增强推理模块
 * 
 * 实现V3的核心推理能力：
 * 1. 任务类型识别和分类
 * 2. 专门化的推理器处理不同任务类型
 * 3. 自我纠错机制
 * 4. 置信度评估
 * 5. 多步推理验证
 * 6. 专家建议整合
 * 
 * @author leavesfly
 * @version 1.0
 */
public class V3ReasoningModule extends LayerAble {
    
    // ========== 配置参数 ==========
    private int dModel;                          // 模型维度
    private int numReasoningSteps;               // 推理步骤数
    private double confidenceThreshold;          // 置信度阈值
    
    // ========== 网络组件 ==========
    private LinearLayer taskClassifier;               // 任务类型分类器
    private Map<TaskType, LayerAble> reasoningEncoders; // 专门化推理器
    private LayerAble selfCorrection;            // 自我纠错模块
    private LinearLayer confidenceEstimator;          // 置信度评估器
    private LayerAble verifier;                  // 验证器
    
    // ========== 运行时状态 ==========
    private List<V3ReasoningStep> currentReasoningChain; // 当前推理链
    private TaskType dominantTaskType;                    // 主导任务类型
    private Map<TaskType, Double> taskTypeDistribution;  // 任务类型分布
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param dModel 模型维度
     * @param numReasoningSteps 推理步骤数
     */
    public V3ReasoningModule(String name, int dModel, int numReasoningSteps) {
        this.name = name;
        this.dModel = dModel;
        this.numReasoningSteps = numReasoningSteps;
        this.confidenceThreshold = 0.8;
        
        // 设置输入输出形状
        this.inputShape = Shape.of(-1, -1, dModel);  // [batch, seq, dModel]
        this.outputShape = Shape.of(-1, dModel);     // [batch, dModel] - 推理状态
        
        init();
    }
    
    /**
     * 默认构造函数
     */
    public V3ReasoningModule(String name, int dModel) {
        this(name, dModel, 7); // 默认7步推理
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化任务类型分类器
            int numTaskTypes = TaskType.values().length;
            taskClassifier = new LinearLayer(name + "_task_classifier", dModel, numTaskTypes, false);
            taskClassifier.init();
            
            // 2. 初始化专门化推理器
            reasoningEncoders = new HashMap<>();
            for (TaskType taskType : TaskType.values()) {
                LayerAble encoder = createTaskSpecificReasoner(taskType);
                encoder.init();
                reasoningEncoders.put(taskType, encoder);
            }
            
            // 3. 初始化自我纠错模块
            selfCorrection = new SelfCorrectionModule(name + "_self_correction", dModel);
            selfCorrection.init();
            
            // 4. 初始化置信度评估器
            confidenceEstimator = new LinearLayer(name + "_confidence", dModel, 1, false);
            confidenceEstimator.init();
            
            // 5. 初始化验证器
            verifier = new VerificationModule(name + "_verifier", dModel);
            verifier.init();
            
            // 初始化运行时状态
            currentReasoningChain = new ArrayList<>();
            taskTypeDistribution = new HashMap<>();
            
            alreadyInit = true;
        }
    }
    
    /**
     * 创建任务特定的推理器
     * 
     * @param taskType 任务类型
     * @return 推理器网络
     */
    private LayerAble createTaskSpecificReasoner(TaskType taskType) {
        String reasonerName = name + "_reasoner_" + taskType.getValue();
        
        switch (taskType) {
            case REASONING:
                return new ReasoningTaskReasoner(reasonerName, dModel);
            case CODING:
                return new CodingTaskReasoner(reasonerName, dModel);
            case MATH:
                return new MathTaskReasoner(reasonerName, dModel);
            case MULTIMODAL:
                return new MultimodalTaskReasoner(reasonerName, dModel);
            case GENERAL:
            default:
                return new GeneralTaskReasoner(reasonerName, dModel);
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputEmbedding = inputs[0]; // [batch, seq, dModel]
        
        // 清空当前推理链
        currentReasoningChain.clear();
        
        // 1. 获取输入的平均表示作为推理状态
        Variable currentState = computeMeanPooling(inputEmbedding); // [batch, dModel]
        
        // 2. 识别任务类型
        TaskType taskType = identifyTaskType(currentState);
        this.dominantTaskType = taskType;
        
        // 3. 执行多步推理
        for (int step = 0; step < numReasoningSteps; step++) {
            V3ReasoningStep reasoningStep = executeReasoningStep(currentState, taskType, step);
            currentReasoningChain.add(reasoningStep);
            
            // 更新推理状态
            currentState = updateReasoningState(currentState, reasoningStep);
        }
        
        return currentState;
    }
    
    /**
     * 计算平均池化
     * 
     * @param input 输入张量 [batch, seq, dModel]
     * @return 池化后的张量 [batch, dModel]
     */
    private Variable computeMeanPooling(Variable input) {
        NdArray inputData = input.getValue();
        Shape inputShape = inputData.getShape();
        
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int dModel = inputShape.getDimension(2);
        
        double[] inputArray = inputData.toDoubleArray();
        double[] pooledArray = new double[batchSize * dModel];
        
        // 计算序列维度的平均值
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dModel; d++) {
                double sum = 0.0;
                for (int s = 0; s < seqLen; s++) {
                    int inputIdx = (b * seqLen + s) * dModel + d;
                    sum += inputArray[inputIdx];
                }
                int pooledIdx = b * dModel + d;
                pooledArray[pooledIdx] = sum / seqLen;
            }
        }
        
        return new Variable(NdArray.of(pooledArray).reshape(Shape.of(batchSize, dModel)));
    }
    
    /**
     * 识别任务类型
     * 
     * @param currentState 当前状态
     * @return 识别的任务类型
     */
    private TaskType identifyTaskType(Variable currentState) {
        // 1. 任务分类
        Variable taskProbs = taskClassifier.layerForward(currentState);
        taskProbs = applySoftmax(taskProbs);
        
        // 2. 找到概率最高的任务类型
        NdArray probsData = taskProbs.getValue();
        double[] probsArray = probsData.toDoubleArray();
        
        int maxIdx = 0;
        double maxProb = probsArray[0];
        for (int i = 1; i < probsArray.length; i++) {
            if (probsArray[i] > maxProb) {
                maxProb = probsArray[i];
                maxIdx = i;
            }
        }
        
        // 3. 更新任务类型分布
        TaskType[] taskTypes = TaskType.values();
        taskTypeDistribution.clear();
        for (int i = 0; i < taskTypes.length && i < probsArray.length; i++) {
            taskTypeDistribution.put(taskTypes[i], (double) probsArray[i]);
        }
        
        return taskTypes[maxIdx];
    }
    
    /**
     * 执行单步推理
     * 
     * @param currentState 当前推理状态
     * @param taskType 任务类型
     * @param stepNumber 步骤编号
     * @return 推理步骤结果
     */
    private V3ReasoningStep executeReasoningStep(Variable currentState, TaskType taskType, int stepNumber) {
        long startTime = System.currentTimeMillis();
        
        // 1. 使用任务特定的推理器
        LayerAble reasoner = reasoningEncoders.get(taskType);
        if (reasoner == null) {
            reasoner = reasoningEncoders.get(TaskType.GENERAL);
        }
        
        Variable thoughtState = reasoner.layerForward(currentState);
        
        // 2. 自我纠错
        Variable correctionInput = concatenateStates(currentState, thoughtState);
        Variable correctionOutput = selfCorrection.layerForward(correctionInput);
        Variable correctedState = applyCorrectionWeights(thoughtState, currentState, correctionOutput);
        
        // 3. 置信度评估
        double confidence = estimateConfidence(correctedState);
        
        // 4. 验证
        Variable verificationInput = concatenateThreeStates(currentState, thoughtState, correctedState);
        Variable verificationOutput = verifier.layerForward(verificationInput);
        double verificationScore = extractVerificationScore(verificationOutput);
        
        // 5. 生成专家建议（模拟）
        Map<String, Double> expertAdvice = generateExpertAdvice(taskType, confidence);
        
        long endTime = System.currentTimeMillis();
        
        // 6. 创建推理步骤
        V3ReasoningStep step = new V3ReasoningStep(
            "V3 Step " + (stepNumber + 1) + " - " + taskType.getValue() + " 思考",
            "V3 Step " + (stepNumber + 1) + " - 专门化处理",
            confidence,
            "V3 验证分数: " + String.format("%.3f", verificationScore),
            taskType,
            expertAdvice,
            "应用纠错权重: " + String.format("%.3f", extractCorrectionWeight(correctionOutput)),
            stepNumber,
            endTime - startTime
        );
        
        return step;
    }
    
    /**
     * 更新推理状态
     * 
     * @param currentState 当前状态
     * @param reasoningStep 推理步骤
     * @return 更新后的状态
     */
    private Variable updateReasoningState(Variable currentState, V3ReasoningStep reasoningStep) {
        // 简化实现：基于置信度调整状态更新强度
        double updateStrength = reasoningStep.getConfidence() * 0.1;
        
        // 创建更新向量（这里使用随机更新作为简化）
        NdArray currentData = currentState.getValue();
        double[] currentArray = currentData.toDoubleArray();
        double[] updatedArray = currentArray.clone();
        
        // 应用小幅更新
        for (int i = 0; i < updatedArray.length; i++) {
            updatedArray[i] += updateStrength * (Math.random() - 0.5) * 0.01;
        }
        
        return new Variable(NdArray.of(updatedArray).reshape(currentData.getShape()));
    }
    
    /**
     * 应用Softmax
     */
    private Variable applySoftmax(Variable x) {
        return x.softMax();
    }
    
    /**
     * 连接两个状态张量
     */
    private Variable concatenateStates(Variable state1, Variable state2) {
        NdArray data1 = state1.getValue();
        NdArray data2 = state2.getValue();
        
        double[] array1 = data1.toDoubleArray();
        double[] array2 = data2.toDoubleArray();
        double[] concatenated = new double[array1.length + array2.length];
        
        System.arraycopy(array1, 0, concatenated, 0, array1.length);
        System.arraycopy(array2, 0, concatenated, array1.length, array2.length);
        
        return new Variable(NdArray.of(concatenated).reshape(Shape.of(data1.getShape().getDimension(0), dModel * 2)));
    }
    
    /**
     * 连接三个状态张量
     */
    private Variable concatenateThreeStates(Variable state1, Variable state2, Variable state3) {
        NdArray data1 = state1.getValue();
        NdArray data2 = state2.getValue();
        NdArray data3 = state3.getValue();
        
        double[] array1 = data1.toDoubleArray();
        double[] array2 = data2.toDoubleArray();
        double[] array3 = data3.toDoubleArray();
        double[] concatenated = new double[array1.length + array2.length + array3.length];
        
        System.arraycopy(array1, 0, concatenated, 0, array1.length);
        System.arraycopy(array2, 0, concatenated, array1.length, array2.length);
        System.arraycopy(array3, 0, concatenated, array1.length + array2.length, array3.length);
        
        return new Variable(NdArray.of(concatenated).reshape(Shape.of(data1.getShape().getDimension(0), dModel * 3)));
    }
    
    /**
     * 应用纠错权重
     */
    private Variable applyCorrectionWeights(Variable thoughtState, Variable originalState, Variable correctionOutput) {
        double correctionWeight = extractCorrectionWeight(correctionOutput);
        double originalWeight = 1.0 - correctionWeight;
        
        // correctedState = correctionWeight * thoughtState + originalWeight * originalState
        Variable weighted1 = multiplyByScalar(thoughtState, correctionWeight);
        Variable weighted2 = multiplyByScalar(originalState, originalWeight);
        
        return weighted1.add(weighted2);
    }
    
    /**
     * 提取纠错权重
     */
    private double extractCorrectionWeight(Variable correctionOutput) {
        NdArray data = correctionOutput.getValue();
        double[] array = data.toDoubleArray();
        
        // 使用Sigmoid将输出映射到[0,1]
        double sum = 0.0;
        for (double value : array) {
            sum += 1.0 / (1.0 + Math.exp(-value));
        }
        return Math.max(0.0, Math.min(1.0, sum / array.length));
    }
    
    /**
     * 标量乘法
     */
    private Variable multiplyByScalar(Variable var, double scalar) {
        NdArray data = var.getValue();
        double[] array = data.toDoubleArray();
        
        for (int i = 0; i < array.length; i++) {
            array[i] *= scalar;
        }
        
        return new Variable(NdArray.of(array).reshape(data.getShape()));
    }
    
    /**
     * 估计置信度
     */
    private double estimateConfidence(Variable state) {
        Variable confidenceOutput = confidenceEstimator.layerForward(state);
        NdArray data = confidenceOutput.getValue();
        double[] array = data.toDoubleArray();
        
        // 使用Sigmoid激活
        double confidence = 0.0;
        for (double value : array) {
            confidence += 1.0 / (1.0 + Math.exp(-value));
        }
        confidence /= array.length;
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    /**
     * 提取验证分数
     */
    private double extractVerificationScore(Variable verificationOutput) {
        NdArray data = verificationOutput.getValue();
        double[] array = data.toDoubleArray();
        
        // 使用Sigmoid激活
        double score = 0.0;
        for (double value : array) {
            score += 1.0 / (1.0 + Math.exp(-value));
        }
        return score / array.length;
    }
    
    /**
     * 生成专家建议
     */
    private Map<String, Double> generateExpertAdvice(TaskType taskType, double confidence) {
        Map<String, Double> advice = new HashMap<>();
        
        // 基于任务类型和置信度生成模拟的专家建议
        advice.put("reasoning", taskType.isComplexReasoning() ? 0.8 : 0.3);
        advice.put("coding", taskType == TaskType.CODING ? 0.9 : 0.2);
        advice.put("math", taskType == TaskType.MATH ? 0.9 : 0.3);
        advice.put("general", 0.5);
        advice.put("multimodal", taskType == TaskType.MULTIMODAL ? 0.8 : 0.1);
        
        // 根据置信度调整权重
        for (Map.Entry<String, Double> entry : advice.entrySet()) {
            entry.setValue(entry.getValue() * confidence);
        }
        
        return advice;
    }
    
    /**
     * 获取推理链摘要
     * 
     * @return 推理链的摘要信息
     */
    public String getReasoningChainSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== V3 推理链摘要 ===\n");
        sb.append(String.format("主导任务类型: %s\n", dominantTaskType != null ? dominantTaskType.getValue() : "未知"));
        sb.append(String.format("推理步骤数: %d\n", currentReasoningChain.size()));
        
        double avgConfidence = currentReasoningChain.stream()
                .mapToDouble(V3ReasoningStep::getConfidence)
                .average()
                .orElse(0.0);
        sb.append(String.format("平均置信度: %.3f\n", avgConfidence));
        
        long totalTime = currentReasoningChain.stream()
                .mapToLong(V3ReasoningStep::getExecutionTimeMs)
                .sum();
        sb.append(String.format("总推理时间: %d毫秒\n", totalTime));
        
        sb.append("任务类型分布:\n");
        if (taskTypeDistribution != null) {
            taskTypeDistribution.forEach((type, prob) ->
                sb.append(String.format("  - %s: %.3f\n", type.getValue(), prob)));
        }
        
        sb.append("推理步骤详情:\n");
        for (int i = 0; i < currentReasoningChain.size(); i++) {
            V3ReasoningStep step = currentReasoningChain.get(i);
            sb.append(String.format("  %d. %s\n", i + 1, step.getSummary()));
        }
        
        sb.append("===================");
        return sb.toString();
    }
    
    @Override
    public void clearGrads() {
        taskClassifier.clearGrads();
        for (LayerAble reasoner : reasoningEncoders.values()) {
            reasoner.clearGrads();
        }
        selfCorrection.clearGrads();
        confidenceEstimator.clearGrads();
        verifier.clearGrads();
    }
    
    // ========== Inner Classes ==========
    
    /**
     * 自我纠错模块
     */
    private static class SelfCorrectionModule extends LayerAble {
        protected LinearLayer layer1, layer2;
        
        public SelfCorrectionModule(String name, int dModel) {
            this.name = name;
            this.inputShape = Shape.of(-1, dModel * 2);
            this.outputShape = Shape.of(-1, dModel);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                layer1 = new LinearLayer(name + "_l1", inputShape.getDimension(-1), inputShape.getDimension(-1) / 2, false);
                layer2 = new LinearLayer(name + "_l2", inputShape.getDimension(-1) / 2, outputShape.getDimension(-1), false);
                layer1.init();
                layer2.init();
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // 使用ReLU激活
            x = layer2.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            layer1.clearGrads();
            layer2.clearGrads();
        }
    }
    
    /**
     * 验证模块
     */
    private static class VerificationModule extends LayerAble {
        private LinearLayer layer1, layer2, layer3;
        
        public VerificationModule(String name, int dModel) {
            this.name = name;
            this.inputShape = Shape.of(-1, dModel * 3);
            this.outputShape = Shape.of(-1, 1);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                layer1 = new LinearLayer(name + "_l1", inputShape.getDimension(-1), inputShape.getDimension(-1) / 2, false);
                layer2 = new LinearLayer(name + "_l2", inputShape.getDimension(-1) / 2, 64, false);
                layer3 = new LinearLayer(name + "_l3", 64, 1, false);
                layer1.init();
                layer2.init();
                layer3.init();
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // ReLU激活
            x = layer2.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // ReLU激活
            x = layer3.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            layer1.clearGrads();
            layer2.clearGrads();
            layer3.clearGrads();
        }
    }
    
    /**
     * 通用任务推理器
     */
    private static class GeneralTaskReasoner extends LayerAble {
        protected LinearLayer layer1, layer2;
        
        public GeneralTaskReasoner(String name, int dModel) {
            this.name = name;
            this.inputShape = Shape.of(-1, dModel);
            this.outputShape = Shape.of(-1, dModel);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                layer1 = new LinearLayer(name + "_l1", inputShape.getDimension(-1), inputShape.getDimension(-1) * 2, false);
                layer2 = new LinearLayer(name + "_l2", inputShape.getDimension(-1) * 2, outputShape.getDimension(-1), false);
                layer1.init();
                layer2.init();
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // 使用ReLU激活
            x = layer2.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            layer1.clearGrads();
            layer2.clearGrads();
        }
    }
    
    /**
     * 推理任务推理器
     */
    private static class ReasoningTaskReasoner extends GeneralTaskReasoner {
        public ReasoningTaskReasoner(String name, int dModel) {
            super(name, dModel);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                // 推理任务使用更深的网络
                layer1 = new LinearLayer(name + "_l1", inputShape.getDimension(-1), inputShape.getDimension(-1) * 3, false);
                layer2 = new LinearLayer(name + "_l2", inputShape.getDimension(-1) * 3, outputShape.getDimension(-1), false);
                layer1.init();
                layer2.init();
                alreadyInit = true;
            }
        }
    }
    
    /**
     * 代码任务推理器
     */
    private static class CodingTaskReasoner extends GeneralTaskReasoner {
        public CodingTaskReasoner(String name, int dModel) {
            super(name, dModel);
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new GELU().forward(x.getValue())); // 代码任务使用GELU
            x = layer2.layerForward(x);
            return x;
        }
    }
    
    /**
     * 数学任务推理器
     */
    private static class MathTaskReasoner extends GeneralTaskReasoner {
        private LinearLayer layer3;
        
        public MathTaskReasoner(String name, int dModel) {
            super(name, dModel);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                // 数学任务需要更多容量
                layer1 = new LinearLayer(name + "_l1", inputShape.getDimension(-1), inputShape.getDimension(-1) * 3, false);
                layer2 = new LinearLayer(name + "_l2", inputShape.getDimension(-1) * 3, inputShape.getDimension(-1) * 2, false);
                layer3 = new LinearLayer(name + "_l3", inputShape.getDimension(-1) * 2, outputShape.getDimension(-1), false);
                layer1.init();
                layer2.init();
                layer3.init();
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // ReLU激活
            x = layer2.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // ReLU激活
            x = layer3.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            super.clearGrads();
            if (layer3 != null) layer3.clearGrads();
        }
    }
    
    /**
     * 多模态任务推理器
     */
    private static class MultimodalTaskReasoner extends GeneralTaskReasoner {
        public MultimodalTaskReasoner(String name, int dModel) {
            super(name, dModel);
        }
    }
    
    // ========== Getter Methods ==========
    
    public List<V3ReasoningStep> getCurrentReasoningChain() { return currentReasoningChain; }
    public TaskType getDominantTaskType() { return dominantTaskType; }
    public Map<TaskType, Double> getTaskTypeDistribution() { return taskTypeDistribution; }
    public int getNumReasoningSteps() { return numReasoningSteps; }
    public double getConfidenceThreshold() { return confidenceThreshold; }
}