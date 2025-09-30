package io.leavesfly.tinyai.nlp.deepseekR1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Tanh;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * 思维链(Chain of Thought)推理层
 * 
 * 这个层实现了DeepSeek-R1的核心思维链推理能力，
 * 能够将复杂问题分解为多个连续的推理步骤，
 * 每个步骤都生成中间推理状态。
 * 
 * 思维链推理层的核心功能：
 * 1. 问题分解：将复杂问题分解为子问题
 * 2. 步骤推理：为每个子问题生成推理步骤
 * 3. 状态跟踪：跟踪推理过程中的中间状态
 * 4. 置信度评估：评估每个推理步骤的置信度
 * 5. 步骤融合：将多个推理步骤的结果融合
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CoTReasoningLayer extends Layer {
    
    // 推理相关参数
    private int dModel;                    // 模型维度
    private int maxReasoningSteps;         // 最大推理步骤数
    private double reasoningThreshold;     // 推理置信度阈值
    private int reasoningHeads;            // 推理注意力头数
    
    // 推理组件
    private Parameter problemDecompWeight;  // 问题分解权重
    private Parameter problemDecompBias;    // 问题分解偏置
    private Parameter stepReasoningWeight;  // 步骤推理权重
    private Parameter stepReasoningBias;    // 步骤推理偏置
    private Parameter confidenceWeight;     // 置信度评估权重
    private Parameter confidenceBias;       // 置信度评估偏置
    private Parameter fusionWeight;         // 步骤融合权重
    private Parameter fusionBias;           // 步骤融合偏置
    
    // 层归一化
    private LayerNorm preReasoningNorm;     // 推理前归一化
    private LayerNorm postReasoningNorm;    // 推理后归一化
    
    // 推理状态
    private List<Variable> reasoningStates; // 推理中间状态
    private List<Double> stepConfidences;   // 每步置信度
    
    /**
     * 构造思维链推理层
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param maxReasoningSteps 最大推理步骤数
     * @param reasoningThreshold 推理置信度阈值
     */
    public CoTReasoningLayer(String name, int dModel, int maxReasoningSteps, double reasoningThreshold) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel <= 0 || maxReasoningSteps <= 0) {
            throw new IllegalArgumentException("dModel和maxReasoningSteps必须大于0");
        }
        if (reasoningThreshold < 0.0 || reasoningThreshold > 1.0) {
            throw new IllegalArgumentException("推理置信度阈值必须在0.0到1.0之间");
        }
        
        this.dModel = dModel;
        this.maxReasoningSteps = maxReasoningSteps;
        this.reasoningThreshold = reasoningThreshold;
        this.reasoningHeads = Math.max(4, dModel / 64); // 推理注意力头数
        
        this.reasoningStates = new ArrayList<>();
        this.stepConfidences = new ArrayList<>();
        
        init();
    }
    
    /**
     * 初始化权重矩阵
     * 
     * @param shape 权重形状
     * @return 初始化的权重矩阵
     */
    private NdArray initializeWeights(Shape shape) {
        NdArray weights = NdArray.of(shape);
        // 使用Xavier初始化
        double fanIn = shape.getDimension(0);
        double fanOut = shape.getDimension(1);
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        // 随机初始化
        for (int i = 0; i < weights.getShape().size(); i++) {
            double value = (Math.random() * 2.0 - 1.0) * limit;
            weights.set((float) value, i);
        }
        
        return weights;
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化问题分解权重和偏置
            problemDecompWeight = new Parameter(
                initializeWeights(Shape.of(dModel, dModel))
            );
            problemDecompBias = new Parameter(
                NdArray.zeros(Shape.of(dModel))
            );
            params.put("problemDecompWeight", problemDecompWeight);
            params.put("problemDecompBias", problemDecompBias);
            
            // 2. 初始化步骤推理权重和偏置
            stepReasoningWeight = new Parameter(
                initializeWeights(Shape.of(dModel, dModel))
            );
            stepReasoningBias = new Parameter(
                NdArray.zeros(Shape.of(dModel))
            );
            params.put("stepReasoningWeight", stepReasoningWeight);
            params.put("stepReasoningBias", stepReasoningBias);
            
            // 3. 初始化置信度评估权重和偏置
            confidenceWeight = new Parameter(
                initializeWeights(Shape.of(dModel, 1))
            );
            confidenceBias = new Parameter(
                NdArray.zeros(Shape.of(1))
            );
            params.put("confidenceWeight", confidenceWeight);
            params.put("confidenceBias", confidenceBias);
            
            // 4. 初始化步骤融合权重和偏置
            fusionWeight = new Parameter(
                initializeWeights(Shape.of(dModel * 2, dModel))
            );
            fusionBias = new Parameter(
                NdArray.zeros(Shape.of(dModel))
            );
            params.put("fusionWeight", fusionWeight);
            params.put("fusionBias", fusionBias);
            
            // 5. 初始化层归一化
            preReasoningNorm = new LayerNorm(name + "_pre_reasoning_norm", dModel);
            postReasoningNorm = new LayerNorm(name + "_post_reasoning_norm", dModel);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len, dModel)
        
        // 清空推理状态
        reasoningStates.clear();
        stepConfidences.clear();
        
        // 1. 推理前归一化
        Variable normalized = preReasoningNorm.layerForward(input);
        
        // 2. 问题分解
        Variable decomposed = performProblemDecomposition(normalized);
        
        // 3. 多步推理
        Variable reasoned = performMultiStepReasoning(decomposed);
        
        // 4. 推理后归一化
        Variable output = postReasoningNorm.layerForward(reasoned);
        
        // 5. 残差连接
        output = addResidualConnection(input, output);
        
        return output;
    }
    
    /**
     * 执行问题分解
     * 将输入问题分解为更小的子问题
     * 
     * @param input 输入张量
     * @return 分解后的表示
     */
    private Variable performProblemDecomposition(Variable input) {
        // 线性变换进行问题分解
        Variable weights = new Variable(problemDecompWeight.getValue());
        Variable bias = new Variable(problemDecompBias.getValue());
        
        // 矩阵乘法: input @ weight + bias
        MatMul matMul = new MatMul();
        NdArray output = matMul.forward(input.getValue(), weights.getValue());
        Variable outputVar = new Variable(output);
        outputVar = addBias(outputVar, bias);
        
        // 使用Tanh激活函数
        Tanh tanh = new Tanh();
        NdArray tanhOutput = tanh.forward(outputVar.getValue());
        outputVar = new Variable(tanhOutput);
        
        return outputVar;
    }
    
    /**
     * 执行多步推理
     * 迭代地进行推理步骤，直到达到置信度阈值或最大步数
     * 
     * @param input 分解后的问题表示
     * @return 推理后的表示
     */
    private Variable performMultiStepReasoning(Variable input) {
        Variable currentState = input;
        
        for (int step = 0; step < maxReasoningSteps; step++) {
            // 执行单步推理
            Variable stepResult = performSingleReasoningStep(currentState, step);
            
            // 计算置信度
            double confidence = calculateStepConfidence(stepResult);
            stepConfidences.add(confidence);
            
            // 保存推理状态
            reasoningStates.add(stepResult);
            
            // 检查是否达到置信度阈值
            if (confidence >= reasoningThreshold) {
                break;
            }
            
            // 更新当前状态
            currentState = stepResult;
        }
        
        // 融合所有推理步骤
        return fuseReasoningSteps();
    }
    
    /**
     * 执行单步推理
     * 
     * @param state 当前推理状态
     * @param stepIndex 推理步骤索引
     * @return 推理结果
     */
    private Variable performSingleReasoningStep(Variable state, int stepIndex) {
        // 使用步骤推理权重进行变换
        Variable weights = new Variable(stepReasoningWeight.getValue());
        Variable bias = new Variable(stepReasoningBias.getValue());
        
        // 线性变换
        MatMul matMul = new MatMul();
        NdArray output = matMul.forward(state.getValue(), weights.getValue());
        Variable outputVar = new Variable(output);
        outputVar = addBias(outputVar, bias);
        
        // 激活函数
        Tanh tanh = new Tanh();
        NdArray tanhOutput = tanh.forward(outputVar.getValue());
        outputVar = new Variable(tanhOutput);
        
        return outputVar;
    }
    
    /**
     * 计算推理步骤的置信度
     * 
     * @param stepResult 推理步骤结果
     * @return 置信度分数(0-1)
     */
    private double calculateStepConfidence(Variable stepResult) {
        // 使用置信度权重计算置信度分数
        Variable weights = new Variable(confidenceWeight.getValue());
        Variable bias = new Variable(confidenceBias.getValue());
        
        // 计算置信度: stepResult @ weights + bias
        MatMul matMul = new MatMul();
        NdArray confidenceArray = matMul.forward(stepResult.getValue(), weights.getValue());
        Variable confidence = new Variable(confidenceArray);
        confidence = addBias(confidence, bias);
        
        // 对最后一个时间步进行平均
        NdArray confArray = confidence.getValue();
        int batchSize = confArray.getShape().getDimension(0);
        int seqLen = confArray.getShape().getDimension(1);
        
        double sum = 0.0;
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                sum += confArray.get(b, t, 0);
            }
        }
        
        double avgConfidence = sum / (batchSize * seqLen);
        
        // 使用sigmoid将结果映射到[0,1]
        return 1.0 / (1.0 + Math.exp(-avgConfidence));
    }
    
    /**
     * 融合所有推理步骤的结果
     * 
     * @return 融合后的推理结果
     */
    private Variable fuseReasoningSteps() {
        if (reasoningStates.isEmpty()) {
            throw new IllegalStateException("没有推理步骤可以融合");
        }
        
        // 如果只有一个步骤，直接返回
        if (reasoningStates.size() == 1) {
            return reasoningStates.get(0);
        }
        
        // 加权融合多个推理步骤
        Variable fusedResult = null;
        double totalWeight = 0.0;
        
        for (int i = 0; i < reasoningStates.size(); i++) {
            Variable state = reasoningStates.get(i);
            double confidence = stepConfidences.get(i);
            
            if (fusedResult == null) {
                // 初始化融合结果
                fusedResult = multiplyByScalar(state, confidence);
            } else {
                // 加权累加
                Variable weightedState = multiplyByScalar(state, confidence);
                fusedResult = addVariables(fusedResult, weightedState);
            }
            
            totalWeight += confidence;
        }
        
        // 归一化
        if (totalWeight > 0.0) {
            fusedResult = multiplyByScalar(fusedResult, 1.0 / totalWeight);
        }
        
        return fusedResult;
    }
    
    /**
     * 添加偏置
     * 
     * @param input 输入变量
     * @param bias 偏置变量
     * @return 添加偏置后的结果
     */
    private Variable addBias(Variable input, Variable bias) {
        NdArray inputArray = input.getValue();
        NdArray biasArray = bias.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        int batchSize = inputArray.getShape().getDimension(0);
        int seqLen = inputArray.getShape().getDimension(1);
        int features = inputArray.getShape().getDimension(2);
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int f = 0; f < features; f++) {
                    float value = inputArray.get(b, t, f) + biasArray.get(f);
                    result.set(value, b, t, f);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 变量乘以标量
     * 
     * @param var 输入变量
     * @param scalar 标量值
     * @return 乘法结果
     */
    private Variable multiplyByScalar(Variable var, double scalar) {
        NdArray array = var.getValue();
        NdArray result = NdArray.of(array.getShape());
        
        for (int i = 0; i < array.getShape().size(); i++) {
            result.set((float)(array.get(i) * scalar), i);
        }
        
        return new Variable(result);
    }
    
    /**
     * 变量相加
     * 
     * @param var1 第一个变量
     * @param var2 第二个变量
     * @return 相加结果
     */
    private Variable addVariables(Variable var1, Variable var2) {
        NdArray array1 = var1.getValue();
        NdArray array2 = var2.getValue();
        NdArray result = NdArray.of(array1.getShape());
        
        for (int i = 0; i < array1.getShape().size(); i++) {
            result.set(array1.get(i) + array2.get(i), i);
        }
        
        return new Variable(result);
    }
    
    /**
     * 添加残差连接
     * 
     * @param input 原始输入
     * @param output 层输出
     * @return 残差连接结果
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        return addVariables(input, output);
    }
    
    /**
     * 获取推理步骤数量
     * 
     * @return 推理步骤数量
     */
    public int getNumReasoningSteps() {
        return reasoningStates.size();
    }
    
    /**
     * 获取推理步骤置信度列表
     * 
     * @return 置信度列表
     */
    public List<Double> getStepConfidences() {
        return new ArrayList<>(stepConfidences);
    }
    
    /**
     * 获取平均置信度
     * 
     * @return 平均置信度
     */
    public double getAverageConfidence() {
        if (stepConfidences.isEmpty()) {
            return 0.0;
        }
        
        double sum = stepConfidences.stream().mapToDouble(Double::doubleValue).sum();
        return sum / stepConfidences.size();
    }
    
    /**
     * 获取最高置信度
     * 
     * @return 最高置信度
     */
    public double getMaxConfidence() {
        return stepConfidences.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
    }
    
    /**
     * 重置推理状态
     */
    public void resetReasoningState() {
        reasoningStates.clear();
        stepConfidences.clear();
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getMaxReasoningSteps() { return maxReasoningSteps; }
    public double getReasoningThreshold() { return reasoningThreshold; }
    public int getReasoningHeads() { return reasoningHeads; }
    
    public List<Variable> getReasoningStates() { 
        return new ArrayList<>(reasoningStates); 
    }
}