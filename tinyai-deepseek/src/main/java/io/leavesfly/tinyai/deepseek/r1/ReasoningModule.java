package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Tanh;
import io.leavesfly.tinyai.func.math.Sigmoid;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * 推理模块 - DeepSeek R1的核心组件
 * 
 * 实现多步推理能力，将复杂问题分解为多个推理步骤，
 * 每个步骤包含思维状态编码、行动预测、置信度评估和验证。
 * 
 * 推理过程：
 * 1. 思维状态编码：将当前理解编码为内部表示
 * 2. 行动预测：基于当前状态预测下一步行动
 * 3. 置信度评估：评估当前推理步骤的可信度
 * 4. 验证检查：验证推理步骤的合理性
 * 5. 状态更新：更新推理状态继续下一步
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ReasoningModule extends Layer {
    
    private int dModel;                    // 模型维度
    private int numReasoningSteps;         // 最大推理步骤数
    private double reasoningThreshold;     // 推理置信度阈值
    
    // 思维状态编码器权重
    private Parameter thoughtEncoderW1;    // 思维编码第一层权重
    private Parameter thoughtEncoderB1;    // 思维编码第一层偏置
    private Parameter thoughtEncoderW2;    // 思维编码第二层权重
    private Parameter thoughtEncoderB2;    // 思维编码第二层偏置
    
    // 行动预测器权重
    private Parameter actionPredictorW1;   // 行动预测第一层权重
    private Parameter actionPredictorB1;   // 行动预测第一层偏置
    private Parameter actionPredictorW2;   // 行动预测第二层权重
    private Parameter actionPredictorB2;   // 行动预测第二层偏置
    
    // 置信度评估器权重
    private Parameter confidenceW1;        // 置信度评估第一层权重
    private Parameter confidenceB1;        // 置信度评估第一层偏置
    private Parameter confidenceW2;        // 置信度评估第二层权重
    private Parameter confidenceB2;        // 置信度评估第二层偏置
    
    // 验证器权重
    private Parameter verifierW1;          // 验证器第一层权重
    private Parameter verifierB1;          // 验证器第一层偏置
    private Parameter verifierW2;          // 验证器第二层权重
    private Parameter verifierB2;          // 验证器第二层偏置
    
    // 推理状态
    private List<Variable> reasoningStates;  // 推理中间状态
    private List<ReasoningStep> reasoningSteps; // 推理步骤记录
    
    /**
     * 构造推理模块
     * 
     * @param name 模块名称
     * @param dModel 模型维度
     * @param numReasoningSteps 最大推理步骤数
     * @param reasoningThreshold 推理置信度阈值
     */
    public ReasoningModule(String name, int dModel, int numReasoningSteps, double reasoningThreshold) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel <= 0 || numReasoningSteps <= 0) {
            throw new IllegalArgumentException("dModel和numReasoningSteps必须大于0");
        }
        if (reasoningThreshold < 0.0 || reasoningThreshold > 1.0) {
            throw new IllegalArgumentException("推理置信度阈值必须在0.0到1.0之间");
        }
        
        this.dModel = dModel;
        this.numReasoningSteps = numReasoningSteps;
        this.reasoningThreshold = reasoningThreshold;
        this.reasoningStates = new ArrayList<>();
        this.reasoningSteps = new ArrayList<>();
        
        init();
    }
    
    /**
     * 默认构造函数
     */
    public ReasoningModule(String name, int dModel, int numReasoningSteps) {
        this(name, dModel, numReasoningSteps, 0.7);
    }
    
    /**
     * 初始化权重矩阵
     */
    private NdArray initializeWeights(Shape shape) {
        NdArray weights = NdArray.of(shape);
        double fanIn = shape.getDimension(0);
        double fanOut = shape.getDimension(1);
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        for (int i = 0; i < weights.getShape().size(); i++) {
            double value = (Math.random() * 2.0 - 1.0) * limit;
            weights.set((float) value, i);
        }
        
        return weights;
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化思维状态编码器
            int hiddenDim = dModel * 2; // 扩展隐藏维度
            thoughtEncoderW1 = new Parameter(initializeWeights(Shape.of(dModel, hiddenDim)));
            thoughtEncoderB1 = new Parameter(NdArray.zeros(Shape.of(hiddenDim)));
            thoughtEncoderW2 = new Parameter(initializeWeights(Shape.of(hiddenDim, dModel)));
            thoughtEncoderB2 = new Parameter(NdArray.zeros(Shape.of(dModel)));
            
            params.put("thoughtEncoderW1", thoughtEncoderW1);
            params.put("thoughtEncoderB1", thoughtEncoderB1);
            params.put("thoughtEncoderW2", thoughtEncoderW2);
            params.put("thoughtEncoderB2", thoughtEncoderB2);
            
            // 初始化行动预测器
            actionPredictorW1 = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            actionPredictorB1 = new Parameter(NdArray.zeros(Shape.of(dModel)));
            actionPredictorW2 = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            actionPredictorB2 = new Parameter(NdArray.zeros(Shape.of(dModel)));
            
            params.put("actionPredictorW1", actionPredictorW1);
            params.put("actionPredictorB1", actionPredictorB1);
            params.put("actionPredictorW2", actionPredictorW2);
            params.put("actionPredictorB2", actionPredictorB2);
            
            // 初始化置信度评估器
            int confidenceDim = 64; // 小维度输出
            confidenceW1 = new Parameter(initializeWeights(Shape.of(dModel, confidenceDim)));
            confidenceB1 = new Parameter(NdArray.zeros(Shape.of(confidenceDim)));
            confidenceW2 = new Parameter(initializeWeights(Shape.of(confidenceDim, 1)));
            confidenceB2 = new Parameter(NdArray.zeros(Shape.of(1)));
            
            params.put("confidenceW1", confidenceW1);
            params.put("confidenceB1", confidenceB1);
            params.put("confidenceW2", confidenceW2);
            params.put("confidenceB2", confidenceB2);
            
            // 初始化验证器
            int verifierInputDim = dModel * 2; // 思维状态和行动状态组合
            verifierW1 = new Parameter(initializeWeights(Shape.of(verifierInputDim, dModel)));
            verifierB1 = new Parameter(NdArray.zeros(Shape.of(dModel)));
            verifierW2 = new Parameter(initializeWeights(Shape.of(dModel, 1)));
            verifierB2 = new Parameter(NdArray.zeros(Shape.of(1)));
            
            params.put("verifierW1", verifierW1);
            params.put("verifierB1", verifierB1);
            params.put("verifierW2", verifierW2);
            params.put("verifierB2", verifierB2);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputEmbedding = inputs[0];  // shape: (batch_size, seq_len, dModel)
        
        // 清空推理状态
        reasoningStates.clear();
        reasoningSteps.clear();
        
        // 获取平均池化的输入表示
        Variable currentState = meanPooling(inputEmbedding);  // shape: (batch_size, dModel)
        
        // 执行多步推理
        for (int step = 0; step < numReasoningSteps; step++) {
            ReasoningStepResult stepResult = performSingleReasoningStep(currentState, step);
            
            // 保存推理状态和步骤
            reasoningStates.add(stepResult.newState);
            reasoningSteps.add(stepResult.reasoningStep);
            
            // 检查是否达到置信度阈值
            if (stepResult.reasoningStep.getConfidence() >= reasoningThreshold) {
                break;
            }
            
            // 更新当前状态
            currentState = stepResult.newState;
        }
        
        // 返回最终推理状态，扩展为序列格式
        return expandToSequence(currentState, inputEmbedding.getValue().getShape());
    }
    
    /**
     * 执行单步推理
     * 
     * @param state 当前推理状态
     * @param stepIndex 步骤索引
     * @return 推理步骤结果
     */
    private ReasoningStepResult performSingleReasoningStep(Variable state, int stepIndex) {
        // 1. 编码思维状态
        Variable thoughtState = encodeThoughtState(state);
        
        // 2. 预测下一步行动
        Variable actionState = predictAction(thoughtState);
        
        // 3. 评估置信度
        double confidence = evaluateConfidence(actionState);
        
        // 4. 验证步骤
        double verificationScore = verifyStep(thoughtState, actionState);
        
        // 5. 更新状态（残差连接）
        Variable newState = updateState(state, actionState);
        
        // 6. 创建推理步骤记录
        ReasoningStep step = ReasoningStep.create(
            stepIndex,
            String.format("步骤 %d 思考", stepIndex + 1),
            String.format("步骤 %d 行动", stepIndex + 1),
            confidence
        );
        step.setVerification(String.format("验证分数: %.3f", verificationScore));
        
        return new ReasoningStepResult(newState, step);
    }
    
    /**
     * 编码思维状态
     */
    private Variable encodeThoughtState(Variable state) {
        // 第一层：线性变换 + ReLU激活
        Variable layer1 = linearTransform(state, thoughtEncoderW1, thoughtEncoderB1);
        layer1 = relu(layer1);
        
        // 第二层：线性变换
        Variable layer2 = linearTransform(layer1, thoughtEncoderW2, thoughtEncoderB2);
        
        return layer2;
    }
    
    /**
     * 预测下一步行动
     */
    private Variable predictAction(Variable thoughtState) {
        // 第一层：线性变换 + ReLU激活
        Variable layer1 = linearTransform(thoughtState, actionPredictorW1, actionPredictorB1);
        layer1 = relu(layer1);
        
        // 第二层：线性变换
        Variable layer2 = linearTransform(layer1, actionPredictorW2, actionPredictorB2);
        
        return layer2;
    }
    
    /**
     * 评估置信度
     */
    private double evaluateConfidence(Variable actionState) {
        // 第一层：线性变换 + ReLU激活
        Variable layer1 = linearTransform(actionState, confidenceW1, confidenceB1);
        layer1 = relu(layer1);
        
        // 第二层：线性变换 + Sigmoid激活
        Variable layer2 = linearTransform(layer1, confidenceW2, confidenceB2);
        layer2 = sigmoid(layer2);
        
        // 返回平均置信度
        return getMeanValue(layer2);
    }
    
    /**
     * 验证推理步骤
     */
    private double verifyStep(Variable thoughtState, Variable actionState) {
        // 组合思维状态和行动状态
        Variable combinedState = concatenate(thoughtState, actionState);
        
        // 第一层：线性变换 + ReLU激活
        Variable layer1 = linearTransform(combinedState, verifierW1, verifierB1);
        layer1 = relu(layer1);
        
        // 第二层：线性变换 + Sigmoid激活
        Variable layer2 = linearTransform(layer1, verifierW2, verifierB2);
        layer2 = sigmoid(layer2);
        
        // 返回验证分数
        return getMeanValue(layer2);
    }
    
    /**
     * 更新推理状态
     */
    private Variable updateState(Variable currentState, Variable actionState) {
        // 简单的残差连接和缩放
        Variable scaled = multiplyByScalar(actionState, 0.1);
        return addVariables(currentState, scaled);
    }
    
    /**
     * 线性变换：input @ weight + bias
     */
    private Variable linearTransform(Variable input, Parameter weight, Parameter bias) {
        MatMul matMul = new MatMul();
        NdArray result = matMul.forward(input.getValue(), weight.getValue());
        return addBias(new Variable(result), bias);
    }
    
    /**
     * ReLU激活函数
     */
    private Variable relu(Variable input) {
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            float value = inputArray.get(i);
            result.set(Math.max(0.0f, value), i);
        }
        
        return new Variable(result);
    }
    
    /**
     * Sigmoid激活函数
     */
    private Variable sigmoid(Variable input) {
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            float x = inputArray.get(i);
            float sigmoidValue = (float) (1.0 / (1.0 + Math.exp(-x)));
            result.set(sigmoidValue, i);
        }
        
        return new Variable(result);
    }
    
    /**
     * 平均池化
     */
    private Variable meanPooling(Variable input) {
        NdArray inputArray = input.getValue();
        int batchSize = inputArray.getShape().getDimension(0);
        int seqLen = inputArray.getShape().getDimension(1);
        int dModel = inputArray.getShape().getDimension(2);
        
        NdArray result = NdArray.of(Shape.of(batchSize, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dModel; d++) {
                float sum = 0.0f;
                for (int t = 0; t < seqLen; t++) {
                    sum += inputArray.get(b, t, d);
                }
                result.set(sum / seqLen, b, d);
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 扩展为序列格式
     */
    private Variable expandToSequence(Variable pooled, Shape originalShape) {
        NdArray pooledArray = pooled.getValue();
        int batchSize = originalShape.getDimension(0);
        int seqLen = originalShape.getDimension(1);
        int dModel = originalShape.getDimension(2);
        
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, dModel));
        
        // 将池化结果复制到每个时间步
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int d = 0; d < dModel; d++) {
                    result.set(pooledArray.get(b, d), b, t, d);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 添加偏置
     */
    private Variable addBias(Variable input, Parameter bias) {
        NdArray inputArray = input.getValue();
        NdArray biasArray = bias.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        // 广播添加偏置
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            int[] indices = inputArray.getShape().getIndices(i);
            int biasIndex = indices[indices.length - 1]; // 最后一个维度索引
            float value = inputArray.get(i) + biasArray.get(biasIndex);
            result.set(value, i);
        }
        
        return new Variable(result);
    }
    
    /**
     * 连接两个变量
     */
    private Variable concatenate(Variable var1, Variable var2) {
        NdArray array1 = var1.getValue();
        NdArray array2 = var2.getValue();
        
        int batchSize = array1.getShape().getDimension(0);
        int dim1 = array1.getShape().getDimension(1);
        int dim2 = array2.getShape().getDimension(1);
        
        NdArray result = NdArray.of(Shape.of(batchSize, dim1 + dim2));
        
        for (int b = 0; b < batchSize; b++) {
            // 复制第一个变量
            for (int d = 0; d < dim1; d++) {
                result.set(array1.get(b, d), b, d);
            }
            // 复制第二个变量
            for (int d = 0; d < dim2; d++) {
                result.set(array2.get(b, d), b, dim1 + d);
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 变量相加
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
     * 标量乘法
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
     * 获取变量的平均值
     */
    private double getMeanValue(Variable var) {
        NdArray array = var.getValue();
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < array.getShape().size(); i++) {
            sum += array.get(i);
            count++;
        }
        
        return count > 0 ? sum / count : 0.0;
    }
    
    /**
     * 推理步骤结果内部类
     */
    private static class ReasoningStepResult {
        final Variable newState;
        final ReasoningStep reasoningStep;
        
        ReasoningStepResult(Variable newState, ReasoningStep reasoningStep) {
            this.newState = newState;
            this.reasoningStep = reasoningStep;
        }
    }
    
    /**
     * 重置推理状态
     */
    public void resetReasoningState() {
        reasoningStates.clear();
        reasoningSteps.clear();
    }
    
    /**
     * 获取推理步骤记录
     */
    public List<ReasoningStep> getReasoningSteps() {
        return new ArrayList<>(reasoningSteps);
    }
    
    /**
     * 获取推理状态历史
     */
    public List<Variable> getReasoningStates() {
        return new ArrayList<>(reasoningStates);
    }
    
    /**
     * 获取平均置信度
     */
    public double getAverageConfidence() {
        if (reasoningSteps.isEmpty()) {
            return 0.0;
        }
        
        double sum = reasoningSteps.stream()
                                  .mapToDouble(ReasoningStep::getConfidence)
                                  .sum();
        return sum / reasoningSteps.size();
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getNumReasoningSteps() { return numReasoningSteps; }
    public double getReasoningThreshold() { return reasoningThreshold; }
    public int getCurrentStepCount() { return reasoningSteps.size(); }
    
    @Override
    public String toString() {
        return String.format("ReasoningModule(dModel=%d, maxSteps=%d, threshold=%.3f)", 
                           dModel, numReasoningSteps, reasoningThreshold);
    }
}