package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.activate.SigmoidLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 推理模块Block - DeepSeek R1的核心推理组件
 * 
 * 该模块负责执行多步推理过程，包括：
 * 1. 思维状态编码
 * 2. 行动预测
 * 3. 置信度评估
 * 4. 验证机制
 * 
 * 基于Python实现中的ReasoningModule，使用TinyAI架构重新实现
 */
public class ReasoningBlock extends Block {
    
    private int dModel;
    private int numReasoningSteps;
    
    // 思维状态编码器
    private LinearLayer thoughtEncoderLayer1;
    private ReLuLayer thoughtActivation;
    private LinearLayer thoughtEncoderLayer2;
    
    // 行动预测器
    private LinearLayer actionPredictorLayer1;
    private ReLuLayer actionActivation;
    private LinearLayer actionPredictorLayer2;
    
    // 置信度评估器
    private LinearLayer confidenceLayer1;
    private ReLuLayer confidenceActivation;
    private LinearLayer confidenceLayer2;
    private SigmoidLayer confidenceSigmoid;
    
    // 验证器
    private LinearLayer verifierLayer1;
    private ReLuLayer verifierActivation;
    private LinearLayer verifierLayer2;
    private SigmoidLayer verifierSigmoid;
    
    /**
     * 构造推理Block
     * 
     * @param name 组件名称
     * @param dModel 模型维度
     * @param numReasoningSteps 推理步骤数量，默认为5
     */
    public ReasoningBlock(String name, int dModel, int numReasoningSteps) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, dModel));
        this.dModel = dModel;
        this.numReasoningSteps = numReasoningSteps;
        init();
    }
    
    /**
     * 使用默认推理步骤数的构造函数
     */
    public ReasoningBlock(String name, int dModel) {
        this(name, dModel, 5);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 思维状态编码器：dModel -> dModel*2 -> dModel
            thoughtEncoderLayer1 = new LinearLayer(name + "_thought_encoder1", dModel, dModel * 2, true);
            thoughtActivation = new ReLuLayer(name + "_thought_relu", Shape.of(-1, dModel * 2));
            thoughtEncoderLayer2 = new LinearLayer(name + "_thought_encoder2", dModel * 2, dModel, true);
            
            addLayer(thoughtEncoderLayer1);
            addLayer(thoughtActivation);
            addLayer(thoughtEncoderLayer2);
            
            // 行动预测器：dModel -> dModel -> dModel
            actionPredictorLayer1 = new LinearLayer(name + "_action_predictor1", dModel, dModel, true);
            actionActivation = new ReLuLayer(name + "_action_relu", Shape.of(-1, dModel));
            actionPredictorLayer2 = new LinearLayer(name + "_action_predictor2", dModel, dModel, true);
            
            addLayer(actionPredictorLayer1);
            addLayer(actionActivation);
            addLayer(actionPredictorLayer2);
            
            // 置信度评估器：dModel -> 64 -> 1
            confidenceLayer1 = new LinearLayer(name + "_confidence1", dModel, 64, true);
            confidenceActivation = new ReLuLayer(name + "_confidence_relu", Shape.of(-1, 64));
            confidenceLayer2 = new LinearLayer(name + "_confidence2", 64, 1, true);
            confidenceSigmoid = new SigmoidLayer(name + "_confidence_sigmoid");
            
            addLayer(confidenceLayer1);
            addLayer(confidenceActivation);
            addLayer(confidenceLayer2);
            addLayer(confidenceSigmoid);
            
            // 验证器：dModel*2 -> dModel -> 1
            verifierLayer1 = new LinearLayer(name + "_verifier1", dModel * 2, dModel, true);
            verifierActivation = new ReLuLayer(name + "_verifier_relu", Shape.of(-1, dModel));
            verifierLayer2 = new LinearLayer(name + "_verifier2", dModel, 1, true);
            verifierSigmoid = new SigmoidLayer(name + "_verifier_sigmoid");
            
            addLayer(verifierLayer1);
            addLayer(verifierActivation);
            addLayer(verifierLayer2);
            addLayer(verifierSigmoid);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputEmbedding = inputs[0];
        NdArray inputData = inputEmbedding.getValue();
        
        // 获取输入维度：[batch_size, seq_len, d_model]
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 计算序列的平均值作为初始状态：[batch_size, d_model]
        Variable currentState = meanAlongSequence(inputEmbedding);
        
        // 存储推理步骤信息（简化版）
        List<ReasoningStepInfo> reasoningSteps = new ArrayList<>();
        
        // 执行多步推理
        for (int step = 0; step < numReasoningSteps; step++) {
            // 1. 编码思维状态
            Variable thoughtState = encodeThoughtState(currentState);
            
            // 2. 预测下一步行动
            Variable actionState = predictAction(thoughtState);
            
            // 3. 评估置信度
            Variable confidence = assessConfidence(actionState);
            
            // 4. 验证步骤
            Variable combinedState = combineStates(thoughtState, actionState);
            Variable verificationScore = verifyStep(combinedState);
            
            // 5. 更新状态（加权更新）
            currentState = updateState(currentState, actionState, 0.1f);
            
            // 记录推理步骤
            ReasoningStepInfo stepInfo = new ReasoningStepInfo(
                step + 1,
                extractConfidenceValue(confidence),
                extractVerificationValue(verificationScore)
            );
            reasoningSteps.add(stepInfo);
        }
        
        return currentState;
    }
    
    /**
     * 计算序列维度的平均值
     */
    private Variable meanAlongSequence(Variable input) {
        NdArray inputData = input.getValue();
        Shape inputShape = inputData.getShape();
        
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int dModel = inputShape.getDimension(2);
        
        // 创建输出数组：[batch_size, d_model]
        NdArray output = NdArray.zeros(Shape.of(batchSize, dModel));
        
        // 计算每个batch的序列平均值
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dModel; d++) {
                float sum = 0.0f;
                for (int s = 0; s < seqLen; s++) {
                    sum += inputData.get(b, s, d);
                }
                output.set(sum / seqLen, b, d);
            }
        }
        
        return new Variable(output);
    }
    
    /**
     * 编码思维状态
     */
    private Variable encodeThoughtState(Variable currentState) {
        Variable hidden = thoughtEncoderLayer1.layerForward(currentState);
        Variable activated = thoughtActivation.layerForward(hidden);
        return thoughtEncoderLayer2.layerForward(activated);
    }
    
    /**
     * 预测行动
     */
    private Variable predictAction(Variable thoughtState) {
        Variable hidden = actionPredictorLayer1.layerForward(thoughtState);
        Variable activated = actionActivation.layerForward(hidden);
        return actionPredictorLayer2.layerForward(activated);
    }
    
    /**
     * 评估置信度
     */
    private Variable assessConfidence(Variable actionState) {
        Variable hidden = confidenceLayer1.layerForward(actionState);
        Variable activated = confidenceActivation.layerForward(hidden);
        Variable linear = confidenceLayer2.layerForward(activated);
        return confidenceSigmoid.layerForward(linear);
    }
    
    /**
     * 组合状态
     */
    private Variable combineStates(Variable thoughtState, Variable actionState) {
        // 拼接两个状态：[batch_size, dModel * 2]
        NdArray thoughtData = thoughtState.getValue();
        NdArray actionData = actionState.getValue();
        
        int batchSize = thoughtData.getShape().getDimension(0);
        NdArray combined = NdArray.zeros(Shape.of(batchSize, dModel * 2));
        
        // 拼接操作
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dModel; d++) {
                combined.set(thoughtData.get(b, d), b, d);
                combined.set(actionData.get(b, d), b, d + dModel);
            }
        }
        
        return new Variable(combined);
    }
    
    /**
     * 验证步骤
     */
    private Variable verifyStep(Variable combinedState) {
        Variable hidden = verifierLayer1.layerForward(combinedState);
        Variable activated = verifierActivation.layerForward(hidden);
        Variable linear = verifierLayer2.layerForward(activated);
        return verifierSigmoid.layerForward(linear);
    }
    
    /**
     * 更新状态
     */
    private Variable updateState(Variable currentState, Variable actionState, float updateRate) {
        // currentState + updateRate * actionState
        Variable scaledAction = new Variable(actionState.getValue().mulNum(updateRate));
        return currentState.add(scaledAction);
    }
    
    /**
     * 提取置信度数值
     */
    private float extractConfidenceValue(Variable confidence) {
        NdArray data = confidence.getValue();
        // 使用getNumber()方法获取标量值或第一个元素
        return data.getNumber().floatValue();
    }
    
    /**
     * 提取验证分数数值
     */
    private float extractVerificationValue(Variable verification) {
        NdArray data = verification.getValue();
        // 使用getNumber()方法获取标量值或第一个元素
        return data.getNumber().floatValue();
    }
    
    /**
     * 推理步骤信息类
     */
    public static class ReasoningStepInfo {
        private int stepNumber;
        private float confidence;
        private float verificationScore;
        private String thought;
        private String action;
        
        public ReasoningStepInfo(int stepNumber, float confidence, float verificationScore) {
            this.stepNumber = stepNumber;
            this.confidence = confidence;
            this.verificationScore = verificationScore;
            this.thought = "推理步骤 " + stepNumber + " 思考";
            this.action = "推理步骤 " + stepNumber + " 行动";
        }
        
        // Getters
        public int getStepNumber() { return stepNumber; }
        public float getConfidence() { return confidence; }
        public float getVerificationScore() { return verificationScore; }
        public String getThought() { return thought; }
        public String getAction() { return action; }
        
        @Override
        public String toString() {
            return String.format("步骤%d: 思考=[%s], 行动=[%s], 置信度=%.3f, 验证分数=%.3f",
                    stepNumber, thought, action, confidence, verificationScore);
        }
    }
    
    /**
     * 获取推理步骤数量
     */
    public int getNumReasoningSteps() {
        return numReasoningSteps;
    }
    
    /**
     * 获取模型维度
     */
    public int getDModel() {
        return dModel;
    }
}