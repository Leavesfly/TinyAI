package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.activate.SigmoidLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
// import io.leavesfly.tinyai.nnet.layer.norm.DropoutLayer; // 暂时不可用

/**
 * 自我反思模块Block - DeepSeek R1的反思和质量评估组件
 * 
 * 该模块负责对推理过程进行自我反思，包括：
 * 1. 推理质量评估
 * 2. 改进建议生成
 * 3. 确定是否需要进一步优化
 * 
 * 基于Python实现中的ReflectionModule，使用TinyAI架构重新实现
 */
public class ReflectionBlock extends Block {
    
    private int dModel;
    private double qualityThreshold;
    
    // 反思评估器
    private LinearLayer reflectionEvaluatorLayer1;
    private ReLuLayer reflectionActivation1;
    // private DropoutLayer reflectionDropout; // 暂时不使用dropout
    private LinearLayer reflectionEvaluatorLayer2;
    private ReLuLayer reflectionActivation2;
    private LinearLayer reflectionEvaluatorLayer3;
    private SigmoidLayer reflectionSigmoid;
    
    // 改进建议生成器
    private LinearLayer improvementGeneratorLayer1;
    private ReLuLayer improvementActivation;
    private LinearLayer improvementGeneratorLayer2;
    
    /**
     * 构造反思Block
     * 
     * @param name 组件名称
     * @param dModel 模型维度
     * @param qualityThreshold 质量阈值，低于此值需要改进，默认0.7
     */
    public ReflectionBlock(String name, int dModel, double qualityThreshold) {
        super(name, Shape.of(-1, dModel * 2), Shape.of(-1, dModel));
        this.dModel = dModel;
        this.qualityThreshold = qualityThreshold;
        init();
    }
    
    /**
     * 使用默认质量阈值的构造函数
     */
    public ReflectionBlock(String name, int dModel) {
        this(name, dModel, 0.7);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 反思评估器：dModel*2 -> dModel -> dModel/2 -> 1
            reflectionEvaluatorLayer1 = new LinearLayer(name + "_reflection_eval1", dModel * 2, dModel, true);
            reflectionActivation1 = new ReLuLayer(name + "_reflection_relu1", Shape.of(-1, dModel));
            // reflectionDropout = new DropoutLayer(name + "_reflection_dropout", Shape.of(-1, dModel), 0.1f); // 暂时不使用
            reflectionEvaluatorLayer2 = new LinearLayer(name + "_reflection_eval2", dModel, dModel / 2, true);
            reflectionActivation2 = new ReLuLayer(name + "_reflection_relu2", Shape.of(-1, dModel / 2));
            reflectionEvaluatorLayer3 = new LinearLayer(name + "_reflection_eval3", dModel / 2, 1, true);
            reflectionSigmoid = new SigmoidLayer(name + "_reflection_sigmoid");
            
            addLayer(reflectionEvaluatorLayer1);
            addLayer(reflectionActivation1);
            // addLayer(reflectionDropout); // 暂时跳过dropout
            addLayer(reflectionEvaluatorLayer2);
            addLayer(reflectionActivation2);
            addLayer(reflectionEvaluatorLayer3);
            addLayer(reflectionSigmoid);
            
            // 改进建议生成器：dModel -> dModel -> dModel
            improvementGeneratorLayer1 = new LinearLayer(name + "_improvement1", dModel, dModel, true);
            improvementActivation = new ReLuLayer(name + "_improvement_relu", Shape.of(-1, dModel));
            improvementGeneratorLayer2 = new LinearLayer(name + "_improvement2", dModel, dModel, true);
            
            addLayer(improvementGeneratorLayer1);
            addLayer(improvementActivation);
            addLayer(improvementGeneratorLayer2);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        if (inputs.length < 2) {
            throw new IllegalArgumentException("ReflectionBlock需要两个输入：推理输出和原始输入");
        }
        
        Variable reasoningOutput = inputs[0];
        Variable originalInput = inputs[1];
        
        // 组合推理输出和原始输入
        Variable combinedInput = combineInputs(reasoningOutput, originalInput);
        
        // 评估推理质量
        Variable qualityScore = evaluateQuality(combinedInput);
        
        // 生成改进建议
        Variable improvementSuggestion = generateImprovement(reasoningOutput);
        
        // 创建反思结果（返回改进建议作为主要输出）
        return improvementSuggestion;
    }
    
    /**
     * 进行完整的反思分析
     * 返回反思结果信息
     */
    public ReflectionResult performReflection(Variable reasoningOutput, Variable originalInput) {
        // 组合输入
        Variable combinedInput = combineInputs(reasoningOutput, originalInput);
        
        // 评估质量
        Variable qualityScore = evaluateQuality(combinedInput);
        float qualityValue = extractScalarValue(qualityScore);
        
        // 生成改进建议
        Variable improvementSuggestion = generateImprovement(reasoningOutput);
        
        // 判断是否需要改进
        boolean needsRefinement = qualityValue < qualityThreshold;
        
        return new ReflectionResult(
            qualityValue,
            improvementSuggestion,
            needsRefinement,
            generateQualityDescription(qualityValue)
        );
    }
    
    /**
     * 组合推理输出和原始输入
     */
    private Variable combineInputs(Variable reasoningOutput, Variable originalInput) {
        NdArray reasoningData = reasoningOutput.getValue();
        NdArray originalData = originalInput.getValue();
        
        // 确保两个输入的batch维度一致
        int batchSize = reasoningData.getShape().getDimension(0);
        int originalBatch = originalData.getShape().getDimension(0);
        
        if (batchSize != originalBatch) {
            throw new IllegalArgumentException(
                String.format("批次大小不匹配：推理输出=%d, 原始输入=%d", batchSize, originalBatch)
            );
        }
        
        // 创建组合数组：[batch_size, dModel * 2]
        NdArray combined = NdArray.zeros(Shape.of(batchSize, dModel * 2));
        
        // 拼接操作
        for (int b = 0; b < batchSize; b++) {
            // 复制推理输出
            for (int d = 0; d < dModel; d++) {
                combined.set(reasoningData.get(b, d), b, d);
            }
            // 复制原始输入
            for (int d = 0; d < dModel; d++) {
                combined.set(originalData.get(b, d), b, d + dModel);
            }
        }
        
        return new Variable(combined);
    }
    
    /**
     * 评估推理质量
     */
    private Variable evaluateQuality(Variable combinedInput) {
        Variable hidden1 = reflectionEvaluatorLayer1.layerForward(combinedInput);
        Variable activated1 = reflectionActivation1.layerForward(hidden1);
        Variable dropped = activated1; // 跳过dropout，直接使用激活后的结果
        Variable hidden2 = reflectionEvaluatorLayer2.layerForward(dropped);
        Variable activated2 = reflectionActivation2.layerForward(hidden2);
        Variable output = reflectionEvaluatorLayer3.layerForward(activated2);
        return reflectionSigmoid.layerForward(output);
    }
    
    /**
     * 生成改进建议
     */
    private Variable generateImprovement(Variable reasoningOutput) {
        Variable hidden = improvementGeneratorLayer1.layerForward(reasoningOutput);
        Variable activated = improvementActivation.layerForward(hidden);
        return improvementGeneratorLayer2.layerForward(activated);
    }
    
    /**
     * 提取标量值
     */
    private float extractScalarValue(Variable variable) {
        NdArray data = variable.getValue();
        // 使用getNumber()方法获取标量值或第一个元素
        return data.getNumber().floatValue();
    }
    
    /**
     * 生成质量描述
     */
    private String generateQualityDescription(float qualityScore) {
        if (qualityScore >= 0.9f) {
            return "推理质量优秀，逻辑清晰且结论可靠";
        } else if (qualityScore >= 0.8f) {
            return "推理质量良好，大部分步骤合理";
        } else if (qualityScore >= 0.7f) {
            return "推理质量中等，可以接受但有改进空间";
        } else if (qualityScore >= 0.6f) {
            return "推理质量一般，需要进一步改进";
        } else if (qualityScore >= 0.5f) {
            return "推理质量较差，存在明显问题";
        } else {
            return "推理质量不佳，需要重新思考";
        }
    }
    
    /**
     * 反思结果类
     */
    public static class ReflectionResult {
        private float qualityScore;
        private Variable improvementSuggestion;
        private boolean needsRefinement;
        private String qualityDescription;
        
        public ReflectionResult(float qualityScore, Variable improvementSuggestion, 
                              boolean needsRefinement, String qualityDescription) {
            this.qualityScore = qualityScore;
            this.improvementSuggestion = improvementSuggestion;
            this.needsRefinement = needsRefinement;
            this.qualityDescription = qualityDescription;
        }
        
        // Getters
        public float getQualityScore() { return qualityScore; }
        public Variable getImprovementSuggestion() { return improvementSuggestion; }
        public boolean needsRefinement() { return needsRefinement; }
        public String getQualityDescription() { return qualityDescription; }
        
        @Override
        public String toString() {
            return String.format("反思结果: 质量分数=%.3f, 需要改进=%s, 描述=[%s]",
                    qualityScore, needsRefinement ? "是" : "否", qualityDescription);
        }
    }
    
    /**
     * 获取质量阈值
     */
    public double getQualityThreshold() {
        return qualityThreshold;
    }
    
    /**
     * 设置质量阈值
     */
    public void setQualityThreshold(double qualityThreshold) {
        this.qualityThreshold = qualityThreshold;
    }
    
    /**
     * 获取模型维度
     */
    public int getDModel() {
        return dModel;
    }
}