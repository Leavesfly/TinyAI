package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * 自我反思模块
 * 
 * 对推理过程进行质量评估和改进建议，是DeepSeek R1
 * 实现自我纠错和持续改进的关键组件。
 * 
 * 反思功能：
 * 1. 质量评估：评估推理过程的质量和可信度
 * 2. 一致性检查：检查推理步骤之间的逻辑一致性
 * 3. 完整性验证：验证推理过程的完整性
 * 4. 改进建议：生成改进推理质量的建议
 * 5. 风险识别：识别潜在的推理风险点
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ReflectionModule extends Layer {
    
    private int dModel;                    // 模型维度
    private double qualityThreshold;       // 质量阈值
    
    // 反思评估器权重
    private Parameter reflectionEvaluatorW1;  // 反思评估第一层权重
    private Parameter reflectionEvaluatorB1;  // 反思评估第一层偏置
    private Parameter reflectionEvaluatorW2;  // 反思评估第二层权重
    private Parameter reflectionEvaluatorB2;  // 反思评估第二层偏置
    private Parameter reflectionEvaluatorW3;  // 反思评估第三层权重
    private Parameter reflectionEvaluatorB3;  // 反思评估第三层偏置
    
    // 改进建议生成器权重
    private Parameter improvementGeneratorW1; // 改进建议第一层权重
    private Parameter improvementGeneratorB1; // 改进建议第一层偏置
    private Parameter improvementGeneratorW2; // 改进建议第二层权重
    private Parameter improvementGeneratorB2; // 改进建议第二层偏置
    
    // 一致性检查器权重
    private Parameter consistencyCheckerW1;   // 一致性检查第一层权重
    private Parameter consistencyCheckerB1;   // 一致性检查第一层偏置
    private Parameter consistencyCheckerW2;   // 一致性检查第二层权重
    private Parameter consistencyCheckerB2;   // 一致性检查第二层偏置
    
    // 风险识别器权重
    private Parameter riskDetectorW1;         // 风险识别第一层权重
    private Parameter riskDetectorB1;         // 风险识别第一层偏置
    private Parameter riskDetectorW2;         // 风险识别第二层权重
    private Parameter riskDetectorB2;         // 风险识别第二层偏置
    
    /**
     * 构造自我反思模块
     * 
     * @param name 模块名称
     * @param dModel 模型维度
     * @param qualityThreshold 质量阈值
     */
    public ReflectionModule(String name, int dModel, double qualityThreshold) {
        super(name, Shape.of(-1, dModel * 2), Shape.of(-1, dModel));
        
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel必须大于0");
        }
        if (qualityThreshold < 0.0 || qualityThreshold > 1.0) {
            throw new IllegalArgumentException("质量阈值必须在0.0到1.0之间");
        }
        
        this.dModel = dModel;
        this.qualityThreshold = qualityThreshold;
        
        init();
    }
    
    /**
     * 默认构造函数
     */
    public ReflectionModule(String name, int dModel) {
        this(name, dModel, 0.7);
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
            int inputDim = dModel * 2; // 推理输出和原始输入组合
            int hiddenDim = dModel;
            int smallDim = dModel / 2;
            
            // 初始化反思评估器
            reflectionEvaluatorW1 = new Parameter(initializeWeights(Shape.of(inputDim, hiddenDim)));
            reflectionEvaluatorB1 = new Parameter(NdArray.zeros(Shape.of(hiddenDim)));
            reflectionEvaluatorW2 = new Parameter(initializeWeights(Shape.of(hiddenDim, smallDim)));
            reflectionEvaluatorB2 = new Parameter(NdArray.zeros(Shape.of(smallDim)));
            reflectionEvaluatorW3 = new Parameter(initializeWeights(Shape.of(smallDim, 1)));
            reflectionEvaluatorB3 = new Parameter(NdArray.zeros(Shape.of(1)));
            
            params.put("reflectionEvaluatorW1", reflectionEvaluatorW1);
            params.put("reflectionEvaluatorB1", reflectionEvaluatorB1);
            params.put("reflectionEvaluatorW2", reflectionEvaluatorW2);
            params.put("reflectionEvaluatorB2", reflectionEvaluatorB2);
            params.put("reflectionEvaluatorW3", reflectionEvaluatorW3);
            params.put("reflectionEvaluatorB3", reflectionEvaluatorB3);
            
            // 初始化改进建议生成器
            improvementGeneratorW1 = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            improvementGeneratorB1 = new Parameter(NdArray.zeros(Shape.of(dModel)));
            improvementGeneratorW2 = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            improvementGeneratorB2 = new Parameter(NdArray.zeros(Shape.of(dModel)));
            
            params.put("improvementGeneratorW1", improvementGeneratorW1);
            params.put("improvementGeneratorB1", improvementGeneratorB1);
            params.put("improvementGeneratorW2", improvementGeneratorW2);
            params.put("improvementGeneratorB2", improvementGeneratorB2);
            
            // 初始化一致性检查器
            consistencyCheckerW1 = new Parameter(initializeWeights(Shape.of(inputDim, hiddenDim)));
            consistencyCheckerB1 = new Parameter(NdArray.zeros(Shape.of(hiddenDim)));
            consistencyCheckerW2 = new Parameter(initializeWeights(Shape.of(hiddenDim, 1)));
            consistencyCheckerB2 = new Parameter(NdArray.zeros(Shape.of(1)));
            
            params.put("consistencyCheckerW1", consistencyCheckerW1);
            params.put("consistencyCheckerB1", consistencyCheckerB1);
            params.put("consistencyCheckerW2", consistencyCheckerW2);
            params.put("consistencyCheckerB2", consistencyCheckerB2);
            
            // 初始化风险识别器
            riskDetectorW1 = new Parameter(initializeWeights(Shape.of(inputDim, hiddenDim)));
            riskDetectorB1 = new Parameter(NdArray.zeros(Shape.of(hiddenDim)));
            riskDetectorW2 = new Parameter(initializeWeights(Shape.of(hiddenDim, 1)));
            riskDetectorB2 = new Parameter(NdArray.zeros(Shape.of(1)));
            
            params.put("riskDetectorW1", riskDetectorW1);
            params.put("riskDetectorB1", riskDetectorB1);
            params.put("riskDetectorW2", riskDetectorW2);
            params.put("riskDetectorB2", riskDetectorB2);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable reasoningOutput = inputs[0];  // shape: (batch_size, dModel)
        Variable originalInput = inputs[1];    // shape: (batch_size, dModel)
        
        // 返回推理输出（简化实现）
        // 实际应用中可以返回改进后的推理结果
        return reasoningOutput;
    }
    
    /**
     * 执行自我反思
     * 
     * @param reasoningOutput 推理模块的输出
     * @param originalInput 原始输入
     * @return 反思结果
     */
    public ReflectionResult performReflection(Variable reasoningOutput, Variable originalInput) {
        // 组合推理输出和原始输入
        Variable combinedInput = concatenate(reasoningOutput, originalInput);
        
        // 1. 评估推理质量
        double qualityScore = evaluateQuality(combinedInput);
        
        // 2. 检查一致性
        double consistencyScore = checkConsistency(combinedInput);
        
        // 3. 识别风险
        double riskScore = detectRisk(combinedInput);
        
        // 4. 生成改进建议
        Variable improvementSuggestion = generateImprovement(reasoningOutput);
        
        // 5. 综合评估
        boolean needsRefinement = (qualityScore < qualityThreshold) || 
                                  (consistencyScore < 0.8) || 
                                  (riskScore > 0.3);
        
        // 构建反思结果
        ReflectionResult result = new ReflectionResult();
        result.setQualityScore(qualityScore);
        result.setConsistencyScore(consistencyScore);
        result.setRiskScore(riskScore);
        result.setImprovementSuggestion(improvementSuggestion);
        result.setNeedsRefinement(needsRefinement);
        result.setOverallScore(calculateOverallScore(qualityScore, consistencyScore, riskScore));
        
        // 添加详细分析
        result.addAnalysis("质量分析", analyzeQuality(qualityScore));
        result.addAnalysis("一致性分析", analyzeConsistency(consistencyScore));
        result.addAnalysis("风险分析", analyzeRisk(riskScore));
        result.addAnalysis("改进建议", generateImprovementText(qualityScore, consistencyScore, riskScore));
        
        return result;
    }
    
    /**
     * 评估推理质量
     */
    private double evaluateQuality(Variable combinedInput) {
        // 三层网络评估质量
        Variable layer1 = linearTransform(combinedInput, reflectionEvaluatorW1, reflectionEvaluatorB1);
        layer1 = relu(layer1);
        
        Variable layer2 = linearTransform(layer1, reflectionEvaluatorW2, reflectionEvaluatorB2);
        layer2 = relu(layer2);
        
        Variable layer3 = linearTransform(layer2, reflectionEvaluatorW3, reflectionEvaluatorB3);
        layer3 = sigmoid(layer3);
        
        return getMeanValue(layer3);
    }
    
    /**
     * 检查一致性
     */
    private double checkConsistency(Variable combinedInput) {
        Variable layer1 = linearTransform(combinedInput, consistencyCheckerW1, consistencyCheckerB1);
        layer1 = relu(layer1);
        
        Variable layer2 = linearTransform(layer1, consistencyCheckerW2, consistencyCheckerB2);
        layer2 = sigmoid(layer2);
        
        return getMeanValue(layer2);
    }
    
    /**
     * 识别风险
     */
    private double detectRisk(Variable combinedInput) {
        Variable layer1 = linearTransform(combinedInput, riskDetectorW1, riskDetectorB1);
        layer1 = relu(layer1);
        
        Variable layer2 = linearTransform(layer1, riskDetectorW2, riskDetectorB2);
        layer2 = sigmoid(layer2);
        
        return getMeanValue(layer2);
    }
    
    /**
     * 生成改进建议
     */
    private Variable generateImprovement(Variable reasoningOutput) {
        Variable layer1 = linearTransform(reasoningOutput, improvementGeneratorW1, improvementGeneratorB1);
        layer1 = relu(layer1);
        
        Variable layer2 = linearTransform(layer1, improvementGeneratorW2, improvementGeneratorB2);
        
        return layer2;
    }
    
    /**
     * 计算综合评分
     */
    private double calculateOverallScore(double quality, double consistency, double risk) {
        // 加权计算综合评分
        double qualityWeight = 0.5;
        double consistencyWeight = 0.3;
        double riskWeight = 0.2;
        
        return quality * qualityWeight + 
               consistency * consistencyWeight + 
               (1.0 - risk) * riskWeight; // 风险越低，评分越高
    }
    
    /**
     * 分析质量评分
     */
    private String analyzeQuality(double qualityScore) {
        if (qualityScore >= 0.9) {
            return "推理质量优秀，逻辑清晰，结论可信";
        } else if (qualityScore >= 0.7) {
            return "推理质量良好，存在小幅改进空间";
        } else if (qualityScore >= 0.5) {
            return "推理质量一般，需要进一步优化";
        } else {
            return "推理质量较差，建议重新推理";
        }
    }
    
    /**
     * 分析一致性评分
     */
    private String analyzeConsistency(double consistencyScore) {
        if (consistencyScore >= 0.9) {
            return "推理步骤高度一致，逻辑连贯";
        } else if (consistencyScore >= 0.7) {
            return "推理步骤基本一致，偶有跳跃";
        } else if (consistencyScore >= 0.5) {
            return "推理步骤存在不一致，需要检查";
        } else {
            return "推理步骤严重不一致，逻辑混乱";
        }
    }
    
    /**
     * 分析风险评分
     */
    private String analyzeRisk(double riskScore) {
        if (riskScore <= 0.1) {
            return "风险极低，结果高度可信";
        } else if (riskScore <= 0.3) {
            return "风险较低，结果基本可信";
        } else if (riskScore <= 0.6) {
            return "风险中等，需要谨慎对待";
        } else {
            return "风险较高，建议进一步验证";
        }
    }
    
    /**
     * 生成改进建议文本
     */
    private String generateImprovementText(double quality, double consistency, double risk) {
        StringBuilder suggestions = new StringBuilder();
        
        if (quality < 0.7) {
            suggestions.append("建议：增强推理深度，提供更多支撑论据；");
        }
        if (consistency < 0.8) {
            suggestions.append("建议：检查推理步骤的逻辑连贯性；");
        }
        if (risk > 0.3) {
            suggestions.append("建议：验证关键假设和结论的可靠性；");
        }
        
        if (suggestions.length() == 0) {
            suggestions.append("推理质量良好，继续保持。");
        }
        
        return suggestions.toString();
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
     * 添加偏置
     */
    private Variable addBias(Variable input, Parameter bias) {
        NdArray inputArray = input.getValue();
        NdArray biasArray = bias.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        // 广播添加偏置
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            // 获取多维索引
            int[] indices = getIndicesFromFlat(inputArray.getShape(), i);
            int biasIndex = indices[indices.length - 1]; // 最后一个维度索引
            float value = inputArray.get(i) + biasArray.get(biasIndex);
            result.set(value, i);
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
     * 从平坦索引获取多维索引
     */
    private static int[] getIndicesFromFlat(Shape shape, int flatIndex) {
        int[] indices = new int[shape.getDimNum()];
        int remaining = flatIndex;
        
        for (int dim = shape.getDimNum() - 1; dim >= 0; dim--) {
            int dimSize = shape.getDimension(dim);
            indices[dim] = remaining % dimSize;
            remaining /= dimSize;
        }
        
        return indices;
    }
    
    /**
     * 反思结果类
     */
    public static class ReflectionResult {
        private double qualityScore;           // 质量评分
        private double consistencyScore;       // 一致性评分
        private double riskScore;              // 风险评分
        private double overallScore;           // 综合评分
        private Variable improvementSuggestion; // 改进建议向量
        private boolean needsRefinement;       // 是否需要改进
        private Map<String, String> analysis;  // 详细分析
        
        public ReflectionResult() {
            this.analysis = new HashMap<>();
        }
        
        public void addAnalysis(String key, String value) {
            analysis.put(key, value);
        }
        
        /**
         * 获取反思报告
         */
        public String getReflectionReport() {
            StringBuilder report = new StringBuilder();
            report.append("=== 推理反思报告 ===\n");
            report.append(String.format("质量评分: %.3f\n", qualityScore));
            report.append(String.format("一致性评分: %.3f\n", consistencyScore));
            report.append(String.format("风险评分: %.3f\n", riskScore));
            report.append(String.format("综合评分: %.3f\n", overallScore));
            report.append(String.format("需要改进: %s\n", needsRefinement ? "是" : "否"));
            report.append("\n详细分析:\n");
            
            for (Map.Entry<String, String> entry : analysis.entrySet()) {
                report.append(String.format("%s: %s\n", entry.getKey(), entry.getValue()));
            }
            
            return report.toString();
        }
        
        // Getter和Setter方法
        public double getQualityScore() { return qualityScore; }
        public void setQualityScore(double qualityScore) { this.qualityScore = qualityScore; }
        
        public double getConsistencyScore() { return consistencyScore; }
        public void setConsistencyScore(double consistencyScore) { this.consistencyScore = consistencyScore; }
        
        public double getRiskScore() { return riskScore; }
        public void setRiskScore(double riskScore) { this.riskScore = riskScore; }
        
        public double getOverallScore() { return overallScore; }
        public void setOverallScore(double overallScore) { this.overallScore = overallScore; }
        
        public Variable getImprovementSuggestion() { return improvementSuggestion; }
        public void setImprovementSuggestion(Variable improvementSuggestion) { this.improvementSuggestion = improvementSuggestion; }
        
        public boolean isNeedsRefinement() { return needsRefinement; }
        public void setNeedsRefinement(boolean needsRefinement) { this.needsRefinement = needsRefinement; }
        
        public Map<String, String> getAnalysis() { return new HashMap<>(analysis); }
        public void setAnalysis(Map<String, String> analysis) { this.analysis = new HashMap<>(analysis); }
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public double getQualityThreshold() { return qualityThreshold; }
    
    @Override
    public String toString() {
        return String.format("ReflectionModule(dModel=%d, qualityThreshold=%.3f)", 
                           dModel, qualityThreshold);
    }
}