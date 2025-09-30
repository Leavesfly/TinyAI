package io.leavesfly.tinyai.nlp.deepseekR1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

/**
 * 推理输出头
 * 
 * 这个层负责将DeepSeek-R1的推理结果转换为最终的输出格式，
 * 包括生成答案token概率分布和推理置信度信息。
 * 
 * 推理输出头的特点：
 * 1. 双输出模式：同时输出答案和推理过程
 * 2. 置信度分数：为每个输出提供置信度评估
 * 3. 推理步骤标记：标识输出是否来自推理过程
 * 4. 自适应阈值：根据推理质量调整输出阈值
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ReasoningOutputHead extends Layer {
    
    private int dModel;                 // 模型维度
    private int vocabSize;              // 词汇表大小
    private int maxReasoningSteps;      // 最大推理步骤数
    private boolean useBias;            // 是否使用偏置
    
    // 输出参数
    private Parameter answerProjectionWeight;    // 答案投影权重
    private Parameter answerProjectionBias;      // 答案投影偏置
    private Parameter reasoningProjectionWeight; // 推理投影权重
    private Parameter reasoningProjectionBias;   // 推理投影偏置
    private Parameter confidenceWeight;          // 置信度权重
    private Parameter confidenceBias;            // 置信度偏置
    
    /**
     * 构造推理输出头
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param vocabSize 词汇表大小
     * @param maxReasoningSteps 最大推理步骤数
     * @param useBias 是否使用偏置
     */
    public ReasoningOutputHead(String name, int dModel, int vocabSize, int maxReasoningSteps, boolean useBias) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, vocabSize));
        
        if (dModel <= 0 || vocabSize <= 0 || maxReasoningSteps <= 0) {
            throw new IllegalArgumentException("所有参数必须大于0");
        }
        
        this.dModel = dModel;
        this.vocabSize = vocabSize;
        this.maxReasoningSteps = maxReasoningSteps;
        this.useBias = useBias;
        
        init();
    }
    
    /**
     * 简化构造函数（默认使用偏置）
     */
    public ReasoningOutputHead(String name, int dModel, int vocabSize, int maxReasoningSteps) {
        this(name, dModel, vocabSize, maxReasoningSteps, true);
    }
    
    /**
     * 初始化权重矩阵
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
            // 1. 初始化答案投影权重
            answerProjectionWeight = new Parameter(
                initializeWeights(Shape.of(dModel, vocabSize))
            );
            params.put("answerProjectionWeight", answerProjectionWeight);
            
            if (useBias) {
                answerProjectionBias = new Parameter(
                    NdArray.zeros(Shape.of(vocabSize))
                );
                params.put("answerProjectionBias", answerProjectionBias);
            }
            
            // 2. 初始化推理投影权重（用于生成推理token）
            reasoningProjectionWeight = new Parameter(
                initializeWeights(Shape.of(dModel, vocabSize))
            );
            params.put("reasoningProjectionWeight", reasoningProjectionWeight);
            
            if (useBias) {
                reasoningProjectionBias = new Parameter(
                    NdArray.zeros(Shape.of(vocabSize))
                );
                params.put("reasoningProjectionBias", reasoningProjectionBias);
            }
            
            // 3. 初始化置信度权重
            confidenceWeight = new Parameter(
                initializeWeights(Shape.of(dModel, 1))
            );
            confidenceBias = new Parameter(
                NdArray.zeros(Shape.of(1))
            );
            params.put("confidenceWeight", confidenceWeight);
            params.put("confidenceBias", confidenceBias);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len, dModel)
        
        // 1. 计算答案logits
        Variable answerLogits = computeAnswerLogits(input);
        
        // 2. 计算推理logits（可选）
        Variable reasoningLogits = computeReasoningLogits(input);
        
        // 3. 计算置信度分数
        Variable confidenceScores = computeConfidenceScores(input);
        
        // 4. 融合答案和推理logits
        Variable finalLogits = fuseLogits(answerLogits, reasoningLogits, confidenceScores);
        
        return finalLogits;
    }
    
    /**
     * 计算答案logits
     * 
     * @param input 输入特征
     * @return 答案logits
     */
    private Variable computeAnswerLogits(Variable input) {
        Variable weights = new Variable(answerProjectionWeight.getValue());
        
        MatMul matMul = new MatMul();
        NdArray logits = matMul.forward(input.getValue(), weights.getValue());
        Variable result = new Variable(logits);
        
        // 添加偏置（如果使用）
        if (useBias && answerProjectionBias != null) {
            Variable bias = new Variable(answerProjectionBias.getValue());
            result = addBias(result, bias);
        }
        
        return result;
    }
    
    /**
     * 计算推理logits
     * 
     * @param input 输入特征
     * @return 推理logits
     */
    private Variable computeReasoningLogits(Variable input) {
        Variable weights = new Variable(reasoningProjectionWeight.getValue());
        
        MatMul matMul = new MatMul();
        NdArray logits = matMul.forward(input.getValue(), weights.getValue());
        Variable result = new Variable(logits);
        
        // 添加偏置（如果使用）
        if (useBias && reasoningProjectionBias != null) {
            Variable bias = new Variable(reasoningProjectionBias.getValue());
            result = addBias(result, bias);
        }
        
        return result;
    }
    
    /**
     * 计算置信度分数
     * 
     * @param input 输入特征
     * @return 置信度分数
     */
    private Variable computeConfidenceScores(Variable input) {
        Variable weights = new Variable(confidenceWeight.getValue());
        Variable bias = new Variable(confidenceBias.getValue());
        
        MatMul matMul = new MatMul();
        NdArray confidence = matMul.forward(input.getValue(), weights.getValue());
        Variable result = new Variable(confidence);
        result = addBias(result, bias);
        
        // 使用sigmoid激活函数将置信度映射到[0,1]
        result = applySigmoid(result);
        
        return result;
    }
    
    /**
     * 融合答案和推理logits
     * 
     * @param answerLogits 答案logits
     * @param reasoningLogits 推理logits
     * @param confidenceScores 置信度分数
     * @return 融合后的logits
     */
    private Variable fuseLogits(Variable answerLogits, Variable reasoningLogits, Variable confidenceScores) {
        // 基于置信度分数动态融合答案和推理logits
        NdArray answerArray = answerLogits.getValue();
        NdArray reasoningArray = reasoningLogits.getValue();
        NdArray confidenceArray = confidenceScores.getValue();
        NdArray fusedArray = NdArray.of(answerArray.getShape());
        
        int batchSize = answerArray.getShape().getDimension(0);
        int seqLen = answerArray.getShape().getDimension(1);
        int vocabSize = answerArray.getShape().getDimension(2);
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // 获取当前位置的置信度
                float confidence = confidenceArray.get(b, t, 0);
                
                for (int v = 0; v < vocabSize; v++) {
                    float answerLogit = answerArray.get(b, t, v);
                    float reasoningLogit = reasoningArray.get(b, t, v);
                    
                    // 基于置信度加权融合
                    float fusedLogit = confidence * answerLogit + (1.0f - confidence) * reasoningLogit;
                    fusedArray.set(fusedLogit, b, t, v);
                }
            }
        }
        
        return new Variable(fusedArray);
    }
    
    /**
     * 应用sigmoid激活函数
     * 
     * @param input 输入变量
     * @return sigmoid结果
     */
    private Variable applySigmoid(Variable input) {
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            float value = inputArray.get(i);
            float sigmoidValue = (float)(1.0 / (1.0 + Math.exp(-value)));
            result.set(sigmoidValue, i);
        }
        
        return new Variable(result);
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
     * 生成推理专用的输出
     * 
     * @param input 输入特征
     * @return 推理输出
     */
    public Variable generateReasoningOutput(Variable input) {
        return computeReasoningLogits(input);
    }
    
    /**
     * 生成答案专用的输出
     * 
     * @param input 输入特征
     * @return 答案输出
     */
    public Variable generateAnswerOutput(Variable input) {
        return computeAnswerLogits(input);
    }
    
    /**
     * 获取输出置信度
     * 
     * @param input 输入特征
     * @return 置信度分数
     */
    public Variable getOutputConfidence(Variable input) {
        return computeConfidenceScores(input);
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getVocabSize() { return vocabSize; }
    public int getMaxReasoningSteps() { return maxReasoningSteps; }
    public boolean isUseBias() { return useBias; }
    
    public Parameter getAnswerProjectionWeight() { return answerProjectionWeight; }
    public Parameter getAnswerProjectionBias() { return answerProjectionBias; }
    public Parameter getReasoningProjectionWeight() { return reasoningProjectionWeight; }
    public Parameter getReasoningProjectionBias() { return reasoningProjectionBias; }
    public Parameter getConfidenceWeight() { return confidenceWeight; }
    public Parameter getConfidenceBias() { return confidenceBias; }
}