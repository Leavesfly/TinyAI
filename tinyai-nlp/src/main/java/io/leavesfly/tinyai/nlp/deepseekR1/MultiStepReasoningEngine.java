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
import java.util.Map;
import java.util.HashMap;

/**
 * 多步推理引擎
 * 
 * 这个引擎实现了DeepSeek-R1的多步推理能力，
 * 能够自适应地确定推理深度，动态调整推理策略，
 * 并为每个推理步骤提供详细的中间结果。
 * 
 * 多步推理引擎的核心功能：
 * 1. 自适应推理深度：根据问题复杂度动态确定推理步数
 * 2. 推理路径规划：为复杂问题规划最优推理路径
 * 3. 中间结果缓存：保存和管理推理中间结果
 * 4. 推理质量评估：评估每步推理的质量和可信度
 * 5. 推理策略调整：根据中间结果调整后续推理策略
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MultiStepReasoningEngine extends Layer {
    
    // 推理引擎参数
    private int dModel;                     // 模型维度
    private int maxReasoningSteps;          // 最大推理步骤数
    private double reasoningThreshold;      // 推理置信度阈值
    private int numReasoningHeads;          // 推理注意力头数量
    private double adaptiveThreshold;       // 自适应调整阈值
    
    // 推理策略参数
    private Parameter depthPredictorWeight;     // 深度预测器权重
    private Parameter depthPredictorBias;       // 深度预测器偏置
    private Parameter qualityEvaluatorWeight;   // 质量评估器权重
    private Parameter qualityEvaluatorBias;     // 质量评估器偏置
    private Parameter strategyAdjusterWeight;   // 策略调整器权重
    private Parameter strategyAdjusterBias;     // 策略调整器偏置
    private Parameter pathPlannerWeight;        // 路径规划器权重
    private Parameter pathPlannerBias;          // 路径规划器偏置
    
    // 层归一化组件
    private LayerNorm preReasoningNorm;         // 推理前归一化
    private LayerNorm postReasoningNorm;        // 推理后归一化
    private LayerNorm intermediateNorm;         // 中间步骤归一化
    
    // 推理状态管理
    private List<Variable> reasoningPath;       // 推理路径
    private List<Double> stepQualities;         // 每步质量分数
    private Map<String, Object> reasoningCache; // 推理缓存
    private int currentStep;                    // 当前推理步骤
    private double currentConfidence;           // 当前置信度
    
    /**
     * 构造多步推理引擎
     * 
     * @param name 引擎名称
     * @param dModel 模型维度
     * @param maxReasoningSteps 最大推理步骤数
     * @param reasoningThreshold 推理置信度阈值
     */
    public MultiStepReasoningEngine(String name, int dModel, int maxReasoningSteps, double reasoningThreshold) {
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
        this.numReasoningHeads = Math.max(4, dModel / 64);
        this.adaptiveThreshold = 0.1;
        
        // 初始化推理状态
        this.reasoningPath = new ArrayList<>();
        this.stepQualities = new ArrayList<>();
        this.reasoningCache = new HashMap<>();
        this.currentStep = 0;
        this.currentConfidence = 0.0;
        
        init();
    }
    
    /**
     * 简化构造函数
     * 
     * @param name 引擎名称
     * @param dModel 模型维度
     * @param maxReasoningSteps 最大推理步骤数
     */
    public MultiStepReasoningEngine(String name, int dModel, int maxReasoningSteps) {
        this(name, dModel, maxReasoningSteps, 0.7);
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
            // 1. 初始化深度预测器
            depthPredictorWeight = new Parameter(
                initializeWeights(Shape.of(dModel, 1))
            );
            depthPredictorBias = new Parameter(
                NdArray.zeros(Shape.of(1))
            );
            params.put("depthPredictorWeight", depthPredictorWeight);
            params.put("depthPredictorBias", depthPredictorBias);
            
            // 2. 初始化质量评估器
            qualityEvaluatorWeight = new Parameter(
                initializeWeights(Shape.of(dModel, 1))
            );
            qualityEvaluatorBias = new Parameter(
                NdArray.zeros(Shape.of(1))
            );
            params.put("qualityEvaluatorWeight", qualityEvaluatorWeight);
            params.put("qualityEvaluatorBias", qualityEvaluatorBias);
            
            // 3. 初始化策略调整器
            strategyAdjusterWeight = new Parameter(
                initializeWeights(Shape.of(dModel, dModel))
            );
            strategyAdjusterBias = new Parameter(
                NdArray.zeros(Shape.of(dModel))
            );
            params.put("strategyAdjusterWeight", strategyAdjusterWeight);
            params.put("strategyAdjusterBias", strategyAdjusterBias);
            
            // 4. 初始化路径规划器
            pathPlannerWeight = new Parameter(
                initializeWeights(Shape.of(dModel, dModel))
            );
            pathPlannerBias = new Parameter(
                NdArray.zeros(Shape.of(dModel))
            );
            params.put("pathPlannerWeight", pathPlannerWeight);
            params.put("pathPlannerBias", pathPlannerBias);
            
            // 5. 初始化层归一化
            preReasoningNorm = new LayerNorm(name + "_pre_reasoning_norm", dModel);
            postReasoningNorm = new LayerNorm(name + "_post_reasoning_norm", dModel);
            intermediateNorm = new LayerNorm(name + "_intermediate_norm", dModel);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len, dModel)
        
        // 重置推理状态
        resetReasoningState();
        
        // 1. 推理前归一化
        Variable normalized = preReasoningNorm.layerForward(input);
        
        // 2. 预测推理深度
        int predictedDepth = predictReasoningDepth(normalized);
        
        // 3. 规划推理路径
        Variable pathPlan = planReasoningPath(normalized, predictedDepth);
        
        // 4. 执行多步推理
        Variable reasoningResult = executeMultiStepReasoning(pathPlan, predictedDepth);
        
        // 5. 推理后归一化
        Variable output = postReasoningNorm.layerForward(reasoningResult);
        
        // 6. 残差连接
        output = addVariables(input, output);
        
        return output;
    }
    
    /**
     * 预测推理深度
     * 根据输入复杂度预测需要的推理步骤数
     * 
     * @param input 输入表示
     * @return 预测的推理深度
     */
    private int predictReasoningDepth(Variable input) {
        // 使用深度预测器计算推理深度
        Variable weights = new Variable(depthPredictorWeight.getValue());
        Variable bias = new Variable(depthPredictorBias.getValue());
        
        MatMul matMul = new MatMul();
        NdArray depthScore = matMul.forward(input.getValue(), weights.getValue());
        Variable depthVar = new Variable(depthScore);
        depthVar = addBias(depthVar, bias);
        
        // 计算平均深度分数
        NdArray depthArray = depthVar.getValue();
        double avgDepth = calculateMean(depthArray);
        
        // 使用sigmoid将分数映射到[0,1]，然后缩放到[1, maxReasoningSteps]
        double normalizedDepth = 1.0 / (1.0 + Math.exp(-avgDepth));
        int predictedDepth = (int) Math.ceil(normalizedDepth * maxReasoningSteps);
        
        // 确保深度在合理范围内
        predictedDepth = Math.max(1, Math.min(predictedDepth, maxReasoningSteps));
        
        return predictedDepth;
    }
    
    /**
     * 规划推理路径
     * 为多步推理制定路径规划
     * 
     * @param input 输入表示
     * @param depth 推理深度
     * @return 路径规划结果
     */
    private Variable planReasoningPath(Variable input, int depth) {
        Variable weights = new Variable(pathPlannerWeight.getValue());
        Variable bias = new Variable(pathPlannerBias.getValue());
        
        MatMul matMul = new MatMul();
        NdArray pathOutput = matMul.forward(input.getValue(), weights.getValue());
        Variable pathVar = new Variable(pathOutput);
        pathVar = addBias(pathVar, bias);
        
        // 使用Tanh激活函数
        Tanh tanh = new Tanh();
        NdArray tanhOutput = tanh.forward(pathVar.getValue());
        pathVar = new Variable(tanhOutput);
        
        return pathVar;
    }
    
    /**
     * 执行多步推理
     * 
     * @param pathPlan 路径规划
     * @param depth 推理深度
     * @return 推理结果
     */
    private Variable executeMultiStepReasoning(Variable pathPlan, int depth) {
        Variable currentState = pathPlan;
        reasoningPath.add(currentState);
        
        for (int step = 0; step < depth; step++) {
            currentStep = step;
            
            // 执行单步推理
            Variable stepResult = performSingleStep(currentState, step);
            
            // 中间归一化
            stepResult = intermediateNorm.layerForward(stepResult);
            
            // 评估步骤质量
            double quality = evaluateStepQuality(stepResult);
            stepQualities.add(quality);
            
            // 更新置信度
            currentConfidence = Math.max(currentConfidence, quality);
            
            // 缓存中间结果
            reasoningCache.put("step_" + step, stepResult);
            reasoningPath.add(stepResult);
            
            // 检查是否满足早停条件
            if (quality >= reasoningThreshold && step >= 1) {
                break;
            }
            
            // 自适应调整策略
            currentState = adjustReasoningStrategy(stepResult, quality, step);
        }
        
        // 融合所有推理步骤
        return fuseReasoningSteps();
    }
    
    /**
     * 执行单步推理
     * 
     * @param state 当前状态
     * @param step 步骤索引
     * @return 单步推理结果
     */
    private Variable performSingleStep(Variable state, int step) {
        // 使用当前状态进行推理变换
        Variable weights = new Variable(strategyAdjusterWeight.getValue());
        Variable bias = new Variable(strategyAdjusterBias.getValue());
        
        MatMul matMul = new MatMul();
        NdArray stepOutput = matMul.forward(state.getValue(), weights.getValue());
        Variable stepVar = new Variable(stepOutput);
        stepVar = addBias(stepVar, bias);
        
        // 激活函数
        Tanh tanh = new Tanh();
        NdArray tanhOutput = tanh.forward(stepVar.getValue());
        stepVar = new Variable(tanhOutput);
        
        return stepVar;
    }
    
    /**
     * 评估步骤质量
     * 
     * @param stepResult 步骤结果
     * @return 质量分数
     */
    private double evaluateStepQuality(Variable stepResult) {
        Variable weights = new Variable(qualityEvaluatorWeight.getValue());
        Variable bias = new Variable(qualityEvaluatorBias.getValue());
        
        MatMul matMul = new MatMul();
        NdArray qualityOutput = matMul.forward(stepResult.getValue(), weights.getValue());
        Variable qualityVar = new Variable(qualityOutput);
        qualityVar = addBias(qualityVar, bias);
        
        // 计算平均质量分数
        double avgQuality = calculateMean(qualityVar.getValue());
        
        // 使用sigmoid映射到[0,1]
        return 1.0 / (1.0 + Math.exp(-avgQuality));
    }
    
    /**
     * 调整推理策略
     * 
     * @param stepResult 当前步骤结果
     * @param quality 当前质量
     * @param step 当前步骤
     * @return 调整后的状态
     */
    private Variable adjustReasoningStrategy(Variable stepResult, double quality, int step) {
        // 如果质量较低，尝试不同的推理策略
        if (quality < adaptiveThreshold) {
            // 策略调整：增加探索性
            return addExplorationNoise(stepResult);
        } else {
            // 策略保持：继续当前方向
            return stepResult;
        }
    }
    
    /**
     * 添加探索性噪声
     * 
     * @param state 当前状态
     * @return 添加噪声后的状态
     */
    private Variable addExplorationNoise(Variable state) {
        NdArray stateArray = state.getValue();
        NdArray noiseArray = NdArray.of(stateArray.getShape());
        
        // 添加小幅度的高斯噪声
        for (int i = 0; i < noiseArray.getShape().size(); i++) {
            double noise = (Math.random() - 0.5) * 0.01; // 1% 噪声
            float newValue = stateArray.get(i) + (float) noise;
            noiseArray.set(newValue, i);
        }
        
        return new Variable(noiseArray);
    }
    
    /**
     * 融合推理步骤
     * 
     * @return 融合结果
     */
    private Variable fuseReasoningSteps() {
        if (reasoningPath.isEmpty()) {
            throw new IllegalStateException("没有推理步骤可以融合");
        }
        
        if (reasoningPath.size() == 1) {
            return reasoningPath.get(0);
        }
        
        // 加权融合：后面的步骤权重更高
        Variable fusedResult = null;
        double totalWeight = 0.0;
        
        for (int i = 0; i < reasoningPath.size(); i++) {
            Variable step = reasoningPath.get(i);
            double weight = (i + 1.0) / reasoningPath.size(); // 线性增长权重
            
            if (i < stepQualities.size()) {
                weight *= stepQualities.get(i); // 质量加权
            }
            
            if (fusedResult == null) {
                fusedResult = multiplyByScalar(step, weight);
            } else {
                Variable weightedStep = multiplyByScalar(step, weight);
                fusedResult = addVariables(fusedResult, weightedStep);
            }
            
            totalWeight += weight;
        }
        
        // 归一化
        if (totalWeight > 0.0) {
            fusedResult = multiplyByScalar(fusedResult, 1.0 / totalWeight);
        }
        
        return fusedResult;
    }
    
    /**
     * 重置推理状态
     */
    private void resetReasoningState() {
        reasoningPath.clear();
        stepQualities.clear();
        reasoningCache.clear();
        currentStep = 0;
        currentConfidence = 0.0;
    }
    
    /**
     * 计算数组的平均值
     * 
     * @param array 输入数组
     * @return 平均值
     */
    private double calculateMean(NdArray array) {
        double sum = 0.0;
        int size = array.getShape().size();
        
        for (int i = 0; i < size; i++) {
            sum += array.get(i);
        }
        
        return sum / size;
    }
    
    /**
     * 添加偏置
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
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getMaxReasoningSteps() { return maxReasoningSteps; }
    public double getReasoningThreshold() { return reasoningThreshold; }
    public int getCurrentStep() { return currentStep; }
    public double getCurrentConfidence() { return currentConfidence; }
    public List<Variable> getReasoningPath() { return new ArrayList<>(reasoningPath); }
    public List<Double> getStepQualities() { return new ArrayList<>(stepQualities); }
    public Map<String, Object> getReasoningCache() { return new HashMap<>(reasoningCache); }
}