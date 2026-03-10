package io.leavesfly.tinyai.nl.optimizer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nl.core.NestedOptimizationLevel;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 嵌套Adam优化器（NestedAdam）
 * 实现多层级的Adam优化
 * 
 * <p>Adam (Adaptive Moment Estimation) 结合了动量和自适应学习率。
 * 该实现支持多个嵌套层级，每个层级维护独立的一阶和二阶动量估计。</p>
 * 
 * @author TinyAI Team
 */
public class NestedAdam extends DeepOptimizer {
    
    /**
     * 一阶动量衰减率（beta1）
     */
    private float beta1;
    
    /**
     * 二阶动量衰减率（beta2）
     */
    private float beta2;
    
    /**
     * 数值稳定性常数
     */
    private float epsilon;
    
    /**
     * 权重衰减系数
     */
    private float weightDecay;
    
    /**
     * 是否使用AMSGrad变体
     */
    private boolean amsgrad;
    
    /**
     * 一阶动量缓存（m_t），使用 "levelIndex_paramIndex" 作为 key
     */
    private Map<String, Variable> firstMoments;
    
    /**
     * 二阶动量缓存（v_t），使用 "levelIndex_paramIndex" 作为 key
     */
    private Map<String, Variable> secondMoments;
    
    /**
     * AMSGrad的最大二阶动量缓存
     */
    private Map<String, Variable> maxSecondMoments;
    
    /**
     * 每个参数的时间步
     */
    private Map<String, Integer> timeSteps;
    
    /**
     * 构造函数
     * 
     * @param globalLearningRate 全局学习率
     * @param beta1 一阶动量衰减率
     * @param beta2 二阶动量衰减率
     * @param epsilon 数值稳定性常数
     * @param weightDecay 权重衰减系数
     * @param amsgrad 是否使用AMSGrad
     */
    public NestedAdam(float globalLearningRate, float beta1, float beta2, 
                      float epsilon, float weightDecay, boolean amsgrad) {
        super(globalLearningRate);
        this.beta1 = Math.max(0.0f, Math.min(1.0f, beta1));
        this.beta2 = Math.max(0.0f, Math.min(1.0f, beta2));
        this.epsilon = Math.max(1e-8f, epsilon);
        this.weightDecay = Math.max(0.0f, weightDecay);
        this.amsgrad = amsgrad;
        
        this.firstMoments = new HashMap<>();
        this.secondMoments = new HashMap<>();
        this.maxSecondMoments = new HashMap<>();
        this.timeSteps = new HashMap<>();
    }
    
    /**
     * 标准构造函数（使用默认参数）
     * 
     * @param globalLearningRate 全局学习率
     */
    public NestedAdam(float globalLearningRate) {
        this(globalLearningRate, 0.9f, 0.999f, 1e-8f, 0.0f, false);
    }
    
    /**
     * 带权重衰减的构造函数
     * 
     * @param globalLearningRate 全局学习率
     * @param weightDecay 权重衰减系数
     */
    public NestedAdam(float globalLearningRate, float weightDecay) {
        this(globalLearningRate, 0.9f, 0.999f, 1e-8f, weightDecay, false);
    }
    
    @Override
    protected void updateLevel(NestedOptimizationLevel level, List<Variable> gradients) {
        if (level == null || gradients == null || gradients.isEmpty()) {
            return;
        }
        
        List<Variable> parameters = level.getParameters();
        float levelLearningRate = level.getLearningRate();
        
        if (levelLearningRate == 0.0f) {
            levelLearningRate = globalLearningRate;
        }
        
        int count = Math.min(parameters.size(), gradients.size());
        for (int i = 0; i < count; i++) {
            Variable param = parameters.get(i);
            Variable grad = gradients.get(i);
            
            if (param == null || grad == null) {
                continue;
            }
            
            // 应用权重衰减: grad = grad + weightDecay * param
            if (weightDecay > 0.0f) {
                grad = grad.add(param.mul(new Variable(weightDecay)));
            }
            
            // 使用 (levelIndex, paramIndex) 作为 key 执行 Adam 更新
            Variable newParam = adamUpdate(level.getLevelIndex(), i, param, grad, levelLearningRate);
            parameters.set(i, newParam);
        }
    }
    
    /**
     * 执行 Adam 更新
     * 使用 (levelIndex, paramIndex) 作为动量状态的 key，避免参数引用丢失
     * 
     * @param levelIndex 层级索引
     * @param paramIndex 参数索引
     * @param param 参数
     * @param grad 梯度
     * @param learningRate 学习率
     * @return 更新后的参数
     */
    private Variable adamUpdate(int levelIndex, int paramIndex, 
                                 Variable param, Variable grad, float learningRate) {
        String stateKey = levelIndex + "_" + paramIndex;
        
        // 获取或初始化时间步
        int t = timeSteps.getOrDefault(stateKey, 0) + 1;
        timeSteps.put(stateKey, t);
        
        // 获取或初始化一阶动量
        Variable m = firstMoments.get(stateKey);
        if (m == null) {
            m = new Variable(NdArray.zeros(param.getValue().getShape()));
        }
        
        // 获取或初始化二阶动量
        Variable v = secondMoments.get(stateKey);
        if (v == null) {
            v = new Variable(NdArray.zeros(param.getValue().getShape()));
        }
        
        // 更新一阶动量：m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        m = m.mul(new Variable(beta1)).add(grad.mul(new Variable(1.0f - beta1)));
        
        // 更新二阶动量：v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        Variable gradSquared = grad.mul(grad);
        v = v.mul(new Variable(beta2)).add(gradSquared.mul(new Variable(1.0f - beta2)));
        
        // 保存更新后的动量
        firstMoments.put(stateKey, m);
        secondMoments.put(stateKey, v);
        
        // 偏差修正
        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, t);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, t);
        
        Variable mHat = m.mul(new Variable(1.0f / biasCorrection1));
        Variable vHat = v.mul(new Variable(1.0f / biasCorrection2));
        
        // AMSGrad 变体
        if (amsgrad) {
            Variable vMax = maxSecondMoments.get(stateKey);
            if (vMax == null) {
                vMax = vHat;
            } else {
                vMax = elementwiseMax(vMax, vHat);
            }
            maxSecondMoments.put(stateKey, vMax);
            vHat = vMax;
        }
        
        // 计算更新：delta = learningRate * mHat / (sqrt(vHat) + epsilon)
        Variable sqrtV = sqrt(vHat);
        Variable denominator = sqrtV.add(new Variable(epsilon));
        Variable update = mHat.mul(new Variable(learningRate)).div(denominator);
        
        // param = param - update
        return param.sub(update);
    }
    
    /**
     * 计算平方根（元素级）
     * 
     * @param v 输入变量
     * @return 平方根
     */
    private Variable sqrt(Variable v) {
        if (v == null) {
            return null;
        }
        
        NdArray data = v.getValue();
        int[] shape = data.getShape().getShapeDims();
        int totalSize = data.getShape().size();
        float[] result = new float[totalSize];
        
        NdArray flat = data.flatten();
        int ndim = flat.getShape().getDimNum();
        
        for (int i = 0; i < totalSize; i++) {
            int[] idx = (ndim == 1) ? new int[]{i} : new int[]{0, i};
            float val = flat.get(idx);
            result[i] = (float) Math.sqrt(Math.max(0.0f, val));
        }
        
        NdArray sqrtData = NdArray.of(result, io.leavesfly.tinyai.ndarr.Shape.of(shape));
        return new Variable(sqrtData);
    }
    
    /**
     * 元素级最大值
     * 
     * @param v1 变量1
     * @param v2 变量2
     * @return 元素级最大值
     */
    private Variable elementwiseMax(Variable v1, Variable v2) {
        if (v1 == null) return v2;
        if (v2 == null) return v1;
        
        NdArray data1 = v1.getValue();
        NdArray data2 = v2.getValue();
        int[] shape = data1.getShape().getShapeDims();
        int totalSize = data1.getShape().size();
        float[] result = new float[totalSize];
        
        NdArray flat1 = data1.flatten();
        NdArray flat2 = data2.flatten();
        int ndim1 = flat1.getShape().getDimNum();
        int ndim2 = flat2.getShape().getDimNum();
        
        for (int i = 0; i < totalSize; i++) {
            int[] idx1 = (ndim1 == 1) ? new int[]{i} : new int[]{0, i};
            int[] idx2 = (ndim2 == 1) ? new int[]{i} : new int[]{0, i};
            result[i] = Math.max(flat1.get(idx1), flat2.get(idx2));
        }
        
        NdArray maxData = NdArray.of(result, io.leavesfly.tinyai.ndarr.Shape.of(shape));
        return new Variable(maxData);
    }
    
    @Override
    public void reset() {
        super.reset();
        firstMoments.clear();
        secondMoments.clear();
        maxSecondMoments.clear();
        timeSteps.clear();
    }
    
    /**
     * 获取优化器配置信息
     * 
     * @return 配置字符串
     */
    public String getConfig() {
        StringBuilder sb = new StringBuilder();
        sb.append("NestedAdam配置:\n");
        sb.append(String.format("  全局学习率: %.6f\n", globalLearningRate));
        sb.append(String.format("  Beta1: %.4f\n", beta1));
        sb.append(String.format("  Beta2: %.4f\n", beta2));
        sb.append(String.format("  Epsilon: %.2e\n", epsilon));
        sb.append(String.format("  权重衰减: %.6f\n", weightDecay));
        sb.append(String.format("  AMSGrad: %s\n", amsgrad ? "是" : "否"));
        sb.append(String.format("  梯度裁剪: %s", enableGradientClipping ? 
            String.format("是 (阈值=%.2f)", gradientClipThreshold) : "否"));
        
        return sb.toString();
    }
    
    // Getters and Setters
    
    public float getBeta1() {
        return beta1;
    }
    
    public void setBeta1(float beta1) {
        this.beta1 = Math.max(0.0f, Math.min(1.0f, beta1));
    }
    
    public float getBeta2() {
        return beta2;
    }
    
    public void setBeta2(float beta2) {
        this.beta2 = Math.max(0.0f, Math.min(1.0f, beta2));
    }
    
    public float getEpsilon() {
        return epsilon;
    }
    
    public void setEpsilon(float epsilon) {
        this.epsilon = Math.max(1e-8f, epsilon);
    }
    
    public float getWeightDecay() {
        return weightDecay;
    }
    
    public void setWeightDecay(float weightDecay) {
        this.weightDecay = Math.max(0.0f, weightDecay);
    }
    
    public boolean isAmsgrad() {
        return amsgrad;
    }
    
    public void setAmsgrad(boolean amsgrad) {
        this.amsgrad = amsgrad;
    }
}
