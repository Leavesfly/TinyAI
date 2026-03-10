package io.leavesfly.tinyai.nl.optimizer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nl.core.NestedOptimizationLevel;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 嵌套随机梯度下降优化器（NestedSGD）
 * 实现多层级的SGD优化
 * 
 * <p>每个嵌套层级使用独立的SGD更新，支持动量和权重衰减。
 * 不同层级可以有不同的学习率和更新频率。</p>
 * 
 * @author TinyAI Team
 */
public class NestedSGD extends DeepOptimizer {
    
    /**
     * 动量系数
     */
    private float momentum;
    
    /**
     * 权重衰减系数（L2正则化）
     */
    private float weightDecay;
    
    /**
     * 是否使用Nesterov动量
     */
    private boolean nesterov;
    
    /**
     * 动量缓存：使用 "levelIndex_paramIndex" 作为 key，避免 Variable 引用丢失
     */
    private Map<String, Variable> velocities;
    
    /**
     * 构造函数
     * 
     * @param globalLearningRate 全局学习率
     * @param momentum 动量系数
     * @param weightDecay 权重衰减系数
     * @param nesterov 是否使用Nesterov动量
     */
    public NestedSGD(float globalLearningRate, float momentum, float weightDecay, boolean nesterov) {
        super(globalLearningRate);
        this.momentum = Math.max(0.0f, Math.min(1.0f, momentum));
        this.weightDecay = Math.max(0.0f, weightDecay);
        this.nesterov = nesterov;
        this.velocities = new HashMap<>();
    }
    
    /**
     * 简化构造函数（无动量）
     * 
     * @param globalLearningRate 全局学习率
     */
    public NestedSGD(float globalLearningRate) {
        this(globalLearningRate, 0.0f, 0.0f, false);
    }
    
    /**
     * 带动量的构造函数
     * 
     * @param globalLearningRate 全局学习率
     * @param momentum 动量系数
     */
    public NestedSGD(float globalLearningRate, float momentum) {
        this(globalLearningRate, momentum, 0.0f, false);
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
            
            // 应用权重衰减（L2正则化）: grad = grad + weightDecay * param
            if (weightDecay > 0.0f) {
                grad = grad.add(param.mul(new Variable(weightDecay)));
            }
            
            Variable newParam;
            if (momentum > 0.0f) {
                // 使用动量更新
                newParam = applyMomentumUpdate(level.getLevelIndex(), i, param, grad, levelLearningRate);
            } else {
                // 无动量的标准 SGD: param = param - lr * grad
                newParam = param.sub(grad.mul(new Variable(levelLearningRate)));
            }
            
            parameters.set(i, newParam);
        }
    }
    
    /**
     * 使用动量进行参数更新
     * 标准动量公式: v = μ * v + lr * grad; param = param - v
     * Nesterov 动量: v = μ * v + lr * grad; param = param - (μ * v + lr * grad)
     * 
     * @param levelIndex 层级索引
     * @param paramIndex 参数索引
     * @param param 当前参数
     * @param grad 梯度
     * @param learningRate 学习率
     * @return 更新后的参数
     */
    private Variable applyMomentumUpdate(int levelIndex, int paramIndex, 
                                          Variable param, Variable grad, float learningRate) {
        // 使用 (levelIndex, paramIndex) 作为 key，避免 Variable 引用丢失
        String velocityKey = levelIndex + "_" + paramIndex;
        
        Variable velocity = velocities.get(velocityKey);
        if (velocity == null) {
            velocity = new Variable(NdArray.zeros(param.getValue().getShape()));
        }
        
        // v = μ * v + lr * grad
        Variable scaledGrad = grad.mul(new Variable(learningRate));
        velocity = velocity.mul(new Variable(momentum)).add(scaledGrad);
        
        // 保存速度
        velocities.put(velocityKey, velocity);
        
        if (nesterov) {
            // Nesterov: param = param - (μ * v + lr * grad)
            Variable nesterovUpdate = velocity.mul(new Variable(momentum)).add(scaledGrad);
            return param.sub(nesterovUpdate);
        } else {
            // 标准动量: param = param - v
            return param.sub(velocity);
        }
    }
    
    @Override
    public void reset() {
        super.reset();
        velocities.clear();
    }
    
    /**
     * 获取优化器配置信息
     * 
     * @return 配置字符串
     */
    public String getConfig() {
        StringBuilder sb = new StringBuilder();
        sb.append("NestedSGD配置:\n");
        sb.append(String.format("  全局学习率: %.6f\n", globalLearningRate));
        sb.append(String.format("  动量: %.4f\n", momentum));
        sb.append(String.format("  权重衰减: %.6f\n", weightDecay));
        sb.append(String.format("  Nesterov: %s\n", nesterov ? "是" : "否"));
        sb.append(String.format("  梯度裁剪: %s", enableGradientClipping ? 
            String.format("是 (阈值=%.2f)", gradientClipThreshold) : "否"));
        
        return sb.toString();
    }
    
    // Getters and Setters
    
    public float getMomentum() {
        return momentum;
    }
    
    public void setMomentum(float momentum) {
        this.momentum = Math.max(0.0f, Math.min(1.0f, momentum));
    }
    
    public float getWeightDecay() {
        return weightDecay;
    }
    
    public void setWeightDecay(float weightDecay) {
        this.weightDecay = Math.max(0.0f, weightDecay);
    }
    
    public boolean isNesterov() {
        return nesterov;
    }
    
    public void setNesterov(boolean nesterov) {
        this.nesterov = nesterov;
    }
    
    public Map<String, Variable> getVelocities() {
        return new HashMap<>(velocities);
    }
}
