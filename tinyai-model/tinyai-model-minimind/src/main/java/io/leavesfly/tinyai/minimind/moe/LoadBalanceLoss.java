package io.leavesfly.tinyai.minimind.moe;

/**
 * Load Balance Loss - 负载均衡损失
 * 
 * 确保专家使用均衡,避免某些专家过载而其他专家闲置
 * 
 * 核心公式:
 * L_balance = α · importance_loss + β · load_loss
 * 
 * importance_loss = num_experts · Σ(importance_i · load_i)
 * load_loss = CV(load) = std(load) / mean(load)
 * 
 * 其中:
 * - importance_i: 专家i的重要性(所有样本的权重和)
 * - load_i: 专家i的负载(被选中的次数占比)
 * - CV: 变异系数(Coefficient of Variation)
 * 
 * @author leavesfly
 * @since 2024
 */
public class LoadBalanceLoss {
    
    private final float importanceCoef;  // 重要性损失系数
    private final float loadCoef;        // 负载损失系数
    
    /**
     * 构造函数
     * 
     * @param importanceCoef 重要性损失系数(默认0.01)
     * @param loadCoef 负载损失系数(默认0.01)
     */
    public LoadBalanceLoss(float importanceCoef, float loadCoef) {
        this.importanceCoef = importanceCoef;
        this.loadCoef = loadCoef;
    }
    
    /**
     * 默认构造函数
     */
    public LoadBalanceLoss() {
        this(0.01f, 0.01f);
    }
    
    /**
     * 计算负载均衡损失
     * 
     * @param stats 负载均衡统计
     * @param numExperts 专家数量
     * @return 负载均衡损失
     */
    public float computeLoss(MoELayer.LoadBalanceStats stats, int numExperts) {
        float[] importance = stats.getImportance();
        float[] load = stats.getLoad();
        
        // 1. Switch Transformer 标准的辅助损失
        // Auxiliary Loss = numExperts * sum(fraction_i * probability_i)
        // 其中 fraction_i 是分配给专家 i 的 token 比例
        // probability_i 是路由到专家 i 的平均概率
        float auxiliaryLoss = 0.0f;
        for (int i = 0; i < numExperts; i++) {
            auxiliaryLoss += load[i] * importance[i];
        }
        auxiliaryLoss *= numExperts;
        
        // 2. Load Loss: 变异系数CV(load)
        float loadLoss = coefficientOfVariation(load);
        
        // 3. 总损失：使用辅助损失替代原来的 importance loss
        float totalLoss = importanceCoef * auxiliaryLoss + loadCoef * loadLoss;
        
        return totalLoss;
    }
    
    /**
     * 计算变异系数 CV = std / mean
     */
    private float coefficientOfVariation(float[] values) {
        int n = values.length;
        if (n == 0) return 0.0f;
        
        // 计算均值
        float mean = 0.0f;
        for (float v : values) {
            mean += v;
        }
        mean /= n;
        
        if (mean == 0.0f) return 0.0f;
        
        // 计算标准差
        float variance = 0.0f;
        for (float v : values) {
            variance += (v - mean) * (v - mean);
        }
        variance /= n;
        float std = (float) Math.sqrt(variance);
        
        // 变异系数
        return std / mean;
    }
    
    /**
     * 获取重要性系数
     */
    public float getImportanceCoef() {
        return importanceCoef;
    }
    
    /**
     * 获取负载系数
     */
    public float getLoadCoef() {
        return loadCoef;
    }
    
    @Override
    public String toString() {
        return String.format("LoadBalanceLoss(importance_coef=%.4f, load_coef=%.4f)",
            importanceCoef, loadCoef);
    }
}