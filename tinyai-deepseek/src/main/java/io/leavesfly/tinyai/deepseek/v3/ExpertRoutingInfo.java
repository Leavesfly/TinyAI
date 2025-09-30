package io.leavesfly.tinyai.deepseek.v3;

import java.util.List;
import java.util.Map;

/**
 * 专家路由信息
 * 
 * 包含专家混合模型(MoE)的路由决策信息，
 * 用于记录哪些专家被选中以及相关的负载均衡统计。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ExpertRoutingInfo {
    
    /**
     * 专家权重数组
     * 每个专家的激活权重，决定其对最终输出的贡献度
     */
    private double[] expertWeights;
    
    /**
     * 被选中的专家索引列表
     * 记录哪些专家被选中参与当前计算
     */
    private List<Integer> selectedExperts;
    
    /**
     * 路由损失
     * 衡量路由决策的质量和不确定性
     */
    private double routingLoss;
    
    /**
     * 负载均衡损失
     * 衡量专家使用的均衡程度，防止少数专家过载
     */
    private double loadBalanceLoss;
    
    /**
     * 专家使用计数
     * 记录每个专家在当前批次中被使用的次数
     */
    private long[] expertUsageCount;
    
    /**
     * 任务类型偏置信息
     * 记录针对不同任务类型的专家选择偏置
     */
    private Map<TaskType, Double> taskTypeBias;
    
    /**
     * 构造函数
     * 
     * @param expertWeights 专家权重数组
     * @param selectedExperts 选中的专家索引列表
     * @param routingLoss 路由损失
     * @param loadBalanceLoss 负载均衡损失
     */
    public ExpertRoutingInfo(double[] expertWeights, List<Integer> selectedExperts,
                           double routingLoss, double loadBalanceLoss) {
        this.expertWeights = expertWeights;
        this.selectedExperts = selectedExperts;
        this.routingLoss = routingLoss;
        this.loadBalanceLoss = loadBalanceLoss;
    }
    
    /**
     * 完整构造函数
     * 
     * @param expertWeights 专家权重数组
     * @param selectedExperts 选中的专家索引列表
     * @param routingLoss 路由损失
     * @param loadBalanceLoss 负载均衡损失
     * @param expertUsageCount 专家使用计数
     * @param taskTypeBias 任务类型偏置
     */
    public ExpertRoutingInfo(double[] expertWeights, List<Integer> selectedExperts,
                           double routingLoss, double loadBalanceLoss,
                           long[] expertUsageCount, Map<TaskType, Double> taskTypeBias) {
        this.expertWeights = expertWeights;
        this.selectedExperts = selectedExperts;
        this.routingLoss = routingLoss;
        this.loadBalanceLoss = loadBalanceLoss;
        this.expertUsageCount = expertUsageCount;
        this.taskTypeBias = taskTypeBias;
    }
    
    /**
     * 计算专家选择的熵
     * 衡量专家选择的多样性
     * 
     * @return 专家选择熵值
     */
    public double calculateExpertEntropy() {
        if (expertWeights == null || expertWeights.length == 0) {
            return 0.0;
        }
        
        double entropy = 0.0;
        for (double weight : expertWeights) {
            if (weight > 0) {
                entropy -= weight * Math.log(weight);
            }
        }
        return entropy;
    }
    
    /**
     * 获取最大权重专家的索引
     * 
     * @return 权重最大的专家索引
     */
    public int getDominantExpertIndex() {
        if (expertWeights == null || expertWeights.length == 0) {
            return -1;
        }
        
        int maxIndex = 0;
        double maxWeight = expertWeights[0];
        for (int i = 1; i < expertWeights.length; i++) {
            if (expertWeights[i] > maxWeight) {
                maxWeight = expertWeights[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    /**
     * 计算负载分布的标准差
     * 衡量专家使用的不均衡程度
     * 
     * @return 负载分布标准差
     */
    public double calculateLoadStandardDeviation() {
        if (expertUsageCount == null || expertUsageCount.length == 0) {
            return 0.0;
        }
        
        // 计算平均值
        double mean = 0.0;
        for (long count : expertUsageCount) {
            mean += count;
        }
        mean /= expertUsageCount.length;
        
        // 计算方差
        double variance = 0.0;
        for (long count : expertUsageCount) {
            variance += Math.pow(count - mean, 2);
        }
        variance /= expertUsageCount.length;
        
        return Math.sqrt(variance);
    }
    
    /**
     * 生成路由信息摘要
     * 
     * @return 路由信息的文字描述
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("专家路由信息摘要:\n");
        sb.append(String.format("  - 选中专家数量: %d\n", selectedExperts != null ? selectedExperts.size() : 0));
        sb.append(String.format("  - 路由损失: %.6f\n", routingLoss));
        sb.append(String.format("  - 负载均衡损失: %.6f\n", loadBalanceLoss));
        sb.append(String.format("  - 专家选择熵: %.6f\n", calculateExpertEntropy()));
        sb.append(String.format("  - 主导专家索引: %d\n", getDominantExpertIndex()));
        
        if (expertUsageCount != null) {
            sb.append("  - 专家使用分布: [");
            for (int i = 0; i < expertUsageCount.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(expertUsageCount[i]);
            }
            sb.append("]\n");
            sb.append(String.format("  - 负载标准差: %.2f\n", calculateLoadStandardDeviation()));
        }
        
        return sb.toString();
    }
    
    // ========== Getter and Setter Methods ==========
    
    public double[] getExpertWeights() {
        return expertWeights;
    }
    
    public void setExpertWeights(double[] expertWeights) {
        this.expertWeights = expertWeights;
    }
    
    public List<Integer> getSelectedExperts() {
        return selectedExperts;
    }
    
    public void setSelectedExperts(List<Integer> selectedExperts) {
        this.selectedExperts = selectedExperts;
    }
    
    public double getRoutingLoss() {
        return routingLoss;
    }
    
    public void setRoutingLoss(double routingLoss) {
        this.routingLoss = routingLoss;
    }
    
    public double getLoadBalanceLoss() {
        return loadBalanceLoss;
    }
    
    public void setLoadBalanceLoss(double loadBalanceLoss) {
        this.loadBalanceLoss = loadBalanceLoss;
    }
    
    public long[] getExpertUsageCount() {
        return expertUsageCount;
    }
    
    public void setExpertUsageCount(long[] expertUsageCount) {
        this.expertUsageCount = expertUsageCount;
    }
    
    public Map<TaskType, Double> getTaskTypeBias() {
        return taskTypeBias;
    }
    
    public void setTaskTypeBias(Map<TaskType, Double> taskTypeBias) {
        this.taskTypeBias = taskTypeBias;
    }
}