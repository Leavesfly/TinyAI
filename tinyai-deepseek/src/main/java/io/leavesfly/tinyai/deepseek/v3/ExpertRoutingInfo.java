package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.ndarr.NdArray;
import java.util.List;

/**
 * 专家路由信息
 * 
 * 包含MoE层的专家选择和路由相关信息
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ExpertRoutingInfo {
    
    /**
     * 专家权重分布
     */
    private final NdArray expertWeights;
    
    /**
     * 选中的专家索引列表
     */
    private final List<Integer> selectedExperts;
    
    /**
     * 路由损失
     */
    private final float routingLoss;
    
    /**
     * 负载均衡损失
     */
    private final float loadBalanceLoss;
    
    /**
     * 构造函数
     * 
     * @param expertWeights 专家权重
     * @param selectedExperts 选中的专家列表
     * @param routingLoss 路由损失
     * @param loadBalanceLoss 负载均衡损失
     */
    public ExpertRoutingInfo(NdArray expertWeights, List<Integer> selectedExperts, 
                            float routingLoss, float loadBalanceLoss) {
        this.expertWeights = expertWeights;
        this.selectedExperts = selectedExperts;
        this.routingLoss = routingLoss;
        this.loadBalanceLoss = loadBalanceLoss;
    }
    
    // Getters
    public NdArray getExpertWeights() {
        return expertWeights;
    }
    
    public List<Integer> getSelectedExperts() {
        return selectedExperts;
    }
    
    public float getRoutingLoss() {
        return routingLoss;
    }
    
    public float getLoadBalanceLoss() {
        return loadBalanceLoss;
    }
    
    /**
     * 获取总的MoE损失
     */
    public float getTotalMoELoss() {
        return routingLoss + loadBalanceLoss;
    }
    
    @Override
    public String toString() {
        return String.format("ExpertRoutingInfo{routingLoss=%.4f, loadBalanceLoss=%.4f, selectedExperts=%s}", 
                           routingLoss, loadBalanceLoss, selectedExperts);
    }
}