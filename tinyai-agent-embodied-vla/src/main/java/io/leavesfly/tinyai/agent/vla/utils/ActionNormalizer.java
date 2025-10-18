package io.leavesfly.tinyai.agent.vla.utils;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 动作归一化工具
 * 用于动作空间的归一化和反归一化
 * 
 * @author TinyAI
 */
public class ActionNormalizer {
    
    private final double[] actionLow;
    private final double[] actionHigh;
    private final int actionDim;
    
    /**
     * 构造函数
     * 
     * @param actionLow 动作空间下界
     * @param actionHigh 动作空间上界
     */
    public ActionNormalizer(double[] actionLow, double[] actionHigh) {
        if (actionLow.length != actionHigh.length) {
            throw new IllegalArgumentException("Action bounds must have same dimension");
        }
        
        this.actionLow = actionLow;
        this.actionHigh = actionHigh;
        this.actionDim = actionLow.length;
    }
    
    /**
     * 归一化动作到 [-1, 1] 范围
     * 
     * @param action 原始动作
     * @return 归一化后的动作
     */
    public NdArray normalize(NdArray action) {
        float[] normalized = new float[actionDim];
        
        for (int i = 0; i < actionDim; i++) {
            double val = action.get(i);
            normalized[i] = (float) (2.0 * (val - actionLow[i]) / (actionHigh[i] - actionLow[i]) - 1.0);
            
            // Clamp到 [-1, 1]
            normalized[i] = (float) Math.max(-1.0, Math.min(1.0, normalized[i]));
        }
        
        return  NdArray.of(normalized);
    }
    
    /**
     * 反归一化动作到真实动作空间
     * 
     * @param normalizedAction 归一化的动作（范围 [-1, 1]）
     * @return 真实动作
     */
    public NdArray denormalize(NdArray normalizedAction) {
        double[] denormalized = new double[actionDim];
        
        for (int i = 0; i < actionDim; i++) {
            double val = normalizedAction.get(i);
            denormalized[i] = actionLow[i] + (val + 1.0) / 2.0 * (actionHigh[i] - actionLow[i]);
            
            // Clamp到动作空间范围
            denormalized[i] = Math.max(actionLow[i], Math.min(actionHigh[i], denormalized[i]));
        }
        
        return  NdArray.of(denormalized);
    }
    
    /**
     * 获取动作维度
     */
    public int getActionDim() {
        return actionDim;
    }
    
    /**
     * 获取动作下界
     */
    public double[] getActionLow() {
        return actionLow;
    }
    
    /**
     * 获取动作上界
     */
    public double[] getActionHigh() {
        return actionHigh;
    }
}
