package io.leavesfly.tinyai.ml.training;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.visual.ValidationUtils;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.Map;

/**
 * 梯度裁剪工具，用于防止梯度爆炸，通过限制梯度大小稳定训练过程。
 *
 * @author TinyAI
 * @version 1.0
 */
public class GradientClipper {
    
    /**
     * 裁剪类型
     */
    public enum ClipType {
        /**
         * L2范数裁剪：如果梯度的L2范数超过阈值，则按比例缩放
         */
        NORM,
        
        /**
         * 值裁剪：直接将梯度值裁剪到指定范围内
         */
        VALUE
    }
    
    private final ClipType clipType;
    private final float maxValue;
    
    /**
     * 构造函数
     * 
     * @param clipType 裁剪类型（NORM或VALUE）
     * @param maxValue 最大阈值（对于NORM是L2范数阈值，对于VALUE是绝对值上限）
     */
    public GradientClipper(ClipType clipType, float maxValue) {
        ValidationUtils.requireNonNull(clipType, "clipType");
        ValidationUtils.requirePositive(maxValue, "maxValue");
        
        this.clipType = clipType;
        this.maxValue = maxValue;
    }
    
    /**
     * 对模型的梯度进行裁剪
     * 
     * @param model 模型
     */
    public void clipGradients(Model model) {
        ValidationUtils.requireNonNull(model, "model");
        
        Map<String, Parameter> params = model.getAllParams();
        
        if (clipType == ClipType.NORM) {
            clipByNorm(params);
        } else {
            clipByValue(params);
        }
    }
    
    /**
     * L2范数裁剪
     * 计算所有参数梯度的总L2范数，如果超过阈值则按比例缩放
     * 
     * @param params 参数映射
     */
    private void clipByNorm(Map<String, Parameter> params) {
        // 计算总L2范数
        double totalNormSquared = 0.0;
        int paramCount = 0;
        
        for (Parameter param : params.values()) {
            if (param != null && param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradArray = grad.getArray();
                
                for (float value : gradArray) {
                    totalNormSquared += value * value;
                }
                paramCount++;
            }
        }
        
        if (paramCount == 0) {
            return;
        }
        
        double totalNorm = Math.sqrt(totalNormSquared);
        
        // 如果总范数超过阈值，按比例缩放
        if (totalNorm > maxValue) {
            float clipCoeff = (float) (maxValue / (totalNorm + 1e-6));
            
            for (Parameter param : params.values()) {
                if (param != null && param.getGrad() != null) {
                    NdArray grad = param.getGrad();
                    // 缩放梯度
                    grad.mulNum(clipCoeff);
                }
            }
        }
    }
    
    /**
     * 值裁剪
     * 直接将每个梯度值裁剪到[-maxValue, maxValue]范围内
     * 
     * @param params 参数映射
     */
    private void clipByValue(Map<String, Parameter> params) {
        for (Parameter param : params.values()) {
            if (param != null && param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradArray = grad.getArray();
                
                // 裁剪每个值
                for (int i = 0; i < gradArray.length; i++) {
                    if (gradArray[i] > maxValue) {
                        gradArray[i] = maxValue;
                    } else if (gradArray[i] < -maxValue) {
                        gradArray[i] = -maxValue;
                    }
                }
            }
        }
    }
    
    /**
     * 获取裁剪类型
     * 
     * @return 裁剪类型
     */
    public ClipType getClipType() {
        return clipType;
    }
    
    /**
     * 获取最大阈值
     * 
     * @return 最大阈值
     */
    public float getMaxValue() {
        return maxValue;
    }
}

