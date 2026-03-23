package io.leavesfly.tinyai.banana.training;

import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Banana训练器基类
 * 
 * 提供训练过程中常用的公共方法，包括梯度裁剪、参数计算和检查点目录创建
 * 
 * @author TinyAI
 * @since 2024
 */
public class BaseBananaTrainer {
    
    /**
     * 梯度裁剪
     * 
     * 防止梯度爆炸，将梯度范数裁剪到指定阈值内
     * 
     * @param model 模型
     * @param maxGradNorm 最大梯度范数阈值
     */
    protected static void clipGradients(BananaModel model, float maxGradNorm) {
        // 计算梯度范数
        double totalNorm = 0.0;
        
        for (var param : model.getAllParams().values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradData = grad.getArray();
                
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 如果超过阈值,进行裁剪
        if (totalNorm > maxGradNorm) {
            float clipCoef = maxGradNorm / (float) totalNorm;
            
            for (var param : model.getAllParams().values()) {
                if (param.getGrad() != null) {
                    NdArray grad = param.getGrad();
                    float[] gradData = grad.getArray();
                    
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= clipCoef;
                    }
                }
            }
        }
    }
    
    /**
     * 计算模型总参数量
     * 
     * @param model 模型
     * @return 总参数量
     */
    protected static long calculateTotalParams(BananaModel model) {
        long totalParams = 0;
        for (var param : model.getAllParams().values()) {
            int[] dims = param.getValue().getShape().getShapeDims();
            long size = 1;
            for (int d : dims) size *= d;
            totalParams += size;
        }
        return totalParams;
    }
    
    /**
     * 创建检查点目录
     * 
     * @param checkpointDir 检查点目录路径
     */
    protected static void createCheckpointDir(String checkpointDir) {
        try {
            Path path = Paths.get(checkpointDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
}
