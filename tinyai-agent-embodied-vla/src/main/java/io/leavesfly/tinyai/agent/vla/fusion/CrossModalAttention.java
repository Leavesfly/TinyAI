package io.leavesfly.tinyai.agent.vla.fusion;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 跨模态注意力机制
 * 实现视觉、语言、本体感知三种模态的交叉注意力
 * 
 * @author TinyAI
 */
public class CrossModalAttention extends Block {
    
    private final int hiddenDim;
    private final int numHeads;
    private final int headDim;
    
    // Query, Key, Value投影层
    private LinearLayer qProj;
    private LinearLayer kProj;
    private LinearLayer vProj;
    private LinearLayer outProj;
    
    /**
     * 构造函数
     * 
     * @param hiddenDim 隐藏维度
     * @param numHeads 注意力头数
     */
    public CrossModalAttention(int hiddenDim, int numHeads) {
        super("CrossModalAttention", null);
        this.hiddenDim = hiddenDim;
        this.numHeads = numHeads;
        this.headDim = hiddenDim / numHeads;
        
        // 初始化投影层
        this.qProj = new LinearLayer("qProj", hiddenDim, hiddenDim, false);
        this.kProj = new LinearLayer("kProj", hiddenDim, hiddenDim, false);
        this.vProj = new LinearLayer("vProj", hiddenDim, hiddenDim, false);
        this.outProj = new LinearLayer("outProj", hiddenDim, hiddenDim, false);
    }
    
    @Override
    public void init() {
        // 初始化已在构造函数中完成
    }
    
    /**
     * 计算跨模态注意力
     * 
     * @param query Query特征（通常是语言特征）
     * @param keyValue Key/Value特征（视觉+本体感知特征）
     * @return 融合后的特征
     */
    public Variable computeAttention(Variable query, Variable keyValue) {
        // 投影Q, K, V
        Variable q = qProj.layerForward(query);
        Variable k = kProj.layerForward(keyValue);
        Variable v = vProj.layerForward(keyValue);
        
        // 获取维度
        NdArray qData = q.getValue();
        NdArray kData = k.getValue();
        NdArray vData = v.getValue();
        
        int queryLen = qData.getShape().getDimension(0);
        int kvLen = kData.getShape().getDimension(0);
        
        // 计算注意力分数：Q @ K^T / sqrt(d_k)
        double[][] scores = new double[queryLen][kvLen];
        double scale = Math.sqrt(headDim);
        
        for (int i = 0; i < queryLen; i++) {
            for (int j = 0; j < kvLen; j++) {
                double score = 0.0;
                for (int d = 0; d < hiddenDim; d++) {
                    score += qData.get(i * hiddenDim + d) * kData.get(j * hiddenDim + d);
                }
                scores[i][j] = score / scale;
            }
        }
        
        // Softmax归一化
        double[][] attnWeights = softmax(scores);
        
        // 加权求和：attnWeights @ V
        double[][] output = new double[queryLen][hiddenDim];
        for (int i = 0; i < queryLen; i++) {
            for (int d = 0; d < hiddenDim; d++) {
                double sum = 0.0;
                for (int j = 0; j < kvLen; j++) {
                    sum += attnWeights[i][j] * vData.get(j * hiddenDim + d);
                }
                output[i][d] = sum;
            }
        }
        
        // 输出投彡
        NdArray outputArray = NdArray.of(output);
        Variable result = outProj.layerForward(new Variable(outputArray));
        
        return result;
    }
    
    /**
     * Softmax函数
     */
    private double[][] softmax(double[][] scores) {
        int rows = scores.length;
        int cols = scores[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            // 找到最大值（数值稳定性）
            double max = scores[i][0];
            for (int j = 1; j < cols; j++) {
                max = Math.max(max, scores[i][j]);
            }
            
            // 计算exp和sum
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.exp(scores[i][j] - max);
                sum += result[i][j];
            }
            
            // 归一化
            for (int j = 0; j < cols; j++) {
                result[i][j] /= sum;
            }
        }
        
        return result;
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 默认实现：自注意力
        return computeAttention(inputs[0], inputs[0]);
    }
}
