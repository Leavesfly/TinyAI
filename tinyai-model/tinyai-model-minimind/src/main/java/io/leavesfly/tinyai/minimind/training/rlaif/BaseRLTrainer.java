package io.leavesfly.tinyai.minimind.training.rlaif;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.util.Map;

/**
 * 强化学习训练器基类
 * 
 * 提供 RL 训练器共有的工具方法：
 * - logSoftmax: 对数 softmax 计算
 * - computeLogProb: 计算对数概率
 * - clipGradients: 梯度裁剪（支持传入 Object）
 * 
 * @author leavesfly
 * @since 2024
 */
public class BaseRLTrainer extends BaseTrainer {
    
    /**
     * 构造函数
     * 
     * @param model 模型
     */
    public BaseRLTrainer(MiniMindModel model) {
        super(model);
    }
    
    /**
     * Log Softmax
     * 
     * @param x 输入变量
     * @return 对数 softmax 结果
     */
    protected Variable logSoftmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        Variable logSumExp = sumExp.log();
        return x.sub(logSumExp);
    }
    
    /**
     * 计算对数概率
     * 
     * @param logits logit 输出
     * @param labels 标签
     * @return 对数概率
     */
    protected Variable computeLogProb(Variable logits, Variable labels) {
        Variable logProbs = logSoftmax(logits);
        Variable meanLogProb = logProbs.mean(0, true);
        return meanLogProb;
    }
    
    /**
     * 梯度裁剪（支持传入 Object）
     * 
     * @param model 模型对象（MiniMindModel 或 ValueNetwork）
     * @param maxNorm 最大梯度范数
     */
    protected void clipGradients(Object model, float maxNorm) {
        if (maxNorm <= 0) return;
        
        Map<String, Parameter> params;
        if (model instanceof MiniMindModel) {
            params = ((MiniMindModel) model).getAllParams();
        } else if (model instanceof io.leavesfly.tinyai.minimind.training.rlaif.ppo.ValueNetwork) {
            // ValueNetwork 返回 v2.core.ParameterV1，暂不支持
            return;
        } else {
            return;
        }
        
        float totalNorm = 0.0f;
        for (Parameter param : params.values()) {
            if (param.getGrad() != null) {
                float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) param.getGrad()).buffer;
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = (float) Math.sqrt(totalNorm);
        
        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (Parameter param : params.values()) {
                if (param.getGrad() != null) {
                    float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) param.getGrad()).buffer;
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= scale;
                    }
                }
            }
        }
    }
}
