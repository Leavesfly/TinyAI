package io.leavesfly.tinyai.minimind.training.rlaif;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;

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
public abstract class BaseRLTrainer extends BaseTrainer {
    
    /**
     * 构造函数
     * 
     * @param model 模型
     */
    public BaseRLTrainer(MiniMindModel model) {
        super(model);
    }
    
    /**
     * Log Softmax（使用框架内置实现,在vocab维度axis=-1上计算）
     * 
     * @param x 输入变量
     * @return 对数 softmax 结果
     */
    protected Variable logSoftmax(Variable x) {
        return x.logSoftmax();
    }
    
    /**
     * 计算对数概率: log π(y|x)
     * 
     * 通过 softmaxCrossEntropy 正确计算 token 级别的对数概率,
     * 内部完成 logSoftmax → gather(labels) → NLL,保持计算图连通。
     * 
     * @param logits logit 输出 [batch_size, seq_len, vocab_size]
     * @param labels 标签 [batch_size, seq_len]
     * @return 平均对数概率(标量Variable)
     */
    protected Variable computeLogProb(Variable logits, Variable labels) {
        // SoftmaxCE 只支持 2 维输入 [N, vocab_size]
        // 如果 logits 是 3 维 [batch_size, seq_len, vocab_size]，需要先 reshape 成 2 维
        int dims = logits.getValue().getShape().getDimNum();
        Variable flatLogits = logits;
        Variable flatLabels = labels;
        if (dims == 3) {
            int[] shape = logits.getValue().getShape().getShapeDims();
            int batchSize = shape[0];
            int seqLen = shape[1];
            int vocabSize = shape[2];
            flatLogits = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
            flatLabels = labels.reshape(Shape.of(batchSize * seqLen, 1));
        }
        Variable crossEntropy = flatLogits.softmaxCrossEntropy(flatLabels);
        return crossEntropy.neg();
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
