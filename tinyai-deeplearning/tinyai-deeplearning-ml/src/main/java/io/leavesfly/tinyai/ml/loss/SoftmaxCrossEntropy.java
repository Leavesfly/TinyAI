package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;

/**
 * Softmax 交叉熵损失函数，用于分类任务，结合 Softmax 激活函数与交叉熵损失。
 * 支持 PyTorch 风格的 reduction 参数。
 *
 * @author TinyDL
 * @version 2.0
 */
public class SoftmaxCrossEntropy extends Loss {
    
    /**
     * 默认构造函数(使用MEAN归约)
     */
    public SoftmaxCrossEntropy() {
        super(Reduction.MEAN);
    }
    
    /**
     * 带归约方式的构造函数
     *
     * @param reduction 归约方式
     */
    public SoftmaxCrossEntropy(Reduction reduction) {
        super(reduction);
    }
    
    /**
     * 完整参数构造函数
     *
     * @param reduction   归约方式
     * @param weight      类别权重
     * @param ignoreIndex 忽略的索引值
     */
    public SoftmaxCrossEntropy(Reduction reduction, Variable weight, int ignoreIndex) {
        super(reduction, weight, ignoreIndex);
    }
    
    @Override
    public Variable loss(Variable y, Variable predict) {
        // 计算Softmax交叉熄
        Variable loss = predict.softmaxCrossEntropy(y);
        
        // 应用归约
        return applyReduction(loss);
    }
}