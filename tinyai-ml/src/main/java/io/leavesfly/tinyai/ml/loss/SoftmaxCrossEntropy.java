package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;

/**
 * Softmax交叉熵损失函数
 * 
 * 用于分类任务的损失计算，结合了Softmax激活函数和交叉熵损失。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class SoftmaxCrossEntropy extends Loss {
    @Override
    public Variable loss(Variable y, Variable predict) {
        return predict.softmaxCrossEntropy(y);
    }
}