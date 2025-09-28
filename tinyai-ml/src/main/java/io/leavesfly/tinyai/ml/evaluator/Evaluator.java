package io.leavesfly.tinyai.ml.evaluator;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.dataset.DataSet;

/**
 * 模型效果评估器抽象类
 * 
 * 该类是所有模型评估器的基类，定义了模型评估的基本接口。
 * 子类需要实现具体的评估逻辑，如准确率评估、回归损失评估等。
 * 
 * @author TinyDL
 * @version 1.0
 */
public abstract class Evaluator {

    protected Model model;
    protected DataSet dataSet;

    /**
     * 执行模型评估
     */
    public abstract void evaluate();

}