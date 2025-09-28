package io.leavesfly.tinyai.ml.optimize;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.Map;

/**
 * 参数优化器抽象类
 * <p>
 * 该类是所有参数优化器实现的基类，定义了参数更新的基本接口和流程。
 * 子类需要实现具体的参数更新逻辑。
 *
 * @author TinyDL
 * @version 1.0
 */
public abstract class Optimizer {

    private Model target;

    /**
     * 构造函数
     *
     * @param target 目标模型
     */
    public Optimizer(Model target) {
        this.target = target;
    }

    /**
     * 更新所有参数
     */
    public void update() {
        Map<String, Parameter> parameterMap = target.getAllParams();
        for (Parameter parameter : parameterMap.values()) {
            updateOne(parameter);
        }
    }

    /**
     * 更新单个参数
     *
     * @param parameter 参数
     */
    public abstract void updateOne(Parameter parameter);

}