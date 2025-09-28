package io.leavesfly.tinyai.ml.inference;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;

/**
 * 模型推理器
 * <p>
 * 与训练器Trainer对应，用于模型训练完成后的推理预测。
 * 通过Translator实现输入数据和输出数据的转换，完成端到端的预测流程。
 *
 * @param <I> 输入数据类型
 * @param <O> 输出数据类型
 * @author TinyDL
 * @version 1.0
 */
public class Predictor<I, O> {

    private Translator<I, O> translator;

    private Model model;

    /**
     * 构造函数
     *
     * @param translator 数据转换器
     * @param model      模型
     */
    public Predictor(Translator<I, O> translator, Model model) {
        this.translator = translator;
        this.model = model;
    }

    /**
     * 执行预测
     *
     * @param input 输入数据
     * @return 预测结果
     */
    public O predict(I input) {
        NdArray _input = translator.input2NdArray(input);
        Config.train = false;

        Variable _output = model.forward(new Variable(_input));

        return translator.ndArray2Output(_output.getValue());
    }
}