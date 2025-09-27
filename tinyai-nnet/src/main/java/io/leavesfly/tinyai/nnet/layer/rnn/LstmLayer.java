package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.RnnLayer;
import io.leavesfly.tinyai.nnet.layer.activate.SigmoidLayer;

import java.util.Objects;

/**
 * 长短期记忆网络层 (LSTM Layer)
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * LSTM是一种特殊的循环神经网络，能够学习长期依赖信息，有效缓解梯度消失问题。
 * LSTM通过三个门控机制来控制信息的流动：
 * 1. 遗忘门 (Forget Gate) - 控制从细胞状态中丢弃什么信息
 * 2. 输入门 (Input Gate) - 控制哪些新信息被存储在细胞状态中
 * 3. 输出门 (Output Gate) - 控制基于细胞状态输出什么信息
 * <p>
 * LSTM 公式:
 * f_t = σ(W_f * [h_{t-1}, x_t] + b_f)   // 遗忘门
 * i_t = σ(W_i * [h_{t-1}, x_t] + b_i)   // 输入门
 * o_t = σ(W_o * [h_{t-1}, x_t] + b_o)   // 输出门
 * ũ_t = tanh(W_u * [h_{t-1}, x_t] + b_u) // 候选细胞状态
 * C_t = f_t * C_{t-1} + i_t * ũ_t       // 细胞状态
 * h_t = o_t * tanh(C_t)                 // 隐藏状态
 * <p>
 * 其中:
 * - f_t 是遗忘门输出
 * - i_t 是输入门输出
 * - o_t 是输出门输出
 * - ũ_t 是候选细胞状态
 * - C_t 是当前细胞状态
 * - h_t 是当前隐藏状态
 * - σ 是sigmoid激活函数
 * - * 表示矩阵乘法
 * - [h_{t-1}, x_t] 表示隐藏状态和输入的拼接
 */
public class LstmLayer extends RnnLayer {

    /**
     * 当前时间步的隐藏状态 (h_t)
     */
    private Variable state;

    /**
     * 当前时间步的细胞状态 (C_t)
     */
    private Variable candidate;

    /**
     * 隐藏层大小
     */
    private int hiddenSize;

    /**
     * 构造一个LSTM层实例
     *
     * @param name         层名称
     * @param xInputShape  输入形状 (batch_size, input_size)
     * @param yOutputShape 输出形状 (batch_size, hidden_size)
     */
    public LstmLayer(String name, Shape xInputShape, Shape yOutputShape) {
        super(name, xInputShape, yOutputShape);
        this.hiddenSize = yOutputShape.getColumn();

        // 初始化遗忘门参数 (Forget Gate)
        // x2f: 输入到遗忘门的权重矩阵
        NdArray initWeight = NdArray.likeRandomN(Shape.of(xInputShape.getColumn(), hiddenSize))
                .mulNum(Math.sqrt((double) 1 / xInputShape.getColumn()));
        Parameter x2f = new Parameter(initWeight);
        x2f.setName(getName() + ".x2f");
        addParam(x2f.getName(), x2f);

        // x2f-b: 遗忘门的偏置项
        Parameter b = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b.setName(getName() + ".x2f-b");
        addParam(b.getName(), b);

        // 初始化输入门参数 (Input Gate)
        // x2i: 输入到输入门的权重矩阵
        initWeight = NdArray.likeRandomN(Shape.of(xInputShape.getColumn(), hiddenSize))
                .mulNum(Math.sqrt((double) 1 / xInputShape.getColumn()));
        Parameter x2i = new Parameter(initWeight);
        x2i.setName(getName() + ".x2i");
        addParam(x2i.getName(), x2i);

        // x2i-b: 输入门的偏置项
        b = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b.setName(getName() + ".x2i-b");
        addParam(b.getName(), b);

        // 初始化输出门参数 (Output Gate)
        // x2o: 输入到输出门的权重矩阵
        initWeight = NdArray.likeRandomN(Shape.of(xInputShape.getColumn(), hiddenSize))
                .mulNum(Math.sqrt((double) 1 / xInputShape.getColumn()));
        Parameter x2o = new Parameter(initWeight);
        x2o.setName(getName() + ".x2o");
        addParam(x2o.getName(), x2o);

        // x2o-b: 输出门的偏置项
        b = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b.setName(getName() + ".x2o-b");
        addParam(b.getName(), b);

        // 初始化候选细胞状态参数
        // x2u: 输入到候选细胞状态的权重矩阵
        initWeight = NdArray.likeRandomN(Shape.of(xInputShape.getColumn(), hiddenSize))
                .mulNum(Math.sqrt((double) 1 / xInputShape.getColumn()));
        Parameter x2u = new Parameter(initWeight);
        x2u.setName(getName() + ".x2u");
        addParam(x2u.getName(), x2u);

        // x2u-b: 候选细胞状态的偏置项
        b = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b.setName(getName() + ".x2u-b");
        addParam(b.getName(), b);

        // =======================

        // 初始化隐藏状态到各门的权重矩阵

        // h2f: 隐藏状态到遗忘门的权重矩阵
        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize))
                .mulNum(Math.sqrt((double) 1 / hiddenSize));
        Parameter h2f = new Parameter(initWeight);
        h2f.setName(getName() + ".h2f");
        addParam(h2f.getName(), h2f);

        // h2i: 隐藏状态到输入门的权重矩阵
        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize))
                .mulNum(Math.sqrt((double) 1 / hiddenSize));
        Parameter h2i = new Parameter(initWeight);
        h2i.setName(getName() + ".h2i");
        addParam(h2i.getName(), h2i);

        // h2o: 隐藏状态到输出门的权重矩阵
        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize))
                .mulNum(Math.sqrt((double) 1 / hiddenSize));
        Parameter h2o = new Parameter(initWeight);
        h2o.setName(getName() + ".h2o");
        addParam(h2o.getName(), h2o);

        // h2u: 隐藏状态到候选细胞状态的权重矩阵
        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize))
                .mulNum(Math.sqrt((double) 1 / hiddenSize));
        Parameter h2u = new Parameter(initWeight);
        h2u.setName(getName() + ".h2u");
        addParam(h2u.getName(), h2u);

        resetState();
    }

    /**
     * 重置LSTM层的内部状态
     * 在处理新序列之前应调用此方法
     */
    @Override
    public void resetState() {
        state = null;
        candidate = null;
    }

    /**
     * 初始化方法（空实现，参数已在构造函数中初始化）
     */
    @Override
    public void init() {
        // 参数已在构造函数中初始化
    }

    /**
     * LSTM层的前向传播方法
     *
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 当前时间步的隐藏状态
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];

        Variable fGate = null;  // 遗忘门
        Variable iGate = null;  // 输入门
        Variable oGate = null;  // 输出门
        Variable uState = null; // 候选细胞状态

        // 第一次前向传播，没有前一时间步的隐藏状态
        if (Objects.isNull(state)) {
            // 计算遗忘门: f_t = σ(W_f * x_t + b_f)
            fGate = x.linear(getParamBy(getName() + ".x2f"), getParamBy(getName() + ".x2f-b"));
            fGate = new SigmoidLayer("").call(fGate);

            // 计算输入门: i_t = σ(W_i * x_t + b_i)
            iGate = x.linear(getParamBy(getName() + ".x2i"), getParamBy(getName() + ".x2i-b"));
            iGate = new SigmoidLayer("").call(iGate);

            // 计算输出门: o_t = σ(W_o * x_t + b_o)
            oGate = x.linear(getParamBy(getName() + ".x2o"), getParamBy(getName() + ".x2o-b"));
            oGate = new SigmoidLayer("").call(oGate);

            // 计算候选细胞状态: ũ_t = tanh(W_u * x_t + b_u)
            uState = x.linear(getParamBy(getName() + ".x2u"), getParamBy(getName() + ".x2u-b")).tanh();

        } else {
            // 后续前向传播，包含前一时间步的隐藏状态

            // 计算遗忘门: f_t = σ(W_f * x_t + W_hf * h_{t-1} + b_f)
            fGate = x.linear(getParamBy(getName() + ".x2f"), getParamBy(getName() + ".x2f-b"))
                    .add(state.linear(getParamBy(getName() + ".h2f"), null));
            fGate = new SigmoidLayer("").call(fGate);

            // 计算输入门: i_t = σ(W_i * x_t + W_hi * h_{t-1} + b_i)
            iGate = x.linear(getParamBy(getName() + ".x2i"), getParamBy(getName() + ".x2i-b"))
                    .add(state.linear(getParamBy(getName() + ".h2i"), null));
            iGate = new SigmoidLayer("").call(iGate);

            // 计算输出门: o_t = σ(W_o * x_t + W_ho * h_{t-1} + b_o)
            oGate = x.linear(getParamBy(getName() + ".x2o"), getParamBy(getName() + ".x2o-b"))
                    .add(state.linear(getParamBy(getName() + ".h2o"), null));
            oGate = new SigmoidLayer("").call(oGate);

            // 计算候选细胞状态: ũ_t = tanh(W_u * x_t + W_hu * h_{t-1} + b_u)
            uState = x.linear(getParamBy(getName() + ".x2u"), getParamBy(getName() + ".x2u-b"))
                    .add(state.linear(getParamBy(getName() + ".h2u"), null)).tanh();
        }

        // 更新细胞状态: C_t = f_t * C_{t-1} + i_t * ũ_t
        if (Objects.isNull(candidate)) {
            // 第一次，没有前一时间步的细胞状态
            candidate = iGate.mul(uState);
        } else {
            // 后续时间步，包含前一时间步的细胞状态
            candidate = fGate.mul(candidate).add(iGate.mul(uState));
        }

        // 计算隐藏状态: h_t = o_t * tanh(C_t)
        state = oGate.mul(candidate.tanh());
        return state;
    }
}