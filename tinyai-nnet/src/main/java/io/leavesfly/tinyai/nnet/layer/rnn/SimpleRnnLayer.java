package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.RnnLayer;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 简单递归网络层实现 (Simple RNN Layer)
 * <p>
 * 这是一个使用 tanh 作为激活函数的标准循环神经网络层实现。
 * 该层维护一个内部状态，在序列处理中传递信息。
 * <p>
 * RNN 公式:
 * h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
 * <p>
 * 其中:
 * - h_t 是当前时间步的隐藏状态
 * - x_t 是当前时间步的输入
 * - h_{t-1} 是前一个时间步的隐藏状态
 * - W_xh 是输入到隐藏状态的权重矩阵
 * - W_hh 是隐藏状态到隐藏状态的权重矩阵
 * - b 是偏置项
 */
public class SimpleRnnLayer extends RnnLayer {

    /**
     * 输入到隐藏状态的权重矩阵参数
     * 形状: (input_size, hidden_size)
     */
    Parameter x2h;

    /**
     * 隐藏状态到隐藏状态的权重矩阵参数
     * 形状: (hidden_size, hidden_size)
     */
    Parameter h2h;

    /**
     * 偏置参数
     * 形状: (1, hidden_size)
     */
    Parameter b;

    /**
     * 当前时间步的隐藏状态变量
     */
    private Variable state;

    /**
     * 当前时间步的隐藏状态值（NdArray形式）
     */
    private NdArray stateValue;

    // 用于反向传播的缓存变量
    /**
     * 前一个时间步的隐藏状态变量
     */
    private Variable prevState;

    /**
     * tanh激活函数之前的值，用于反向传播计算
     */
    private Variable preTanh;

    /**
     * 输入线性变换结果 (x * W_xh + b)
     */
    private Variable xLinear;

    /**
     * 隐藏状态线性变换结果 (h_{t-1} * W_hh)
     */
    private Variable hLinear;

    /**
     * 隐藏层大小
     */
    private int hiddenSize;

    /**
     * 构造一个SimpleRnnLayer实例
     *
     * @param name         层名称
     * @param xInputShape  输入形状 (batch_size, input_size)
     * @param yOutputShape 输出形状 (batch_size, hidden_size)
     */
    public SimpleRnnLayer(String name, Shape xInputShape, Shape yOutputShape) {
        super(name, xInputShape, yOutputShape);
        this.hiddenSize = yOutputShape.getColumn();
        init();
    }

    /**
     * 重置RNN层的内部状态
     * 在处理新序列之前应调用此方法
     */
    @Override
    public void resetState() {
        state = null;
        stateValue = null;
    }

    /**
     * 初始化RNN层的参数
     * 包括输入到隐藏状态权重、隐藏状态到隐藏状态权重和偏置项
     */
    @Override
    public void init() {
        int inputSize = inputShape.getColumn();

        // 初始化输入到隐藏状态的权重矩阵，使用Xavier初始化
        NdArray initWeight = NdArray.likeRandomN(Shape.of(inputSize, hiddenSize))
                .mulNum(Math.sqrt((double) 1 / inputSize));
        x2h = new Parameter(initWeight);
        x2h.setName(getName() + ".x2h");
        addParam(x2h.getName(), x2h);

        // 初始化隐藏状态到隐藏状态的权重矩阵，使用Xavier初始化
        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize)).mulNum(Math.sqrt((double) 1 / hiddenSize));
        h2h = new Parameter(initWeight);
        h2h.setName(getName() + ".h2h");
        addParam(h2h.getName(), h2h);

        // 初始化偏置项为零
        b = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b.setName(getName() + ".b");
        addParam(b.getName(), b);
    }

    /**
     * 基于Variable的前向传播方法
     *
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 当前时间步的隐藏状态
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];

        // 第一次前向传播，没有前一时间步的隐藏状态
        if (Objects.isNull(state)) {
            prevState = null;
            xLinear = x.linear(x2h, b);
            state = xLinear.tanh();
            stateValue = state.getValue();
            preTanh = state;
        } else {
            // 后续前向传播，包含前一时间步的隐藏状态
            prevState = state;
            xLinear = x.linear(x2h, b);
            hLinear = new Variable(stateValue).linear(h2h, null);
            state = xLinear.add(hLinear).tanh();
            stateValue = state.getValue();
            preTanh = state;
        }
        return state;
    }

    /**
     * 基于NdArray的前向传播方法
     *
     * @param inputs 输入数组，通常只包含一个输入数组
     * @return 当前时间步的隐藏状态值
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];

        // 第一次前向传播，没有前一时间步的隐藏状态
        if (stateValue == null) {
            NdArray xLinear = x.dot(x2h.getValue());
            Shape xLinearShape = xLinear.getShape();
            NdArray bias = b.getValue().broadcastTo(xLinearShape);
            NdArray linearResult = xLinear.add(bias);
            stateValue = linearResult.tanh();
        } else {
            // 后续前向传播，包含前一时间步的隐藏状态
            NdArray xLinear = x.dot(x2h.getValue());
            Shape xLinearShape = xLinear.getShape();
            NdArray bias = b.getValue().broadcastTo(xLinearShape);
            NdArray xLinearWithBias = xLinear.add(bias);
            NdArray hLinear = stateValue.dot(h2h.getValue());
            NdArray linearResult = xLinearWithBias.add(hLinear);
            stateValue = linearResult.tanh();
        }
        return stateValue;
    }

    /**
     * 反向传播方法
     *
     * @param yGrad 上游传来的梯度
     * @return 输入和参数的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 计算tanh的梯度: grad * (1 - tanh^2(x))
        NdArray tanhGrad = yGrad.mul(NdArray.ones(preTanh.getValue().getShape()).sub(preTanh.getValue().square()));

        // 计算线性变换的梯度
        NdArray xLinearGrad = tanhGrad;
        NdArray hLinearGrad = tanhGrad;

        // 计算输入x的梯度: grad * W_xh^T
        NdArray xGrad = xLinearGrad.dot(x2h.getValue().transpose());

        // 计算参数梯度
        NdArray x2hGrad = inputs[0].getValue().transpose().dot(xLinearGrad);  // W_xh的梯度
        NdArray bGrad = xLinearGrad.sumTo(b.getValue().getShape());           // b的梯度

        // 如果有前一状态，计算h2h的梯度和前一状态的梯度
        if (prevState != null) {
            // 计算h2h的梯度: h_{t-1}^T * grad
            NdArray h2hGrad = prevState.getValue().transpose().dot(hLinearGrad);
            // 计算前一状态的梯度: grad * W_hh^T
            NdArray hGrad = hLinearGrad.dot(h2h.getValue().transpose());
            // 将输入梯度和前一状态梯度相加
            xGrad = xGrad.add(hGrad);
            return Arrays.asList(xGrad, x2hGrad, h2hGrad, bGrad);
        } else {
            return Arrays.asList(xGrad, x2hGrad, bGrad);
        }
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}