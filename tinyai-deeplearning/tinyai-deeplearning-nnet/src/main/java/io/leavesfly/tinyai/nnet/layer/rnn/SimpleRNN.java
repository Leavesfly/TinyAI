package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的SimpleRNN层
 * <p>
 * Simple Recurrent Neural Network (SimpleRNN) 最基础的循环神经网络层
 * <p>
 * SimpleRNN使用简单的循环结构来处理序列数据：
 * - 接收当前输入和前一时刻的隐藏状态
 * - 生成新的隐藏状态
 * <p>
 * 公式：
 * h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
 *
 * @author leavesfly
 * @version 2.0
 */
public class SimpleRNN extends Module {

    // 权重参数
    private Parameter W_ih;  // 输入到隐藏的权重
    private Parameter W_hh;  // 隐藏到隐藏的权重
    private Parameter b;     // 偏置

    // 状态缓冲区
    private NdArray hiddenState;  // 隐藏状态 h_t（用于序列化和状态保存）
    private transient Variable hiddenStateVar;  // 隐藏状态 Variable（用于计算图）

    private final int inputSize;
    private final int hiddenSize;
    private final boolean useBias;
    private final String activation;  // 激活函数类型 (tanh, relu)

    /**
     * 构造函数
     *
     * @param name       层名称
     * @param inputSize  输入特征数
     * @param hiddenSize 隐藏状态维度
     * @param useBias    是否使用偏置
     * @param activation 激活函数 (tanh, relu)
     */
    public SimpleRNN(String name, int inputSize, int hiddenSize, boolean useBias, String activation) {
        super(name);
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.useBias = useBias;
        this.activation = activation != null ? activation : "tanh";

        initializeParameters();
        init();
    }

    /**
     * 构造函数（默认使用tanh激活和偏置）
     *
     * @param name       层名称
     * @param inputSize  输入特征数
     * @param hiddenSize 隐藏状态维度
     */
    public SimpleRNN(String name, int inputSize, int hiddenSize) {
        this(name, inputSize, hiddenSize, true, "tanh");
    }

    /**
     * 构造函数
     *
     * @param name       层名称
     * @param inputSize  输入特征数
     * @param hiddenSize 隐藏状态维度
     * @param useBias    是否使用偏置
     */
    public SimpleRNN(String name, int inputSize, int hiddenSize, boolean useBias) {
        this(name, inputSize, hiddenSize, useBias, "tanh");
    }

    /**
     * 初始化参数
     */
    private void initializeParameters() {
        // 权重参数
        W_ih = registerParameter("W_ih", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_hh = registerParameter("W_hh", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        if (useBias) {
            b = registerParameter("b", new Parameter(NdArray.of(Shape.of(hiddenSize))));
        }

        // 注册状态缓冲区（使用Buffer机制）
        registerBuffer("hidden_state", null);
    }

    @Override
    public void resetParameters() {
        // 使用Xavier初始化所有权重
        Initializers.xavierUniform(W_ih.data());
        Initializers.xavierUniform(W_hh.data());

        // 偏置初始化为0
        if (useBias) {
            Initializers.zeros(b.data());
        }
    }

    /**
     * 重置隐藏状态
     */
    public void resetState() {
        hiddenState = null;
        hiddenStateVar = null;
        _buffers.put("hidden_state", null);
    }

    /**
     * 初始化状态（如果尚未初始化）
     *
     * @param batchSize 批次大小
     */
    private void initializeStateIfNeeded(int batchSize) {
        if (hiddenState == null || hiddenState.getShape().getDimension(0) != batchSize) {
            hiddenState = NdArray.zeros(Shape.of(batchSize, hiddenSize));
            _buffers.put("hidden_state", hiddenState);
            
            // 同时初始化 Variable，用于计算图
            hiddenStateVar = new Variable(hiddenState);
            hiddenStateVar.setRequireGrad(false);
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 获取batch_size用于初始化状态，使用 Variable 的 size 方法
        int batchSize = x.size(0);

        // 初始化状态
        initializeStateIfNeeded(batchSize);

        // 直接使用 hiddenStateVar 参与计算，保持计算图连接
        Variable h = hiddenStateVar;

        // 计算新的隐藏状态: h_t = activation(W_ih @ x_t + W_hh @ h_{t-1} + b)
        // 使用Variable层级的操作
        Variable h_new = x.matMul(W_ih.transpose()).add(h.matMul(W_hh.transpose()));

        if (useBias) {
            h_new = h_new.add(b);
        }

        // 应用激活函数
        h_new = applyActivation(h_new);

        // 更新状态：使用 detach 断开历史计算图，避免无限增长
        hiddenStateVar = h_new.detach();
        // 同步更新 buffer（用于序列化和状态保存）
        hiddenState = h_new.getValue();
        _buffers.put("hidden_state", hiddenState);

        return h_new;
    }

    /**
     * 应用激活函数
     *
     * @param x 输入变量
     * @return 激活后的变量
     */
    private Variable applyActivation(Variable x) {
        switch (activation.toLowerCase()) {
            case "tanh":
                return x.tanh();
            case "relu":
                return x.relu();
            default:
                throw new IllegalArgumentException("Unsupported activation: " + activation);
        }
    }

    /**
     * 获取当前隐藏状态
     *
     * @return 隐藏状态
     */
    public NdArray getHiddenState() {
        return hiddenState;
    }

    /**
     * 设置隐藏状态
     *
     * @param hiddenState 新的隐藏状态
     */
    public void setHiddenState(NdArray hiddenState) {
        this.hiddenState = hiddenState;
        _buffers.put("hidden_state", hiddenState);
        // 同时更新 Variable
        this.hiddenStateVar = new Variable(hiddenState);
        this.hiddenStateVar.setRequireGrad(false);
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public String getActivation() {
        return activation;
    }

    @Override
    public String toString() {
        return "SimpleRNN{" +
                "name='" + name + '\'' +
                ", inputSize=" + inputSize +
                ", hiddenSize=" + hiddenSize +
                ", useBias=" + useBias +
                ", activation='" + activation + '\'' +
                '}';
    }
}