package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的GRU层
 * <p>
 * Gated Recurrent Unit (GRU) 循环神经网络层
 * <p>
 * GRU通过两个门控机制（重置门、更新门）来控制信息流，
 * 相比LSTM更简单但性能相近：
 * - 重置门(r): 决定如何将新输入与之前的记忆结合
 * - 更新门(z): 决定保留多少之前的记忆
 * <p>
 * 公式：
 * r_t = sigmoid(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
 * z_t = sigmoid(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
 * n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_{t-1}) + b_n)
 * h_t = (1 - z_t) * n_t + z_t * h_{t-1}
 *
 * @author leavesfly
 * @version 2.0
 */
public class GRU extends Module {

    // 重置门参数
    private Parameter W_ir;  // 输入到重置门的权重
    private Parameter W_hr;  // 隐藏到重置门的权重
    private Parameter b_r;   // 重置门的偏置

    // 更新门参数
    private Parameter W_iz;  // 输入到更新门的权重
    private Parameter W_hz;  // 隐藏到更新门的权重
    private Parameter b_z;   // 更新门的偏置

    // 新记忆参数
    private Parameter W_in;  // 输入到新记忆的权重
    private Parameter W_hn;  // 隐藏到新记忆的权重
    private Parameter b_n;   // 新记忆的偏置

    // 状态缓冲区
    private NdArray hiddenState;  // 隐藏状态 h_t

    private final int inputSize;
    private final int hiddenSize;
    private final boolean bias;

    /**
     * 构造函数
     *
     * @param name       层名称
     * @param inputSize  输入特征数
     * @param hiddenSize 隐藏状态维度
     * @param bias       是否使用偏置
     */
    public GRU(String name, int inputSize, int hiddenSize, boolean bias) {
        super(name);
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.bias = bias;

        initializeParameters();
        init();
    }

    /**
     * 构造函数（默认使用偏置）
     *
     * @param name       层名称
     * @param inputSize  输入特征数
     * @param hiddenSize 隐藏状态维度
     */
    public GRU(String name, int inputSize, int hiddenSize) {
        this(name, inputSize, hiddenSize, true);
    }

    /**
     * 初始化参数
     */
    private void initializeParameters() {
        // 重置门参数
        W_ir = registerParameter("W_ir", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_hr = registerParameter("W_hr", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        // 更新门参数
        W_iz = registerParameter("W_iz", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_hz = registerParameter("W_hz", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        // 新记忆参数
        W_in = registerParameter("W_in", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_hn = registerParameter("W_hn", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        if (bias) {
            b_r = registerParameter("b_r", new Parameter(NdArray.of(Shape.of(hiddenSize))));
            b_z = registerParameter("b_z", new Parameter(NdArray.of(Shape.of(hiddenSize))));
            b_n = registerParameter("b_n", new Parameter(NdArray.of(Shape.of(hiddenSize))));
        }

        // 注册状态缓冲区（使用Buffer机制）
        registerBuffer("hidden_state", null);
    }

    @Override
    public void resetParameters() {
        // 使用Xavier初始化所有权重
        Initializers.xavierUniform(W_ir.data());
        Initializers.xavierUniform(W_hr.data());
        Initializers.xavierUniform(W_iz.data());
        Initializers.xavierUniform(W_hz.data());
        Initializers.xavierUniform(W_in.data());
        Initializers.xavierUniform(W_hn.data());

        // 偏置初始化为0
        if (bias) {
            Initializers.zeros(b_r.data());
            Initializers.zeros(b_z.data());
            Initializers.zeros(b_n.data());
        }
    }

    /**
     * 重置隐藏状态
     */
    public void resetState() {
        hiddenState = null;
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
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        int batchSize = x.getValue().getShape().getDimension(0);

        // 初始化状态
        initializeStateIfNeeded(batchSize);

        Variable h = new Variable(hiddenState);

        // 重置门: r_t = sigmoid(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
        Variable r_t = x.matMul(W_ir.transpose()).add(h.matMul(W_hr.transpose()));
        if (bias) {
            r_t = r_t.add(b_r);
        }
        r_t = r_t.sigmoid();

        // 更新门: z_t = sigmoid(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
        Variable z_t = x.matMul(W_iz.transpose()).add(h.matMul(W_hz.transpose()));
        if (bias) {
            z_t = z_t.add(b_z);
        }
        z_t = z_t.sigmoid();

        // 新记忆: n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_{t-1}) + b_n)
        Variable n_t = x.matMul(W_in.transpose()).add(r_t.mul(h.matMul(W_hn.transpose())));
        if (bias) {
            n_t = n_t.add(b_n);
        }
        n_t = n_t.tanh();

        // 新隐藏状态: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        Variable oneMinusZ = new Variable(1.0f).sub(z_t);
        Variable h_new = oneMinusZ.mul(n_t).add(z_t.mul(h));

        // 更新缓冲区状态
        hiddenState = h_new.getValue();
        _buffers.put("hidden_state", hiddenState);

        return h_new;
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
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    @Override
    public String toString() {
        return "GRU{" +
                "name='" + name + '\'' +
                ", inputSize=" + inputSize +
                ", hiddenSize=" + hiddenSize +
                ", bias=" + bias +
                '}';
    }
}
