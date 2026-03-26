package io.leavesfly.tinyai.nnet.v2.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

/**
 * V2版本的LSTM层
 * <p>
 * Long Short-Term Memory (LSTM) 循环神经网络层
 * <p>
 * LSTM通过三个门控机制（遗忘门、输入门、输出门）来控制信息流：
 * - 遗忘门(f): 决定从细胞状态中丢弃什么信息
 * - 输入门(i): 决定什么新信息被存储到细胞状态
 * - 输出门(o): 决定输出什么值
 * <p>
 * 公式：
 * f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
 * i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
 * o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
 * c_tilde_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)
 * c_t = f_t * c_{t-1} + i_t * c_tilde_t
 * h_t = o_t * tanh(c_t)
 *
 * @author leavesfly
 * @version 2.0
 */
public class LSTM extends Module {

    // 输入门参数
    private Parameter W_ii;  // 输入到输入门的权重
    private Parameter W_hi;  // 隐藏到输入门的权重
    private Parameter b_i;   // 输入门的偏置

    // 遗忘门参数
    private Parameter W_if;  // 输入到遗忘门的权重
    private Parameter W_hf;  // 隐藏到遗忘门的权重
    private Parameter b_f;   // 遗忘门的偏置

    // 细胞门参数
    private Parameter W_ig;  // 输入到细胞门的权重
    private Parameter W_hg;  // 隐藏到细胞门的权重
    private Parameter b_g;   // 细胞门的偏置

    // 输出门参数
    private Parameter W_io;  // 输入到输出门的权重
    private Parameter W_ho;  // 隐藏到输出门的权重
    private Parameter b_o;   // 输出门的偏置

    // 状态缓冲区
    private NdArray hiddenState;  // 隐藏状态 h_t
    private NdArray cellState;    // 细胞状态 c_t

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
    public LSTM(String name, int inputSize, int hiddenSize, boolean bias) {
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
    public LSTM(String name, int inputSize, int hiddenSize) {
        this(name, inputSize, hiddenSize, true);
    }

    /**
     * 初始化参数
     */
    private void initializeParameters() {
        // 输入门参数
        W_ii = registerParameter("W_ii", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_hi = registerParameter("W_hi", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        // 遗忘门参数
        W_if = registerParameter("W_if", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_hf = registerParameter("W_hf", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        // 细胞门参数
        W_ig = registerParameter("W_ig", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_hg = registerParameter("W_hg", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        // 输出门参数
        W_io = registerParameter("W_io", new Parameter(NdArray.of(Shape.of(hiddenSize, inputSize))));
        W_ho = registerParameter("W_ho", new Parameter(NdArray.of(Shape.of(hiddenSize, hiddenSize))));

        if (bias) {
            b_i = registerParameter("b_i", new Parameter(NdArray.of(Shape.of(hiddenSize))));
            b_f = registerParameter("b_f", new Parameter(NdArray.of(Shape.of(hiddenSize))));
            b_g = registerParameter("b_g", new Parameter(NdArray.of(Shape.of(hiddenSize))));
            b_o = registerParameter("b_o", new Parameter(NdArray.of(Shape.of(hiddenSize))));
        }

        // 注册状态缓冲区（使用Buffer机制）
        registerBuffer("hidden_state", null);
        registerBuffer("cell_state", null);
    }

    @Override
    public void resetParameters() {
        // 使用Xavier初始化所有权重
        Initializers.xavierUniform(W_ii.data());
        Initializers.xavierUniform(W_hi.data());
        Initializers.xavierUniform(W_if.data());
        Initializers.xavierUniform(W_hf.data());
        Initializers.xavierUniform(W_ig.data());
        Initializers.xavierUniform(W_hg.data());
        Initializers.xavierUniform(W_io.data());
        Initializers.xavierUniform(W_ho.data());

        // 偏置初始化为0
        if (bias) {
            Initializers.zeros(b_i.data());
            Initializers.zeros(b_f.data());
            Initializers.zeros(b_g.data());
            Initializers.zeros(b_o.data());
        }
    }

    /**
     * 重置隐藏状态和细胞状态
     */
    public void resetState() {
        hiddenState = null;
        cellState = null;
        _buffers.put("hidden_state", null);
        _buffers.put("cell_state", null);
    }

    /**
     * 初始化状态（如果尚未初始化）
     *
     * @param batchSize 批次大小
     */
    private void initializeStateIfNeeded(int batchSize) {
        if (hiddenState == null) {
            hiddenState = NdArray.zeros(Shape.of(batchSize, hiddenSize));
            cellState = NdArray.zeros(Shape.of(batchSize, hiddenSize));
            _buffers.put("hidden_state", hiddenState);
            _buffers.put("cell_state", cellState);
        } else if (hiddenState.getShape().getDimension(0) != batchSize) {
            // batchSize 变化时重新初始化状态
            hiddenState = NdArray.zeros(Shape.of(batchSize, hiddenSize));
            cellState = NdArray.zeros(Shape.of(batchSize, hiddenSize));
            _buffers.put("hidden_state", hiddenState);
            _buffers.put("cell_state", cellState);
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        int batchSize = x.getShape().getDimension(0);

        // 初始化状态
        initializeStateIfNeeded(batchSize);

        Variable h = new Variable(hiddenState);
        Variable c = new Variable(cellState);

        // 计算四个门
        // 输入门: i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
        Variable i_t = x.matMul(W_ii.transpose()).add(h.matMul(W_hi.transpose()));
        if (bias) {
            i_t = i_t.add(b_i);
        }
        i_t = i_t.sigmoid();

        // 遗忘门: f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)
        Variable f_t = x.matMul(W_if.transpose()).add(h.matMul(W_hf.transpose()));
        if (bias) {
            f_t = f_t.add(b_f);
        }
        f_t = f_t.sigmoid();

        // 细胞门: g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
        Variable g_t = x.matMul(W_ig.transpose()).add(h.matMul(W_hg.transpose()));
        if (bias) {
            g_t = g_t.add(b_g);
        }
        g_t = g_t.tanh();

        // 输出门: o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)
        Variable o_t = x.matMul(W_io.transpose()).add(h.matMul(W_ho.transpose()));
        if (bias) {
            o_t = o_t.add(b_o);
        }
        o_t = o_t.sigmoid();

        // 更新细胞状态: c_t = f_t * c_{t-1} + i_t * g_t
        Variable c_new = f_t.mul(c).add(i_t.mul(g_t));

        // 更新隐藏状态: h_t = o_t * tanh(c_t)
        Variable h_new = o_t.mul(c_new.tanh());

        // 更新缓冲区状态
        hiddenState = h_new.getValue();
        cellState = c_new.getValue();
        _buffers.put("hidden_state", hiddenState);
        _buffers.put("cell_state", cellState);

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
     * 获取当前细胞状态
     *
     * @return 细胞状态
     */
    public NdArray getCellState() {
        return cellState;
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

    /**
     * 设置细胞状态
     *
     * @param cellState 新的细胞状态
     */
    public void setCellState(NdArray cellState) {
        this.cellState = cellState;
        _buffers.put("cell_state", cellState);
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    @Override
    public String toString() {
        return "LSTM{" +
                "name='" + name + '\'' +
                ", inputSize=" + inputSize +
                ", hiddenSize=" + hiddenSize +
                ", bias=" + bias +
                '}';
    }
}
