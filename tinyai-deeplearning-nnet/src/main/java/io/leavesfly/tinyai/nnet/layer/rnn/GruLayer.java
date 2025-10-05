package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.activate.SigmoidLayer;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 门控循环单元层(GRU)
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * GRU是LSTM的简化版本，包含更新门和重置门，能够有效处理序列数据中的长期依赖问题。
 * GRU通过两个门控机制来控制信息的流动：
 * 1. 更新门 (Update Gate) - 控制前一状态信息的保留程度
 * 2. 重置门 (Reset Gate) - 控制前一状态信息对当前候选状态的影响
 * <p>
 * GRU 公式:
 * z_t = σ(W_z * x_t + U_z * h_{t-1} + b_z)   // 更新门
 * r_t = σ(W_r * x_t + U_r * h_{t-1} + b_r)   // 重置门
 * ũ_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h) // 候选状态
 * h_t = (1 - z_t) ⊙ ũ_t + z_t ⊙ h_{t-1}     // 当前隐藏状态
 * <p>
 * 其中:
 * - z_t 是更新门输出
 * - r_t 是重置门输出
 * - ũ_t 是候选状态
 * - h_t 是当前隐藏状态
 * - σ 是sigmoid激活函数
 * - ⊙ 表示逐元素乘法
 */
public class GruLayer extends Layer {

    /**
     * 当前时间步的隐藏状态
     */
    private Variable state;

    /**
     * 当前时间步的隐藏状态值（NdArray形式）
     */
    private NdArray stateValue;

    // GRU的参数
    // 更新门参数
    /**
     * 输入到更新门的权重矩阵
     * 形状: (input_size, hidden_size)
     */
    private Parameter w_z;

    /**
     * 隐藏状态到更新门的权重矩阵
     * 形状: (hidden_size, hidden_size)
     */
    private Parameter u_z;

    /**
     * 更新门的偏置项
     * 形状: (1, hidden_size)
     */
    private Parameter b_z;

    // 重置门参数
    /**
     * 输入到重置门的权重矩阵
     * 形状: (input_size, hidden_size)
     */
    private Parameter w_r;

    /**
     * 隐藏状态到重置门的权重矩阵
     * 形状: (hidden_size, hidden_size)
     */
    private Parameter u_r;

    /**
     * 重置门的偏置项
     * 形状: (1, hidden_size)
     */
    private Parameter b_r;

    // 候选状态参数
    /**
     * 输入到候选状态的权重矩阵
     * 形状: (input_size, hidden_size)
     */
    private Parameter w_h;

    /**
     * 隐藏状态到候选状态的权重矩阵
     * 形状: (hidden_size, hidden_size)
     */
    private Parameter u_h;

    /**
     * 候选状态的偏置项
     * 形状: (1, hidden_size)
     */
    private Parameter b_h;

    /**
     * 隐藏层大小
     */
    private int hiddenSize;
    
    /**
     * 当前状态的批大小
     * 用于检测批大小变化并重置状态
     */
    private int currentBatchSize = -1;

    // 用于反向传播的缓存变量
    /**
     * 更新门变量
     */
    private Variable zGate;

    /**
     * 重置门变量
     */
    private Variable rGate;

    /**
     * 候选状态变量
     */
    private Variable hCandidate;

    /**
     * 重置后的状态变量
     */
    private Variable resetState;

    // 前向传播中的中间变量
    /**
     * 输入到更新门的线性变换结果
     */
    private Variable x_z;

    /**
     * 隐藏状态到更新门的线性变换结果
     */
    private Variable h_z;

    /**
     * 输入到重置门的线性变换结果
     */
    private Variable x_r;

    /**
     * 隐藏状态到重置门的线性变换结果
     */
    private Variable h_r;

    /**
     * 输入到候选状态的线性变换结果
     */
    private Variable x_h;

    /**
     * 隐藏状态到候选状态的线性变换结果
     */
    private Variable h_h;

    /**
     * (1 - 更新门)的值，用于计算当前状态
     */
    private Variable oneMinusZ;

    /**
     * 构造一个GRU层实例
     *
     * @param _name         层名称
     * @param _xInputShape  输入形状 (batch_size, input_size)
     * @param _yOutputShape 输出形状 (batch_size, hidden_size)
     */
    public GruLayer(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
        hiddenSize = _yOutputShape.getColumn();
        init();
    }

    /**
     * 重置GRU层的内部状态
     * 在处理新序列之前应调用此方法
     */
    public void resetState() {
        state = null;
        stateValue = null;
        currentBatchSize = -1; // 重置批大小记录
    }

    /**
     * 初始化GRU层的参数
     * 包括更新门、重置门和候选状态的权重矩阵及偏置项
     */
    @Override
    public void init() {
        int inputSize = inputShape.getColumn();

        // 初始化更新门参数
        NdArray initWeight = NdArray.likeRandomN(Shape.of(inputSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (inputSize + hiddenSize)));
        w_z = new Parameter(initWeight);
        w_z.setName("w_z");
        addParam(w_z.getName(), w_z);

        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (hiddenSize + hiddenSize)));
        u_z = new Parameter(initWeight);
        u_z.setName("u_z");
        addParam(u_z.getName(), u_z);

        b_z = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b_z.setName("b_z");
        addParam(b_z.getName(), b_z);

        // 初始化重置门参数
        initWeight = NdArray.likeRandomN(Shape.of(inputSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (inputSize + hiddenSize)));
        w_r = new Parameter(initWeight);
        w_r.setName("w_r");
        addParam(w_r.getName(), w_r);

        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (hiddenSize + hiddenSize)));
        u_r = new Parameter(initWeight);
        u_r.setName("u_r");
        addParam(u_r.getName(), u_r);

        b_r = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b_r.setName("b_r");
        addParam(b_r.getName(), b_r);

        // 初始化候选状态参数
        initWeight = NdArray.likeRandomN(Shape.of(inputSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (inputSize + hiddenSize)));
        w_h = new Parameter(initWeight);
        w_h.setName("w_h");
        addParam(w_h.getName(), w_h);

        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize))
                .mulNum((float) Math.sqrt(2.0 / (hiddenSize + hiddenSize)));
        u_h = new Parameter(initWeight);
        u_h.setName("u_h");
        addParam(u_h.getName(), u_h);

        b_h = new Parameter(NdArray.zeros(Shape.of(1, hiddenSize)));
        b_h.setName("b_h");
        addParam(b_h.getName(), b_h);
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        int inputBatchSize = x.getValue().getShape().getRow();
        
        // 检测批大小变化，如果变化则重置状态
        if (currentBatchSize != -1 && currentBatchSize != inputBatchSize) {
            // 批大小变化，重置状态以适应新的批大小
            resetState();
        }
        currentBatchSize = inputBatchSize;

        if (Objects.isNull(state)) {
            // 第一次前向传播
            // 计算更新门
            x_z = x.linear(w_z, b_z);
            zGate = x_z.sigmoid();

            // 计算重置门
            x_r = x.linear(w_r, b_r);
            rGate = x_r.sigmoid();

            // 计算候选状态
            x_h = x.linear(w_h, b_h);
            hCandidate = x_h.tanh();

            // 计算当前状态
            oneMinusZ = new Variable(NdArray.ones(zGate.getValue().getShape())).sub(zGate);
            state = oneMinusZ.mul(hCandidate);
            stateValue = state.getValue();
        } else {
            // 后续前向传播
            // 检查状态形状是否与当前输入匹配
            if (state.getValue().getShape().getRow() != inputBatchSize) {
                // 状态批大小不匹配，重新初始化状态
                resetState();
                return layerForward(inputs); // 递归调用处理重置后的状态
            }
            
            // 计算更新门
            x_z = x.linear(w_z, b_z);
            h_z = state.linear(u_z, null);
            zGate = x_z.add(h_z).sigmoid();

            // 计算重置门
            x_r = x.linear(w_r, b_r);
            h_r = state.linear(u_r, null);
            rGate = x_r.add(h_r).sigmoid();

            // 重置前一状态
            resetState = rGate.mul(state);

            // 计算候选状态
            x_h = x.linear(w_h, b_h);
            h_h = resetState.linear(u_h, null);
            hCandidate = x_h.add(h_h).tanh();

            // 计算当前状态
            oneMinusZ = new Variable(NdArray.ones(zGate.getValue().getShape())).sub(zGate);
            state = zGate.mul(state).add(oneMinusZ.mul(hCandidate));
            stateValue = state.getValue();
        }

        return state;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];

        if (stateValue == null) {
            // 第一次前向传播
            // 计算更新门
            NdArray x_z = x.dot(w_z.getValue()).add(b_z.getValue().broadcastTo(x.getShape()));
            NdArray zGateValue = x_z.sigmoid();

            // 计算重置门
            NdArray x_r = x.dot(w_r.getValue()).add(b_r.getValue().broadcastTo(x.getShape()));
            NdArray rGateValue = x_r.sigmoid();

            // 计算候选状态
            NdArray x_h = x.dot(w_h.getValue()).add(b_h.getValue().broadcastTo(x.getShape()));
            NdArray hCandidateValue = x_h.tanh();

            // 计算当前状态
            NdArray oneMinusZ = NdArray.ones(zGateValue.getShape()).sub(zGateValue);
            stateValue = oneMinusZ.mul(hCandidateValue);
        } else {
            // 后续前向传播
            // 计算更新门
            NdArray x_z = x.dot(w_z.getValue()).add(b_z.getValue().broadcastTo(x.getShape()));
            NdArray h_z = stateValue.dot(u_z.getValue());
            NdArray zGateValue = x_z.add(h_z).sigmoid();

            // 计算重置门
            NdArray x_r = x.dot(w_r.getValue()).add(b_r.getValue().broadcastTo(x.getShape()));
            NdArray h_r = stateValue.dot(u_r.getValue());
            NdArray rGateValue = x_r.add(h_r).sigmoid();

            // 重置前一状态
            NdArray resetStateValue = rGateValue.mul(stateValue);

            // 计算候选状态
            NdArray x_h = x.dot(w_h.getValue()).add(b_h.getValue().broadcastTo(x.getShape()));
            NdArray h_h = resetStateValue.dot(u_h.getValue());
            NdArray hCandidateValue = x_h.add(h_h).tanh();

            // 计算当前状态
            NdArray oneMinusZ = NdArray.ones(zGateValue.getShape()).sub(zGateValue);
            stateValue = zGateValue.mul(stateValue).add(oneMinusZ.mul(hCandidateValue));
        }

        return stateValue;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // GRU反向传播实现
        // 根据GRU的数学公式计算梯度

        // 保存当前状态的梯度
        NdArray dhNext = yGrad;

        // 计算更新门的梯度
        NdArray dz = dhNext.mul(stateValue.sub(hCandidate.getValue()));
        // 应用sigmoid导数
        dz = dz.mul(zGate.getValue()).mul(NdArray.ones(zGate.getValue().getShape()).sub(zGate.getValue()));

        // 计算候选状态的梯度
        NdArray dhCandidate = dhNext.mul(oneMinusZ.getValue());
        // 应用tanh导数
        dhCandidate = dhCandidate.mul(NdArray.ones(hCandidate.getValue().getShape()).sub(hCandidate.getValue().square()));

        // 计算重置门的梯度
        NdArray dr = null;
        if (resetState != null) {
            dr = dhCandidate.dot(u_h.getValue().transpose()).mul(state.getValue());
            // 应用sigmoid导数
            dr = dr.mul(rGate.getValue()).mul(NdArray.ones(rGate.getValue().getShape()).sub(rGate.getValue()));
        } else {
            dr = NdArray.zeros(rGate.getValue().getShape());
        }

        // 计算输入x的梯度
        NdArray dx = null;
        NdArray dhPrev = null;

        // 计算参数梯度
        NdArray dw_z = null, du_z = null, db_z = null;
        NdArray dw_r = null, du_r = null, db_r = null;
        NdArray dw_h = null, du_h = null, db_h = null;

        if (resetState != null) {
            // 后续时间步的反向传播

            // 输入到更新门的梯度
            dw_z = inputs[0].getValue().transpose().dot(dz);
            du_z = state.getValue().transpose().dot(dz);
            db_z = dz.sumTo(b_z.getValue().getShape());

            // 输入到重置门的梯度
            dw_r = inputs[0].getValue().transpose().dot(dr);
            du_r = state.getValue().transpose().dot(dr);
            db_r = dr.sumTo(b_r.getValue().getShape());

            // 输入到候选状态的梯度
            dw_h = inputs[0].getValue().transpose().dot(dhCandidate);
            du_h = resetState.getValue().transpose().dot(dhCandidate);
            db_h = dhCandidate.sumTo(b_h.getValue().getShape());

            // 输入梯度
            dx = dz.dot(w_z.getValue().transpose())
                    .add(dr.dot(w_r.getValue().transpose()))
                    .add(dhCandidate.dot(w_h.getValue().transpose()));

            // 前一状态的梯度
            dhPrev = dz.dot(u_z.getValue().transpose())
                    .add(dr.dot(u_r.getValue().transpose()))
                    .add(dhCandidate.dot(u_h.getValue().transpose()).mul(rGate.getValue()));

            // 加上通过更新门传递的梯度
            dhPrev = dhPrev.add(dhNext.mul(zGate.getValue()));
        } else {
            // 第一个时间步的反向传播

            // 输入到更新门的梯度
            dw_z = inputs[0].getValue().transpose().dot(dz);
            db_z = dz.sumTo(b_z.getValue().getShape());
            du_z = null; // 第一个时间步没有前一状态

            // 输入到重置门的梯度
            dw_r = inputs[0].getValue().transpose().dot(dr);
            db_r = dr.sumTo(b_r.getValue().getShape());
            du_r = null; // 第一个时间步没有前一状态

            // 输入到候选状态的梯度
            dw_h = inputs[0].getValue().transpose().dot(dhCandidate);
            db_h = dhCandidate.sumTo(b_h.getValue().getShape());
            du_h = null; // 第一个时间步没有前一状态

            // 输入梯度
            dx = dz.dot(w_z.getValue().transpose())
                    .add(dr.dot(w_r.getValue().transpose()))
                    .add(dhCandidate.dot(w_h.getValue().transpose()));

            // 前一状态的梯度
            dhPrev = null; // 第一个时间步没有前一状态
        }

        // 返回梯度列表，顺序需要与参数顺序一致
        if (dhPrev != null) {
            return Arrays.asList(dx, dw_z, du_z, db_z, dw_r, du_r, db_r, dw_h, du_h, db_h);
        } else {
            // 第一个时间步的情况
            return Arrays.asList(dx, dw_z, db_z, dw_r, db_r, dw_h, db_h);
        }
    }
}