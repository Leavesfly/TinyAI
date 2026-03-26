package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.func.base.*;
import io.leavesfly.tinyai.func.loss.MeanSE;
import io.leavesfly.tinyai.func.loss.SoftmaxCE;
import io.leavesfly.tinyai.func.math.*;
import io.leavesfly.tinyai.func.matrix.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.Serializable;
import java.util.*;

/**
 * Variable - 自动微分系统的核心单元
 *
 * Variable 封装了值（NdArray）、梯度、计算历史三要素，是构建计算图的基本节点。
 * 每次算子调用都会在计算图中创建新节点，支持链式自动微分。
 *
 * 核心概念：
 * - 值（value）：存储数值的 NdArray
 * - 梯度（grad）：反向传播计算的梯度
 * - 创建者（creator）：生成该变量的 Function，用于追溯计算图
 *
 * Variable vs NdArray：
 * - NdArray：纯数值存储，用于数据和计算
 * - Variable：值 + 梯度 + 计算历史，用于模型参数和自动微分
 *
 * 计算图示例：
 * <pre>
 *     x ──┐
 *         ├──→ [Mul] ──→ z ──┐
 *     y ──┘                   ├──→ [Add] ──→ loss
 *                          w ──┘
 * </pre>
 *
 * 使用示例：
 * <pre>{@code
 * // 创建变量
 * Variable x = new Variable(NdArray.of(2.0f));
 * Variable y = new Variable(NdArray.of(3.0f));
 *
 * // 构建计算图
 * Variable z = x.mul(y).add(x.squ());  // z = x*y + x²
 *
 * // 反向传播
 * z.backward();
 *
 * // 查看梯度
 * x.getGrad();  // ∂z/∂x = y + 2x = 7
 * y.getGrad();  // ∂z/∂y = x = 2
 * }</pre>
 *
 * @author TinyAI Team
 * @see Function
 * @see NdArray
 */
public class Variable implements Serializable {

    private static final long serialVersionUID = 1L;

    // ==================== 常量定义 ====================

    /** LeakyReLU 默认负斜率 */
    public static final float DEFAULT_LEAKY_RELU_SLOPE = 0.01f;

    /** ELU 默认 alpha 参数 */
    public static final float DEFAULT_ELU_ALPHA = 1.0f;

    /** RMSNorm 默认 epsilon 值 */
    public static final float DEFAULT_RMS_NORM_EPS = 1e-6f;

    /** LogSoftmax 默认计算轴 */
    public static final int DEFAULT_LOG_SOFTMAX_AXIS = -1;

    /** 下三角矩阵默认对角线偏移 */
    public static final int DEFAULT_TRIL_DIAGONAL = 0;

    // ==================== 核心字段 ====================

    /** 变量名称，用于调试和可视化 */
    private String name;

    /** 变量的值 */
    private NdArray value;

    /** 变量的梯度 */
    private NdArray grad;

    /** 创建该变量的函数，用于构建计算图（transient 避免序列化循环） */
    private transient Function creator;

    /** 是否需要计算梯度 */
    protected boolean requireGrad = true;

    // ==================== 构造函数 ====================

    /**
     * 创建 Variable
     *
     * @param value NdArray 值，不能为 null
     */
    public Variable(NdArray value) {
        this(value, null, true);
    }

    /**
     * 创建标量 Variable
     *
     * @param number 数值
     */
    public Variable(Number number) {
        this(NdArray.of(requireNonNull(number, "number")), null, true);
    }

    /**
     * 创建带名称的 Variable
     *
     * @param value NdArray 值
     * @param name  变量名称
     */
    public Variable(NdArray value, String name) {
        this(value, name, true);
    }

    /**
     * 完整构造函数
     *
     * @param value       NdArray 值，不能为 null
     * @param name        变量名称
     * @param requireGrad 是否需要梯度
     */
    public Variable(NdArray value, String name, boolean requireGrad) {
        this.value = requireNonNull(value, "NdArray value");
        this.name = name;
        this.requireGrad = requireGrad;
    }

    /** 私有辅助方法：非空校验 */
    private static <T> T requireNonNull(T obj, String name) {
        if (obj == null) {
            throw new RuntimeException(name + " is null!");
        }
        return obj;
    }

    // ==================== 属性访问 ====================

    /** 获取变量值 */
    public NdArray getValue() {
        return value;
    }

    /** 设置变量值 */
    public void setValue(NdArray value) {
        this.value = value;
    }

    /** 获取梯度 */
    public NdArray getGrad() {
        return grad;
    }

    /**
     * 设置梯度
     *
     * @param grad 梯度，形状必须与 value 一致
     */
    public void setGrad(NdArray grad) {
        if (grad == null) {
            return;
        }
        if (!grad.getShape().equals(value.getShape())) {
            throw new RuntimeException("grad shape must equal value shape!");
        }
        this.grad = requireGrad ? grad : null;
    }

    /** 获取变量名称 */
    public String getName() {
        return name;
    }

    /** 设置变量名称（链式调用） */
    public Variable setName(String name) {
        this.name = name;
        return this;
    }

    /** 获取创建该变量的函数 */
    public Function getCreator() {
        return creator;
    }

    /** 设置创建函数（框架内部使用） */
    public void setCreator(Function creator) {
        this.creator = creator;
    }

    /** 是否需要计算梯度 */
    public boolean isRequireGrad() {
        return requireGrad;
    }

    /** 设置是否需要梯度（链式调用） */
    public Variable setRequireGrad(boolean requireGrad) {
        this.requireGrad = requireGrad;
        return this;
    }

    // ==================== 形状与维度 ====================

    /** 获取 Shape 对象 */
    public Shape getShape() {
        return value.getShape();
    }

    /** 获取维度数量 */
    public int ndim() {
        return value.getShape().getDimNum();
    }

    /**
     * 获取指定维度大小（支持负数索引）
     *
     * @param dim 维度索引，-1 表示最后一维
     */
    public int size(int dim) {
        Shape shape = value.getShape();
        int ndim = shape.getDimNum();

        // 处理负数索引
        int actualDim = dim < 0 ? ndim + dim : dim;

        if (actualDim < 0 || actualDim >= ndim) {
            throw new IndexOutOfBoundsException(String.format(
                    "Dimension out of range (expected [-%d, %d), got %d)", ndim, ndim, dim));
        }
        return shape.getDimension(actualDim);
    }

    /** 获取所有维度大小数组 */
    public int[] sizes() {
        return value.getShape().getShapeDims();
    }

    /** sizes() 的别名 */
    public int[] shape() {
        return sizes();
    }

    /** 获取元素总数 */
    public int getElementCount() {
        return value.getShape().size();
    }

    /** 是否为标量（单元素） */
    public boolean isScalar() {
        return value.getShape().size() == 1;
    }

    /** 是否为矩阵（2维） */
    public boolean isMatrix() {
        return value.getShape().isMatrix();
    }

    /** 是否为向量（1维） */
    public boolean isVector() {
        return value.getShape().getDimNum() == 1;
    }

    /** 获取数据类型（当前固定为 float32） */
    public String dtype() {
        return "float32";
    }

    // ==================== 自动微分 ====================

    /**
     * 反向传播 - 计算所有上游变量的梯度
     *
     * 从当前变量开始，沿计算图反向传播梯度。使用链式法则：
     * 对于 y = f(g(x))，有 ∂y/∂x = ∂y/∂g × ∂g/∂x
     *
     * 使用步骤：
     * 1. 构建计算图（前向传播）
     * 2. 调用 loss.backward()
     * 3. 获取梯度 var.getGrad()
     * 4. 下一轮迭代前调用 clearGrad()
     */
    public void backward() {
        // 使用 IdentityHashMap 提升性能（基于引用比较）
        Set<Variable> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        backwardInternal(visited);
    }

    /** 递归反向传播实现 */
    private void backwardInternal(Set<Variable> visited) {
        // 已访问则跳过（防止重复处理共享变量）
        if (visited.contains(this)) {
            return;
        }
        visited.add(this);

        if (!requireGrad) {
            this.grad = null;
            return;
        }

        // 初始化梯度为 1
        if (grad == null) {
            setGrad(NdArray.ones(value.getShape()));
        }

        Function func = creator;
        if (func == null) {
            return;
        }

        Variable[] funcInputs = func.getInputs();
        List<NdArray> grads = func.isMultiOutput()
                ? buildOutputGradsForMulti(func, this)
                : func.backward(grad);

        if (funcInputs.length != grads.size()) {
            throw new RuntimeException(String.format(
                    "backward grads size error! Function: %s, inputs: %d, grads: %d",
                    func.getClass().getSimpleName(), funcInputs.length, grads.size()));
        }

        for (int i = 0; i < funcInputs.length; i++) {
            Variable input = funcInputs[i];
            NdArray inputGrad = grads.get(i);

            // 梯度为 null 表示不可导，跳过
            if (inputGrad == null) {
                continue;
            }

            // 累加梯度（支持梯度复用）
            if (input.getGrad() != null) {
                input.setGrad(input.getGrad().add(inputGrad));
            } else {
                input.setGrad(inputGrad);
            }
            input.backwardInternal(visited);
        }
    }

    /**
     * 迭代式反向传播
     *
     * 使用栈模拟递归，避免深层网络栈溢出。
     */
    public void backwardIterative() {
        if (!requireGrad) {
            this.grad = null;
            return;
        }

        if (grad == null) {
            setGrad(NdArray.ones(value.getShape()));
        }

        Stack<Variable> stack = new Stack<>();
        stack.push(this);

        while (!stack.isEmpty()) {
            Variable current = stack.pop();
            Function func = current.getCreator();

            if (func == null) {
                continue;
            }

            Variable[] funcInputs = func.getInputs();
            List<NdArray> grads = func.isMultiOutput()
                    ? buildOutputGradsForMulti(func, current)
                    : func.backward(current.getGrad());

            if (funcInputs.length != grads.size()) {
                throw new RuntimeException("backward grads size error!");
            }

            for (int i = 0; i < funcInputs.length; i++) {
                Variable input = funcInputs[i];
                NdArray inputGrad = grads.get(i);

                if (inputGrad == null) {
                    continue;
                }

                if (input.getGrad() != null) {
                    input.setGrad(input.getGrad().add(inputGrad));
                } else {
                    input.setGrad(inputGrad);
                }

                if (input.getCreator() != null) {
                    stack.push(input);
                }
            }
        }
    }

    /** 为多输出函数构造梯度列表 */
    private List<NdArray> buildOutputGradsForMulti(Function func, Variable current) {
        Variable[] outs = func.getOutputs();
        if (outs == null || outs.length == 0) {
            throw new RuntimeException("Multi-output function has no outputs captured.");
        }

        List<NdArray> yGrads = new ArrayList<>(outs.length);
        for (Variable out : outs) {
            NdArray g = (out == current) ? current.getGrad() : out.getGrad();
            yGrads.add(g != null ? g : NdArray.zeros(out.getValue().getShape()));
        }
        return func.backwardMulti(yGrads);
    }

    /** 清理梯度 */
    public void clearGrad() {
        grad = null;
    }

    /** zeroGrad 的别名，与 PyTorch 风格一致 */
    public void zeroGrad() {
        clearGrad();
    }

    /** 切断计算图（用于 RNN 截断 BPTT） */
    public void unChainBackward() {
        Function func = creator;
        if (func != null) {
            Variable[] xs = func.getInputs();
            unChain();
            for (Variable x : xs) {
                x.unChainBackward();
            }
        }
    }

    private void unChain() {
        creator = null;
    }

    /** @deprecated 不再需要，backward() 内部使用局部变量 */
    @Deprecated
    public static void resetBackwardCounter() {
        // 保留用于向后兼容
    }

    // ==================== 四则运算 ====================

    /** 加法：this + other */
    public Variable add(Variable other) {
        return new Add().call(this, other);
    }

    /** 减法：this - other */
    public Variable sub(Variable other) {
        return new Sub().call(this, other);
    }

    /** 乘法（逐元素）：this * other */
    public Variable mul(Variable other) {
        return new Mul().call(this, other);
    }

    /** 除法（逐元素）：this / other */
    public Variable div(Variable other) {
        return new Div().call(this, other);
    }

    /** 取反：-this */
    public Variable neg() {
        return new Neg().call(this);
    }

    // ==================== 基础数学函数 ====================

    /** 平方：x² */
    public Variable squ() {
        return new Square().call(this);
    }

    /** squ() 的别名 */
    public Variable square() {
        return squ();
    }

    /** 幂运算：x^pow */
    public Variable pow(float pow) {
        return new Pow(pow).call(this);
    }

    /** 自然指数：e^x */
    public Variable exp() {
        return new Exp().call(this);
    }

    /** 自然对数：ln(x) */
    public Variable log() {
        return new Log().call(this);
    }

    /** 平方根：√x */
    public Variable sqrt() {
        return new Sqrt().call(this);
    }

    /** 正弦：sin(x) */
    public Variable sin() {
        return new Sin().call(this);
    }

    /** 余弦：cos(x) */
    public Variable cos() {
        return new Cos().call(this);
    }

    /** 双曲正切：tanh(x) */
    public Variable tanh() {
        return new Tanh().call(this);
    }

    /** Sigmoid：1/(1+e^{-x}) */
    public Variable sigmoid() {
        return new Sigmoid().call(this);
    }

    /** Softmax（全局归一化）：e^{xᵢ}/Σe^{xⱼ} */
    public Variable softMax() {
        return new SoftMax().call(this);
    }

    // ==================== 激活函数 ====================

    /** ReLU：max(0, x) */
    public Variable relu() {
        return new ReLU().call(this);
    }

    /** GELU：高斯误差线性单元 */
    public Variable gelu() {
        return new GELU().call(this);
    }

    /** SiLU（Swish）：x * sigmoid(x) */
    public Variable silu() {
        return new SiLU().call(this);
    }

    /** LeakyReLU：x > 0 ? x : negativeSlope * x */
    public Variable leakyRelu(float negativeSlope) {
        return new LeakyReLU(negativeSlope).call(this);
    }

    /** LeakyReLU（默认斜率 0.01） */
    public Variable leakyRelu() {
        return leakyRelu(DEFAULT_LEAKY_RELU_SLOPE);
    }

    /** ELU：x >= 0 ? x : alpha*(e^x - 1) */
    public Variable elu(float alpha) {
        return new ELU(alpha).call(this);
    }

    /** ELU（默认 alpha=1.0） */
    public Variable elu() {
        return elu(DEFAULT_ELU_ALPHA);
    }

    /** LogSoftmax：log(softmax(x)) */
    public Variable logSoftmax(int axis) {
        return new LogSoftmax(axis).call(this);
    }

    /** LogSoftmax（默认 axis=-1） */
    public Variable logSoftmax() {
        return logSoftmax(DEFAULT_LOG_SOFTMAX_AXIS);
    }

    /** 裁剪：将值限制在 [min, max] */
    public Variable clip(float min, float max) {
        return new Clip(min, max).call(this);
    }

    // ==================== 统计与归约 ====================

    /** 全局求和 */
    public Variable sum() {
        return new Sum().call(this);
    }

    /** 归约到指定形状（广播的逆操作） */
    public Variable sumTo(Shape shape) {
        return new SumTo(shape).call(this);
    }

    /** 沿指定轴计算均值 */
    public Variable mean(int axis, boolean keepdims) {
        return new Mean(axis, keepdims).call(this);
    }

    /** 沿指定轴计算最大值 */
    public Variable max(int axis, boolean keepdims) {
        return new Max(axis, keepdims).call(this);
    }

    /** 沿指定轴计算最小值 */
    public Variable min(int axis, boolean keepdims) {
        return new Min(axis, keepdims).call(this);
    }

    /** 沿指定轴计算方差 */
    public Variable var(int axis, boolean keepdims) {
        return new Variance(axis, keepdims).call(this);
    }

    // ==================== 矩阵与形状操作 ====================

    /** 矩阵乘法 */
    public Variable matMul(Variable other) {
        return new MatMul().call(this, other);
    }

    /** 批量矩阵乘法：[B,N,M] @ [B,M,P] → [B,N,P] */
    public Variable bmm(Variable other) {
        return new BMM().call(this, other);
    }

    /** 矩阵转置 */
    public Variable transpose() {
        return new Transpose().call(this);
    }

    /** 重塑形状 */
    public Variable reshape(Shape shape) {
        return new Reshape(shape).call(this);
    }

    /** 广播到目标形状 */
    public Variable broadcastTo(Shape shape) {
        return new BroadcastTo(shape).call(this);
    }

    /** 扩展维度为1的轴（不复制数据） */
    public Variable expand(Shape shape) {
        return new Expand(shape).call(this);
    }

    /** 沿各维度重复（复制数据） */
    public Variable repeat(int... repeats) {
        return new Repeat(repeats).call(this);
    }

    /** 移除所有大小为1的维度 */
    public Variable squeeze() {
        return new Squeeze().call(this);
    }

    /** 移除指定维度（大小必须为1） */
    public Variable squeeze(int dim) {
        return new Squeeze(dim).call(this);
    }

    /** 在指定位置插入大小为1的维度 */
    public Variable unsqueeze(int dim) {
        return new Unsqueeze(dim).call(this);
    }

    /** 维度重排 */
    public Variable permute(int... order) {
        return new Permute(order).call(this);
    }

    /** 下三角矩阵 */
    public Variable tril(int k) {
        return new Tril(k).call(this);
    }

    /** 下三角矩阵（默认 k=0） */
    public Variable tril() {
        return tril(DEFAULT_TRIL_DIAGONAL);
    }

    /** 线性变换：y = xW + b */
    public Variable linear(Variable w, Variable b) {
        return b == null
                ? new Linear().call(this, w)
                : new Linear().call(this, w, b);
    }

    // ==================== 索引与切片 ====================

    /** 行列索引取子集 */
    public Variable getItem(int[] rowSlices, int[] colSlices) {
        return new GetItem(rowSlices, colSlices).call(this);
    }

    /** getItem() 的别名 */
    public Variable slice(int[] rowSlices, int[] colSlices) {
        return getItem(rowSlices, colSlices);
    }

    /** 在指定维度选择单个索引 */
    public Variable select(int dim, int index) {
        return new Select(dim, index).call(this);
    }

    /** 在指定维度进行范围切片 [start, end) */
    public Variable sliceRange(int dim, int start, int end) {
        return new SliceRange(dim, start, end).call(this);
    }

    /** 沿指定维度按大小分割 */
    public Variable[] split(int splitSize, int dim) {
        return new SplitBySize(splitSize, dim).callMulti(this);
    }

    /** 沿指定维度按索引选取 */
    public Variable indexSelect(int dim, Variable index) {
        return new IndexSelect(dim).call(this, index);
    }

    /** 按索引分散累加 */
    public Variable scatterAdd(int dim, Variable index, Variable src) {
        return new ScatterAdd(dim).call(this, index, src);
    }

    /** Gather（Embedding 查找） */
    public Variable gather(Variable indices) {
        return new Gather().call(this, indices);
    }

    // ==================== 条件与比较 ====================

    /** 条件选择：condition ? x : y */
    public static Variable where(Variable condition, Variable x, Variable y) {
        return new Where().call(condition, x, y);
    }

    /** 掩码填充 */
    public Variable maskedFill(Variable mask, float value) {
        Variable fillValue = new Variable(value);
        fillValue.setRequireGrad(false);
        return Variable.where(mask, fillValue, this);
    }

    /** 大于比较（不参与梯度） */
    public Variable gt(Variable other) {
        Variable result = new Variable(value.gt(other.getValue()));
        result.setRequireGrad(false);
        return result;
    }

    /** 小于比较（不参与梯度） */
    public Variable lt(Variable other) {
        Variable result = new Variable(value.lt(other.getValue()));
        result.setRequireGrad(false);
        return result;
    }

    /** 等于比较（不参与梯度） */
    public Variable eq(Variable other) {
        Variable result = new Variable(value.eq(other.getValue()));
        result.setRequireGrad(false);
        return result;
    }

    // ==================== 归一化 ====================

    /** RMSNorm：x / sqrt(mean(x²) + eps) * weight */
    public Variable rmsNorm(int[] normalizedShape, float eps, Variable weight) {
        return new RMSNorm(normalizedShape, eps).call(this, weight);
    }

    /** RMSNorm（默认 eps=1e-6） */
    public Variable rmsNorm(int[] normalizedShape, Variable weight) {
        return rmsNorm(normalizedShape, DEFAULT_RMS_NORM_EPS, weight);
    }

    // ==================== 损失函数 ====================

    /** 均方误差损失 */
    public Variable meanSquaredError(Variable other) {
        return new MeanSE().call(this, other);
    }

    /** Softmax 交叉熵损失 */
    public Variable softmaxCrossEntropy(Variable other) {
        return new SoftmaxCE().call(this, other);
    }

    // ==================== 卷积操作 ====================

    /** 2D 卷积 */
    public Variable conv2d(Variable kernel, int stride, int padding) {
        return new Conv2d(stride, padding).call(this, kernel);
    }

    /** 2D 卷积（默认 stride=1, padding=0） */
    public Variable conv2d(Variable kernel) {
        return conv2d(kernel, 1, 0);
    }

    // ==================== 高级算子 ====================

    /**
     * Top-K 操作
     *
     * @return [values, indices]
     */
    public Variable[] topK(int k, int axis, boolean largest, boolean sorted) {
        return new TopK(k, axis, largest, sorted).callMulti(this);
    }

    /** Top-K（默认取最大值、排序） */
    public Variable[] topK(int k, int axis) {
        return topK(k, axis, true, true);
    }

    /** 旋转位置编码（RoPE） */
    public Variable applyRope(int dim, int maxSeqLen, float theta) {
        return new RotaryEmbedding(dim, maxSeqLen, theta).call(this);
    }

    /** RoPE（默认 theta=10000） */
    public Variable applyRope(int dim, int maxSeqLen) {
        return new RotaryEmbedding(dim, maxSeqLen).call(this);
    }

    // ==================== 工具方法 ====================

    /** 从计算图分离（停止梯度） */
    public Variable detach() {
        Variable result = new Detach().call(this);
        result.setRequireGrad(false);
        return result;
    }

    /** 深拷贝张量值 */
    public Variable clone() {
        return new Clone().call(this);
    }

    /** 创建同形状全1张量 */
    public Variable onesLike() {
        return new OnesLike().call(this);
    }

    /** 创建同形状全0张量 */
    public Variable zerosLike() {
        return new ZerosLike().call(this);
    }

    /** 创建同形状指定值张量 */
    public Variable fullLike(float value) {
        Variable result = new Variable(NdArray.like(getShape(), value));
        result.setRequireGrad(false);
        return result;
    }

    /** fullLike() 的别名 */
    public Variable fill(float value) {
        return fullLike(value);
    }

    /** 创建同形状空张量（返回全零） */
    public Variable newLike() {
        return zerosLike();
    }

    // ==================== 静态创建方法 ====================

    /** 全零张量 */
    public static Variable zeros(Shape shape) {
        return new Variable(NdArray.zeros(shape));
    }

    /** 全一张量 */
    public static Variable ones(Shape shape) {
        return new Variable(NdArray.ones(shape));
    }

    /** 指定值张量 */
    public static Variable full(Shape shape, float value) {
        return new Full(shape, value).call();
    }

    /** 均匀分布随机张量 [0, 1) */
    public static Variable rand(Shape shape) {
        Variable result = new Variable(NdArray.likeRandom(0, 1, shape));
        result.setRequireGrad(false);
        return result;
    }

    /** 标准正态分布随机张量 */
    public static Variable randn(Shape shape) {
        return new Variable(NdArray.randn(shape));
    }

    /** 沿指定维度拼接 */
    public static Variable cat(Variable[] variables, int dim) {
        return new Concat(dim).call(variables);
    }
}
