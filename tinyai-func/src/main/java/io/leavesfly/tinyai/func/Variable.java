package io.leavesfly.tinyai.func;


import io.leavesfly.tinyai.func.base.*;
import io.leavesfly.tinyai.func.loss.MeanSE;
import io.leavesfly.tinyai.func.loss.SoftmaxCE;
import io.leavesfly.tinyai.func.math.*;
import io.leavesfly.tinyai.func.matrix.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;
import java.util.Stack;

/**
 * 数学中的变量的抽象表示
 * <p>
 * 在TinyDL深度学习框架中，Variable类是对数学变量的抽象表示。
 * 它不仅包含变量的值(NdArray)，还包含变量的梯度、生成该变量的函数等信息。
 * Variable是自动微分系统的核心组件，负责构建和维护计算图。
 */
public class Variable implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 变量的名称
     * 用于标识变量，便于调试和可视化
     */
    private String name;

    /**
     * 变量的值
     * 存储变量的实际数值，使用NdArray表示
     */
    private NdArray value;

    /**
     * 变量的梯度
     * 存储反向传播计算得到的梯度值，用于参数更新
     */
    private NdArray grad;

    /**
     * 记录是什么函数生成的当前Variable
     * 指向生成该变量的函数，用于构建计算图
     * 使用transient关键字标记，序列化时不保存，避免循环引用
     */
    private transient Function creator;

    /**
     * 是否需要计算当前变量的梯度
     * 当设置为false时，反向传播过程中不会计算和存储该变量的梯度
     */
    private boolean requireGrad = true;

    /**
     * 构造函数
     * <p>
     * 使用指定的NdArray值创建Variable实例
     *
     * @param _value 变量的值，不能为null
     * @throws RuntimeException 当_value为null时抛出异常
     */
    public Variable(NdArray _value) {
        if (Objects.isNull(_value)) {
            throw new RuntimeException("NdArray value is null!");
        }
        this.value = _value;
    }

    public Variable(Number number) {
        if (Objects.isNull(number)) {
            throw new RuntimeException("NdArray number is null!");
        }
        this.value = NdArray.of(number);
    }

    public Variable(NdArray _value, String _name) {
        if (Objects.isNull(_value)) {
            throw new RuntimeException("NdArray _value is null!");
        }
        this.value = _value;
        this.name = _name;
    }

    public Variable(NdArray _value, String _name, boolean _requireGrad) {
        if (Objects.isNull(_value)) {
            throw new RuntimeException("NdArray _value is null!");
        }
        this.value = _value;
        this.name = _name;
        this.requireGrad = _requireGrad;
    }

    public Variable setRequireGrad(boolean _requireGrad) {
        this.requireGrad = _requireGrad;
        return this;
    }

    /**
     * 获取变量是否需要计算梯度
     *
     * @return 是否需要计算梯度
     */
    public boolean isRequireGrad() {
        return requireGrad;
    }

    /**
     * 变量的反向传播（递归实现）
     * <p>
     * 根据正向传播时构建的计算图，从当前变量开始反向传播计算每个变量的梯度。
     * 如果变量不需要计算梯度，则直接返回。
     * 如果梯度未初始化，则初始化为全1的数组。
     * 然后递归地调用生成该变量的函数的backward方法计算输入变量的梯度。
     */
    public void backward() {
        if (!requireGrad) {
            this.grad = null;
            return;
        }
        //初始化为1
        if (Objects.isNull(grad)) {
            setGrad(NdArray.ones(this.getValue().getShape()));
        }
        // 当前采用的是递归调用，为了效率可用堆栈循环
        Function _creator = creator;
        if (!Objects.isNull(_creator)) {
            Variable[] _inputs = _creator.getInputs();
            List<NdArray> grads = _creator.backward(grad);
            if (_inputs.length != grads.size()) {
                throw new RuntimeException("Variable backward grads size error!");
            }
            int index = 0;
            for (Variable input : _inputs) {
                // 累加梯度而不是直接设置，支持梯度复用
                if (input.getGrad() != null) {
                    input.setGrad(input.getGrad().add(grads.get(index)));
                } else {
                    input.setGrad(grads.get(index));
                }
                input.backward();
                index++;
            }
        }
    }

    /**
     * 变量的反向传播（迭代实现）
     * <p>
     * 使用栈来实现迭代的反向传播，避免递归调用可能导致的栈溢出问题。
     * 特别适用于深层网络或RNN等场景。
     */
    public void backwardIterative() {
        if (!requireGrad) {
            this.grad = null;
            return;
        }

        // 初始化梯度为1
        if (Objects.isNull(grad)) {
            setGrad(NdArray.ones(this.getValue().getShape()));
        }

        // 使用栈来模拟递归过程
        Stack<Variable> stack = new Stack<>();
        stack.push(this);

        while (!stack.isEmpty()) {
            Variable currentVar = stack.pop();

            Function currentCreator = currentVar.getCreator();
            if (Objects.isNull(currentCreator)) {
                continue;
            }

            Variable[] inputs = currentCreator.getInputs();
            List<NdArray> grads = currentCreator.backward(currentVar.getGrad());

            if (inputs.length != grads.size()) {
                throw new RuntimeException("Variable backward grads size error!");
            }

            for (int i = 0; i < inputs.length; i++) {
                Variable input = inputs[i];
                NdArray grad = grads.get(i);

                // 累加梯度而不是直接设置，支持梯度复用
                if (input.getGrad() != null) {
                    input.setGrad(input.getGrad().add(grad));
                } else {
                    input.setGrad(grad);
                }

                // 如果输入变量有创建者函数，将其加入栈中继续处理
                if (input.getCreator() != null) {
                    stack.push(input);
                }
            }
        }
    }

    /**
     * 切断计算图
     * <p>
     * 用于RNN中切断计算图，防止梯度回传过长导致的梯度消失或爆炸问题。
     * 该方法会清除当前变量的creator引用，并递归地对输入变量调用unChainBackward。
     */
    public void unChainBackward() {
        Function creatorFunc = creator;
        if (!Objects.isNull(creatorFunc)) {
            Variable[] xs = creatorFunc.getInputs();
            unChain();
            for (Variable x : xs) {
                x.unChainBackward();
            }
        }
    }

    /**
     * 清理梯度
     * <p>
     * 将变量的梯度设置为null，释放梯度占用的内存。
     * 通常在每次训练迭代开始前调用，以确保梯度不会累积。
     */
    public void clearGrad() {
        grad = null;
    }

    public NdArray getValue() {
        return value;
    }

    private void unChain() {
        creator = null;
    }

    public void setValue(NdArray value) {
        this.value = value;
    }

    public NdArray getGrad() {
        return grad;
    }

    public void setGrad(NdArray _grad) {
        if (_grad == null) {
            return;
        }
        if (!_grad.getShape().equals(value.getShape())) {
            throw new RuntimeException("_grad shape must equal value shape!");
        }
        if (requireGrad) {
            this.grad = _grad;
        } else {
            this.grad = null;
        }
    }

    public Function getCreator() {
        return creator;
    }

    public void setCreator(Function creator) {
        this.creator = creator;
    }

    public String getName() {
        return name;
    }

    public Variable setName(String name) {
        this.name = name;
        return this;
    }

    //    # =============================================================================
//            # 1，四则运算
//    # =============================================================================

    /**
     * 加法运算
     * <p>
     * 对当前变量与另一个变量执行加法运算
     *
     * @param other 参与运算的另一个变量
     * @return 加法运算结果的新变量
     */
    public Variable add(Variable other) {
        Function function = new Add();
        return function.call(this, other);
    }

    /**
     * 减法运算
     * <p>
     * 对当前变量与另一个变量执行减法运算
     *
     * @param other 参与运算的另一个变量
     * @return 减法运算结果的新变量
     */
    public Variable sub(Variable other) {
        Function function = new Sub();
        return function.call(this, other);
    }

    /**
     * 乘法运算
     * <p>
     * 对当前变量与另一个变量执行乘法运算
     *
     * @param other 参与运算的另一个变量
     * @return 乘法运算结果的新变量
     */
    public Variable mul(Variable other) {
        Function function = new Mul();
        return function.call(this, other);
    }

    /**
     * 除法运算
     * <p>
     * 对当前变量与另一个变量执行除法运算
     *
     * @param other 参与运算的另一个变量
     * @return 除法运算结果的新变量
     */
    public Variable div(Variable other) {
        Function function = new Div();
        return function.call(this, other);
    }

    /**
     * 取反操作
     * <p>
     * 对变量执行取反运算，返回一个新的变量，其值为原变量值的相反数。
     *
     * @return 取反后的变量
     */
    public Variable neg() {
        Function function = new Neg();
        return function.call(this);
    }

    //    # =============================================================================
//            # 2，基本数学函数
//    # =============================================================================

    /**
     * 平方运算
     * <p>
     * 对变量执行平方运算
     *
     * @return 平方运算结果的新变量
     */
    public Variable squ() {
        Function function = new Squ();
        return function.call(this);
    }

    /**
     * 幂运算
     * <p>
     * 对变量执行幂运算
     *
     * @param pow 幂指数
     * @return 幂运算结果的新变量
     */
    public Variable pow(float pow) {
        Function function = new Pow(pow);
        return function.call(this);
    }

    /**
     * 指数运算
     * <p>
     * 对变量执行自然指数运算(e^x)
     *
     * @return 指数运算结果的新变量
     */
    public Variable exp() {
        Function function = new Exp();
        return function.call(this);
    }

    /**
     * 正弦运算
     * <p>
     * 对变量执行正弦运算
     *
     * @return 正弦运算结果的新变量
     */
    public Variable sin() {
        Function function = new Sin();
        return function.call(this);
    }

    /**
     * 余弦运算
     * <p>
     * 对变量执行余弦运算
     *
     * @return 余弦运算结果的新变量
     */
    public Variable cos() {
        Function function = new Cos();
        return function.call(this);
    }

    /**
     * 对数运算
     * <p>
     * 对变量执行自然对数运算(ln(x))
     *
     * @return 对数运算结果的新变量
     */
    public Variable log() {
        Function function = new Log();
        return function.call(this);
    }

    /**
     * 双曲正切运算
     * <p>
     * 对变量执行双曲正切运算
     *
     * @return 双曲正切运算结果的新变量
     */
    public Variable tanh() {
        Function function = new Tanh();
        return function.call(this);
    }

    /**
     * Sigmoid运算
     * <p>
     * 对变量执行Sigmoid运算，将值映射到(0,1)区间
     *
     * @return Sigmoid运算结果的新变量
     */
    public Variable sigmoid() {
        Function function = new Sigmoid();
        return function.call(this);
    }

    /**
     * SoftMax运算
     * <p>
     * 对变量执行SoftMax运算，常用于多分类问题的输出层
     *
     * @return SoftMax运算结果的新变量
     */
    public Variable softMax() {
        Function function = new SoftMax();
        return function.call(this);
    }

    /**
     * ReLU运算
     * <p>
     * 对变量执行ReLU运算，将负值置为0
     *
     * @return ReLU运算结果的新变量
     */
    public Variable relu() {
        Function function = new ReLu();
        return function.call(this);
    }

    /**
     * 裁剪运算
     * <p>
     * 将变量的值限制在指定范围内
     *
     * @param min 最小值
     * @param max 最大值
     * @return 裁剪后的新变量
     */
    public Variable clip(float min, float max) {
        Function function = new Clip(min, max);
        return function.call(this);
    }

    /**
     * 最大值运算
     * <p>
     * 沿指定轴计算变量的最大值
     *
     * @param _axis     轴索引
     * @param _keepdims 是否保持维度
     * @return 最大值运算结果的新变量
     */
    public Variable max(int _axis, boolean _keepdims) {
        Function function = new Max(_axis, _keepdims);
        return function.call(this);
    }

    /**
     * 最小值运算
     * <p>
     * 沿指定轴计算变量的最小值
     *
     * @param _axis     轴索引
     * @param _keepdims 是否保持维度
     * @return 最小值运算结果的新变量
     */
    public Variable min(int _axis, boolean _keepdims) {
        Function function = new Min(_axis, _keepdims);
        return function.call(this);
    }

    //    # =============================================================================
//            # 3，张量的变形操作
//    # =============================================================================

    /**
     * 广播操作
     * <p>
     * 将变量广播到指定形状
     *
     * @param shape 目标形状
     * @return 广播后的新变量
     */
    public Variable broadcastTo(Shape shape) {
        Function function = new BroadcastTo(shape);
        return function.call(this);
    }

    /**
     * 矩阵乘法
     * <p>
     * 对当前变量与另一个变量执行矩阵乘法运算
     *
     * @param other 参与运算的另一个变量
     * @return 矩阵乘法结果的新变量
     */
    public Variable matMul(Variable other) {
        Function function = new MatMul();
        return function.call(this, other);
    }

    /**
     * 重塑操作
     * <p>
     * 改变变量的形状
     *
     * @param shape 新的形状
     * @return 重塑后的新变量
     */
    public Variable reshape(Shape shape) {
        Function function = new Reshape(shape);
        return function.call(this);
    }

    /**
     * 求和运算
     * <p>
     * 对变量的所有元素求和
     *
     * @return 求和结果的新变量
     */
    public Variable sum() {
        Function function = new Sum();
        return function.call(this);
    }

    /**
     * 求和到指定形状
     * <p>
     * 将变量求和到指定形状
     *
     * @param shape 目标形状
     * @return 求和后的新变量
     */
    public Variable sumTo(Shape shape) {
        Function function = new SumTo(shape);
        return function.call(this);
    }

    /**
     * 转置操作
     * <p>
     * 对变量执行转置操作
     *
     * @return 转置后的新变量
     */
    public Variable transpose() {
        Function function = new Transpose();
        return function.call(this);
    }

    /**
     * 线性变换
     * <p>
     * 对变量执行线性变换 y = xW + b
     *
     * @param w 权重变量
     * @param b 偏置变量，可为null
     * @return 线性变换结果的新变量
     */
    public Variable linear(Variable w, Variable b) {
        Function function = new Linear();
        if (Objects.isNull(b)) {
            return function.call(this, w);
        }
        return function.call(this, w, b);
    }

    /**
     * 索引操作
     * <p>
     * 根据指定的行列索引获取变量的子集
     *
     * @param _rowSlices 行索引数组
     * @param _colSlices 列索引数组
     * @return 索引操作结果的新变量
     */
    public Variable getItem(int[] _rowSlices, int[] _colSlices) {
        Function function = new GetItem(_rowSlices, _colSlices);
        return function.call(this);
    }

    //    # =============================================================================
//            # 4，loss函数
//    # =============================================================================

    /**
     * 均方误差损失
     * <p>
     * 计算当前变量与目标变量之间的均方误差损失
     *
     * @param other 目标变量
     * @return 均方误差损失值
     */
    public Variable meanSquaredError(Variable other) {
        Function function = new MeanSE();
        return function.call(this, other);
    }

    /**
     * Softmax交叉熵损失
     * <p>
     * 计算当前变量与目标变量之间的Softmax交叉熵损失
     *
     * @param other 目标变量
     * @return Softmax交叉熵损失值
     */
    public Variable softmaxCrossEntropy(Variable other) {
        return new SoftmaxCE().call(this, other);
    }
}