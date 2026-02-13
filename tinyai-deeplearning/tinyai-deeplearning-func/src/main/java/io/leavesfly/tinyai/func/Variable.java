package io.leavesfly.tinyai.func;


import io.leavesfly.tinyai.func.base.*;
import io.leavesfly.tinyai.func.loss.MeanSE;
import io.leavesfly.tinyai.func.loss.SoftmaxCE;
import io.leavesfly.tinyai.func.math.*;
import io.leavesfly.tinyai.func.math.Mean;
import io.leavesfly.tinyai.func.math.Variance;
import io.leavesfly.tinyai.func.math.Sqrt;
import io.leavesfly.tinyai.func.matrix.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.Serializable;
import java.util.*;

/**
 * Variable - 自动微分系统的核心，构建计算图的基本单元
 *
 * <p><b>核心概念：</b>
 * Variable（变量）是深度学习框架中最基础的概念之一。它不仅仅是一个数值，
 * 而是封装了<b>值</b>、<b>梯度</b>和<b>计算历史</b>的完整对象。
 * 这种设计使得自动微分（Automatic Differentiation）成为可能。
 *
 * <p><b>Variable vs NdArray：</b>
 * <table border="1">
 *   <tr><th>特性</th><th>NdArray</th><th>Variable</th></tr>
 *   <tr><td>存储内容</td><td>纯数值</td><td>数值 + 梯度 + 计算历史</td></tr>
 *   <tr><td>是否可微</td><td>否</td><td>是</td></tr>
 *   <tr><td>使用场景</td><td>数据存储、数值计算</td><td>模型参数、中间结果</td></tr>
 *   <tr><td>类比</td><td>float[] / numpy.array</td><td>torch.Tensor / tf.Variable</td></tr>
 * </table>
 *
 * <p><b>计算图（Computational Graph）：</b>
 * <pre>
 *     x ──┐
 *         ├──→ [乘法] ──→ z ──┐
 *     y ──┘                   ├──→ [加法] ──→ loss
 *                         w ──┘
 * </pre>
 * 每个 Variable 记录了它是如何被计算出来的（creator函数），
 * 这形成了一个有向无环图（DAG），反向传播时沿着这个图计算梯度。
 *
 * <p><b>快速入门示例：</b>
 * <pre>{@code
 * // 1. 创建变量（需要梯度的参数）
 * Variable x = new Variable(NdArray.of(2.0f));  // x = 2
 * Variable y = new Variable(NdArray.of(3.0f));  // y = 3
 *
 * // 2. 构建计算图
 * Variable z = x.mul(y);      // z = x * y = 6
 * Variable w = z.add(x);      // w = z + x = 8
 *
 * // 3. 反向传播
 * w.backward();               // 计算所有梯度
 *
 * // 4. 查看梯度
 * System.out.println(x.getGrad());  // ∂w/∂x = y + 1 = 4
 * System.out.println(y.getGrad());  // ∂w/∂y = x = 2
 * }</pre>
 *
 * <p><b>自动微分原理：</b>
 * <ol>
 *   <li><b>前向传播：</b>计算输出值，同时记录计算图</li>
 *   <li><b>反向传播：</b>从输出开始，沿着计算图反向传播梯度</li *   <li><b>链式法则：</b>复合函数的导数 = 外层导数 × 内层导数</li>
 * </ol>
 *
 * <p><b>关键术语：</b>
 * <ul>
 *   <li><b>RequireGrad：</b>是否需要计算该变量的梯度（模型参数设为true，输入数据设为false）</li>
 *   <li><b>Creator：</b>生成该变量的函数，用于反向传播时找到上游节点</li>
 *   <li><b>Leaf Node：</b>叶子节点，通常是用户直接创建的变量（如模型参数）</li>
 * </ul>
 *
 * <p><b>Java vs PyTorch 对比：</b>
 * <table border="1">
 *   <tr><th>操作</th><th>PyTorch</th><th>TinyAI (Java)</th></tr>
 *   <tr><td>创建变量</td><td>{@code x = torch.tensor(2.0, requires_grad=True)}</td><td>{@code new Variable(NdArray.of(2.0f))}</td></tr>
 *   <tr><td>查看值</td><td>{@code x.item()}</td><td>{@code x.getValue()}</td></tr>
 *   <tr><td>查看梯度</td><td>{@code x.grad}</td><td>{@code x.getGrad()}</td></tr>
 *   <tr><td>反向传播</td><td>{@code loss.backward()}</td><td>{@code loss.backward()}</td></tr>
 *   <tr><td>清零梯度</td><td>{@code x.grad.zero_()}</td><td>{@code x.zeroGrad()}</td></tr>
 * </table>
 *
 * @author TinyAI Team
 * @see Function
 * @see NdArray
 */
public class Variable implements Serializable {

    private static final long serialVersionUID = 1L;

    // =============================================================================
    // 常量定义
    // =============================================================================
    
    /**
     * 激活函数默认参数 - LeakyReLU 负斜率
     */
    public static final float DEFAULT_LEAKY_RELU_SLOPE = 0.01f;
    
    /**
     * 激活函数默认参数 - ELU alpha 参数
     */
    public static final float DEFAULT_ELU_ALPHA = 1.0f;
    
    /**
     * 归一化默认参数 - RMS Norm epsilon 值
     */
    public static final float DEFAULT_RMS_NORM_EPS = 1e-6f;
    
    /**
     * LogSoftmax 默认计算轴
     */
    public static final int DEFAULT_LOG_SOFTMAX_AXIS = -1;
    
    /**
     * 下三角矩阵默认对角线偏移
     */
    public static final int DEFAULT_TRIL_DIAGONAL = 0;

    // =============================================================================
    // 核心字段
    // =============================================================================

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
    protected boolean requireGrad = true;

    /**
     * 构造函数 - 创建一个新的Variable实例
     *
     * <p><b>使用场景：</b>
     * 这是创建Variable的最常用方式，适用于模型参数、输入数据等。
     *
     * <p><b>示例：</b>
     * <pre>{@code
     * // 创建模型权重参数
     * Variable weights = new Variable(NdArray.randn(Shape.of(784, 256)));
     * weights.setName("layer1_weights");
     *
     * // 创建输入数据（不需要梯度）
     * Variable input = new Variable(NdArray.of(data));
     * input.setRequireGrad(false);  // 输入数据不需要计算梯度
     * }</pre>
     *
     * @param _value 变量的值，使用NdArray存储，不能为null
     * @throws RuntimeException 当_value为null时抛出异常
     * @see #Variable(NdArray, String)
     * @see #Variable(NdArray, String, boolean)
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

    // =============================================================================
    // 形状和维度相关的便捷属性方法（类似 PyTorch 的设计）
    // 避免在 Module 中直接调用 getValue().getShape()
    // 注意：getShape() 方法已在其他位置定义
    // =============================================================================

    /**
     * 获取变量的维度数量
     * <p>
     * 例如：标量返回1（TinyAI中标量也是1维），向量返回1，矩阵返回2，3D张量返回3
     *
     * @return 维度数量
     */
    public int ndim() {
        return value.getShape().getDimNum();
    }

    /**
     * 获取指定维度的大小
     * <p>
     * 支持负数索引：-1 表示最后一维，-2 表示倒数第二维
     *
     * @param dim 维度索引
     * @return 该维度的大小
     */
    public int size(int dim) {
        Shape shape = value.getShape();
        int ndim = shape.getDimNum();
        
        // 处理负数索引
        if (dim < 0) {
            dim = ndim + dim;
        }
        
        if (dim < 0 || dim >= ndim) {
            throw new IndexOutOfBoundsException(
                    String.format("Dimension out of range (expected to be in range of [-%d, %d), but got %d)",
                            ndim, ndim, dim));
        }
        
        return shape.getDimension(dim);
    }

    /**
     * 获取所有维度的大小数组
     * <p>
     * 返回一个包含所有维度大小的数组
     *
     * @return 维度大小数组
     */
    public int[] sizes() {
        return value.getShape().getShapeDims();
    }

    /**
     * 获取张量中元素的总数
     *
     * @return 元素总数
     */
    public int numel() {
        return value.getShape().size();
    }

    /**
     * 判断变量是否是标量（只有一个元素）
     *
     * @return 如果是标量返回true
     */
    public boolean isScalar() {
        return value.getShape().size() == 1;
    }

    /**
     * 判断变量是否是矩阵（2维张量）
     *
     * @return 如果是矩阵返回true
     */
    public boolean isMatrix() {
        return value.getShape().isMatrix();
    }

    /**
     * 判断变量是否是向量（1维张量）
     *
     * @return 如果是向量返回true
     */
    public boolean isVector() {
        return value.getShape().getDimNum() == 1;
    }

    // 访问标记，防止重复处理
    private static Set<Variable> visitedInBackward = new HashSet<>();
    
    /**
     * 重置反向传播访问标记
     */
    public static void resetBackwardCounter() {
        visitedInBackward.clear();
    }
    
    /**
     * 反向传播 - 自动计算所有相关变量的梯度
     *
     * <p><b>核心原理 - 链式法则（Chain Rule）：</b>
     * 如果 y = f(g(x))，那么 ∂y/∂x = ∂y/∂g × ∂g/∂x
     * <pre>
     * 计算图示例：
     *     x ──┐
     *         ├──→ [*] ──→ z ──┐
     *     y ──┘                ├──→ [+] ──→ loss
     *                        w ──┘
     *
     * 反向传播过程：
     * 1. 从 loss 开始，初始化 grad = 1
     * 2. 传播到 [+] 节点：∂loss/∂z = 1, ∂loss/∂w = 1
     * 3. 传播到 [*] 节点：∂loss/∂x = ∂loss/∂z × y = y
     *                     ∂loss/∂y = ∂loss/∂z × x = x
     * </pre>
     *
     * <p><b>使用步骤：</b>
     * <ol>
     *   <li>构建计算图（前向传播）</li>
     *   <li>调用 loss.backward() 触发反向传播</li>
     *   <li>使用 var.getGrad() 获取各变量的梯度</li>
     *   <li>使用 optimizer.step() 更新参数</li>
     *   <li>调用 zeroGrad() 清零梯度（准备下一轮）</li>
     * </ol>
     *
     * <p><b>完整示例：</b>
     * <pre>{@code
     * // 1. 创建变量
     * Variable x = new Variable(NdArray.of(2.0f));
     * Variable y = new Variable(NdArray.of(3.0f));
     *
     * // 2. 前向传播
     * Variable z = x.mul(y).add(x.square());  // z = x*y + x²
     *
     * // 3. 反向传播
     * z.backward();
     *
     * // 4. 查看梯度
     * System.out.println("∂z/∂x = " + x.getGrad());  // y + 2x = 3 + 4 = 7
     * System.out.println("∂z/∂y = " + y.getGrad());  // x = 2
     * }</pre>
     *
     * <p><b>注意事项：</b>
     * <ul>
     *   <li><b>梯度累积：</b>默认会累加梯度，每次迭代前需要调用 zeroGrad()</li>
     *   <li><b>内存管理：</b>计算图在backward后会被释放，如需保留请设置retain_graph</li>
     *   <li><b>叶子节点：</b>只有叶子节点（直接创建的Variable）的梯度会被保留</li>
     * </ul>
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>PyTorch: {@code loss.backward()}</li>
     *   <li>TensorFlow: {@code tape.gradient(loss, variables)}</li>
     * </ul>
     *
     * @see #zeroGrad()
     * @see #getGrad()
     * @see #setRequireGrad(boolean)
     */
    public void backward() {
        // 重置访问记录，避免内存泄漏
        visitedInBackward.clear();
        backwardInternal();
        // backward完成后立即清空，释放引用
        visitedInBackward.clear();
    }
    
    /**
     * 内部递归backward实现
     */
    private void backwardInternal() {
        // 如果已经访问过，跳过（防止重复处理共享变量）
        if (visitedInBackward.contains(this)) {
            return;
        }
        visitedInBackward.add(this);

        if (!requireGrad) {
            this.grad = null;
            return;
        }
        // 初始化为1
        if (Objects.isNull(grad)) {
            setGrad(NdArray.ones(this.getValue().getShape()));
        }
        // 当前采用的是递归调用，为了效率可用堆栈循环
        Function _creator = creator;
        if (!Objects.isNull(_creator)) {
            Variable[] _inputs = _creator.getInputs();
            
            List<NdArray> grads = _creator.isMultiOutput()
                    ? buildOutputGradsForMulti(_creator, this)
                    : _creator.backward(grad);

            if (_inputs.length != grads.size()) {
                throw new RuntimeException("Variable backward grads size error!");
            }

            int index = 0;
            for (Variable input : _inputs) {
                NdArray inputGrad = grads.get(index);
                // 如果梯度为null，跳过该输入（例如索引不可导）
                if (inputGrad == null) {
                    index++;
                    continue;
                }
                
                // 累加梯度而不是直接设置，支持梯度复用
                if (input.getGrad() != null) {
                    input.setGrad(input.getGrad().add(inputGrad));
                } else {
                    input.setGrad(inputGrad);
                }
                input.backwardInternal();
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
            List<NdArray> grads = currentCreator.isMultiOutput()
                    ? buildOutputGradsForMulti(currentCreator, currentVar)
                    : currentCreator.backward(currentVar.getGrad());

            if (inputs.length != grads.size()) {
                throw new RuntimeException("Variable backward grads size error!");
            }

            for (int i = 0; i < inputs.length; i++) {
                Variable input = inputs[i];
                NdArray grad = grads.get(i);
                
                // 如果梯度为null，跳过该输入（例如索引不可导）
                if (grad == null) {
                    continue;
                }

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
     * 为多输出函数构造上游梯度列表
     * <p>
     * 该方法用于处理具有多个输出的函数(如split操作)的反向传播
     * 
     * @param creatorFunc 创建当前变量的函数
     * @param currentVar 当前正在反向传播的变量
     * @return 所有输出变量的梯度列表
     */
    private List<NdArray> buildOutputGradsForMulti(Function creatorFunc, Variable currentVar) {
        Variable[] outs = creatorFunc.getOutputs();
        if (Objects.isNull(outs) || outs.length == 0) {
            throw new RuntimeException("Multi-output function has no outputs captured.");
        }
        List<NdArray> yGrads = new ArrayList<>(outs.length);
        for (Variable out : outs) {
            NdArray g = (out == currentVar) ? currentVar.getGrad() : out.getGrad();
            if (Objects.isNull(g)) {
                g = NdArray.zeros(out.getValue().getShape());
            }
            yGrads.add(g);
        }
        return creatorFunc.backwardMulti(yGrads);
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

    /**
     * 获取变量的值
     * 
     * @return 变量的NdArray值
     */
    public NdArray getValue() {
        return value;
    }

    /**
     * 切断计算图连接
     * <p>
     * 将creator设置为null,断开与前驱节点的连接
     */
    private void unChain() {
        creator = null;
    }

    /**
     * 设置变量的值
     * 
     * @param value 新的NdArray值
     */
    public void setValue(NdArray value) {
        this.value = value;
    }

    /**
     * 获取变量的梯度
     * 
     * @return 变量的梯度,如果未计算则为null
     */
    public NdArray getGrad() {
        return grad;
    }

    /**
     * 设置变量的梯度
     * <p>
     * 梯度的形状必须与变量值的形状一致
     * 
     * @param _grad 要设置的梯度
     * @throws RuntimeException 当梯度形状与变量形状不匹配时抛出
     */
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

    /**
     * 获取变量的形状
     * 
     * @return 变量的Shape对象
     */
    public Shape getShape() {
        return value.getShape();
    }

    /**
     * 获取创建该变量的函数
     * 
     * @return 创建函数,如果是叶子节点则为null
     */
    public Function getCreator() {
        return creator;
    }

    /**
     * 设置创建该变量的函数
     * <p>
     * 用于构建计算图,记录变量的创建来源
     * 
     * @param creator 创建函数
     */
    public void setCreator(Function creator) {
        this.creator = creator;
    }

    /**
     * 获取变量名称
     * 
     * @return 变量名称
     */
    public String getName() {
        return name;
    }

    /**
     * 设置变量名称
     * <p>
     * 支持链式调用
     * 
     * @param name 变量名称
     * @return 当前Variable实例
     */
    public Variable setName(String name) {
        this.name = name;
        return this;
    }

    // =============================================================================
    // 四则运算操作
    // =============================================================================
    // 提供基础的算术运算:加、减、乘、除、取反
    // 所有运算都会创建新的Variable并构建计算图以支持自动微分
    // =============================================================================

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

    // =============================================================================
    // 数学函数操作
    // =============================================================================
    // 提供常用数学函数:指数、对数、三角函数、激活函数等
    // 包括:exp, log, sin, cos, tanh, sigmoid, relu, gelu等
    // =============================================================================

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
     * GELU激活函数运算
     * <p>
     * Gaussian Error Linear Unit，常用于Transformer模型
     * GELU(x) = x * Φ(x) ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     *
     * @return GELU运算结果的新变量
     */
    public Variable gelu() {
        Function function = new GELU();
        return function.call(this);
    }

    /**
     * SiLU激活函数运算（Swish）
     * <p>
     * Sigmoid Linear Unit，自门控激活函数
     * SiLU(x) = x * sigmoid(x)
     *
     * @return SiLU运算结果的新变量
     */
    public Variable silu() {
        Function function = new SiLU();
        return function.call(this);
    }

    /**
     * LeakyReLU激活函数运算
     * <p>
     * ReLU的改进版本，解决神经元死亡问题
     * LeakyReLU(x) = max(negative_slope * x, x)
     *
     * @param negativeSlope 负斜率参数，默认0.01
     * @return LeakyReLU运算结果的新变量
     */
    public Variable leakyRelu(float negativeSlope) {
        Function function = new LeakyReLU(negativeSlope);
        return function.call(this);
    }

    /**
     * LeakyReLU激活函数运算（默认negativeSlope=0.01）
     *
     * @return LeakyReLU运算结果的新变量
     */
    public Variable leakyRelu() {
        return leakyRelu(DEFAULT_LEAKY_RELU_SLOPE);
    }

    /**
     * ELU激活函数运算
     * <p>
     * Exponential Linear Unit，具有负值饱和特性
     * ELU(x) = x if x >= 0, else alpha * (exp(x) - 1)
     *
     * @param alpha alpha参数，默认1.0
     * @return ELU运算结果的新变量
     */
    public Variable elu(float alpha) {
        Function function = new ELU(alpha);
        return function.call(this);
    }

    /**
     * ELU激活函数运算（默认alpha=1.0）
     *
     * @return ELU运算结果的新变量
     */
    public Variable elu() {
        return elu(DEFAULT_ELU_ALPHA);
    }

    /**
     * LogSoftmax激活函数运算
     * <p>
     * Softmax的对数形式，常用于NLLLoss组合
     * LogSoftmax(x) = log(softmax(x))
     *
     * @param axis 计算轴，默认-1表示最后一维
     * @return LogSoftmax运算结果的新变量
     */
    public Variable logSoftmax(int axis) {
        Function function = new LogSoftmax(axis);
        return function.call(this);
    }

    /**
     * LogSoftmax激活函数运算（默认axis=-1）
     *
     * @return LogSoftmax运算结果的新变量
     */
    public Variable logSoftmax() {
        return logSoftmax(DEFAULT_LOG_SOFTMAX_AXIS);
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

    /**
     * 均值运算
     * <p>
     * 沿指定轴计算变量的均值
     *
     * @param _axis     轴索引
     * @param _keepdims 是否保持维度
     * @return 均值运算结果的新变量
     */
    public Variable mean(int _axis, boolean _keepdims) {
        Function function = new Mean(_axis, _keepdims);
        return function.call(this);
    }

    /**
     * 方差运算
     * <p>
     * 沿指定轴计算变量的方差
     *
     * @param _axis     轴索引
     * @param _keepdims 是否保持维度
     * @return 方差运算结果的新变量
     */
    public Variable var(int _axis, boolean _keepdims) {
        Function function = new Variance(_axis, _keepdims);
        return function.call(this);
    }

    /**
     * 平方根运算
     * <p>
     * 对变量执行平方根运算
     *
     * @return 平方根运算结果的新变量
     */
    public Variable sqrt() {
        Function function = new Sqrt();
        return function.call(this);
    }

    // =============================================================================
    // 张量形状与矩阵操作
    // =============================================================================
    // 提供张量形状变换和矩阵运算:
    // - 形状变换: reshape, transpose, squeeze, broadcast等
    // - 矩阵运算: matmul, bmm(批量矩阵乘法)
    // - 归约运算: sum, mean, max, min等
    // =============================================================================

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

    // =============================================================================
    // 损失函数
    // =============================================================================
    // 提供常用的损失函数计算
    // =============================================================================

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

    // =============================================================================
    // 高级算子 - Transformer/LLM支持
    // =============================================================================
    // 为大语言模型训练提供的专用算子:
    // - squeeze/detach/clone等张量操作
    // - expand/repeat等广播操作
    // - tril/bmm等矩阵操作
    // - indexSelect/scatterAdd等索引操作
    // - rmsNorm等归一化操作
    // =============================================================================

    /**
     * 压缩维度
     * <p>
     * 移除所有大小为1的维度
     *
     * @return 压缩后的新变量
     */
    public Variable squeeze() {
        Function function = new Squeeze();
        return function.call(this);
    }

    /**
     * 压缩指定维度
     * <p>
     * 移除指定维度（如果大小为1）
     *
     * @param dim 要移除的维度索引（支持负数索引）
     * @return 压缩后的新变量
     */
    public Variable squeeze(int dim) {
        Function function = new Squeeze(dim);
        return function.call(this);
    }

    /**
     * 切断梯度
     * <p>
     * 从计算图中分离变量，停止梯度传播
     *
     * @return 切断梯度后的新变量（requireGrad=false）
     */
    public Variable detach() {
        Function function = new Detach();
        Variable result = function.call(this);
        result.setRequireGrad(false);
        return result;
    }

    /**
     * 创建同形状全1张量
     *
     * @return 与当前变量同形状的全1张量
     */
    public Variable onesLike() {
        Function function = new OnesLike();
        return function.call(this);
    }

    /**
     * 创建同形状全0张量
     *
     * @return 与当前变量同形状的全0张量
     */
    public Variable zerosLike() {
        Function function = new ZerosLike();
        return function.call(this);
    }

    /**
     * 克隆张量
     * <p>
     * 深拷贝张量的值，返回新的叶子节点
     *
     * @return 克隆后的新变量
     */
    public Variable clone() {
        Function function = new Clone();
        return function.call(this);
    }

    /**
     * 扩展维度（广播语义）
     * <p>
     * 将大小为1的维度扩展到指定大小，不复制数据
     *
     * @param shape 目标形状
     * @return 扩展后的新变量
     */
    public Variable expand(Shape shape) {
        Function function = new Expand(shape);
        return function.call(this);
    }

    /**
     * 重复张量
     * <p>
     * 沿指定维度重复张量（复制数据）
     *
     * @param repeats 每个维度重复的次数
     * @return 重复后的新变量
     */
    public Variable repeat(int... repeats) {
        Function function = new Repeat(repeats);
        return function.call(this);
    }

    /**
     * 下三角矩阵
     * <p>
     * 返回下三角部分，上三角部分置为0
     *
     * @param k 对角线偏移（k=0表示主对角线及以下）
     * @return 下三角矩阵
     */
    public Variable tril(int k) {
        Function function = new Tril(k);
        return function.call(this);
    }

    /**
     * 下三角矩阵（默认k=0）
     *
     * @return 下三角矩阵
     */
    public Variable tril() {
        return tril(DEFAULT_TRIL_DIAGONAL);
    }

    /**
     * 批量矩阵乘法
     * <p>
     * 输入: [batch, n, m] @ [batch, m, p] -> [batch, n, p]
     *
     * @param other 另一个变量
     * @return 批量矩阵乘法结果
     */
    public Variable bmm(Variable other) {
        Function function = new BMM();
        return function.call(this, other);
    }

    /**
     * 条件选择
     * <p>
     * where(condition, x, y): condition为true选x，否则选y
     *
     * @param condition 条件变量
     * @param x        条件为true时选择的值
     * @param y        条件为false时选择的值
     * @return 条件选择结果
     */
    public static Variable where(Variable condition, Variable x, Variable y) {
        Function function = new Where();
        return function.call(condition, x, y);
    }

    /**
     * 索引选择
     * <p>
     * 沿指定维度选择索引对应的元素
     *
     * @param dim   维度索引
     * @param index 索引变量
     * @return 索引选择结果
     */
    public Variable indexSelect(int dim, Variable index) {
        Function function = new IndexSelect(dim);
        return function.call(this, index);
    }

    /**
     * 分散累加
     * <p>
     * 将源张量的值根据索引分散到目标张量的指定位置并累加
     *
     * @param dim   维度索引
     * @param index 索引变量
     * @param src   源变量
     * @return 分散累加结果
     */
    public Variable scatterAdd(int dim, Variable index, Variable src) {
        Function function = new ScatterAdd(dim);
        return function.call(this, index, src);
    }

    /**
     * RMS归一化
     * <p>
     * RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
     *
     * @param normalizedShape 归一化的维度形状
     * @param eps            epsilon值
     * @param weight         可学习的缩放权重
     * @return RMS归一化结果
     */
    public Variable rmsNorm(int[] normalizedShape, float eps, Variable weight) {
        Function function = new RMSNorm(normalizedShape, eps);
        return function.call(this, weight);
    }

    /**
     * RMS归一化（默认eps=1e-6）
     *
     * @param normalizedShape 归一化的维度形状
     * @param weight         可学习的缩放权重
     * @return RMS归一化结果
     */
    public Variable rmsNorm(int[] normalizedShape, Variable weight) {
        return rmsNorm(normalizedShape, DEFAULT_RMS_NORM_EPS, weight);
    }

    // =============================================================================
    // 工具方法 - PyTorch风格API
    // =============================================================================
    // 提供与PyTorch类似的便捷方法,减少对NdArray的直接访问:
    // - fullLike/onesLike/zerosLike等创建方法
    // - maskedFill等条件填充
    // - shape/dtype等属性访问
    // =============================================================================

    /**
     * 创建指定值填充的同形状张量
     * <p>
     * 类似 PyTorch 的 full_like
     *
     * @param value 填充值
     * @return 填充后的新变量
     */
    public Variable fullLike(float value) {
        NdArray filledArray = NdArray.like(this.getShape(), value);
        Variable result = new Variable(filledArray);
        result.setRequireGrad(false);  // 常量不需要梯度
        return result;
    }

    /**
     * 使用指定值填充张量（就地操作的Variable版本）
     * <p>
     * 返回新的Variable，值为指定填充值
     *
     * @param value 填充值
     * @return 填充后的新变量
     */
    public Variable fill(float value) {
        return fullLike(value);
    }

    /**
     * 掩码填充
     * <p>
     * 类似 PyTorch 的 masked_fill
     * 在mask为true的位置用指定值填充
     *
     * @param mask  掩码变量（布尔值或0/1）
     * @param value 填充值
     * @return 掩码填充后的新变量
     */
    public Variable maskedFill(Variable mask, float value) {
        // 使用 where 函数实现: where(mask, value, this)
        Variable fillValue = new Variable(value);
        fillValue.setRequireGrad(false);
        return Variable.where(mask, fillValue, this);
    }

    /**
     * 获取数据类型信息
     * <p>
     * 返回底层数据的类型描述（用于调试）
     *
     * @return 数据类型字符串
     */
    public String dtype() {
        return "float32";  // TinyAI 目前主要使用 float32
    }

    /**
     * 创建新的空张量（同形状、同类型）
     * <p>
     * 类似 PyTorch 的 new_empty 或 empty_like
     *
     * @return 新的未初始化变量
     */
    public Variable newLike() {
        return zerosLike();  // 为安全起见，返回全零张量
    }

    /**
     * 批量获取形状（用于替代 getValue().getShape().getShapeDims()）
     * <p>
     * 这是 sizes() 的别名，提供更直观的命名
     *
     * @return 形状维度数组
     */
    public int[] shape() {
        return sizes();
    }

    // =============================================================================
    // 索引与切片操作
    // =============================================================================
    // 提供灵活的张量索引和切片能力:
    // - slice/select/sliceRange等切片操作
    // - cat/split等拼接分割操作
    // =============================================================================

    /**
     * 获取指定索引的子张量（切片操作）
     * <p>
     * 类似 PyTorch 的 x[rowSlices, colSlices]
     *
     * @param rowSlices 行索引数组
     * @param colSlices 列索引数组
     * @return 切片结果
     */
    public Variable slice(int[] rowSlices, int[] colSlices) {
        return getItem(rowSlices, colSlices);
    }

    /**
     * 在指定维度上选择单个索引
     * <p>
     * 类似 PyTorch 的 x.select(dim, index)
     *
     * @param dim   维度索引
     * @param index 要选择的索引位置
     * @return 选择结果（维度会减少1）
     */
    public Variable select(int dim, int index) {
        Function function = new io.leavesfly.tinyai.func.matrix.Select(dim, index);
        return function.call(this);
    }

    /**
     * 在指定维度上进行范围切片
     * <p>
     * 类似 PyTorch 的 x[start:end]
     *
     * @param dim   维度索引
     * @param start 起始索引（包含）
     * @param end   结束索引（不包含）
     * @return 切片结果
     */
    public Variable sliceRange(int dim, int start, int end) {
        Function function = new io.leavesfly.tinyai.func.matrix.SliceRange(dim, start, end);
        return function.call(this);
    }

    /**
     * 创建指定形状和值的常量张量
     * <p>
     * 类似 PyTorch 的 torch.full
     *
     * @param shape 张量形状
     * @param value 常量值
     * @return 常量张量
     */
    public static Variable full(Shape shape, float value) {
        Function function = new io.leavesfly.tinyai.func.matrix.Full(shape, value);
        return function.call();
    }

    /**
     * 创建全零张量
     * <p>
     * 类似 PyTorch 的 torch.zeros
     *
     * @param shape 张量形状
     * @return 全零张量
     */
    public static Variable zeros(Shape shape) {
        return new Variable(NdArray.zeros(shape));
    }

    /**
     * 创建全一张量
     * <p>
     * 类似 PyTorch 的 torch.ones
     *
     * @param shape 张量形状
     * @return 全一张量
     */
    public static Variable ones(Shape shape) {
        return new Variable(NdArray.ones(shape));
    }

    /**
     * 在指定维度上拼接张量
     * <p>
     * 类似 PyTorch 的 torch.cat
     *
     * @param variables 要拼接的变量数组
     * @param dim      拼接的维度
     * @return 拼接结果
     */
    public static Variable cat(Variable[] variables, int dim) {
        Function function = new Concat(dim);
        return function.call(variables);
    }

    /**
     * 在指定维度上分割张量
     * <p>
     * 类似 PyTorch 的 torch.split
     *
     * @param splitSize 每个分块的大小
     * @param dim      分割的维度
     * @return 分割结果数组
     */
    public Variable[] split(int splitSize, int dim) {
        Function function = new io.leavesfly.tinyai.func.matrix.SplitBySize(splitSize, dim);
        return function.callMulti(this);
    }

    // =============================================================================
    // 比较与随机操作
    // =============================================================================
    // 提供比较运算和随机张量生成:
    // - gt/lt/eq等比较操作
    // - rand/randn等随机生成
    // =============================================================================

    /**
     * 大于比较
     * <p>
     * 类似 PyTorch 的 x.gt(y)
     *
     * @param other 比较的另一个变量
     * @return 比较结果（0或1）
     */
    public Variable gt(Variable other) {
        NdArray result = this.value.gt(other.getValue());
        Variable resultVar = new Variable(result);
        resultVar.setRequireGrad(false);  // 比较结果不需要梯度
        return resultVar;
    }

    /**
     * 小于比较
     * <p>
     * 类似 PyTorch 的 x.lt(y)
     *
     * @param other 比较的另一个变量
     * @return 比较结果（0或1）
     */
    public Variable lt(Variable other) {
        NdArray result = this.value.lt(other.getValue());
        Variable resultVar = new Variable(result);
        resultVar.setRequireGrad(false);  // 比较结果不需要梯度
        return resultVar;
    }

    /**
     * 等于比较
     * <p>
     * 类似 PyTorch 的 x.eq(y)
     *
     * @param other 比较的另一个变量
     * @return 比较结果（0或1）
     */
    public Variable eq(Variable other) {
        NdArray result = this.value.eq(other.getValue());
        Variable resultVar = new Variable(result);
        resultVar.setRequireGrad(false);  // 比较结果不需要梯度
        return resultVar;
    }

    /**
     * 创建随机均匀分布张量 [0, 1)
     * <p>
     * 类似 PyTorch 的 torch.rand
     *
     * @param shape 张量形状
     * @return 随机张量
     */
    public static Variable rand(Shape shape) {
        NdArray array = NdArray.likeRandom(0, 1, shape);
        Variable result = new Variable(array);
        result.setRequireGrad(false);  // 随机张量不需要梯度
        return result;
    }

    /**
     * 创建标准正态分布随机张量
     * <p>
     * 类似 PyTorch 的 torch.randn
     *
     * @param shape 张量形状
     * @return 随机张量
     */
    public static Variable randn(Shape shape) {
        return new Variable(NdArray.randn(shape));
    }

    // =============================================================================
    // 卷积操作
    // =============================================================================
    // 提供2D卷积操作，支持深度学习中的卷积神经网络
    // =============================================================================

    /**
     * 2D卷积操作
     * <p>
     * 对当前变量（输入）与卷积核执行2D卷积运算
     * <p>
     * 输入形状: [batch_size, in_channels, height, width]
     * 卷积核形状: [out_channels, in_channels, kernel_h, kernel_w]
     * 输出形状: [batch_size, out_channels, out_h, out_w]
     *
     * @param kernel  卷积核变量
     * @param stride  步长
     * @param padding 填充大小
     * @return 卷积运算结果的新变量
     */
    public Variable conv2d(Variable kernel, int stride, int padding) {
        Function function = new Conv2d(stride, padding);
        return function.call(this, kernel);
    }

    /**
     * 2D卷积操作（默认stride=1, padding=0）
     *
     * @param kernel 卷积核变量
     * @return 卷积运算结果的新变量
     */
    public Variable conv2d(Variable kernel) {
        return conv2d(kernel, 1, 0);
    }
}