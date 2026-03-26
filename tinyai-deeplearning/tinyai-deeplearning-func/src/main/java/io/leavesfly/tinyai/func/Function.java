package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;

import java.io.Serializable;
import java.util.List;

/**
 * Function - 自动微分系统的计算节点基类
 *
 * 采用模板方法设计模式，定义前向传播和反向传播的统一接口。
 * 所有数学运算（加减乘除、矩阵运算、激活函数等）都继承此类。
 *
 * 核心职责：
 * - 执行前向计算：将输入 Variable 转换为输出 Variable
 * - 执行反向传播：根据输出梯度计算输入梯度
 * - 构建计算图：连接 Variable 节点形成 DAG
 *
 * 计算图结构：
 * <pre>
 *   Variable(x) ──→ [Function] ──→ Variable(y)
 *                      ↓
 *              backward(∂L/∂y) → ∂L/∂x
 * </pre>
 *
 * 子类实现要求：
 * - 实现 forward()：定义前向计算逻辑
 * - 实现 backward()：定义梯度计算逻辑
 * - 实现 requireInputNum()：声明所需输入数量
 *
 * 使用示例：
 * <pre>{@code
 * // 子类实现
 * public class Add extends Function {
 *     public NdArray forward(NdArray... inputs) {
 *         return inputs[0].add(inputs[1]);
 *     }
 *     public List<NdArray> backward(NdArray yGrad) {
 *         return Arrays.asList(yGrad, yGrad);
 *     }
 *     public int requireInputNum() { return 2; }
 * }
 *
 * // 使用方式
 * Variable z = new Add().call(x, y);
 * }</pre>
 *
 * @author TinyAI Team
 * @see Variable
 */
public abstract class Function implements Serializable {

    private static final long serialVersionUID = 1L;

    // ==================== 常量定义 ====================

    /** 标识函数可接受任意数量输入的特殊值 */
    protected static final int ARBITRARY_INPUT_NUM = -1;

    // ==================== 核心字段 ====================

    /** 输入变量数组，用于反向传播时追溯计算图 */
    protected Variable[] inputs;

    /** 单输出场景的输出变量 */
    protected Variable output;

    /** 多输出场景的输出变量数组 */
    protected Variable[] outputs;

    // ==================== 公共 API ====================

    /**
     * 单输出函数的调用入口
     *
     * 执行流程：验证输入 → 提取值 → 前向计算 → 创建输出 → 构建计算图
     *
     * @param _inputs 输入变量（可变参数）
     * @return 计算结果的输出变量
     * @throws RuntimeException 输入数量不符或包含 null 时抛出
     */
    public Variable call(Variable... _inputs) {
        // 验证输入参数
        validateInputs(_inputs);

        // 提取 NdArray 值
        NdArray[] ndArrayInputs = extractNdArrays(_inputs);

        // 执行前向传播
        NdArray ndArrayOutput = forward(ndArrayInputs);

        // 创建输出变量
        Variable _output = new Variable(ndArrayOutput);

        // 构建计算图（仅在训练模式且需要梯度时）
        if (shouldBuildGraph(_inputs)) {
            buildComputationGraph(_inputs, _output);
        }

        return _output;
    }

    /**
     * 多输出函数的调用入口
     *
     * 用于 split、topK 等返回多个输出的函数。
     *
     * @param _inputs 输入变量（可变参数）
     * @return 输出变量数组
     * @throws RuntimeException 输入数量不符或包含 null 时抛出
     */
    public Variable[] callMulti(Variable... _inputs) {
        // 验证输入参数
        validateInputs(_inputs);

        // 提取 NdArray 值
        NdArray[] ndArrayInputs = extractNdArrays(_inputs);

        // 执行前向传播（多输出版本）
        NdArray[] ndArrayOutputs = forwardMulti(ndArrayInputs);

        // 创建输出变量数组
        Variable[] _outputs = new Variable[ndArrayOutputs.length];
        for (int i = 0; i < ndArrayOutputs.length; i++) {
            _outputs[i] = new Variable(ndArrayOutputs[i]);
        }

        // 构建计算图
        if (shouldBuildGraph(_inputs)) {
            buildComputationGraphMulti(_inputs, _outputs);
        }

        return _outputs;
    }

    /**
     * 获取输入变量数组
     *
     * @return 输入变量数组，未构建计算图时为 null
     */
    public Variable[] getInputs() {
        return inputs;
    }

    /**
     * 获取输入变量数量
     *
     * @return 输入数量，未设置输入时返回 0
     */
    public int getInputCount() {
        return inputs == null ? 0 : inputs.length;
    }

    /**
     * 设置输入变量数组（框架内部使用）
     */
    public void setInputs(Variable[] inputs) {
        this.inputs = inputs;
    }

    /**
     * 获取单输出变量
     *
     * @return 输出变量，多输出时返回第一个
     */
    public Variable getOutput() {
        return output;
    }

    /**
     * 获取多输出变量数组
     */
    public Variable[] getOutputs() {
        return outputs;
    }

    /**
     * 设置输出变量（框架内部使用）
     */
    public void setOutput(Variable output) {
        this.output = output;
    }

    /**
     * 设置多输出变量数组（框架内部使用）
     */
    public void setOutputs(Variable[] outputs) {
        this.outputs = outputs;
    }

    /**
     * 判断是否为多输出函数
     */
    public boolean isMultiOutput() {
        return outputs != null && outputs.length > 1;
    }

    /**
     * 清理资源，断开计算图连接
     *
     * 用于 RNN 截断式反向传播（TBPTT）或内存优化场景。
     */
    public void unChain() {
        this.inputs = null;
        this.output = null;
        this.outputs = null;
    }

    // ==================== 抽象方法（子类必须实现） ====================

    /**
     * 前向传播计算
     *
     * 实现要求：
     * - 纯函数：相同输入产生相同输出
     * - 无副作用：不修改输入数组
     * - 返回新数组：结果为新创建的 NdArray
     *
     * @param inputs 输入 NdArray 数组
     * @return 计算结果 NdArray
     */
    public abstract NdArray forward(NdArray... inputs);

    /**
     * 反向传播计算梯度
     *
     * 根据链式法则：∂L/∂xᵢ = ∂L/∂y · ∂y/∂xᵢ
     *
     * 实现要求：
     * - 返回列表长度等于输入数量
     * - 不可导的输入对应梯度可为 null
     * - 梯度形状必须与输入形状一致
     *
     * @param yGrad 输出梯度 ∂L/∂y
     * @return 输入梯度列表 [∂L/∂x₁, ∂L/∂x₂, ...]
     */
    public abstract List<NdArray> backward(NdArray yGrad);

    /**
     * 声明所需输入参数个数
     *
     * @return 正整数表示固定数量，-1 表示可变数量
     */
    public abstract int requireInputNum();

    // ==================== 可选重写方法 ====================

    /**
     * 多输出版本的前向传播
     *
     * 默认抛出 UnsupportedOperationException，多输出函数需重写。
     */
    public NdArray[] forwardMulti(NdArray... inputs) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not support multiple outputs");
    }

    /**
     * 多输出版本的反向传播
     *
     * @param yGrads 所有输出的梯度列表
     * @return 输入梯度列表
     */
    public List<NdArray> backwardMulti(List<NdArray> yGrads) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not support multiple outputs backward");
    }

    // ==================== 内部方法 ====================

    /**
     * 验证输入变量的合法性
     *
     * 检查输入数量和 null 值。
     */
    private void validateInputs(Variable[] inputs) {
        int required = requireInputNum();

        // 检查数量
        if (required >= 0 && inputs.length != required) {
            throw new RuntimeException(String.format(
                    "%s requires %d inputs, but got %d",
                    getClass().getSimpleName(), required, inputs.length));
        }

        // 检查 null（使用循环替代 Stream，提升性能）
        for (Variable input : inputs) {
            if (input == null) {
                throw new RuntimeException(
                        getClass().getSimpleName() + " inputs cannot contain null");
            }
        }
    }

    /**
     * 从 Variable 数组提取 NdArray 值
     *
     * 使用循环替代 Stream，减少调用开销。
     */
    private NdArray[] extractNdArrays(Variable[] vars) {
        NdArray[] result = new NdArray[vars.length];
        for (int i = 0; i < vars.length; i++) {
            result[i] = vars[i].getValue();
        }
        return result;
    }

    /**
     * 判断是否需要构建计算图
     *
     * 条件：训练模式 且 至少一个输入需要梯度
     */
    protected boolean shouldBuildGraph(Variable[] vars) {
        if (!Config.train) {
            return false;
        }
        // 使用循环替代 Stream，提升性能
        for (Variable v : vars) {
            if (v != null && v.isRequireGrad()) {
                return true;
            }
        }
        return false;
    }

    /**
     * 构建计算图（单输出）
     */
    private void buildComputationGraph(Variable[] inputs, Variable output) {
        this.inputs = inputs;
        this.output = output;
        this.outputs = new Variable[]{output};
        output.setCreator(this);
    }

    /**
     * 构建计算图（多输出）
     */
    private void buildComputationGraphMulti(Variable[] inputs, Variable[] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.output = outputs.length > 0 ? outputs[0] : null;
        for (Variable out : outputs) {
            out.setCreator(this);
        }
    }
}
