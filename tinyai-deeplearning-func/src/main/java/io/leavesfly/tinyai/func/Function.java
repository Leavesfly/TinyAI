package io.leavesfly.tinyai.func;


import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 抽象的数学函数基类
 * <p>
 * 在TinyDL深度学习框架中，Function类是所有数学函数操作的基类。
 * 它定义了前向传播和反向传播的接口，并负责构建计算图。
 * 每个函数实例都维护输入变量和输出变量之间的关系。
 */
public abstract class Function implements Serializable {

    /**
     * 函数的输入变量数组
     * 存储传递给该函数的所有输入变量
     */
    protected Variable[] inputs;

    /**
     * 函数的输出变量
     * 存储该函数计算结果的输出变量
     */
    protected Variable output;

    /**
     * 函数的执行函数，执行函数的前向传播计算并构建计算图
     * <p>
     * 该方法执行以下操作：
     * 1. 验证输入变量数量是否符合要求
     * 2. 从输入变量中提取NdArray值
     * 3. 调用forward方法执行前向传播计算
     * 4. 创建输出变量
     * 5. 在训练模式下构建计算图
     *
     * @param _inputs 输入变量数组
     * @return 计算结果的输出变量
     * @throws RuntimeException 当输入变量数量不符合要求时抛出异常
     */
    public Variable call(Variable... _inputs) {
        // 输入验证
        if (_inputs.length != requireInputNum() && requireInputNum() > 0) {
            throw new RuntimeException("Function call inputs Variable requireInputNum error! Expected: "
                    + requireInputNum() + ", Actual: " + _inputs.length);
        }

        // 提取NdArray值
        NdArray[] ndArrayInputs = Arrays.stream(_inputs)
                .filter(Objects::nonNull)
                .map(Variable::getValue)
                .toArray(NdArray[]::new);

        // 执行前向传播
        NdArray ndArrayOutput = forward(ndArrayInputs);

        // 创建输出变量
        Variable _output = new Variable(ndArrayOutput);

        // 只在训练模式下构建计算图
        if (Config.train) {
            this.inputs = _inputs;
            this.output = _output;
            _output.setCreator(this);
        }

        return _output;
    }

    /**
     * 函数的前向传播计算
     * <p>
     * 子类必须实现此方法来定义具体的前向传播计算逻辑。
     * 该方法接收NdArray数组作为输入，返回计算结果的NdArray。
     *
     * @param inputs 输入的NdArray数组
     * @return 前向传播计算结果的NdArray
     */
    public abstract NdArray forward(NdArray... inputs);

    /**
     * 函数的反向传播计算（求导）
     * <p>
     * 子类必须实现此方法来定义具体的反向传播计算逻辑。
     * 该方法接收输出变量的梯度，计算并返回输入变量的梯度。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    public abstract List<NdArray> backward(NdArray yGrad);

    /**
     * 获取函数的输入变量数组
     *
     * @return 输入变量数组
     */
    public Variable[] getInputs() {
        return inputs;
    }

    /**
     * 设置函数的输入变量数组
     *
     * @param inputs 输入变量数组
     */
    public void setInputs(Variable[] inputs) {
        this.inputs = inputs;
    }

    /**
     * 获取函数的输出变量
     *
     * @return 输出变量
     */
    public Variable getOutput() {
        return output;
    }

    /**
     * 设置函数的输出变量
     *
     * @param output 输出变量
     */
    public void setOutput(Variable output) {
        this.output = output;
    }

    /**
     * 获取函数所需的输入参数个数
     * <p>
     * 子类实现此方法来指定函数所需的输入变量数量。
     * 返回-1表示函数可以接受任意数量的输入参数。
     *
     * @return 函数所需的输入参数个数
     */
    public abstract int requireInputNum();

    /**
     * 清理函数资源，断开计算图连接
     * <p>
     * 用于RNN中切断计算图，防止梯度回传过长导致的梯度消失或爆炸问题。
     */
    public void unChain() {
        this.inputs = null;
        this.output = null;
    }
}