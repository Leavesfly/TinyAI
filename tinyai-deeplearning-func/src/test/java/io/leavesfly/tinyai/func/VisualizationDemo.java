package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.ComputationGraphVisualizer;
import io.leavesfly.tinyai.util.StepByStepVisualizer;

/**
 * 计算图可视化演示程序
 * <p>
 * 该演示程序展示了ComputationGraphVisualizer和StepByStepVisualizer的使用方法，
 * 包括简单的数学运算、复杂的计算表达式以及反向传播过程的可视化。
 *
 * @author 山泽
 * @version 1.0
 */
public class VisualizationDemo {

    public static void main(String[] args) {
        System.out.println("🎯 计算图可视化演示程序");
        System.out.println("====================================");

        // 演示1：简单的加法和乘法运算
        demo1SimpleArithmetic();

        // 演示2：复杂的数学表达式
        demo2ComplexExpression();

        // 演示3：矩阵运算
        demo3MatrixOperations();

        // 演示4：神经网络相关计算
        demo4NeuralNetwork();
    }

    /**
     * 演示1：简单的加法和乘法运算
     */
    private static void demo1SimpleArithmetic() {
        System.out.println("\n🔸 演示1：简单的加法和乘法运算");
        System.out.println("------------------------------------");

        // 创建输入变量
        Variable x = new Variable(NdArray.of(2.0f), "x");
        Variable y = new Variable(NdArray.of(3.0f), "y");

        // 执行计算: z = x + y
        Variable z = x.add(y);
        z.setName("z = x + y");

        // 显示计算图
        System.out.println("📊 计算表达式: z = x + y");
        ComputationGraphVisualizer.display(z);

        // 执行反向传播并可视化
        System.out.println("🔄 开始反向传播...");
        z.backward();
        StepByStepVisualizer.showBackpropagation(z);

        System.out.println("✅ 演示1完成\n");
    }

    /**
     * 演示2：复杂的数学表达式
     */
    private static void demo2ComplexExpression() {
        System.out.println("\n🔸 演示2：复杂的数学表达式");
        System.out.println("------------------------------------");

        // 创建输入变量
        Variable a = new Variable(NdArray.of(1.0f), "a");
        Variable b = new Variable(NdArray.of(2.0f), "b");
        Variable c = new Variable(NdArray.of(3.0f), "c");

        // 执行复杂计算: result = (a + b) * c
        Variable temp = a.add(b);
        temp.setName("temp = a + b");
        Variable result = temp.mul(c);
        result.setName("result = temp * c");

        // 显示计算图
        System.out.println("📊 计算表达式: result = (a + b) * c");
        ComputationGraphVisualizer.display(result);

        // 重新创建变量用于反向传播演示（因为之前的backward会清空creator）
        Variable a2 = new Variable(NdArray.of(1.0f), "a");
        Variable b2 = new Variable(NdArray.of(2.0f), "b");
        Variable c2 = new Variable(NdArray.of(3.0f), "c");
        Variable temp2 = a2.add(b2);
        temp2.setName("temp = a + b");
        Variable result2 = temp2.mul(c2);
        result2.setName("result = temp * c");

        // 执行反向传播并可视化
        System.out.println("🔄 开始反向传播...");
        StepByStepVisualizer.showBackpropagation(result2);

        System.out.println("✅ 演示2完成\n");
    }

    /**
     * 演示3：矩阵运算
     */
    private static void demo3MatrixOperations() {
        System.out.println("\n🔸 演示3：矩阵运算");
        System.out.println("------------------------------------");

        // 创建矩阵变量
        Variable A = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "A");
        Variable B = new Variable(NdArray.of(new float[][]{{5, 6}, {7, 8}}), "B");

        // 执行矩阵加法: C = A + B
        Variable C = A.add(B);
        C.setName("C = A + B");

        // 显示计算图
        System.out.println("📊 计算表达式: C = A + B (矩阵加法)");
        ComputationGraphVisualizer.display(C);

        // 重新创建用于反向传播
        Variable A2 = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "A");
        Variable B2 = new Variable(NdArray.of(new float[][]{{5, 6}, {7, 8}}), "B");
        Variable C2 = A2.add(B2);
        C2.setName("C = A + B");

        // 执行反向传播并可视化
        System.out.println("🔄 开始反向传播...");
        StepByStepVisualizer.showBackpropagation(C2);

        System.out.println("✅ 演示3完成\n");
    }

    /**
     * 演示4：神经网络相关计算
     */
    private static void demo4NeuralNetwork() {
        System.out.println("\n🔸 演示4：神经网络相关计算");
        System.out.println("------------------------------------");

        // 创建权重和输入
        Variable input = new Variable(NdArray.of(new float[]{1.0f, 2.0f}), "input");
        Variable weight = new Variable(NdArray.of(new float[]{0.5f, -0.3f}), "weight");
        Variable bias = new Variable(NdArray.of(0.1f), "bias");

        // 执行线性变换和激活: output = sigmoid(input * weight + bias)
        Variable linear = input.mul(weight);
        linear.setName("linear = input * weight");

        Variable preActivation = linear.add(bias);
        preActivation.setName("preActivation = linear + bias");

        Variable output = preActivation.sigmoid();
        output.setName("output = sigmoid(preActivation)");

        // 显示计算图
        System.out.println("📊 计算表达式: output = sigmoid(input * weight + bias)");
        ComputationGraphVisualizer.display(output);

        // 重新创建用于反向传播演示
        Variable input2 = new Variable(NdArray.of(new float[]{1.0f, 2.0f}), "input");
        Variable weight2 = new Variable(NdArray.of(new float[]{0.5f, -0.3f}), "weight");
        Variable bias2 = new Variable(NdArray.of(0.1f), "bias");

        Variable linear2 = input2.mul(weight2);
        linear2.setName("linear = input * weight");

        Variable preActivation2 = linear2.add(bias2);
        preActivation2.setName("preActivation = linear + bias");

        Variable output2 = preActivation2.sigmoid();
        output2.setName("output = sigmoid(preActivation)");

        // 执行反向传播并可视化
        System.out.println("🔄 开始反向传播...");
        StepByStepVisualizer.showBackpropagation(output2);

        System.out.println("✅ 演示4完成\n");
    }

    /**
     * 使用说明
     */
    public static void showUsage() {
        System.out.println("📋 使用说明:");
        System.out.println("------------------------------------");
        System.out.println("1. ComputationGraphVisualizer.display(variable)");
        System.out.println("   - 显示以variable为根节点的计算图结构");
        System.out.println("   - 包括所有变量的名称、形状、数值信息");
        System.out.println("   - 显示函数之间的连接关系");
        System.out.println();
        System.out.println("2. variable.backward(); StepByStepVisualizer.showBackpropagation(variable)");
        System.out.println("   - 逐步展示反向传播过程");
        System.out.println("   - 显示每一步的梯度计算");
        System.out.println("   - 展示梯度在计算图中的传播路径");
        System.out.println();
        System.out.println("注意事项:");
        System.out.println("- 变量执行backward()后会清空creator信息");
        System.out.println("- 如需重复演示，请重新创建计算图");
        System.out.println("- 建议为变量设置有意义的名称以便查看");
    }
}