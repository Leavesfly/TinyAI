package io.leavesfly.tinyai.example.v2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;

import java.util.Map;

/**
 * V2模块基础使用示例
 * <p>
 * 本示例展示如何:
 * 1. 创建简单的全连接网络
 * 2. 使用train()和eval()模式切换
 * 3. 访问和管理模型参数
 * 4. 查看子模块结构
 */
public class BasicUsageExample {

    /**
     * 简单的两层全连接网络
     */
    static class SimpleNet extends Module {
        private final Linear fc1;
        private final ReLU relu;
        private final Dropout dropout;
        private final Linear fc2;

        public SimpleNet(String name, int inputSize, int hiddenSize, int outputSize) {
            super(name);

            // 创建子模块
            fc1 = new Linear("fc1", inputSize, hiddenSize, true);
            relu = new ReLU("relu");
            dropout = new Dropout("dropout", 0.5f);
            fc2 = new Linear("fc2", hiddenSize, outputSize, true);

            // 注册子模块（自动收集参数）
            registerModule("fc1", fc1);
            registerModule("relu", relu);
            registerModule("dropout", dropout);
            registerModule("fc2", fc2);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];

            // 前向传播
            x = fc1.forward(x);
            x = relu.forward(x);
            x = dropout.forward(x);
            x = fc2.forward(x);

            return x;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== V2模块基础使用示例 ===\n");

        // 1. 创建模型
        System.out.println("1. 创建模型");
        SimpleNet model = new SimpleNet("simple_net", 784, 256, 10);
        System.out.println("   模型创建成功: " + model.getName());
        System.out.println();

        // 2. 查看模型参数
        System.out.println("2. 模型参数:");
        Map<String, Parameter> params = model.namedParameters();
        System.out.println("   参数数量: " + params.size());
        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
            String name = entry.getKey();
            Parameter param = entry.getValue();
            Shape shape = param.data().getShape();
            System.out.println("   - " + name + ": " + shapeToString(shape));
        }
        System.out.println();

        // 3. 计算参数总数
        System.out.println("3. 参数统计:");
        long totalParams = params.size();
        System.out.println("   总参数数量: " + totalParams);
        System.out.println();

        // 4. 训练模式
        System.out.println("4. 训练模式:");
        model.train();
        System.out.println("   当前模式: " + (model.isTraining() ? "训练" : "推理"));

        // 创建输入数据 (batch_size=4, features=784)
        float[] inputData = new float[4 * 784];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) (Math.random() * 0.1);
        }
        NdArray inputArray = NdArray.of(inputData, Shape.of(4, 784));
        Variable input = new Variable(inputArray);

        // 前向传播
        Variable output = model.forward(input);
        System.out.println("   输出形状: " + shapeToString(output.getValue().getShape()));
        System.out.println("   输出示例 (前5个值): ");
        float[] outputData = output.getValue().getArray();
        for (int i = 0; i < Math.min(5, outputData.length); i++) {
            System.out.printf("     %.4f%n", outputData[i]);
        }
        System.out.println();

        // 5. 推理模式
        System.out.println("5. 推理模式:");
        model.eval();
        System.out.println("   当前模式: " + (model.isTraining() ? "训练" : "推理"));

        // 推理时Dropout应该被禁用
        output = model.forward(input);
        System.out.println("   输出形状: " + shapeToString(output.getValue().getShape()));
        System.out.println("   输出示例 (前5个值): ");
        outputData = output.getValue().getArray();
        for (int i = 0; i < Math.min(5, outputData.length); i++) {
            System.out.printf("     %.4f%n", outputData[i]);
        }
        System.out.println();

        // 6. 访问子模块
        System.out.println("6. 子模块访问:");
        Map<String, Module> submodules = model.namedModules();
        System.out.println("   子模块数量: " + submodules.size());
        for (String name : submodules.keySet()) {
            System.out.println("   - " + name);
        }
        System.out.println();

        System.out.println("=== 示例完成 ===");
    }

    private static String shapeToString(Shape shape) {
        int[] dims = shape.getShapeDims();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(dims[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}

