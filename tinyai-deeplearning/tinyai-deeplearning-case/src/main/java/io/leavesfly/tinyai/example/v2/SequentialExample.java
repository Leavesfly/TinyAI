package io.leavesfly.tinyai.example.v2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.activation.Sigmoid;

/**
 * Sequential容器使用示例
 * <p>
 * 本示例展示如何:
 * 1. 使用Sequential容器快速构建模型
 * 2. 链式调用添加模块
 * 3. Sequential容器的前向传播
 */
public class SequentialExample {

    public static void main(String[] args) {
        System.out.println("=== Sequential容器使用示例 ===\n");

        // 1. 使用链式调用创建Sequential模型
        System.out.println("1. 创建Sequential模型（链式调用）");
        Sequential model = new Sequential("mlp")
                .add(new Linear("fc1", 784, 256, true))
                .add(new ReLU("relu1"))
                .add(new Dropout("dropout1", 0.3f))
                .add(new Linear("fc2", 256, 128, true))
                .add(new ReLU("relu2"))
                .add(new Dropout("dropout2", 0.3f))
                .add(new Linear("fc3", 128, 10, true))
                .add(new Sigmoid("sigmoid"));

        System.out.println("   模型创建成功: " + model.getName());
        System.out.println();

        // 2. 查看模型结构
        System.out.println("2. 模型结构:");
        System.out.println("   子模块列表:");
        for (String name : model.namedModules().keySet()) {
            System.out.println("   - " + name);
        }
        System.out.println();

        // 3. 查看参数
        System.out.println("3. 模型参数:");
        for (String name : model.namedParameters().keySet()) {
            Shape shape = model.namedParameters().get(name).data().getShape();
            System.out.println("   - " + name + ": " + shapeToString(shape));
        }
        System.out.println();

        // 4. 训练模式前向传播
        System.out.println("4. 训练模式前向传播:");
        model.train();

        // 创建输入数据 (batch_size=4, features=784)
        float[] inputData = new float[4 * 784];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) (Math.random() * 0.1);
        }
        NdArray inputArray = NdArray.of(inputData, Shape.of(4, 784));
        Variable input = new Variable(inputArray);

        System.out.println("   输入形状: " + shapeToString(input.getValue().getShape()));
        Variable output = model.forward(input);
        System.out.println("   输出形状: " + shapeToString(output.getValue().getShape()));
        System.out.println("   输出示例 (前5个值): ");
        float[] outputData = output.getValue().getArray();
        for (int i = 0; i < Math.min(5, outputData.length); i++) {
            System.out.printf("     %.4f%n", outputData[i]);
        }
        System.out.println();

        // 5. 推理模式前向传播
        System.out.println("5. 推理模式前向传播:");
        model.eval();
        System.out.println("   当前模式: " + (model.isTraining() ? "训练" : "推理"));

        output = model.forward(input);
        System.out.println("   输出形状: " + shapeToString(output.getValue().getShape()));
        System.out.println("   输出示例 (前5个值): ");
        outputData = output.getValue().getArray();
        for (int i = 0; i < Math.min(5, outputData.length); i++) {
            System.out.printf("     %.4f%n", outputData[i]);
        }
        System.out.println();

        // 6. Sequential的优势
        System.out.println("6. Sequential容器的优势:");
        System.out.println("   - 简化模型定义，代码更简洁");
        System.out.println("   - 支持链式调用，提高可读性");
        System.out.println("   - 自动管理子模块的注册和命名");
        System.out.println("   - 适合构建简单的线性网络结构");
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

