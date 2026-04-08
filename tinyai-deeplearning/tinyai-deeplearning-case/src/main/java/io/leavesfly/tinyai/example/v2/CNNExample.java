package io.leavesfly.tinyai.example.v2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.conv.Conv2d;
import io.leavesfly.tinyai.nnet.layer.conv.MaxPool2d;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.util.UmlPrinter;

/**
 * CNN卷积神经网络示例
 * <p>
 * 本示例展示如何:
 * 1. 构建完整的CNN分类器
 * 2. 使用卷积层、池化层和全连接层
 * 3. 处理图像数据的形状变换
 */
public class CNNExample {

    /**
     * LeNet-5风格的卷积神经网络
     * 用于MNIST手写数字分类
     */
    static class LeNet5 extends Module {
        private final Conv2d conv1;
        private final ReLU relu1;
        private final MaxPool2d pool1;

        private final Conv2d conv2;
        private final ReLU relu2;
        private final MaxPool2d pool2;

        private final Linear fc1;
        private final ReLU relu3;
        private final Dropout dropout;

        private final Linear fc2;
        private final ReLU relu4;

        private final Linear fc3;

        public LeNet5(String name, int numClasses) {
            super(name);

            // 卷积层1: 1 -> 6 channels, 5x5 kernel
            conv1 = new Conv2d("conv1", 1, 6, 5, 5, 1, 0, true);
            relu1 = new ReLU("relu1");
            // 2x2 池化，步长2，无填充，保持注释中的 12x12 输出
            pool1 = new MaxPool2d("pool1", 2, 2, 0);

            // 卷积层2: 6 -> 16 channels, 5x5 kernel
            conv2 = new Conv2d("conv2", 6, 16, 5, 5, 1, 0, true);
            relu2 = new ReLU("relu2");
            // 2x2 池化，步长2，无填充，保持注释中的 4x4 输出
            pool2 = new MaxPool2d("pool2", 2, 2, 0);

            // 全连接层1: 16*4*4 -> 120
            fc1 = new Linear("fc1", 16 * 4 * 4, 120, true);
            relu3 = new ReLU("relu3");
            dropout = new Dropout("dropout", 0.5f);

            // 全连接层2: 120 -> 84
            fc2 = new Linear("fc2", 120, 84, true);
            relu4 = new ReLU("relu4");

            // 输出层: 84 -> numClasses
            fc3 = new Linear("fc3", 84, numClasses, true);

            // 注册所有子模块
            registerModule("conv1", conv1);
            registerModule("relu1", relu1);
            registerModule("pool1", pool1);
            registerModule("conv2", conv2);
            registerModule("relu2", relu2);
            registerModule("pool2", pool2);
            registerModule("fc1", fc1);
            registerModule("relu3", relu3);
            registerModule("dropout", dropout);
            registerModule("fc2", fc2);
            registerModule("relu4", relu4);
            registerModule("fc3", fc3);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];

            // 卷积块1: Conv -> ReLU -> Pool
            // (N, 1, 28, 28) -> (N, 6, 24, 24) -> (N, 6, 12, 12)
            x = conv1.forward(x);
            x = relu1.forward(x);
            x = pool1.forward(x);

            // 卷积块2: Conv -> ReLU -> Pool
            // (N, 6, 12, 12) -> (N, 16, 8, 8) -> (N, 16, 4, 4)
            x = conv2.forward(x);
            x = relu2.forward(x);
            x = pool2.forward(x);

            // 展平: (N, 16, 4, 4) -> (N, 256)
            x = flatten(x);

            // 全连接块1
            x = fc1.forward(x);
            x = relu3.forward(x);
            x = dropout.forward(x);

            // 全连接块2
            x = fc2.forward(x);
            x = relu4.forward(x);

            // 输出层
            x = fc3.forward(x);

            return x;
        }

        /**
         * 将4D张量展平为2D张量
         *
         * @param x 输入变量，形状为 (N, C, H, W)
         * @return 展平后的变量，形状为 (N, C*H*W)
         */
        private Variable flatten(Variable x) {
            NdArray data = x.getValue();
            int[] shape = data.getShape().getShapeDims();

            if (shape.length != 4) {
                throw new IllegalArgumentException("Expected 4D input, got " + shape.length + "D");
            }

            int batchSize = shape[0];
            int flatSize = shape[1] * shape[2] * shape[3];

            // 重塑为 (batch_size, flat_size)
            float[] flatData = data.getArray();
            NdArray flatArray = NdArray.of(flatData, Shape.of(batchSize, flatSize));

            return new Variable(flatArray);
        }
    }

    public static void main(String[] args) {
        System.out.println("=== CNN分类器示例 ===\n");

        // 1. 创建模型
        System.out.println("1. 创建LeNet-5模型");
        LeNet5 model = new LeNet5("lenet5", 10);
        System.out.println("   模型创建成功");
        System.out.println();

        // 2. 查看模型结构
        System.out.println("2. 模型结构:");
        System.out.println("   子模块:");
        for (String name : model.namedModules().keySet()) {
            System.out.println("   - " + name);
        }
        System.out.println();

        // 3. 训练模式前向传播
        System.out.println("3. 训练模式前向传播:");
        model.train();

        // 创建模拟的MNIST图像 (batch_size=4, channels=1, height=28, width=28)
        float[] imageData = new float[4 * 1 * 28 * 28];
        for (int i = 0; i < imageData.length; i++) {
            imageData[i] = (float) (Math.random() * 0.1);
        }
        NdArray imageArray = NdArray.of(imageData, Shape.of(4, 1, 28, 28));
        Variable input = new Variable(imageArray);

        System.out.println("   输入形状: [4, 1, 28, 28]");
        Variable output = model.forward(input);
        System.out.println("   输出形状: " + shapeToString(output.getValue().getShape()));

        // 显示输出（logits）
        float[] outputData = output.getValue().getArray();
        System.out.println("   第一个样本的输出 (10个类别的logits):");
        for (int i = 0; i < 10; i++) {
            System.out.printf("     类别%d: %.4f%n", i, outputData[i]);
        }
        System.out.println();

        // 4. 推理模式前向传播
        System.out.println("4. 推理模式前向传播:");
        model.eval();

        output = model.forward(input);

        System.out.println(UmlPrinter.getDotGraph(output));


        System.out.println("   输出形状: " + shapeToString(output.getValue().getShape()));

        // 找到预测类别
        outputData = output.getValue().getArray();
        System.out.println("   预测结果:");
        for (int n = 0; n < 4; n++) {
            int maxIdx = 0;
            float maxVal = outputData[n * 10];
            for (int i = 1; i < 10; i++) {
                if (outputData[n * 10 + i] > maxVal) {
                    maxVal = outputData[n * 10 + i];
                    maxIdx = i;
                }
            }
            System.out.printf("     样本%d: 预测类别=%d (置信度=%.4f)%n", n, maxIdx, maxVal);
        }
        System.out.println();

        // 5. 模型信息总结
        System.out.println("5. 模型总结:");
        System.out.println("   ----------------------------------------");
        System.out.println("   模型名称: LeNet-5");
        System.out.println("   输入形状: (N, 1, 28, 28)");
        System.out.println("   输出形状: (N, 10)");
        System.out.println("   卷积层数: 2");
        System.out.println("   全连接层数: 3");
        System.out.println("   ----------------------------------------");
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

