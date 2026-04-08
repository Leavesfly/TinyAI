package io.leavesfly.tinyai.nnet.examples;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;

import java.util.Map;

/**
 * 示例5: 模型序列化和加载
 * <p>
 * 本示例展示如何:
 * 1. 使用stateDict保存模型参数
 * 2. 从stateDict加载模型参数
 * 3. 验证保存和加载的正确性
 */
public class ModelSerialization {

    /**
     * 简单的两层网络
     */
    static class SimpleModel extends Module {
        private final Linear fc1;
        private final ReLU relu;
        private final Linear fc2;

        public SimpleModel(String name, int inputSize, int hiddenSize, int outputSize) {
            super(name);

            fc1 = new Linear("fc1", inputSize, hiddenSize, true);
            relu = new ReLU("relu");
            fc2 = new Linear("fc2", hiddenSize, outputSize, true);

            registerModule("fc1", fc1);
            registerModule("relu", relu);
            registerModule("fc2", fc2);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            x = fc1.forward(x);
            x = relu.forward(x);
            x = fc2.forward(x);
            return x;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== 模型序列化示例 ===\n");

        // 1. 创建并训练原始模型
        System.out.println("1. 创建原始模型");
        SimpleModel originalModel = new SimpleModel("original", 10, 20, 5);
        originalModel.train();

        System.out.println("   模型参数:");
        for (String name : originalModel.namedParameters().keySet()) {
            Shape shape = originalModel.namedParameters().get(name).data().getShape();
            System.out.println("   - " + name + ": " + shapeToString(shape));
        }
        System.out.println();

        // 2. 执行一次前向传播（模拟训练后的状态）
        System.out.println("2. 执行前向传播（获取训练后的参数状态）");
        float[] inputData = new float[2 * 10];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) (Math.random() * 0.1);
        }
        Variable input = new Variable(NdArray.of(inputData, Shape.of(2, 10)));
        Variable originalOutput = originalModel.forward(input);

        System.out.println("   输入形状: " + shapeToString(input.getValue().getShape()));
        System.out.println("   输出形状: " + shapeToString(originalOutput.getValue().getShape()));
        System.out.println("   输出示例 (前5个值):");
        float[] outputData = originalOutput.getValue().getArray();
        for (int i = 0; i < Math.min(5, outputData.length); i++) {
            System.out.printf("     %.6f%n", outputData[i]);
        }
        System.out.println();

        // 3. 保存模型参数（stateDict）
        System.out.println("3. 保存模型参数（stateDict）");
        Map<String, NdArray> stateDict = originalModel.stateDict();

        System.out.println("   StateDict内容:");
        for (Map.Entry<String, NdArray> entry : stateDict.entrySet()) {
            String name = entry.getKey();
            Shape shape = entry.getValue().getShape();
            System.out.println("   - " + name + ": " + shapeToString(shape));
        }
        System.out.println("   StateDict条目数: " + stateDict.size());
        System.out.println();

        // 显示一些参数值（验证用）
        System.out.println("   fc1.weight的前5个值:");
        float[] fc1Weight = stateDict.get("fc1.weight").getArray();
        for (int i = 0; i < Math.min(5, fc1Weight.length); i++) {
            System.out.printf("     %.6f%n", fc1Weight[i]);
        }
        System.out.println();

        // 4. 创建新模型并加载参数
        System.out.println("4. 创建新模型并加载参数");
        SimpleModel newModel = new SimpleModel("new", 10, 20, 5);

        System.out.println("   加载前的参数状态（随机初始化）:");
        float[] newFc1WeightBefore = newModel.namedParameters().get("fc1.weight").data().getArray();
        System.out.println("   fc1.weight的前5个值:");
        for (int i = 0; i < Math.min(5, newFc1WeightBefore.length); i++) {
            System.out.printf("     %.6f%n", newFc1WeightBefore[i]);
        }
        System.out.println();

        // 加载参数
        newModel.loadStateDict(stateDict);
        System.out.println("   参数加载完成");
        System.out.println();

        System.out.println("   加载后的参数状态:");
        float[] newFc1WeightAfter = newModel.namedParameters().get("fc1.weight").data().getArray();
        System.out.println("   fc1.weight的前5个值:");
        for (int i = 0; i < Math.min(5, newFc1WeightAfter.length); i++) {
            System.out.printf("     %.6f%n", newFc1WeightAfter[i]);
        }
        System.out.println();

        // 5. 验证参数是否正确加载
        System.out.println("5. 验证参数加载正确性");
        boolean allMatch = true;
        for (String name : originalModel.namedParameters().keySet()) {
            NdArray originalParam = originalModel.namedParameters().get(name).data();
            NdArray newParam = newModel.namedParameters().get(name).data();

            float[] originalData = originalParam.getArray();
            float[] newData = newParam.getArray();

            boolean match = arraysEqual(originalData, newData);
            System.out.println("   " + name + ": " + (match ? "✓ 匹配" : "✗ 不匹配"));

            if (!match) {
                allMatch = false;
            }
        }
        System.out.println("   总体结果: " + (allMatch ? "✓ 所有参数匹配" : "✗ 存在不匹配"));
        System.out.println();

        // 6. 验证输出是否一致
        System.out.println("6. 验证模型输出一致性");
        newModel.eval();
        Variable newOutput = newModel.forward(input);

        float[] originalOutputData = originalOutput.getValue().getArray();
        float[] newOutputData = newOutput.getValue().getArray();

        System.out.println("   原始模型输出 (前5个值):");
        for (int i = 0; i < Math.min(5, originalOutputData.length); i++) {
            System.out.printf("     %.6f%n", originalOutputData[i]);
        }
        System.out.println();

        System.out.println("   加载后模型输出 (前5个值):");
        for (int i = 0; i < Math.min(5, newOutputData.length); i++) {
            System.out.printf("     %.6f%n", newOutputData[i]);
        }
        System.out.println();

        boolean outputMatch = arraysEqual(originalOutputData, newOutputData);
        System.out.println("   输出一致性: " + (outputMatch ? "✓ 一致" : "✗ 不一致"));
        System.out.println();

        // 7. 使用场景说明
        System.out.println("7. 使用场景:");
        System.out.println("   ----------------------------------------");
        System.out.println("   1. 模型保存: 训练后保存最佳模型参数");
        System.out.println("   2. 模型加载: 加载预训练模型用于推理");
        System.out.println("   3. 断点续训: 保存检查点，从断点恢复训练");
        System.out.println("   4. 迁移学习: 加载预训练权重，微调特定任务");
        System.out.println("   5. 模型共享: 在不同环境间共享训练好的模型");
        System.out.println("   ----------------------------------------");
        System.out.println();

        // 8. 注意事项
        System.out.println("8. 注意事项:");
        System.out.println("   ----------------------------------------");
        System.out.println("   1. stateDict只包含参数和buffer，不包含模型结构");
        System.out.println("   2. 加载时需要先创建相同结构的模型");
        System.out.println("   3. 参数名称必须完全匹配");
        System.out.println("   4. 参数形状必须兼容");
        System.out.println("   5. 实际应用中应序列化到文件（如JSON、二进制等）");
        System.out.println("   ----------------------------------------");
        System.out.println();

        System.out.println("=== 示例完成 ===");
    }

    /**
     * 比较两个float数组是否相等（允许小误差）
     */
    private static boolean arraysEqual(float[] a, float[] b) {
        if (a.length != b.length) {
            return false;
        }

        float epsilon = 1e-6f;
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > epsilon) {
                return false;
            }
        }

        return true;
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
