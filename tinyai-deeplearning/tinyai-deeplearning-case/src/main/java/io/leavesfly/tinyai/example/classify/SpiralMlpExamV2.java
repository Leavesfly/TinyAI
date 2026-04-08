package io.leavesfly.tinyai.example.classify;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.simple.SpiralDateSet;
import io.leavesfly.tinyai.ml.evaluator.AccuracyEval;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Classify;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ml.visual.Plot;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.init.Initializers;

import java.util.List;

/**
 * 螺旋数据分类示例 - V2 API版本
 *
 * @author leavesfly
 * @version 0.02
 * <p>
 * 使用V2 API重新实现的螺旋数据集MLP分类器。
 * 螺旋数据集是一个经典的非线性可分数据集，用于测试神经网络的非线性拟合能力。
 * 该示例展示了两种训练方式：
 * 1. 简化的训练方式（使用评估器）
 * 2. 手动实现训练循环的详细训练方式（包含损失和准确率监控，以及训练结果的可视化）
 * 同时演示了训练过程中的模式切换（train/eval）。
 */
public class SpiralMlpExamV2 {

    /**
     * 主函数，执行螺旋数据分类训练
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        test1();  // 使用详细的手动训练方式
    }

    /**
     * 简化的训练方式（V2版本）
     * 使用评估器进行训练和评估，适合快速原型开发
     */
    public static void test() {
        System.out.println("=== Spiral MLP 分类器 (V2 API - 简化训练方式) ===");

        int maxEpoch = 300;
        int batchSize = 10;

        float learRate = 0.1f;  // ReLU需要更小的学习率

        int inputSize = 2;
        int hiddenSize = 30;
        int outputSize = 3;

        // 使用V2 Sequential构建网络
        Sequential sequential = new Sequential("SpiralMlpV2")
                .add(new Linear("fc1", inputSize, hiddenSize))
                .add(new ReLU())
                .add(new Linear("fc2", hiddenSize, hiddenSize))
                .add(new ReLU())
                .add(new Linear("fc3", hiddenSize, outputSize));

        // 参数初始化
        sequential.apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                Initializers.kaimingUniform(linear.getWeight().data());
                if (linear.getBias() != null) {
                    Initializers.zeros(linear.getBias().data());
                }
            }
        });

        // 将V2 Sequential包装为V1 Model，以便与现有组件兼容
        Model model = new Model("SpiralMlpExamV2", sequential);

        ArrayDataset dataSet = new SpiralDateSet(batchSize);
        dataSet.prepare();
        dataSet.shuffle();

        Optimizer optimizer = new SGD(model, learRate);
        Evaluator evaluator = new AccuracyEval(new Classify(), model, dataSet);
        Loss lossFunc = new SoftmaxCrossEntropy();
        Classify accuracy = new Classify();

        // 手动训练循环（V2版本）
        sequential.train();
        List<Batch> batches = dataSet.getBatches();

        System.out.println("训练参数:");
        System.out.println("  轮数: " + maxEpoch);
        System.out.println("  批大小: " + batchSize);
        System.out.println("  学习率: " + learRate);

        float[] lossArray = new float[maxEpoch];
        float[] accArray = new float[maxEpoch];

        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            float sumLoss = 0f;
            float sumAcc = 0f;

            for (Batch batch : batches) {
                Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
                Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

                Variable predict = sequential.forward(variableX);
                Variable loss = lossFunc.loss(variableY, predict);
                float acc = accuracy.accuracyRate(variableY, predict);

                sequential.clearGrads();
                loss.backward();
                optimizer.update();

                sumLoss += loss.getValue().getNumber().floatValue() * batch.getSize();
                sumAcc += acc * batch.getSize();
            }

            sumLoss = sumLoss / dataSet.getSize();
            sumAcc = sumAcc / dataSet.getSize();
            lossArray[epoch] = sumLoss;
            accArray[epoch] = sumAcc;

            if (epoch % (maxEpoch / 10) == 0 || (epoch == maxEpoch - 1)) {
                System.out.printf("epoch = %d, loss: %.6f, accuracy: %.6f%n", epoch, sumLoss, sumAcc);
            }
        }

        // 评估
        sequential.eval();
        evaluator.evaluate();

        System.out.printf("最终损失: %.6f, 最终准确率: %.6f%n", lossArray[maxEpoch - 1], accArray[maxEpoch - 1]);
    }

    /**
     * 手动实现训练循环的详细训练方式
     * 包含损失和准确率监控，以及训练结果可视化
     * 展示V2 API的完整训练流程和模式切换
     */
    public static void test1() {
        System.out.println("=== Spiral MLP 分类器 (V2 API - 详细训练方式) ===");

        //==定义超参数
        int maxEpoch = 300;
        int batchSize = 10;

        int inputSize = 2;
        int hiddenSize = 30;
        int outputSize = 3;

        float learRate = 0.1f;  // ReLU需要更小的学习率

        // 准备数据集
        ArrayDataset dataSet = new SpiralDateSet(batchSize);
        dataSet.prepare();
        dataSet.shuffle();
        List<Batch> batches = dataSet.getBatches();

        // 使用V2 Sequential构建网络
        Sequential sequential = new Sequential("SpiralMlpV2")
                .add(new Linear("fc1", inputSize, hiddenSize))
                .add(new ReLU())
                .add(new Linear("fc2", hiddenSize, hiddenSize))
                .add(new ReLU())
                .add(new Linear("fc3", hiddenSize, outputSize));

        // 参数初始化
        sequential.apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                Initializers.kaimingUniform(linear.getWeight().data());
                if (linear.getBias() != null) {
                    Initializers.zeros(linear.getBias().data());
                }
            }
        });

        // 将V2 Sequential包装为V1 Model，以便与现有组件兼容
        Model model = new Model("SpiralMlpExamV2", sequential);

        // 配置优化器和损失函数
        Optimizer optimizer = new SGD(model, learRate);
        Loss lossFunc = new SoftmaxCrossEntropy();
        Classify accuracy = new Classify();

        // 训练监控数组
        float[] lossArray = new float[maxEpoch];
        float[] accArray = new float[maxEpoch];

        System.out.println("训练参数:");
        System.out.println("  轮数: " + maxEpoch);
        System.out.println("  批大小: " + batchSize);
        System.out.println("  学习率: " + learRate);
        System.out.println("  数据集大小: " + dataSet.getSize());
        System.out.println("  批次数量: " + batches.size());

        // 训练循环
        sequential.train();  // 设置为训练模式
        for (int i = 0; i < maxEpoch; i++) {
            float sumLoss = 0f;
            float sumAcc = 0f;
            int batchCount = 0;

            for (Batch batch : batches) {
                Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
                Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

                Variable predict = sequential.forward(variableX);
                Variable loss = lossFunc.loss(variableY, predict);
                float acc = accuracy.accuracyRate(variableY, predict);

                sequential.clearGrads();
                loss.backward();
                optimizer.update();

                sumLoss += loss.getValue().getNumber().floatValue() * batch.getSize();
                sumAcc += acc * batch.getSize();
                batchCount++;
            }

            sumLoss = sumLoss / dataSet.getSize();
            sumAcc = sumAcc / dataSet.getSize();
            lossArray[i] = sumLoss;
            accArray[i] = sumAcc;

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.printf("i=%d, loss:%.6f, acc:%.6f (批次:%d)%n", i, sumLoss, sumAcc, batchCount);
            }
        }

        System.out.println("\n=== 训练完成 ===");
        System.out.printf("最终损失: %.6f%n", lossArray[maxEpoch - 1]);
        System.out.printf("最终准确率: %.6f%n", accArray[maxEpoch - 1]);

        // 预测与绘制（推理模式）
        System.out.println("\n=== 生成预测结果用于可视化 ===");
        sequential.eval();  // 设置为推理模式

        Variable variableX = new Variable(NdArray.likeRandom(-1, 1, Shape.of(2000, 2)), "input");
        Variable y = sequential.forward(variableX);
        SpiralDateSet spiralDateSet = SpiralDateSet.toSpiralDateSet(variableX, y);

//        System.out.println(UmlPrinter.getDotGraph(y));

        // 可视化
        Plot plot = new Plot();
//        plot.line(Utils.toFloat(Utils.getSeq(maxEpoch)), lossArray, "loss");
//        plot.line(Utils.toFloat(Utils.getSeq(maxEpoch)), accArray, "accuracy");

        int[] type = new int[]{0, 1, 2};
        plot.scatter(dataSet, type);
        plot.scatter(spiralDateSet, type);
        plot.show();

        // 显示模型信息
        System.out.println("\n=== 模型信息 ===");
        System.out.println("模型结构:");
        System.out.println(model);

        System.out.println("\n参数统计:");
        int totalParams = 0;
        var params = sequential.namedParameters();
        for (var entry : params.entrySet()) {
            int paramCount = entry.getValue().data().getShape().size();
            totalParams += paramCount;
            System.out.println(entry.getKey() + ": " + paramCount + " 参数");
        }
        System.out.println("总参数量: " + totalParams);
    }
}
