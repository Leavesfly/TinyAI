package io.leavesfly.tinyai.example;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.visual.Monitor;
import io.leavesfly.tinyai.ml.training.Trainer;
import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.evaluator.AccuracyEval;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Classify;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

import java.util.Random;

/**
 * Trainer Builder API 演示
 * <p>
 * 展示如何使用新的Builder模式来简化Trainer的创建和使用
 *
 * @author TinyAI
 * @version 2.0
 */
public class TrainerBuilderDemo {

    public static void main(String[] args) {
        System.out.println("=== Trainer Builder API 演示 ===\n");

        // 演示1: 最简单的使用方式
        demo1_MinimalUsage();

        System.out.println("\n" + "=".repeat(50) + "\n");

        // 演示2: 完整配置
        demo2_FullConfiguration();

        System.out.println("\n" + "=".repeat(50) + "\n");

        // 演示3: 高级特性
        demo3_AdvancedFeatures();
    }

    /**
     * 演示1: 最简单的使用方式 - 最少配置即可训练
     */
    private static void demo1_MinimalUsage() {
        System.out.println("【演示1】最简单的使用方式");
        System.out.println("只需要4个必需参数: Model, DataSet, Loss, Optimizer\n");

        // 准备数据和模型
        DataSet dataSet = createSimpleDataset();
        Model model = createSimpleModel();
        Loss loss = new SoftmaxCrossEntropy();
        Optimizer optimizer = new Adam(model);

        // 旧方式需要6步: Monitor + Evaluator + Trainer + init + train
        // 新方式只需1步,链式调用完成所有配置并训练!
        Trainer.builder()
                .model(model)
                .dataSet(dataSet)
                .loss(loss)
                .optimizer(optimizer)
                .epochs(10)  // 可选,默认100
                .build()
                .train();    // 使用默认shuffle=true

        System.out.println("✅ 训练完成!");
    }

    /**
     * 演示2: 完整配置 - 自定义所有参数
     */
    private static void demo2_FullConfiguration() {
        System.out.println("【演示2】完整配置 - 自定义所有参数");
        System.out.println("可以灵活配置Monitor、Evaluator等可选参数\n");

        DataSet dataSet = createSimpleDataset();
        dataSet.prepare();
        Model model = createSimpleModel();
        Loss loss = new SoftmaxCrossEntropy();
        Optimizer optimizer = new Adam(model);

        // 自定义Monitor和Evaluator
        Monitor monitor = new Monitor("output/builder_demo.log");
        Evaluator evaluator = new AccuracyEval(new Classify(), model, dataSet);

        // 链式调用配置所有参数
        Trainer trainer = Trainer.builder()
                .model(model)
                .dataSet(dataSet)
                .loss(loss)
                .optimizer(optimizer)
                .epochs(20)
                .monitor(monitor)           // 自定义Monitor
                .evaluator(evaluator)       // 自定义Evaluator
                .shuffle(true)              // 显式设置shuffle
                .validationInterval(5)      // 每5个epoch评估一次
                .build();

        // 训练
        trainer.train();

        // 评估
        System.out.println("\n最终评估:");
        trainer.evaluate();

        System.out.println("✅ 训练和评估完成!");
    }

    /**
     * 演示3: 高级特性 - 早停、梯度裁剪、并行训练
     */
    private static void demo3_AdvancedFeatures() {
        System.out.println("【演示3】高级特性 - 早停、梯度裁剪、并行训练");
        System.out.println("展示Builder模式如何简化复杂功能的配置\n");

        DataSet dataSet = createSimpleDataset();
        Model model = createSimpleModel();
        Loss loss = new SoftmaxCrossEntropy();
        Optimizer optimizer = new Adam(model);

        // 使用Builder一次性配置所有高级特性
        Trainer trainer = Trainer.builder()
                .model(model)
                .dataSet(dataSet)
                .loss(loss)
                .optimizer(optimizer)
                .epochs(100)
                .monitorLogFile("output/advanced_demo.log")  // 快捷方式创建Monitor
                .earlyStopping(10, 0.0001f)   // 早停: patience=10, minDelta=0.0001
                .gradientClipping(5.0f)       // 梯度裁剪: maxNorm=5.0
                .enableParallel(true)         // 启用并行训练(自动线程数)
                .shuffle(true)
                .build();

        System.out.println("配置完成:");
        System.out.println("  - 早停: patience=10, minDelta=0.0001");
        System.out.println("  - 梯度裁剪: maxNorm=5.0 (L2范数)");
        System.out.println("  - 并行训练: 已启用\n");

        // 训练
        trainer.train();

        System.out.println("✅ 高级特性演示完成!");
    }

    /**
     * 创建简单的数据集
     */
    private static DataSet createSimpleDataset() {
        Random random = new Random(42);
        int numSamples = 300;
        int inputDim = 10;
        int numClasses = 3;

        float[][] xData = new float[numSamples][inputDim];
        float[][] yData = new float[numSamples][numClasses];

        for (int i = 0; i < numSamples; i++) {
            int label = i % numClasses;

            // 生成输入数据
            for (int j = 0; j < inputDim; j++) {
                xData[i][j] = (float) (random.nextGaussian() + label);
            }

            // one-hot标签
            for (int j = 0; j < numClasses; j++) {
                yData[i][j] = (j == label) ? 1.0f : 0.0f;
            }
        }

        NdArray[] xsArr = new NdArray[numSamples];
        NdArray[] ysArr = new NdArray[numSamples];

        for (int i = 0; i < numSamples; i++) {
            xsArr[i] = NdArray.of(xData[i], Shape.of(1, inputDim));
            ysArr[i] = NdArray.of(yData[i], Shape.of(1, numClasses));
        }

        // 使用 SimpleArrayDataset 避免匿名内部类初始化顺序问题
        SimpleArrayDataset dataset = new SimpleArrayDataset(32, xsArr, ysArr);
        return dataset;
    }

    /**
     * 简单的 ArrayDataset 实现，避免匿名内部类的复杂性
     */
    private static class SimpleArrayDataset extends ArrayDataset {
        public SimpleArrayDataset(int batchSize, NdArray[] xs, NdArray[] ys) {
            super(batchSize);
            this.xs = xs;
            this.ys = ys;
        }

        @Override
        protected DataSet build(int batchSize, NdArray[] _xs, NdArray[] _ys) {
            return new SimpleArrayDataset(batchSize, _xs, _ys);
        }

        @Override
        public void doPrepare() {
            // 调用 splitDataset 来初始化训练/测试/验证集
            splitDataset(0.8f, 0.1f, 0.1f);
        }
    }

    /**
     * 创建简单的MLP模型
     */
    private static Model createSimpleModel() {
        Module network = new Sequential("mlp")
                .add(new Linear("fc1", 10, 32))
                .add(new ReLU("relu1"))
                .add(new Linear("fc2", 32, 16))
                .add(new ReLU("relu2"))
                .add(new Linear("fc3", 16, 3));

        return new Model("builder_demo", network);
    }
}
