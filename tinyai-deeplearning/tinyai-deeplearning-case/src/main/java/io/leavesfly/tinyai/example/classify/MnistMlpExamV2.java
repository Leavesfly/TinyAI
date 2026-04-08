package io.leavesfly.tinyai.example.classify;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.visual.Monitor;
import io.leavesfly.tinyai.ml.training.Trainer;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.dataset.simple.MnistDataSet;
import io.leavesfly.tinyai.ml.evaluator.AccuracyEval;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Classify;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * 手写数字识别示例 - V2 API版本
 *
 * @author leavesfly
 * @version 0.02
 * <p>
 * 使用V2 API重新实现的MNIST手写数字识别MLP分类器。
 * 展示了V2版本的主要特性：
 * 1. Sequential容器构建网络
 * 2. 统一的参数初始化策略
 * 3. 使用Trainer托管训练循环
 * 4. 训练/推理模式切换
 */
public class MnistMlpExamV2 {

    /**
     * 主函数，执行MNIST手写数字识别训练
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        //===1,定义超参数===
        int maxEpoch = 50;
        int batchSize = 100;

        int inputSize = 28 * 28;
        int hiddenSize1 = 100;
        int hiddenSize2 = 100;
        int outputSize = 10;

        float learRate = 0.1f;

        //===2,定义模型===
        // 使用V2 Sequential构建3层MLP网络
        Sequential sequential = new Sequential("MnistMlpV2")
                .add(new Linear("fc1", inputSize, hiddenSize1))
                .add(new ReLU())
                .add(new Linear("fc2", hiddenSize1, hiddenSize2))
                .add(new ReLU())
                .add(new Linear("fc3", hiddenSize2, outputSize));

        // V2参数初始化：使用Kaiming初始化权重，零初始化偏置
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
        Model model = new Model("MnistMlpExamV2", sequential);

        // 准备数据集及训练组件
        DataSet mnistDataSet = new MnistDataSet(batchSize);
        Evaluator evaluator = new AccuracyEval(new Classify(), model, mnistDataSet);
        Optimizer optimizer = new SGD(model, learRate);
        Loss lossFunc = new SoftmaxCrossEntropy();

        // 使用Trainer管理训练流程
        Trainer trainer = new Trainer(maxEpoch, new Monitor(), evaluator);
        trainer.configureParallelTraining(true, 4); // 启用并行训练，线程数可按需调整
        trainer.init(mnistDataSet, model, lossFunc, optimizer);

        //===3,模型训练==
        System.out.println("开始训练 MNIST MLP 分类器 (V2 API, Trainer版)");
        sequential.train(); // 确保进入训练模式
        trainer.train(true);

        //===4,效果评估==
        sequential.eval();
        trainer.evaluate();

//        model.plot();  // V2版本暂不支持plot，需要后续扩展
    }
}
