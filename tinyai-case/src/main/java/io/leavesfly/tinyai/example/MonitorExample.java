package io.leavesfly.tinyai.example;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.Monitor;
import io.leavesfly.tinyai.ml.Trainer;
import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.simple.SpiralDateSet;
import io.leavesfly.tinyai.ml.evaluator.AccuracyEval;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Classify;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.MlpBlock;

import java.util.List;

/**
 * Monitor监控器使用示例
 * 
 * 该示例演示了如何使用完善后的Monitor类来监控模型训练过程，
 * 包括损失值、准确率、训练时间等信息的收集和可视化。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class MonitorExample {

    /**
     * 主函数，执行Monitor使用示例
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        // 定义超参数
        int maxEpoch = 100;
        int batchSize = 30;
        float learRate = 1.0f;
        int inputSize = 2;
        int hiddenSize = 30;
        int outputSize = 3;

        // 创建模型
        Block block = new MlpBlock("MlpBlock", batchSize, null, inputSize, hiddenSize, hiddenSize, outputSize);
        Model model = new Model("MonitorExample", block);

        // 创建数据集
        ArrayDataset dataSet = new SpiralDateSet(batchSize);

        // 创建优化器、损失函数和评估器
        Optimizer optimizer = new SGD(model, learRate);
        Loss loss = new SoftmaxCrossEntropy();
        Evaluator evaluator = new AccuracyEval(new Classify(), model, dataSet);

        // 创建带日志文件保存功能的Monitor
        Monitor monitor = new Monitor("training_log.txt");

        // 创建训练器并初始化
        Trainer trainer = new Trainer(maxEpoch, monitor, evaluator);
        trainer.init(dataSet, model, loss, optimizer);

        // 开始训练
        System.out.println("开始训练...");
        trainer.train(true);

        // 评估模型
        System.out.println("\n模型评估:");
        trainer.evaluate();

        // 展示Monitor收集的信息
        System.out.println("\n训练信息统计:");
        System.out.println("最佳训练损失: " + String.format("%.6f", monitor.getBestLoss()));
        System.out.println("最佳训练准确率: " + String.format("%.4f", monitor.getBestAccuracy()));
        
        // 获取训练时间信息
        List<Long> timeList = monitor.getTimeList();
        if (!timeList.isEmpty()) {
            long totalTime = timeList.stream().mapToLong(Long::longValue).sum();
            System.out.println("总训练时间: " + totalTime + "ms");
            System.out.println("平均每轮时间: " + String.format("%.2f", (double) totalTime / timeList.size()) + "ms");
        }
        
        // 关闭训练器资源
        trainer.shutdown();
        
        System.out.println("\n训练完成，日志已保存到 training_log.txt");
    }
}