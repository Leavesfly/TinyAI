package io.leavesfly.tinyai.example.parallel;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.dataset.simple.SpiralDateSet;
import io.leavesfly.tinyai.ml.evaluator.AccuracyEval;
import io.leavesfly.tinyai.ml.loss.Classify;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.nnet.block.MlpBlock;
import io.leavesfly.tinyai.util.Config;

/**
 * 并行训练测试示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 该示例演示如何使用TinyDL的并行训练功能来加速模型训练过程。
 * 并行训练通过多线程技术同时处理多个批次的数据，从而提高训练效率。
 */
public class ParallelTrainingTest {

    /**
     * 主函数，执行并行训练测试
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== TinyDL 并行训练测试 ===");

        // 测试参数
        int maxEpoch = 5;
        int batchSize = 16;
        int threadCount = 2;

        try {
            // 准备数据集（螺旋分类数据）
            SpiralDateSet dataSet = new SpiralDateSet(batchSize);

            // 创建模型
            io.leavesfly.tinyai.ml.Model model = createModel("并行训练模型");

            // 创建损失函数和优化器
            SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
            Adam optimizer = new Adam(model, 0.01f, 0.9f, 0.999f, 1e-8f);

            // 测试并行训练
            System.out.println("开始并行训练测试...");
            testParallelTraining(dataSet, model, loss, optimizer, maxEpoch, threadCount);

        } catch (Exception e) {
            System.err.println("测试过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("=== 并行训练测试完成 ===");
    }

    /**
     * 创建MLP模型
     *
     * @param name 模型名称
     * @return 创建的MLP模型
     */
    private static io.leavesfly.tinyai.ml.Model createModel(String name) {
        MlpBlock mlpBlock = new MlpBlock(name, 32, Config.ActiveFunc.ReLU, 2, 16, 16, 3);

        return new io.leavesfly.tinyai.ml.Model(name, mlpBlock);
    }

    /**
     * 测试并行训练
     *
     * @param dataSet     训练数据集
     * @param model       模型
     * @param loss        损失函数
     * @param optimizer   优化器
     * @param maxEpoch    最大训练轮数
     * @param threadCount 并行线程数
     */
    private static void testParallelTraining(SpiralDateSet dataSet, Model model,
                                             SoftmaxCrossEntropy loss, Adam optimizer,
                                             int maxEpoch, int threadCount) {
        long startTime = System.currentTimeMillis();

        io.leavesfly.tinyai.ml.Monitor monitor = new io.leavesfly.tinyai.ml.Monitor();
        Classify classify = new Classify();
        AccuracyEval evaluator = new AccuracyEval(classify, model, dataSet);

        // 创建并行训练器
        io.leavesfly.tinyai.ml.Trainer trainer = new io.leavesfly.tinyai.ml.Trainer(maxEpoch, monitor, evaluator, true, threadCount);
        trainer.init(dataSet, model, loss, optimizer);

        System.out.println("并行训练配置: 线程数=" + trainer.getParallelThreadCount() +
                ", 是否启用=" + trainer.isParallelTrainingEnabled());

        // 执行简化版并行训练（适用于模型不支持序列化的情况）
        trainer.simplifiedParallelTrain(true);

        long endTime = System.currentTimeMillis();
        System.out.println("训练完成，耗时: " + (endTime - startTime) + " ms");

        // 评估结果 - 暂时跳过图表绘制
        // trainer.evaluate();

        // 清理资源
        trainer.shutdown();
    }
}