package io.leavesfly.tinyai.example.classify;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.Monitor;
import io.leavesfly.tinyai.ml.Plot;
import io.leavesfly.tinyai.ml.Trainer;
import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.simple.SpiralDateSet;
import io.leavesfly.tinyai.ml.evaluator.AccuracyEval;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Classify;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.MlpBlock;


import java.util.List;

/**
 * 螺旋数据分类示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 使用MLP神经网络对螺旋数据集进行分类训练的示例。
 * 螺旋数据集是一个经典的非线性可分数据集，用于测试神经网络的非线性拟合能力。
 * 该示例展示了两种训练方式：
 * 1. 使用Trainer类的高级训练方式
 * 2. 手动实现训练循环的低级训练方式
 * 同时演示了训练过程中的损失和准确率监控，以及训练结果的可视化。
 */
public class SpiralMlpExam {

    /**
     * 主函数，执行螺旋数据分类训练
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        test1();
    }

    /**
     * 使用Trainer类的高级训练方式
     */
    public static void test() {

        int maxEpoch = 300;
        int batchSize = 10;

        float learRate = 1.0f;

        int inputSize = 2;
        int hiddenSize = 30;
        int outputSize = 3;

        Block block = new MlpBlock("MlpBlock", batchSize, null, inputSize, hiddenSize, hiddenSize, outputSize);
        Model model = new Model("SpiralMlpExam", block);

        ArrayDataset dataSet = new SpiralDateSet(batchSize);

        Optimizer optimizer = new SGD(model, learRate);
        Evaluator evaluator = new AccuracyEval(new Classify(), model, dataSet);

        Trainer trainer = new Trainer(maxEpoch, new Monitor(), evaluator);
        Loss loss = new SoftmaxCrossEntropy();

        trainer.init(dataSet, model, loss, optimizer);

        trainer.train(true);

        trainer.evaluate();
    }


    /**
     * 手动实现训练循环的低级训练方式
     * 包含损失和准确率监控，以及训练结果可视化
     */
    public static void test1() {
        //==定义超参数
        int maxEpoch = 300;
        int batchSize = 10;

        int inputSize = 2;
        int hiddenSize = 30;
        int outputSize = 3;

        float learRate = 1.0f;

        ArrayDataset dataSet = new SpiralDateSet(batchSize);
        dataSet.prepare();
        dataSet.shuffle();
        List<Batch> batches = dataSet.getBatches();

        Block block = new MlpBlock("MlpBlock", batchSize, null, inputSize, hiddenSize, hiddenSize, outputSize);
        Model model = new Model("SpiralMlpExam", block);

        Optimizer optimizer = new SGD(model, learRate);
        Loss lossFunc = new SoftmaxCrossEntropy();
        Classify accuracy = new Classify();

        float[] lossArray = new float[maxEpoch];
        float[] accArray = new float[maxEpoch];
        for (int i = 0; i < maxEpoch; i++) {
            float sumLoss = 0f;
            float sumAcc = 0f;
            for (Batch batch : batches) {
                Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
                Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);
                Variable predict = model.forward(variableX);
                Variable loss = lossFunc.loss(variableY, predict);
                float acc = accuracy.accuracyRate(variableY, predict);

                model.clearGrads();
                loss.backward();

                optimizer.update();
                sumLoss += loss.getValue().getNumber().floatValue() * batch.getSize();
                sumAcc += acc * batch.getSize();
            }

            sumLoss = sumLoss / dataSet.getSize();
            sumAcc = sumAcc / dataSet.getSize();
            lossArray[i] = sumLoss;
            accArray[i] = sumAcc;

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("i=" + i + ", loss:" + sumLoss + " acc: " + sumAcc);
            }
        }
        //预测与绘制
        Variable variableX = new Variable(NdArray.likeRandom(-1, 1,  Shape.of(2000, 2)));
        Variable y = model.forward(variableX);
        SpiralDateSet spiralDateSet = SpiralDateSet.toSpiralDateSet(variableX, y);

        Plot plot = new Plot();
//        plot.line(Utils.toFloat(Utils.getSeq(maxEpoch)), lossArray, "loss");
//        plot.line(Utils.toFloat(Utils.getSeq(maxEpoch)), accArray, "accuracy");
        int[] type = new int[]{0, 1, 2};
        plot.scatter(dataSet, type);
        plot.scatter(spiralDateSet, type);
        plot.show();
    }
}