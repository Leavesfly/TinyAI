package io.leavesfly.tinyai.example.regress;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.Plot;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.dataset.simple.SinDataSet;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.MlpBlock;
import io.leavesfly.tinyai.util.Config;

import java.util.List;

/**
 * MLP拟合正弦曲线示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 该示例演示如何使用多层感知机(MLP)神经网络拟合带有噪声的正弦曲线数据。
 * MLP是一种前馈神经网络，能够学习非线性函数映射，适用于回归和分类任务。
 */
public class MlpSinExam {

    /**
     * 主函数，执行MLP训练和可视化
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {

        //====== 1,生成数据====
        int batchSize = 100;
        DataSet dataSet = new SinDataSet(batchSize);
        dataSet.prepare();
        List<Batch> batches = dataSet.getBatches();

        Variable variableX = batches.get(0).toVariableX().setName("x").setRequireGrad(false);
        Variable variableY = batches.get(0).toVariableY().setName("y").setRequireGrad(false);

        Block block = new MlpBlock("MlpBlock", batchSize, Config.ActiveFunc.Sigmoid, 1, 10, 1);

        Model model = new Model("MlpSinExam", block);
        Optimizer optimizer = new SGD(model, 0.2f);
        Loss lossFunc = new MeanSquaredLoss();

        //train
        int maxEpoch = 10000;

        for (int i = 0; i < maxEpoch; i++) {
            Variable predictY = model.forward(variableX);
            Variable loss = lossFunc.loss(variableY, predictY);

            model.clearGrads();
            loss.backward();

            optimizer.update();

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("i=" + i + " loss:" + loss.getValue().getNumber());
            }
        }

//        model.plot(variableX);
//
        Variable predictY = model.forward(variableX);
        float[] p_y = predictY.transpose().getValue().getMatrix()[0];
        float[] x = variableX.transpose().getValue().getMatrix()[0];
        float[] y = variableY.transpose().getValue().getMatrix()[0];
        Plot plot = new Plot();
        plot.scatter(x, y);
        plot.line(x, p_y, "line");
        plot.show();
    }
}