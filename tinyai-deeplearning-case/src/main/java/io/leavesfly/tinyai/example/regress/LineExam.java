package io.leavesfly.tinyai.example.regress;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Plot;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Uml;


import java.util.Random;

/**
 * 线性回归示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 该示例演示如何使用梯度下降法拟合带有噪声的线性数据。
 * 线性回归是机器学习中最基础的回归算法之一，用于建立输入特征与目标值之间的线性关系。
 */
public class LineExam {

    /**
     * 主函数，执行线性回归训练和可视化
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {

        //====== 1，生成数据====
        Random random = new Random(0);
        float[] x = new float[100];
        for (int i = 0; i < x.length; i++) {
            x[i] = random.nextFloat();
        }

        float[] y = new float[100];
        for (int i = 0; i < y.length; i++) {
            y[i] = 5 + 2 * x[i] + random.nextFloat();
        }

        Variable w = new Variable(NdArray.of(0), "w");
        Variable b = new Variable(NdArray.of(0), "b");

        Variable variableX = new Variable(NdArray.of(x), "x", false).transpose();
        Variable variableY = new Variable(NdArray.of(y), "y", false).transpose();

        NdArray learnRate = NdArray.of(0.1f);
        int maxEpoch = 100;

        //====== 2，训练模型====
        Variable lastLoss = null;
        for (int i = 0; i < maxEpoch; i++) {
            Variable predict = predict(variableX, w, b);
            Variable loss = meanSquaError(variableY, predict);

            w.clearGrad();
            b.clearGrad();

            loss.setName("loss");
            //反向梯度传播
            loss.backward();

            w.setValue(w.getValue().sub(w.getGrad().mul(learnRate)));
            b.setValue(b.getValue().sub(b.getGrad().mul(learnRate)));

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("i=" + i + " w:" + w.getValue().getNumber() + " b:" + b.getValue().getNumber()
                        + " loss:" + loss.getValue().getNumber().floatValue());
            }
            lastLoss = loss;
        }

        System.out.println(Uml.getDotGraph(lastLoss));

        Variable predictY = predict(variableX, w, b);
        float[] p_y = predictY.transpose().getValue().getMatrix()[0];
        //画图
        Plot plot = new Plot();
        plot.scatter(x, y);
        plot.line(x, p_y, "line");
        plot.show();
    }

    /**
     * 线性预测函数
     *
     * @param x 输入变量
     * @param w 权重变量
     * @param b 偏置变量
     * @return 预测结果
     */
    public static Variable predict(Variable x, Variable w, Variable b) {
        return x.linear(w, b);
    }


    /**
     * 均方误差损失函数
     *
     * @param y 真实值
     * @param x 预测值
     * @return 均方误差
     */
    public static Variable meanSquaError(Variable y, Variable x) {
        return y.sub(x).squ().sum().div(new Variable(y.getValue().getMatrix().length));
    }
}