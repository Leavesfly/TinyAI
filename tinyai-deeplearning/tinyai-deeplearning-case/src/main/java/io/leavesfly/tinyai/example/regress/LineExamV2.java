package io.leavesfly.tinyai.example.regress;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.visual.Plot;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.init.Initializers;

import java.util.Random;

/**
 * 线性回归示例 - V2 API版本
 *
 * @author leavesfly
 * @version 0.02
 * <p>
 * 使用V2 API重新实现的线性回归示例。
 * 该示例演示如何使用梯度下降法拟合带有噪声的线性数据。
 * 线性回归是机器学习中最基础的回归算法之一，用于建立输入特征与目标值之间的线性关系。
 * <p>
 * V2版本特性：
 * 1. 使用Sequential容器和Linear层构建模型
 * 2. 统一的参数初始化策略
 * 3. 手动训练循环展示完整的训练流程
 */
public class LineExamV2 {

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

        Variable variableX = new Variable(NdArray.of(x), "x", false).transpose();
        Variable variableY = new Variable(NdArray.of(y), "y", false).transpose();

        //====== 2，定义模型====
        // 使用V2 Sequential构建线性模型（单层Linear，无激活函数）
        Sequential model = new Sequential("LinearRegressionV2")
                .add(new Linear("fc", 1, 1));

        // V2参数初始化：使用Xavier初始化权重，零初始化偏置
        model.apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                Initializers.xavierUniform(linear.getWeight().data());
                if (linear.getBias() != null) {
                    Initializers.zeros(linear.getBias().data());
                }
            }
        });

        // 将V2 Sequential包装为V1 Model，以便与现有组件兼容
        Model modelWrapper = new Model("LineExamV2", model);
        Optimizer optimizer = new SGD(modelWrapper, 0.1f);

        //====== 3，训练模型====
        int maxEpoch = 100;

        // 使用MeanSquaredLoss（通过手动计算）
        model.train(); // 设置为训练模式

        for (int i = 0; i < maxEpoch; i++) {
            Variable predict = model.forward(variableX);
            Variable loss = meanSquaredError(variableY, predict);

            model.clearGrads();
            loss.backward();
            optimizer.update();

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                // 获取权重和偏置
                Linear linear = (Linear) model.get(0);
                float w = linear.getWeight().getValue().getNumber().floatValue();
                float b = linear.getBias() != null ? linear.getBias().getValue().getNumber().floatValue() : 0f;
                System.out.println("i=" + i + " w:" + w + " b:" + b
                        + " loss:" + loss.getValue().getNumber().floatValue());
            }
        }

        //====== 4，可视化结果====
        model.eval(); // 设置为推理模式
        Variable predictY = model.forward(variableX);
        float[] p_y = predictY.transpose().getValue().getMatrix()[0];
        
        //画图
        Plot plot = new Plot();
        plot.scatter(x, y);
        plot.line(x, p_y, "line");
        plot.show();
    }

    /**
     * 均方误差损失函数
     *
     * @param y 真实值
     * @param predict 预测值
     * @return 均方误差
     */
    public static Variable meanSquaredError(Variable y, Variable predict) {
        return y.sub(predict).squ().sum().div(new Variable(NdArray.of((float) y.getValue().getMatrix().length)));
    }
}

