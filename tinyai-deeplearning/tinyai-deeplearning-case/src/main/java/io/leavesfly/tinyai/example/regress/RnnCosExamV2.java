package io.leavesfly.tinyai.example.regress;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.dataset.simple.CosDataSet;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.rnn.SimpleRNN;
import io.leavesfly.tinyai.nnet.init.Initializers;

import java.util.List;

/**
 * RNN拟合余弦曲线示例 - V2 API版本
 *
 * @author leavesfly
 * @version 0.02
 * <p>
 * 使用V2 API重新实现的RNN拟合余弦曲线示例。
 * 该示例演示如何使用递归神经网络(RNN)拟合序列数据（余弦曲线）。
 * RNN是一种处理序列数据的神经网络，具有记忆能力，能够捕捉时间序列中的依赖关系。
 * <p>
 * V2版本特性：
 * 1. 使用Sequential容器和SimpleRNN层构建网络
 * 2. 统一的参数初始化策略
 * 3. 手动管理RNN的隐藏状态
 * 4. 训练/推理模式切换
 */
public class RnnCosExamV2 {

    /**
     * 主函数，执行RNN训练
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        test0();
    }

    /**
     * RNN训练测试函数
     * <p>
     * 使用SimpleRNN模型训练拟合余弦曲线数据
     */
    public static void test0() {
        //==1，定义超参数
        int maxEpoch = 100;

        //比较特殊表示RNN的隐藏层的大小，
        // 和batchSize得一致，
        // 表示每次预测训练依赖前N个数据
        int bpttLength = 3;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.01f;

        //==2，数据集合
        CosDataSet cosCurveDataSet = new CosDataSet(bpttLength);
        cosCurveDataSet.prepare();
        DataSet trainDataSet = cosCurveDataSet.getTrainDataSet();

        List<Batch> batches = trainDataSet.getBatches();

        //==3，定义网络结构
        // 使用V2 Sequential构建RNN网络：SimpleRNN + Linear
        Sequential sequential = new Sequential("RnnCosV2")
                .add(new SimpleRNN("rnn", inputSize, hiddenSize))
                .add(new Linear("fc", hiddenSize, outputSize));

        // V2参数初始化：使用Xavier初始化权重，零初始化偏置
        sequential.apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                Initializers.xavierUniform(linear.getWeight().data());
                if (linear.getBias() != null) {
                    Initializers.zeros(linear.getBias().data());
                }
            }
            // SimpleRNN内部已经初始化，这里不需要额外处理
        });

        // 将V2 Sequential包装为V1 Model，以便与现有组件兼容
        Model model = new Model("RnnCosExamV2", sequential);
        Optimizer optimizer = new SGD(model, learnRate);
        Loss lossFunc = new MeanSquaredLoss();

        // 获取SimpleRNN层引用，用于管理状态
        SimpleRNN rnnLayer = (SimpleRNN) sequential.get(0);

        //==4，训练网络
        sequential.train(); // 设置为训练模式
        for (int i = 0; i < maxEpoch; i++) {
            //对于递归网络 有状态 每次重新训练的时候要清理中间状态
            rnnLayer.resetState();

            float lossSum = 0f;
            for (Batch batch : batches) {

                NdArray[] xArray = batch.getX();
                NdArray[] yArray = batch.getX();
                Variable loss = new Variable(0f);
                loss.setName("loss");
                for (int j = 0; j < batch.getSize(); j++) {
                    Variable x = new Variable(xArray[j]).setName("x");
                    Variable y = new Variable(yArray[j]).setName("y");
                    Variable predict = sequential.forward(x);
                    loss = loss.add(lossFunc.loss(y, predict));
                    loss.setName("loss" + j);
                }

                sequential.clearGrads();
                loss.backward();
                optimizer.update();

                lossSum += loss.getValue().getNumber().floatValue() / batch.getSize();
                //切断计算图 每批数据要清理重新构建计算图
                loss.unChainBackward();
            }
            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("epoch: " + i + "  avg-loss:" + lossSum / batches.size());
            }
        }
    }
}

