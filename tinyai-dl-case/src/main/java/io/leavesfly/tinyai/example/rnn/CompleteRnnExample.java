package io.leavesfly.tinyai.example.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.dataset.simple.SinDataSet;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.block.GruBlock;
import io.leavesfly.tinyai.nnet.block.LstmBlock;
import io.leavesfly.tinyai.nnet.block.SimpleRnnBlock;

import java.util.List;

/**
 * 完整的RNN示例，比较不同RNN层的性能
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 该示例演示并比较三种不同的递归神经网络架构：
 * 1. SimpleRNN - 基础递归网络
 * 2. LSTM - 长短期记忆网络，能够处理长序列依赖
 * 3. GRU - 门控递归单元，LSTM的简化版本
 * 
 * 所有模型都用于拟合正弦曲线序列数据，以比较它们的性能差异。
 */
public class CompleteRnnExample {
    
    /**
     * 主函数，执行三种RNN模型的训练和比较
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        // 比较不同类型的RNN
        System.out.println("=== Simple RNN ===");
        testSimpleRNN();
        
        System.out.println("\n=== LSTM ===");
        testLSTM();
        
        System.out.println("\n=== GRU ===");
        testGRU();
    }

    /**
     * 测试SimpleRNN
     * 
     * SimpleRNN是最基础的递归网络结构，具有简单的循环连接
     */
    public static void testSimpleRNN() {
        // 定义超参数
        int maxEpoch = 100;
        int bpttLength = 10;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.001f;

        // 数据集合
        SinDataSet sinDataSet = new SinDataSet(bpttLength);
        sinDataSet.prepare();
        DataSet trainDataSet = sinDataSet.getTrainDataSet();
        List<Batch> batches = trainDataSet.getBatches();

        // 定义网络结构
        SimpleRnnBlock rnnBlock = new SimpleRnnBlock("simple-rnn", inputSize, hiddenSize, outputSize);
        Model model = new Model("SimpleRNN", rnnBlock);
        Adam optimizer = new Adam(model, learnRate, 0.9f, 0.999f, 1e-8f);
        Loss lossFunc = new MeanSquaredLoss();

        // 训练网络
        trainModel(model, optimizer, lossFunc, batches, maxEpoch);
    }

    /**
     * 测试LSTM
     * 
     * LSTM(长短期记忆网络)通过门控机制解决梯度消失问题，能够学习长期依赖关系
     */
    public static void testLSTM() {
        // 定义超参数
        int maxEpoch = 100;
        int bpttLength = 10;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.001f;

        // 数据集合
        SinDataSet sinDataSet = new SinDataSet(bpttLength);
        sinDataSet.prepare();
        DataSet trainDataSet = sinDataSet.getTrainDataSet();
        List<Batch> batches = trainDataSet.getBatches();

        // 定义网络结构
        LstmBlock lstmBlock = new LstmBlock("lstm", inputSize, hiddenSize, outputSize);
        Model model = new Model("LSTM", lstmBlock);
        Adam optimizer = new Adam(model, learnRate, 0.9f, 0.999f, 1e-8f);
        Loss lossFunc = new MeanSquaredLoss();

        // 训练网络
        trainModel(model, optimizer, lossFunc, batches, maxEpoch);
    }

    /**
     * 测试GRU
     * 
     * GRU(门控递归单元)是LSTM的简化版本，具有更少的参数和计算复杂度
     */
    public static void testGRU() {
        // 定义超参数
        int maxEpoch = 100;
        int bpttLength = 10;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.001f;

        // 数据集合
        SinDataSet sinDataSet = new SinDataSet(bpttLength);
        sinDataSet.prepare();
        DataSet trainDataSet = sinDataSet.getTrainDataSet();
        List<Batch> batches = trainDataSet.getBatches();

        // 定义网络结构
        GruBlock gruBlock = new GruBlock("gru", inputSize, hiddenSize, outputSize);
        Model model = new Model("GRU", gruBlock);
        Adam optimizer = new Adam(model, learnRate, 0.9f, 0.999f, 1e-8f);
        Loss lossFunc = new MeanSquaredLoss();

        // 训练网络
        trainModel(model, optimizer, lossFunc, batches, maxEpoch);
    }

    /**
     * 通用训练函数
     * 
     * @param model 模型
     * @param optimizer 优化器
     * @param lossFunc 损失函数
     * @param batches 训练批次
     * @param maxEpoch 最大训练轮数
     */
    private static void trainModel(Model model, Adam optimizer, Loss lossFunc, 
                                  List<Batch> batches, int maxEpoch) {
        for (int i = 0; i < maxEpoch; i++) {
            // 对于递归网络有状态，每次重新训练的时候要清理中间状态
            model.resetState();

            float lossSum = 0f;
            int batchCount = 0;
            
            for (Batch batch : batches) {
                NdArray[] xArray = batch.getX();
                NdArray[] yArray = batch.getY();
                
                Variable loss = new Variable(0f);
                loss.setName("loss");
                
                for (int j = 0; j < batch.getSize(); j++) {
                    Variable x = new Variable(xArray[j]).setName("x");
                    Variable y = new Variable(yArray[j]).setName("y");
                    Variable predict = model.forward(x);
                    loss = loss.add(lossFunc.loss(y, predict));
                }

                model.clearGrads();
                loss.backward();
                optimizer.update();

                lossSum += loss.getValue().getNumber().floatValue() / batch.getSize();
                batchCount++;
                
                // 切断计算图，每批数据要清理重新构建计算图
                loss.unChainBackward();
            }
            
            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("epoch: " + i + "  avg-loss:" + lossSum / batchCount);
            }
        }
    }
}