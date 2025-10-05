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

import java.util.List;

/**
 * 简化的GRU测试，用于调试
 */
public class SimpleGruTest {
    
    public static void main(String[] args) {
        System.out.println("=== 开始GRU测试 ===");
        
        // 定义超参数
        int bpttLength = 3; // 减少序列长度
        int inputSize = 1;
        int hiddenSize = 5; // 减少隐藏状态大小
        int outputSize = 1;
        float learnRate = 0.01f;

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

        System.out.println("模型创建成功");
        
        // 只进行第一个batch的前向传播测试
        Batch firstBatch = batches.get(0);
        NdArray[] xArray = firstBatch.getX();
        NdArray[] yArray = firstBatch.getY();
        
        System.out.println("开始前向传播测试...");
        model.resetState();
        
        Variable loss = new Variable(0f);
        loss.setName("loss");
        
        for (int j = 0; j < Math.min(1, firstBatch.getSize()); j++) { // 只处理一个样本
            Variable x = new Variable(xArray[j]).setName("x");
            Variable y = new Variable(yArray[j]).setName("y");
            
            System.out.println("前向传播步骤 " + j + "...");
            Variable predict = model.forward(x);
            System.out.println("前向传播成功，predict shape: " + predict.getValue().getShape());
            
            loss = loss.add(lossFunc.loss(y, predict));
            System.out.println("损失计算成功");
        }

        System.out.println("开始反向传播测试...");
        model.clearGrads();
        
        try {
            loss.backward();
            System.out.println("反向传播成功！");
        } catch (Exception e) {
            System.out.println("反向传播失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}