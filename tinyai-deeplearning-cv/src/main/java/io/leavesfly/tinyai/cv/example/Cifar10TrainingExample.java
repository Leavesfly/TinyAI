package io.leavesfly.tinyai.cv.example;

import io.leavesfly.tinyai.cv.dataset.Cifar10Dataset;
import io.leavesfly.tinyai.cv.model.lenet.LeNet5;
import io.leavesfly.tinyai.cv.preprocess.BatchProcessor;
import io.leavesfly.tinyai.cv.preprocess.ImagePreprocessor;
import io.leavesfly.tinyai.cv.preprocess.ImagePreprocessorImpl;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * CIFAR-10图像分类训练示例
 * 
 * 本示例演示如何使用TinyAI计算机视觉模块进行CIFAR-10图像分类任务：
 * 1. 加载CIFAR-10数据集
 * 2. 预处理图像数据
 * 3. 使用LeNet-5模型进行训练
 * 4. 评估模型性能
 */
public class Cifar10TrainingExample {
    
    // 批次大小
    private static final int BATCH_SIZE = 32;
    
    // 训练轮数
    private static final int EPOCHS = 10;
    
    // 学习率
    private static final float LEARNING_RATE = 0.001f;
    
    public static void main(String[] args) {
        // 创建CIFAR-10数据集加载器
        String datasetPath = "path/to/cifar10"; // 需要替换为实际的数据集路径
        Cifar10Dataset dataset = new Cifar10Dataset(datasetPath);
        
        // 加载训练数据
        System.out.println("Loading training data...");
        if (!dataset.loadTrainData()) {
            System.err.println("Failed to load training data");
            return;
        }
        
        // 加载测试数据
        System.out.println("Loading test data...");
        if (!dataset.loadTestData()) {
            System.err.println("Failed to load test data");
            return;
        }
        
        System.out.println("Training samples: " + dataset.getTrainSize());
        System.out.println("Test samples: " + dataset.getTestSize());
        
        // 创建图像预处理器
        ImagePreprocessor preprocessor = new ImagePreprocessorImpl();
        
        // 创建批处理管道
        BatchProcessor batchProcessor = new BatchProcessor();
        
        // 创建LeNet-5模型
        LeNet5 model = new LeNet5();
        
        System.out.println("Model parameter count: " + model.getParameterCount());
        
        // 训练模型
        trainModel(dataset, model, batchProcessor, preprocessor);
        
        // 评估模型
        evaluateModel(dataset, model, batchProcessor, preprocessor);
    }
    
    /**
     * 训练模型
     * @param dataset 数据集
     * @param model 模型
     * @param batchProcessor 批处理管道
     * @param preprocessor 图像预处理器
     */
    private static void trainModel(Cifar10Dataset dataset, LeNet5 model, 
                                  BatchProcessor batchProcessor, ImagePreprocessor preprocessor) {
        System.out.println("Starting training...");
        
        NdArray trainImages = dataset.getTrainImages();
        NdArray trainLabels = dataset.getTrainLabels();
        
        int trainSize = dataset.getTrainSize();
        int numBatches = (trainSize + BATCH_SIZE - 1) / BATCH_SIZE;
        
        // CIFAR-10图像的均值和标准差（示例值）
        float[] mean = {0.4914f, 0.4822f, 0.4465f};
        float[] std = {0.2470f, 0.2435f, 0.2616f};
        
        // 训练多个轮次
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            System.out.println("Epoch " + (epoch + 1) + "/" + EPOCHS);
            
            float totalLoss = 0.0f;
            int correctPredictions = 0;
            
            // 遍历所有批次
            for (int batch = 0; batch < numBatches; batch++) {
                int startIdx = batch * BATCH_SIZE;
                int endIdx = Math.min(startIdx + BATCH_SIZE, trainSize);
                int batchSize = endIdx - startIdx;
                
                // 提取批次数据
                List<NdArray> batchImages = new ArrayList<>();
                List<Float> batchLabels = new ArrayList<>();
                
                for (int i = startIdx; i < endIdx; i++) {
                    // 提取单个图像
                    NdArray image = extractImage(trainImages, i);
                    batchImages.add(image);
                    
                    // 提取标签
                    batchLabels.add(trainLabels.get(i, 0));
                }
                
                // 预处理批次图像
                NdArray processedBatch = batchProcessor.processBatch(batchImages, mean, std, true);
                
                // 转换为Variable
                Variable input = new Variable(processedBatch);
                
                // 前向传播
                Variable output = model.forward(input);
                
                // 计算损失（简化实现，实际应用中应使用交叉熵损失）
                float loss = computeLoss(output, batchLabels);
                totalLoss += loss;
                
                // 计算准确率
                int correct = computeAccuracy(output, batchLabels);
                correctPredictions += correct;
                
                // 反向传播和参数更新（简化实现）
                // 在实际应用中，这里应该实现完整的反向传播和优化器更新逻辑
                
                // 输出进度
                if ((batch + 1) % 100 == 0) {
                    System.out.println("Batch " + (batch + 1) + "/" + numBatches + 
                                     ", Loss: " + loss + ", Accuracy: " + (correct / (float) batchSize));
                }
            }
            
            // 输出轮次统计信息
            float avgLoss = totalLoss / numBatches;
            float accuracy = correctPredictions / (float) trainSize;
            System.out.println("Epoch " + (epoch + 1) + " completed. " +
                             "Average Loss: " + avgLoss + ", Accuracy: " + accuracy);
        }
        
        System.out.println("Training completed.");
    }
    
    /**
     * 评估模型
     * @param dataset 数据集
     * @param model 模型
     * @param batchProcessor 批处理管道
     * @param preprocessor 图像预处理器
     */
    private static void evaluateModel(Cifar10Dataset dataset, LeNet5 model, 
                                     BatchProcessor batchProcessor, ImagePreprocessor preprocessor) {
        System.out.println("Starting evaluation...");
        
        NdArray testImages = dataset.getTestImages();
        NdArray testLabels = dataset.getTestLabels();
        
        int testSize = dataset.getTestSize();
        int numBatches = (testSize + BATCH_SIZE - 1) / BATCH_SIZE;
        
        // CIFAR-10图像的均值和标准差（示例值）
        float[] mean = {0.4914f, 0.4822f, 0.4465f};
        float[] std = {0.2470f, 0.2435f, 0.2616f};
        
        float totalLoss = 0.0f;
        int correctPredictions = 0;
        
        // 遍历所有批次
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * BATCH_SIZE;
            int endIdx = Math.min(startIdx + BATCH_SIZE, testSize);
            int batchSize = endIdx - startIdx;
            
            // 提取批次数据
            List<NdArray> batchImages = new ArrayList<>();
            List<Float> batchLabels = new ArrayList<>();
            
            for (int i = startIdx; i < endIdx; i++) {
                // 提取单个图像
                NdArray image = extractImage(testImages, i);
                batchImages.add(image);
                
                // 提取标签
                batchLabels.add(testLabels.get(i, 0));
            }
            
            // 预处理批次图像（不进行数据增强）
            NdArray processedBatch = batchProcessor.processBatch(batchImages, mean, std, false);
            
            // 转换为Variable
            Variable input = new Variable(processedBatch);
            
            // 前向传播
            Variable output = model.forward(input);
            
            // 计算损失
            float loss = computeLoss(output, batchLabels);
            totalLoss += loss;
            
            // 计算准确率
            int correct = computeAccuracy(output, batchLabels);
            correctPredictions += correct;
        }
        
        // 输出评估结果
        float avgLoss = totalLoss / numBatches;
        float accuracy = correctPredictions / (float) testSize;
        System.out.println("Evaluation completed. " +
                         "Average Loss: " + avgLoss + ", Accuracy: " + accuracy);
    }
    
    /**
     * 从批次数据中提取单个图像
     * @param batchImages 批次图像数据
     * @param index 图像索引
     * @return 单个图像
     */
    private static NdArray extractImage(NdArray batchImages, int index) {
        // 提取单个图像数据
        // 形状从 (batch_size, 32, 32, 3) 转换为 (32, 32, 3)
        float[][][] imageData = new float[32][32][3];
        
        for (int h = 0; h < 32; h++) {
            for (int w = 0; w < 32; w++) {
                for (int c = 0; c < 3; c++) {
                    imageData[h][w][c] = batchImages.get(index, h, w, c);
                }
            }
        }
        
        return NdArray.of(imageData);
    }
    
    /**
     * 计算损失（简化实现）
     * @param output 模型输出
     * @param labels 真实标签
     * @return 损失值
     */
    private static float computeLoss(Variable output, List<Float> labels) {
        // 简化实现：计算均方误差
        NdArray outputData = output.getValue();
        float loss = 0.0f;
        
        for (int i = 0; i < labels.size(); i++) {
            float label = labels.get(i);
            // 简化：假设标签是类别索引，计算与对应输出的差值
            float predicted = outputData.get(i, (int) label);
            float error = 1.0f - predicted; // 假设正确分类的输出应该是1.0
            loss += error * error;
        }
        
        return loss / labels.size();
    }
    
    /**
     * 计算准确率
     * @param output 模型输出
     * @param labels 真实标签
     * @return 正确预测的数量
     */
    private static int computeAccuracy(Variable output, List<Float> labels) {
        NdArray outputData = output.getValue();
        int correct = 0;
        
        for (int i = 0; i < labels.size(); i++) {
            float label = labels.get(i);
            
            // 找到预测概率最大的类别
            int predictedClass = 0;
            float maxProb = outputData.get(i, 0);
            
            for (int j = 1; j < 10; j++) {
                float prob = outputData.get(i, j);
                if (prob > maxProb) {
                    maxProb = prob;
                    predictedClass = j;
                }
            }
            
            // 检查预测是否正确
            if (predictedClass == (int) label) {
                correct++;
            }
        }
        
        return correct;
    }
}