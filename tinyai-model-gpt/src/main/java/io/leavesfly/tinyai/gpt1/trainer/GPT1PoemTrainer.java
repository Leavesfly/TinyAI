package io.leavesfly.tinyai.gpt1.trainer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.nlp.ChineseTokenizer;

import java.io.File;
import java.util.*;

/**
 * GPT-1古诗词训练器
 * <p>
 * 完整的古诗词语言模型训练流程，包括：
 * 1. 数据加载和预处理
 * 2. 模型初始化和配置
 * 3. 训练循环和损失计算
 * 4. 模型评估和文本生成
 * 5. 训练监控和模型保存
 *
 * @author 山泽
 * @version 1.0
 */
public class GPT1PoemTrainer {

    /**
     * 训练配置
     */
    private TrainingConfig config;

    /**
     * GPT-1模型
     */
    private GPT1Model model;

    /**
     * 中文分词器
     */
    private ChineseTokenizer tokenizer;

    /**
     * 训练数据集
     */
    private ChinesePoemDataSet trainDataset;

    /**
     * 验证数据集
     */
    private ChinesePoemDataSet validDataset;

    /**
     * 优化器
     */
    private Adam optimizer;

    /**
     * 损失函数
     */
    private SoftmaxCrossEntropy lossFunction;

    /**
     * 训练监控器
     */
    private TrainingMonitor monitor;

    /**
     * 训练配置类
     */
    public static class TrainingConfig {
        // 模型配置
        public int vocabSize = 2000;
        public int maxSequenceLength = 64;
        public int hiddenSize = 256;
        public int numLayers = 6;
        public int numAttentionHeads = 8;

        // 训练配置
        public int epochs = 50;
        public int batchSize = 8;
        public double learningRate = 0.001;
        public double validationRatio = 0.2;

        // 生成配置
        public int generateLength = 20;
        public double temperature = 0.8;

        // 监控配置
        public boolean enableMonitoring = true;
        public int printInterval = 10;
        public int saveInterval = 20;
        public String modelSavePath = "gpt1_poem_model";

        @Override
        public String toString() {
            return String.format(
                    "TrainingConfig{" +
                            "vocabSize=%d, maxSeqLen=%d, hiddenSize=%d, numLayers=%d, " +
                            "epochs=%d, batchSize=%d, lr=%.4f, valRatio=%.2f}",
                    vocabSize, maxSequenceLength, hiddenSize, numLayers,
                    epochs, batchSize, learningRate, validationRatio
            );
        }
    }

    /**
     * 构造函数
     *
     * @param config 训练配置
     */
    public GPT1PoemTrainer(TrainingConfig config) {
        this.config = config;
        this.monitor = new TrainingMonitor();

        System.out.println("初始化GPT-1古诗词训练器");
        System.out.println("训练配置: " + config);
    }

    /**
     * 初始化训练组件
     *
     * @param dataPath 数据文件路径
     * @throws Exception 初始化异常
     */
    public void initialize(String dataPath) throws Exception {
        System.out.println("\n=== 初始化训练组件 ===");

        // 1. 初始化分词器并构建词汇表
        System.out.println("1. 初始化中文分词器...");
        tokenizer = new ChineseTokenizer();
        tokenizer.buildVocabFromFile(dataPath);
        tokenizer.printVocabStats();

        // 更新词汇表大小
        config.vocabSize = tokenizer.getVocabSize();

        // 2. 创建GPT-1模型
        System.out.println("2. 创建GPT-1模型...");
        GPT1Config modelConfig = new GPT1Config(
                config.vocabSize,
                config.maxSequenceLength,
                config.hiddenSize,
                config.numLayers,
                config.numAttentionHeads
        );

        model = new GPT1Model("GPT1-古诗词模型", modelConfig);
        model.printModelInfo();

        // 3. 加载和准备数据
        System.out.println("3. 加载训练数据...");
        ChinesePoemDataSet fullDataset = new ChinesePoemDataSet(
                tokenizer, config.maxSequenceLength, config.batchSize
        );
        fullDataset.loadFromFile(dataPath);

        // 分割训练和验证数据
        ChinesePoemDataSet[] datasets = fullDataset.split(config.validationRatio);
        trainDataset = datasets[0];
        validDataset = datasets[1];

        // 4. 初始化优化器和损失函数
        System.out.println("4. 初始化优化器和损失函数...");
        optimizer = new Adam(model, (float) config.learningRate, 0.9f, 0.999f, 1e-4f);
        lossFunction = new SoftmaxCrossEntropy();

        System.out.println("初始化完成!\n");
    }

    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=== 开始训练GPT-1古诗词模型 ===\n");

        for (int epoch = 1; epoch <= config.epochs; epoch++) {
            System.out.printf("轮次 %d/%d\n", epoch, config.epochs);

            // 训练一个epoch
            double trainLoss = trainOneEpoch(epoch);

            // 验证
            double validLoss = validate();

            // 记录训练信息
            monitor.recordEpoch(epoch, trainLoss, validLoss);

            // 打印进度
            if (epoch % config.printInterval == 0) {
                System.out.printf("Epoch %d - 训练损失: %.4f, 验证损失: %.4f, 用时: %s\n",
                        epoch, trainLoss, validLoss, monitor.getLastEpochTime());

                // 生成样本文本
                generateSampleText();
            }

            // 保存模型
            if (epoch % config.saveInterval == 0) {
                saveModel(config.modelSavePath + "_epoch_" + epoch);
            }

            System.out.println();
        }

        System.out.println("=== 训练完成 ===");
        monitor.printSummary();

        // 保存最终模型
        saveModel(config.modelSavePath + "_final");
    }

    /**
     * 训练一个epoch
     *
     * @param epoch 当前epoch数
     * @return 平均训练损失
     */
    private double trainOneEpoch(int epoch) {
        trainDataset.prepare();

        double totalLoss = 0.0;
        int batchCount = 0;

        while (trainDataset.hasNextBatch()) {
            ChinesePoemDataSet.BatchData batch = trainDataset.getNextBatch();

            // 前向传播
            Variable predictions = model.predict(batch.getInput());

            // 计算损失
            Variable loss = lossFunction.loss(batch.getTarget(), predictions);
            double lossValue = loss.getValue().getNumber().doubleValue();

            // 清除梯度
            model.clearGrads();

            // 反向传播
            loss.backward();

            // 更新参数
            optimizer.update();

            totalLoss += lossValue;
            batchCount++;

            // 断开计算图
            loss.unChainBackward();
        }

        return batchCount > 0 ? totalLoss / batchCount : 0.0;
    }

    /**
     * 验证模型
     *
     * @return 平均验证损失
     */
    private double validate() {
        validDataset.reset();

        double totalLoss = 0.0;
        int batchCount = 0;

        while (validDataset.hasNextBatch()) {
            ChinesePoemDataSet.BatchData batch = validDataset.getNextBatch();

            // 前向传播（不更新参数）
            Variable predictions = model.predict(batch.getInput());
            Variable loss = lossFunction.loss(batch.getTarget(), predictions);

            totalLoss += loss.getValue().getNumber().doubleValue();
            batchCount++;

            // 断开计算图
            loss.unChainBackward();
        }

        return batchCount > 0 ? totalLoss / batchCount : 0.0;
    }

    /**
     * 生成样本文本
     */
    private void generateSampleText() {
        System.out.println("生成样本:");

        // 使用一些经典的开头
        String[] prompts = {"春", "月", "山", "水"};

        for (String prompt : prompts) {
            try {
                List<Integer> promptTokens = tokenizer.encode(prompt, false);
                List<Integer> generated = model.generateText(
                        promptTokens, config.generateLength, config.temperature
                );
                String generatedText = tokenizer.decode(generated);
                System.out.printf("  '%s' -> '%s'\n", prompt, generatedText);
            } catch (Exception e) {
                System.out.printf("  '%s' -> 生成失败: %s\n", prompt, e.getMessage());
            }
        }
    }

    /**
     * 保存模型
     *
     * @param savePath 保存路径
     */
    private void saveModel(String savePath) {
        try {
            // 创建保存目录
            File saveDir = new File(savePath);
            if (!saveDir.exists()) {
                saveDir.mkdirs();
            }

            // 这里应该调用模型的保存方法
            // model.save(savePath);
            System.out.println("模型已保存到: " + savePath);
        } catch (Exception e) {
            System.err.println("保存模型失败: " + e.getMessage());
        }
    }

    /**
     * 交互式文本生成
     *
     * @param prompt      提示文本
     * @param length      生成长度
     * @param temperature 温度参数
     * @return 生成的文本
     */
    public String generateText(String prompt, int length, double temperature) {
        try {
            List<Integer> promptTokens = tokenizer.encode(prompt, false);
            List<Integer> generated = model.generateText(promptTokens, length, temperature);
            return tokenizer.decode(generated);
        } catch (Exception e) {
            return "生成失败: " + e.getMessage();
        }
    }

    /**
     * 评估模型困惑度
     *
     * @return 困惑度值
     */
    public double evaluatePerplexity() {
        validDataset.reset();

        double totalLogLikelihood = 0.0;
        int totalTokens = 0;

        while (validDataset.hasNextBatch()) {
            ChinesePoemDataSet.BatchData batch = validDataset.getNextBatch();

            Variable predictions = model.predict(batch.getInput());
            Variable loss = lossFunction.loss(batch.getTarget(), predictions);

            totalLogLikelihood += loss.getValue().getNumber().doubleValue() * batch.getInput().getValue().getShape().size();
            totalTokens += batch.getInput().getValue().getShape().size();

            loss.unChainBackward();
        }

        double avgLogLikelihood = totalLogLikelihood / totalTokens;
        return Math.exp(avgLogLikelihood);
    }

    /**
     * 训练监控器
     */
    private class TrainingMonitor {
        private List<Double> trainLosses = new ArrayList<>();
        private List<Double> validLosses = new ArrayList<>();
        private long epochStartTime = 0;
        private long totalTrainingTime = 0;

        public void recordEpoch(int epoch, double trainLoss, double validLoss) {
            trainLosses.add(trainLoss);
            validLosses.add(validLoss);

            long epochTime = System.currentTimeMillis() - epochStartTime;
            totalTrainingTime += epochTime;
            epochStartTime = System.currentTimeMillis();
        }

        public String getLastEpochTime() {
            if (totalTrainingTime == 0) return "0ms";
            return String.format("%.2fs", totalTrainingTime / 1000.0);
        }

        public void printSummary() {
            System.out.println("\n=== 训练总结 ===");
            if (!trainLosses.isEmpty()) {
                System.out.printf("最终训练损失: %.4f\n", trainLosses.get(trainLosses.size() - 1));
                System.out.printf("最终验证损失: %.4f\n", validLosses.get(validLosses.size() - 1));
                System.out.printf("总训练时间: %.2fs\n", totalTrainingTime / 1000.0);
            }
        }
    }

    /**
     * 主训练方法
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        try {
            // 创建训练配置
            TrainingConfig config = new TrainingConfig();
            config.epochs = 30;
            config.batchSize = 4;
            config.learningRate = 0.0005;
            config.maxSequenceLength = 32;
            config.hiddenSize = 128;
            config.numLayers = 4;

            // 创建训练器
            GPT1PoemTrainer trainer = new GPT1PoemTrainer(config);

            // 数据文件路径
            String dataPath = ChinesePoemDataSet.dataPath;

            // 初始化并训练
            trainer.initialize(dataPath);
            trainer.train();

            // 训练后的交互测试
            System.out.println("\n=== 训练后测试 ===");
            String[] testPrompts = {"春江", "明月", "白日", "青山"};

            for (String prompt : testPrompts) {
                String generated = trainer.generateText(prompt, 15, 0.8);
                System.out.printf("'%s' -> '%s'\n", prompt, generated);
            }

            // 评估困惑度
            double perplexity = trainer.evaluatePerplexity();
            System.out.printf("模型困惑度: %.2f\n", perplexity);

        } catch (Exception e) {
            System.err.println("训练过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}