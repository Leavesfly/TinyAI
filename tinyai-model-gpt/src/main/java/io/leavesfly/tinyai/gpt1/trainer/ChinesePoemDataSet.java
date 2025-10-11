package io.leavesfly.tinyai.gpt1.trainer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nlp.ChineseTokenizer;

import java.util.*;

/**
 * 古诗词数据集类
 * <p>
 * 专门用于处理GPT-1古诗词训练数据
 * 继承自TinyAI的DataSet类，提供：
 * 1. 古诗词数据的加载和预处理
 * 2. 批量数据生成
 * 3. 序列填充和数据增强
 * 4. 训练/验证数据分割
 *
 * @author 山泽
 * @version 1.0
 */
public class ChinesePoemDataSet {

    // 构建词汇表 - 修复文件路径，使用相对于项目根目录的路径
    public static String dataPath = "tinyai-model-gpt/src/main/java/io/leavesfly/tinyai/gpt1/trainer/ci.txt";

    /**
     * 分词器
     */
    private ChineseTokenizer tokenizer;

    /**
     * 训练数据对列表
     */
    private List<ChineseTokenizer.TrainingPair> trainingPairs;

    /**
     * 最大序列长度
     */
    private int maxSequenceLength;

    /**
     * 批量大小
     */
    private int batchSize;

    /**
     * 当前批次索引
     */
    private int currentBatchIndex;

    /**
     * 随机数生成器
     */
    private Random random;

    /**
     * 构造函数
     *
     * @param tokenizer         分词器
     * @param maxSequenceLength 最大序列长度
     * @param batchSize         批量大小
     */
    public ChinesePoemDataSet(ChineseTokenizer tokenizer, int maxSequenceLength, int batchSize) {
        this.tokenizer = tokenizer;
        this.maxSequenceLength = maxSequenceLength;
        this.batchSize = batchSize;
        this.currentBatchIndex = 0;
        this.random = new Random(42);
        this.trainingPairs = new ArrayList<>();
    }

    /**
     * 从文件加载古诗词数据
     *
     * @param filePath 数据文件路径
     * @throws Exception 加载异常
     */
    public void loadFromFile(String filePath) throws Exception {
        System.out.println("正在加载古诗词数据从: " + filePath);

        // 加载并编码数据
        List<List<Integer>> sequences = tokenizer.loadAndEncodeData(filePath);
        System.out.println("加载了 " + sequences.size() + " 个诗句");

        // 创建训练对
        trainingPairs = tokenizer.createTrainingPairs(sequences, maxSequenceLength);
        System.out.println("创建了 " + trainingPairs.size() + " 个训练对");

        // 显示一些统计信息
        printDatasetStats();
    }

    /**
     * 准备数据集
     */
    public void prepare() {
        // 打乱训练数据
        Collections.shuffle(trainingPairs, random);
        currentBatchIndex = 0;

        System.out.println("数据集准备完成，共 " + trainingPairs.size() + " 个训练样本");
    }

    /**
     * 获取下一个批次
     *
     * @return 批次数据，包含输入和目标
     */
    public BatchData getNextBatch() {
        if (currentBatchIndex >= trainingPairs.size()) {
            return null;  // 数据遍历完毕
        }

        int endIndex = Math.min(currentBatchIndex + batchSize, trainingPairs.size());
        List<ChineseTokenizer.TrainingPair> batchPairs =
                trainingPairs.subList(currentBatchIndex, endIndex);

        // 创建批量输入和目标数组
        int actualBatchSize = batchPairs.size();
        float[][] inputBatch = new float[actualBatchSize][maxSequenceLength];
        float[][] targetBatch = new float[actualBatchSize][maxSequenceLength];

        for (int i = 0; i < actualBatchSize; i++) {
            ChineseTokenizer.TrainingPair pair = batchPairs.get(i);

            // 填充输入序列
            List<Integer> paddedInput = tokenizer.padSequence(pair.getInput(), maxSequenceLength);
            for (int j = 0; j < maxSequenceLength; j++) {
                inputBatch[i][j] = paddedInput.get(j);
            }

            // 填充目标序列
            List<Integer> paddedTarget = tokenizer.padSequence(pair.getTarget(), maxSequenceLength);
            for (int j = 0; j < maxSequenceLength; j++) {
                targetBatch[i][j] = paddedTarget.get(j);
            }
        }

        currentBatchIndex = endIndex;

        return new BatchData(
                new Variable(NdArray.of(inputBatch)),
                new Variable(NdArray.of(targetBatch))
        );
    }

    /**
     * 检查是否还有更多批次
     *
     * @return 是否还有批次
     */
    public boolean hasNextBatch() {
        return currentBatchIndex < trainingPairs.size();
    }

    /**
     * 获取总批次数
     *
     * @return 总批次数
     */
    public int getTotalBatches() {
        return (int) Math.ceil((double) trainingPairs.size() / batchSize);
    }

    /**
     * 重置数据集（重新开始遍历）
     */
    public void reset() {
        currentBatchIndex = 0;
    }

    /**
     * 分割数据集为训练集和验证集
     *
     * @param validationRatio 验证集比例
     * @return 包含训练集和验证集的数组
     */
    public ChinesePoemDataSet[] split(double validationRatio) {
        if (validationRatio <= 0 || validationRatio >= 1) {
            throw new IllegalArgumentException("验证集比例必须在(0,1)范围内");
        }

        Collections.shuffle(trainingPairs, random);

        int totalSize = trainingPairs.size();
        int validationSize = (int) (totalSize * validationRatio);
        int trainSize = totalSize - validationSize;

        // 创建训练集
        ChinesePoemDataSet trainSet = new ChinesePoemDataSet(tokenizer, maxSequenceLength, batchSize);
        trainSet.trainingPairs = new ArrayList<>(trainingPairs.subList(0, trainSize));

        // 创建验证集
        ChinesePoemDataSet validationSet = new ChinesePoemDataSet(tokenizer, maxSequenceLength, batchSize);
        validationSet.trainingPairs = new ArrayList<>(trainingPairs.subList(trainSize, totalSize));

        System.out.printf("数据分割完成: 训练集 %d 样本, 验证集 %d 样本\n", trainSize, validationSize);

        return new ChinesePoemDataSet[]{trainSet, validationSet};
    }

    /**
     * 显示数据集统计信息
     */
    public void printDatasetStats() {
        if (trainingPairs.isEmpty()) {
            System.out.println("数据集为空");
            return;
        }

        System.out.println("\n=== 数据集统计信息 ===");
        System.out.println("总训练对数: " + trainingPairs.size());
        System.out.println("最大序列长度: " + maxSequenceLength);
        System.out.println("批量大小: " + batchSize);
        System.out.println("总批次数: " + getTotalBatches());

        // 计算序列长度分布
        Map<Integer, Integer> lengthDistribution = new HashMap<>();
        int totalLength = 0;

        for (ChineseTokenizer.TrainingPair pair : trainingPairs) {
            int length = pair.getInput().size();
            lengthDistribution.put(length, lengthDistribution.getOrDefault(length, 0) + 1);
            totalLength += length;
        }

        double avgLength = (double) totalLength / trainingPairs.size();
        System.out.printf("平均序列长度: %.2f\n", avgLength);

        // 显示长度分布（前10个最常见的长度）
        System.out.println("\n序列长度分布（前10个）:");
        lengthDistribution.entrySet().stream()
                .sorted(Map.Entry.<Integer, Integer>comparingByValue().reversed())
                .limit(10)
                .forEach(entry -> System.out.printf("长度 %d: %d 个序列\n", entry.getKey(), entry.getValue()));

        // 显示一些样本
        System.out.println("\n前3个训练样本:");
        for (int i = 0; i < Math.min(3, trainingPairs.size()); i++) {
            ChineseTokenizer.TrainingPair pair = trainingPairs.get(i);
            String inputText = tokenizer.decode(pair.getInput());
            String targetText = tokenizer.decode(pair.getTarget());
            System.out.printf("样本 %d:\n", i + 1);
            System.out.printf("  输入: %s\n", inputText);
            System.out.printf("  目标: %s\n", targetText);
        }

        System.out.println("========================\n");
    }

    /**
     * 创建用于测试的简单数据样本
     *
     * @param text 文本内容
     * @return 编码后的Variable
     */
    public Variable createSample(String text) {
        List<Integer> encoded = tokenizer.encode(text, false);  // 不添加特殊token
        List<Integer> padded = tokenizer.padSequence(encoded, maxSequenceLength);

        float[][] inputArray = new float[1][maxSequenceLength];
        for (int i = 0; i < maxSequenceLength; i++) {
            inputArray[0][i] = padded.get(i);
        }

        return new Variable(NdArray.of(inputArray));
    }

    /**
     * 批量数据类
     */
    public static class BatchData {
        private final Variable input;
        private final Variable target;

        public BatchData(Variable input, Variable target) {
            this.input = input;
            this.target = target;
        }

        public Variable getInput() {
            return input;
        }

        public Variable getTarget() {
            return target;
        }
    }

    // ==================== Getter方法 ====================

    public ChineseTokenizer getTokenizer() {
        return tokenizer;
    }

    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getDatasetSize() {
        return trainingPairs.size();
    }

    public List<ChineseTokenizer.TrainingPair> getTrainingPairs() {
        return trainingPairs;
    }
}