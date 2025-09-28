package io.leavesfly.tinyai.ml.dataset.simple;

import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;

/**
 * Word2Vec专用数据集
 * 支持Skip-gram和CBOW两种训练模式的数据生成
 */
public class Word2VecDataSet extends ArrayDataset {

    public enum TrainingMode {
        SKIP_GRAM,  // Skip-gram：中心词预测上下文词
        CBOW        // CBOW：上下文词预测中心词
    }

    private final List<String> corpus;           // 原始语料库
    private final int windowSize;               // 上下文窗口大小
    private final TrainingMode mode;            // 训练模式
    private Map<String, Integer> word2idx;      // 词到索引的映射
    private Map<Integer, String> idx2word;      // 索引到词的映射
    private int vocabSize;                      // 词汇表大小

    /**
     * 构造函数
     *
     * @param corpus       语料库（词的列表）
     * @param batchSize    批次大小
     * @param windowSize   上下文窗口大小
     * @param mode         训练模式
     * @param maxVocabSize 最大词汇表大小
     */
    public Word2VecDataSet(List<String> corpus, int batchSize, int windowSize,
                           TrainingMode mode, int maxVocabSize) {
        super(batchSize);
        this.corpus = new ArrayList<>(corpus);
        this.windowSize = windowSize;
        this.mode = mode;
        this.vocabSize = maxVocabSize;

        buildVocabulary();
        generateTrainingData();
    }

    /**
     * 简化构造函数
     */
    public Word2VecDataSet(List<String> corpus, int batchSize) {
        this(corpus, batchSize, 2, TrainingMode.SKIP_GRAM, 1000);
    }

    /**
     * 构建词汇表
     */
    private void buildVocabulary() {
        // 统计词频
        Map<String, Integer> wordCount = new HashMap<>();
        for (String word : corpus) {
            wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
        }

        // 按词频排序，选择最常见的词
        List<Map.Entry<String, Integer>> sortedWords = new ArrayList<>(wordCount.entrySet());
        sortedWords.sort((a, b) -> b.getValue().compareTo(a.getValue()));

        // 构建词汇映射
        word2idx = new HashMap<>();
        idx2word = new HashMap<>();

        int actualVocabSize = Math.min(vocabSize, sortedWords.size());
        for (int i = 0; i < actualVocabSize; i++) {
            String word = sortedWords.get(i).getKey();
            word2idx.put(word, i);
            idx2word.put(i, word);
        }

        this.vocabSize = actualVocabSize;
        System.out.println("词汇表构建完成，包含 " + vocabSize + " 个词");
    }

    /**
     * 生成训练数据
     */
    private void generateTrainingData() {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();

        // 将语料库转换为索引序列
        List<Integer> indexSequence = new ArrayList<>();
        for (String word : corpus) {
            Integer idx = word2idx.get(word);
            if (idx != null) {
                indexSequence.add(idx);
            }
        }

        // 根据模式生成训练样本
        if (mode == TrainingMode.SKIP_GRAM) {
            generateSkipGramSamples(indexSequence, inputList, targetList);
        } else {
            generateCBOWSamples(indexSequence, inputList, targetList);
        }

        // 转换为NdArray数组格式
        this.xs = new NdArray[inputList.size()];
        this.ys = new NdArray[targetList.size()];

        for (int i = 0; i < inputList.size(); i++) {
            this.xs[i] = NdArray.of(new float[][]{inputList.get(i)});
            this.ys[i] = NdArray.of(new float[][]{targetList.get(i)});
        }

        System.out.printf("生成 %s 训练样本 %d 个\n", mode.name(), inputList.size());
    }

    /**
     * 数据准备方法
     */
    @Override
    public void doPrepare() {
        buildVocabulary();
        generateTrainingData();
    }

    /**
     * 构建数据集方法
     */
    @Override
    protected DataSet build(int batchSize, NdArray[] xs, NdArray[] ys) {
        // 创建一个新的Word2VecDataSet实例
        Word2VecDataSet newDataSet = new Word2VecDataSet(this.corpus, batchSize, this.windowSize, this.mode, this.vocabSize);
        newDataSet.xs = xs;
        newDataSet.ys = ys;
        newDataSet.word2idx = this.word2idx;
        newDataSet.idx2word = this.idx2word;
        return newDataSet;
    }

    /**
     * 生成Skip-gram训练样本
     * Skip-gram: 给定中心词，预测上下文词
     */
    private void generateSkipGramSamples(List<Integer> sequence,
                                         List<float[]> inputList,
                                         List<float[]> targetList) {
        for (int i = 0; i < sequence.size(); i++) {
            int centerWord = sequence.get(i);

            // 获取上下文窗口内的所有词
            for (int j = Math.max(0, i - windowSize);
                 j <= Math.min(sequence.size() - 1, i + windowSize); j++) {
                if (i != j) {  // 跳过中心词本身
                    int contextWord = sequence.get(j);

                    // 输入：中心词，输出：上下文词
                    inputList.add(new float[]{centerWord});
                    targetList.add(new float[]{contextWord});
                }
            }
        }
    }

    /**
     * 生成CBOW训练样本
     * CBOW: 给定上下文词，预测中心词
     */
    private void generateCBOWSamples(List<Integer> sequence,
                                     List<float[]> inputList,
                                     List<float[]> targetList) {
        for (int i = windowSize; i < sequence.size() - windowSize; i++) {
            int centerWord = sequence.get(i);

            // 收集上下文词（为简化，这里只取第一个上下文词作为输入）
            // 实际应用中，CBOW会平均所有上下文词的嵌入
            for (int j = Math.max(0, i - windowSize);
                 j <= Math.min(sequence.size() - 1, i + windowSize); j++) {
                if (i != j) {
                    int contextWord = sequence.get(j);

                    // 输入：上下文词，输出：中心词
                    inputList.add(new float[]{contextWord});
                    targetList.add(new float[]{centerWord});
                    break; // 为简化，只取一个上下文词
                }
            }
        }
    }

    /**
     * 添加新的语料到数据集
     */
    public void addCorpus(List<String> newCorpus) {
        this.corpus.addAll(newCorpus);
        buildVocabulary();  // 重新构建词汇表
        generateTrainingData();  // 重新生成训练数据
    }

    /**
     * 获取词汇表映射
     */
    public Map<String, Integer> getWord2idx() {
        return new HashMap<>(word2idx);
    }

    /**
     * 获取反向词汇表映射
     */
    public Map<Integer, String> getIdx2word() {
        return new HashMap<>(idx2word);
    }

    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 获取训练模式
     */
    public TrainingMode getMode() {
        return mode;
    }

    /**
     * 获取窗口大小
     */
    public int getWindowSize() {
        return windowSize;
    }

    /**
     * 从文本文件创建数据集
     */
    public static Word2VecDataSet fromText(String text, int batchSize,
                                           TrainingMode mode, int maxVocabSize) {
        // 简单的文本预处理
        String[] words = text.toLowerCase()
                .replaceAll("[^a-zA-Z0-9\\u4e00-\\u9fa5\\s]", " ")  // 保留中英文和数字
                .split("\\s+");

        List<String> corpus = new ArrayList<>();
        for (String word : words) {
            if (!word.trim().isEmpty()) {
                corpus.add(word.trim());
            }
        }

        return new Word2VecDataSet(corpus, batchSize, 2, mode, maxVocabSize);
    }

    /**
     * 创建中文示例数据集
     */
    public static Word2VecDataSet createChineseExample(int batchSize) {
        List<String> corpus = Arrays.asList(
                "机器", "学习", "是", "人工", "智能", "的", "重要", "分支",
                "深度", "学习", "是", "机器", "学习", "的", "子", "领域",
                "神经", "网络", "是", "深度", "学习", "的", "基础",
                "词", "向量", "可以", "表示", "词语", "的", "语义",
                "Word2Vec", "是", "训练", "词", "向量", "的", "经典", "方法",
                "Skip-gram", "和", "CBOW", "是", "两种", "不同", "的", "训练", "方式",
                "自然", "语言", "处理", "需要", "理解", "词语", "的", "含义",
                "相似", "的", "词语", "应该", "有", "相似", "的", "向量", "表示",
                "训练", "好", "的", "词", "向量", "可以", "用于", "各种", "NLP", "任务"
        );

        return new Word2VecDataSet(corpus, batchSize, 2, TrainingMode.SKIP_GRAM, 100);
    }

    /**
     * 打印数据集统计信息
     */
    public void printStatistics() {
        System.out.println("=== Word2Vec 数据集统计 ===");
        System.out.println("训练模式: " + mode.name());
        System.out.println("词汇表大小: " + vocabSize);
        System.out.println("上下文窗口: " + windowSize);
        System.out.println("原始语料词数: " + corpus.size());
        if (xs != null) {
            System.out.println("训练样本数: " + xs.length);
        }
        System.out.println("批次大小: " + batchSize);

        // 显示最常见的词
        System.out.println("最常见的10个词:");
        int count = 0;
        for (Map.Entry<String, Integer> entry : word2idx.entrySet()) {
            if (count >= 10) break;
            System.out.printf("  %s (索引: %d)\n", entry.getKey(), entry.getValue());
            count++;
        }
    }
}