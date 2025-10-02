package io.leavesfly.tinyai.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nlp.Word2Vec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Word2Vec训练示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 演示如何使用Word2Vec模型进行词向量训练，包括：
 * 1. Skip-gram模式训练
 * 2. CBOW模式训练
 * 3. 负采样训练
 * <p>
 * Word2Vec是Google开发的一种用于生成词向量的神经网络模型，
 * 它能够将词汇映射到连续向量空间中，使得语义相似的词在向量空间中距离较近。
 */
public class Word2VecExample {

    /**
     * 主函数，执行Skip-gram和CBOW模式训练示例
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        // 示例1：Skip-gram模式训练
        System.out.println("=== Skip-gram 模式训练 ===");
        trainSkipGram();

        System.out.println("\n=== CBOW 模式训练 ===");
        // 示例2：CBOW模式训练
        trainCBOW();
    }

    /**
     * Skip-gram模式训练示例
     * <p>
     * Skip-gram模型通过中心词预测上下文词，适用于词汇量大且语料丰富的场景
     */
    public static void trainSkipGram() {
        // 准备简单的语料库
        List<String> corpus = prepareCorpus();

        // 超参数设置
        int vocabSize = 50;      // 词汇表大小
        int embedSize = 10;      // 词向量维度
        int windowSize = 2;      // 上下文窗口大小
        float learningRate = 0.01f;
        int epochs = 100;

        // 创建Word2Vec模型（Skip-gram模式）
        Word2Vec word2vec = new Word2Vec(
                "word2vec_skipgram",
                vocabSize,
                embedSize,
                Word2Vec.TrainingMode.SKIP_GRAM,
                windowSize,
                false,  // 不使用负采样，使用传统softmax
                5
        );

        // 构建词汇表
        word2vec.buildVocab(corpus);

        // 生成训练样本
        List<Word2Vec.TrainingSample> samples = word2vec.generateTrainingSamples(corpus);

        // 创建模型和优化器
        Model model = new Model("word2vec_model", word2vec);
        Optimizer optimizer = new SGD(model, learningRate);
        SoftmaxCrossEntropy lossFunc = new SoftmaxCrossEntropy();

        // 训练循环
        System.out.println("开始Skip-gram训练...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0f;
            int batchCount = 0;

            // 随机打乱样本
            Collections.shuffle(samples);

            for (Word2Vec.TrainingSample sample : samples) {
                // 准备输入数据
                Variable input = new Variable(NdArray.of(new float[][]{{sample.input}}));
                Variable target = new Variable(NdArray.of(new float[][]{{sample.target}}));

                // 前向传播
                Variable output = model.forward(input);
                Variable loss = lossFunc.loss(target, output);

                // 反向传播
                model.clearGrads();
                loss.backward();
                optimizer.update();

                totalLoss += loss.getValue().getNumber().floatValue();
                batchCount++;
            }

            if (epoch % 20 == 0 || epoch == epochs - 1) {
                float avgLoss = totalLoss / batchCount;
                System.out.printf("Epoch %d/%d, Average Loss: %.4f\n", epoch + 1, epochs, avgLoss);
            }
        }

        // 测试词向量效果
        testWordVectors(word2vec);
    }

    /**
     * CBOW模式训练示例
     * <p>
     * CBOW模型通过上下文词预测中心词，适用于小数据集和罕见词的训练
     */
    public static void trainCBOW() {
        // 准备简单的语料库
        List<String> corpus = prepareCorpus();

        // 超参数设置
        int vocabSize = 50;
        int embedSize = 10;
        int windowSize = 2;
        float learningRate = 0.01f;
        int epochs = 50;

        // 创建Word2Vec模型（CBOW模式）
        Word2Vec word2vec = new Word2Vec(
                "word2vec_cbow",
                vocabSize,
                embedSize,
                Word2Vec.TrainingMode.CBOW,
                windowSize,
                false,
                5
        );

        // 构建词汇表
        word2vec.buildVocab(corpus);

        // 生成训练样本
        List<Word2Vec.TrainingSample> samples = word2vec.generateTrainingSamples(corpus);

        // 创建模型和优化器
        Model model = new Model("word2vec_cbow_model", word2vec);
        Optimizer optimizer = new SGD(model, learningRate);
        SoftmaxCrossEntropy lossFunc = new SoftmaxCrossEntropy();

        // 训练循环（简化版本）
        System.out.println("开始CBOW训练...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0f;
            int batchCount = 0;

            Collections.shuffle(samples);

            for (int i = 0; i < Math.min(100, samples.size()); i++) { // 限制训练样本数量
                Word2Vec.TrainingSample sample = samples.get(i);

                Variable input = new Variable(NdArray.of(new float[][]{{sample.input}}));
                Variable target = new Variable(NdArray.of(new float[][]{{sample.target}}));

                Variable output = model.forward(input);
                Variable loss = lossFunc.loss(target, output);

                model.clearGrads();
                loss.backward();
                optimizer.update();

                totalLoss += loss.getValue().getNumber().floatValue();
                batchCount++;
            }

            if (epoch % 10 == 0 || epoch == epochs - 1) {
                float avgLoss = totalLoss / batchCount;
                System.out.printf("Epoch %d/%d, Average Loss: %.4f\n", epoch + 1, epochs, avgLoss);
            }
        }

        // 测试词向量效果
        testWordVectors(word2vec);
    }

    /**
     * 准备示例语料库
     *
     * @return 包含分词后语料的列表
     */
    private static List<String> prepareCorpus() {
        List<String> corpus = new ArrayList<>();

        // 添加一些简单的句子
        String[] sentences = {
                "我 喜欢 机器 学习",
                "深度 学习 很 有趣",
                "神经 网络 是 强大 的 工具",
                "我 正在 学习 Java",
                "TinyDL 是 轻量级 框架",
                "词向量 训练 需要 大量 数据",
                "Word2Vec 是 经典 算法",
                "Skip-gram 和 CBOW 是 两种 模式",
                "机器 学习 包含 很多 算法",
                "深度 学习 属于 机器 学习",
                "神经 网络 可以 学习 复杂 模式",
                "词嵌入 将 词语 映射 到 向量 空间",
                "相似 词语 在 向量 空间 中 距离 较近",
                "训练 词向量 需要 大量 文本 数据"
        };

        // 将句子分解为词
        for (String sentence : sentences) {
            String[] words = sentence.split(" ");
            corpus.addAll(Arrays.asList(words));
        }

        return corpus;
    }

    /**
     * 测试词向量效果
     *
     * @param word2vec 训练好的Word2Vec模型
     */
    private static void testWordVectors(Word2Vec word2vec) {
        System.out.println("\n=== 词向量测试 ===");

        try {
            // 测试获取词向量
            String testWord = "学习";
            if (word2vec.getWord2idx().containsKey(testWord)) {
                NdArray wordVector = word2vec.getWordVector(testWord);
                System.out.printf("词 '%s' 的向量维度: %s\n", testWord, wordVector.getShape());
                System.out.printf("词 '%s' 的向量值: [", testWord);
                float[] vector = wordVector.getMatrix()[0];
                for (int i = 0; i < Math.min(5, vector.length); i++) {
                    System.out.printf("%.3f", vector[i]);
                    if (i < Math.min(4, vector.length - 1)) System.out.print(", ");
                }
                if (vector.length > 5) System.out.print("...");
                System.out.println("]");

                // 测试相似词查找
                List<String> similarWords = word2vec.mostSimilar(testWord, 3);
                System.out.printf("与 '%s' 最相似的词: %s\n", testWord, similarWords);
            }

            // 显示词汇表信息
            System.out.println("词汇表大小: " + word2vec.getWord2idx().size());
            System.out.println("前10个词汇: ");
            int count = 0;
            for (String word : word2vec.getWord2idx().keySet()) {
                if (count >= 10) break;
                System.out.print(word + " ");
                count++;
            }
            System.out.println();

        } catch (Exception e) {
            System.out.println("测试词向量时出现错误: " + e.getMessage());
        }
    }

    /**
     * 负采样训练示例
     * <p>
     * 演示如何使用负采样技术来加速Word2Vec训练
     */
    public static void trainWithNegativeSampling() {
        System.out.println("\n=== 负采样训练示例 ===");

        List<String> corpus = prepareCorpus();

        // 创建使用负采样的Word2Vec模型
        Word2Vec word2vec = new Word2Vec(
                "word2vec_negative_sampling",
                50,
                10,
                Word2Vec.TrainingMode.SKIP_GRAM,
                2,
                true,   // 使用负采样
                5       // 5个负样本
        );

        word2vec.buildVocab(corpus);
        List<Word2Vec.TrainingSample> samples = word2vec.generateTrainingSamples(corpus);

        System.out.println("使用负采样的训练（简化实现）");
        System.out.println("生成的训练样本数量: " + samples.size());

        // 测试负采样功能
        if (!word2vec.getWord2idx().isEmpty()) {
            String firstWord = word2vec.getWord2idx().keySet().iterator().next();
            int targetWordIdx = word2vec.getWord2idx().get(firstWord);

            List<Integer> negatives = word2vec.negativeSampling(targetWordIdx, 3);
            System.out.printf("目标词 '%s' (索引: %d) 的负样本索引: %s\n",
                    firstWord, targetWordIdx, negatives);
        }
    }
}