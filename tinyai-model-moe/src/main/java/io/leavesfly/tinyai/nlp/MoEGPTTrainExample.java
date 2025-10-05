package io.leavesfly.tinyai.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Mixture of Experts (MoE) GPT模型使用示例
 * <p>
 * 这个示例演示了如何：
 * 1. 创建和配置MoE-GPT模型
 * 2. 准备文本数据和训练
 * 3. 监控专家使用情况和负载均衡
 * 4. 进行文本生成推理
 * 5. 分析模型性能和专家分工
 * <p>
 * MoE模型的主要优势：
 * - 大幅增加模型容量而保持计算效率
 * - 专家专业化处理不同类型的语言模式
 * - 可扩展的架构设计
 *
 * @author leavesfly
 * @version 0.01
 */
public class MoEGPTTrainExample {

    /**
     * 训练文本数据
     */
    private static final String[] TRAINING_TEXTS = {
            "人工智能是计算机科学的一个重要分支",
            "深度学习通过神经网络模拟人脑的学习过程",
            "Transformer架构revolutionized自然语言处理领域",
            "机器学习算法可以从数据中自动学习模式",
            "神经网络由多个层次的人工神经元组成",
            "自然语言处理让计算机理解人类语言",
            "卷积神经网络在图像识别中表现优异",
            "循环神经网络适合处理序列数据",
            "注意力机制帮助模型关注重要信息",
            "预训练模型在下游任务中效果显著"
    };

    /**
     * 词汇表大小
     */
    private static final int VOCAB_SIZE = 1000;

    /**
     * 最大序列长度
     */
    private static final int MAX_SEQ_LENGTH = 64;

    /**
     * 批次大小
     */
    private static final int BATCH_SIZE = 4;

    /**
     * 训练轮数
     */
    private static final int EPOCHS = 50;

    /**
     * 学习率
     */
    private static final double LEARNING_RATE = 0.001;

    /**
     * 主函数
     */
    public static void main(String[] args) {
        System.out.println("=== MoE-GPT模型示例开始 ===\n");

        try {
            // 1. 创建MoE-GPT模型
            MoEGPTModel moeModel = createMoEModel();
            moeModel.printModelInfo();

            // 2. 准备训练数据
            SimpleTokenizer tokenizer = new SimpleTokenizer();
            List<int[]> trainingData = prepareTrainingData(tokenizer);

            // 3. 创建模型
            Model model = new Model("MoE_GPT_Model", moeModel);

            // 4. 训练模型（简化版）
            System.out.println("\n=== 开始训练 ===");
            setupAndTrain(model, trainingData, moeModel);

            // 5. 分析专家使用情况
            System.out.println("\n=== 专家使用分析 ===");
            analyzeMoEUsage(moeModel);

            // 6. 进行文本生成测试
            System.out.println("\n=== 文本生成测试 ===");
            testTextGeneration(model, tokenizer);

            // 7. 性能对比分析
            System.out.println("\n=== 性能分析 ===");
            performanceAnalysis(moeModel);

        } catch (Exception e) {
            System.err.println("示例执行出错: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("\n=== MoE-GPT模型示例结束 ===");
    }

    /**
     * 创建MoE-GPT模型
     */
    private static MoEGPTModel createMoEModel() {
        System.out.println("创建MoE-GPT模型...");

        // 创建小规模MoE模型用于演示
        MoEGPTModel model = MoEGPTModel.createSmallModel("demo_moe_gpt", VOCAB_SIZE);

        System.out.println("MoE模型创建完成");
        return model;
    }

    /**
     * 准备训练数据
     */
    private static List<int[]> prepareTrainingData(SimpleTokenizer tokenizer) {
        System.out.println("准备训练数据...");

        List<int[]> trainingData = new ArrayList<>();

        for (String text : TRAINING_TEXTS) {
            // 简化的tokenization（实际应用中需要更复杂的处理）
            int[] tokens = tokenizeText(text, tokenizer);
            if (tokens.length > 1) { // 至少需要2个token才能构成输入-目标对
                trainingData.add(tokens);
            }
        }

        System.out.println("训练数据准备完成，共" + trainingData.size() + "个序列");
        return trainingData;
    }

    /**
     * 简化的文本tokenization
     */
    private static int[] tokenizeText(String text, SimpleTokenizer tokenizer) {
        // 这里使用简化的tokenization
        // 实际应用中应该使用更复杂的分词算法
        String[] words = text.toLowerCase().split("\\s+");
        int[] tokens = new int[Math.min(words.length, MAX_SEQ_LENGTH)];

        Random random = new Random(text.hashCode()); // 使用文本hash作为seed以保证一致性
        for (int i = 0; i < tokens.length; i++) {
            // 简化：使用随机ID代替真实的token ID
            tokens[i] = Math.abs(words[i].hashCode()) % (VOCAB_SIZE - 1) + 1; // 避免使用0（可能是特殊token）
        }

        return tokens;
    }

    /**
     * 设置训练器
     */
    private static void setupAndTrain(Model model, List<int[]> trainingData, MoEGPTModel moeModel) {
        System.out.println("设置训练器并开始简化训练...");

        // 简化的训练循环（不使用Trainer类）
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            float totalLoss = 0.0f;
            int batchCount = 0;

            // 重置专家使用统计
            if (epoch % 10 == 0) {
                moeModel.resetAllExpertStatistics();
            }

            for (int i = 0; i < trainingData.size(); i += BATCH_SIZE) {
                try {
                    // 准备批次数据
                    Variable[] batch = prepareBatch(trainingData, i, BATCH_SIZE);
                    if (batch == null || batch.length < 2) continue;

                    Variable inputs = batch[0];

                    // 前向传播（简化版）
                    Variable outputs = model.forward(inputs);

                    // 计算负载均衡损失
                    float balancingLoss = moeModel.computeTotalLoadBalancingLoss();
                    totalLoss += balancingLoss;

                    batchCount++;

                } catch (Exception e) {
                    System.err.println("批次训练出错: " + e.getMessage());
                }
            }

            // 打印训练进度
            if (epoch % 10 == 0) {
                float avgLoss = batchCount > 0 ? totalLoss / batchCount : 0.0f;
                System.out.printf("Epoch %d: 平均损失 = %.4f\n", epoch, avgLoss);

                // 每隔一段时间打印专家使用情况
                if (epoch % 20 == 0 && epoch > 0) {
                    System.out.println("\n--- Epoch " + epoch + " 专家使用情况 ---");
                    printSimpleExpertUsage(moeModel);
                }
            }
        }

        System.out.println("简化训练完成");
    }

    /**
     * 准备训练批次
     */
    private static Variable[] prepareBatch(List<int[]> trainingData, int startIdx, int batchSize) {
        try {
            List<int[]> batchTokens = new ArrayList<>();
            List<int[]> batchTargets = new ArrayList<>();

            for (int i = 0; i < batchSize && startIdx + i < trainingData.size(); i++) {
                int[] tokens = trainingData.get(startIdx + i);
                if (tokens.length > 1) {
                    // 输入：前n-1个token
                    int[] input = Arrays.copyOfRange(tokens, 0, tokens.length - 1);
                    // 目标：后n-1个token（下一个token预测）
                    int[] target = Arrays.copyOfRange(tokens, 1, tokens.length);

                    batchTokens.add(input);
                    batchTargets.add(target);
                }
            }

            if (batchTokens.isEmpty()) return null;

            // 转换为NdArray
            int realBatchSize = batchTokens.size();
            int seqLen = batchTokens.get(0).length;

            // 输入数据
            NdArray inputArray = NdArray.zeros(Shape.of(realBatchSize, seqLen));
            for (int b = 0; b < realBatchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    inputArray.set((float) batchTokens.get(b)[s], b, s);
                }
            }

            // 目标数据
            NdArray targetArray = NdArray.zeros(Shape.of(realBatchSize, seqLen));
            for (int b = 0; b < realBatchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    targetArray.set((float) batchTargets.get(b)[s], b, s);
                }
            }

            return new Variable[]{new Variable(inputArray), new Variable(targetArray)};

        } catch (Exception e) {
            System.err.println("准备批次数据出错: " + e.getMessage());
            return null;
        }
    }

    /**
     * 分析MoE使用情况
     */
    private static void analyzeMoEUsage(MoEGPTModel moeModel) {
        System.out.println("分析专家使用情况...");

        // 打印详细的专家使用统计
        moeModel.printAllExpertStatistics();

        // 分析专家负载均衡
        analyzeLoadBalancing(moeModel);
    }

    /**
     * 简化的专家使用统计打印
     */
    private static void printSimpleExpertUsage(MoEGPTModel moeModel) {
        List<float[]> allUsageRates = moeModel.getAllLayersExpertUsageRates();

        for (int layer = 0; layer < allUsageRates.size(); layer++) {
            float[] usageRates = allUsageRates.get(layer);
            System.out.printf("第%d层专家使用率: ", layer + 1);
            for (int expert = 0; expert < usageRates.length; expert++) {
                System.out.printf("E%d:%.2f%% ", expert, usageRates[expert] * 100);
            }
            System.out.println();
        }
    }

    /**
     * 分析负载均衡情况
     */
    private static void analyzeLoadBalancing(MoEGPTModel moeModel) {
        System.out.println("\\n--- 负载均衡分析 ---");

        List<float[]> allUsageRates = moeModel.getAllLayersExpertUsageRates();

        for (int layer = 0; layer < allUsageRates.size(); layer++) {
            float[] usageRates = allUsageRates.get(layer);

            // 计算使用率的方差（衡量负载均衡程度）
            float mean = 1.0f / usageRates.length; // 理想的均匀分布
            float variance = 0.0f;
            for (float rate : usageRates) {
                variance += (rate - mean) * (rate - mean);
            }
            variance /= usageRates.length;

            // 计算负载均衡分数（0-100，100表示完全均衡）
            float balanceScore = Math.max(0, 100 * (1 - variance / (mean * mean)));

            System.out.printf("第%d层: 负载均衡分数 = %.1f/100\\n", layer + 1, balanceScore);
        }

        float totalBalancingLoss = moeModel.computeTotalLoadBalancingLoss();
        System.out.printf("总负载均衡损失: %.6f\\n", totalBalancingLoss);
    }

    /**
     * 测试文本生成
     */
    private static void testTextGeneration(Model model, SimpleTokenizer tokenizer) {
        System.out.println("测试文本生成...");

        // 准备一些测试输入
        String[] testPrompts = {
                "人工智能",
                "深度学习",
                "神经网络"
        };

        for (String prompt : testPrompts) {
            System.out.printf("\\n输入提示: \\\"%s\\\"\\n", prompt);

            try {
                // 简化的生成过程
                int[] promptTokens = tokenizeText(prompt, tokenizer);
                String generated = generateText(model, promptTokens, 5); // 生成5个token
                System.out.printf("生成文本: %s\\n", generated);

            } catch (Exception e) {
                System.err.println("文本生成出错: " + e.getMessage());
            }
        }
    }

    /**
     * 简化的文本生成函数
     */
    private static String generateText(Model model, int[] promptTokens, int maxGenTokens) {
        StringBuilder result = new StringBuilder();

        // 复制prompt tokens
        int[] currentTokens = Arrays.copyOf(promptTokens, promptTokens.length + maxGenTokens);
        int currentLength = promptTokens.length;

        for (int i = 0; i < maxGenTokens; i++) {
            try {
                // 准备输入
                int[] input = Arrays.copyOfRange(currentTokens, 0, currentLength);
                NdArray inputArray = NdArray.zeros(Shape.of(1, input.length));
                for (int j = 0; j < input.length; j++) {
                    inputArray.set((float) input[j], 0, j);
                }

                // 生成下一个token
                Variable logits = model.forward(new Variable(inputArray));
                int nextToken = predictNextToken(logits.getValue());

                // 添加到序列中
                currentTokens[currentLength] = nextToken;
                currentLength++;

                // 简化：直接添加token ID到结果（实际应用中需要解码）
                result.append("token_").append(nextToken).append(" ");

            } catch (Exception e) {
                System.err.println("生成第" + i + "个token时出错: " + e.getMessage());
                break;
            }
        }

        return result.toString().trim();
    }

    /**
     * 预测下一个token
     */
    private static int predictNextToken(NdArray logits) {
        int seqLen = logits.getShape().getDimension(1);
        int vocabSize = logits.getShape().getDimension(2);

        // 获取最后一个位置的logits
        float maxLogit = Float.NEGATIVE_INFINITY;
        int bestToken = 0;

        for (int v = 0; v < vocabSize; v++) {
            float logit = logits.get(0, seqLen - 1, v);
            if (logit > maxLogit) {
                maxLogit = logit;
                bestToken = v;
            }
        }

        return bestToken;
    }

    /**
     * 性能分析
     */
    private static void performanceAnalysis(MoEGPTModel moeModel) {
        System.out.println("进行性能分析...");

        long totalParams = moeModel.getParameterCount();
        long activeParams = moeModel.getActiveParameterCount();
        double efficiency = 100.0 * activeParams / totalParams;

        System.out.printf("总参数量: %,d\\n", totalParams);
        System.out.printf("激活参数量: %,d\\n", activeParams);
        System.out.printf("参数效率: %.2f%%\\n", efficiency);

        // 估算传统模型的参数量（用于对比）
        long traditionalParams = estimateTraditionalModelParams(moeModel);
        double capacityIncrease = 100.0 * totalParams / traditionalParams;

        System.out.printf("\\n与传统模型对比:\\n");
        System.out.printf("传统模型参数量: %,d\\n", traditionalParams);
        System.out.printf("容量增加倍数: %.2fx\\n", capacityIncrease / 100);
        System.out.printf("计算效率提升: 激活参数仅为总参数的%.1f%%\\n", efficiency);
    }

    /**
     * 估算传统模型的参数量（用于对比）
     */
    private static long estimateTraditionalModelParams(MoEGPTModel moeModel) {
        // 假设传统模型使用相同的配置，但没有MoE
        long traditionalParams = 0;

        // Token和位置嵌入
        traditionalParams += (long) moeModel.getVocabSize() * moeModel.getDModel();
        traditionalParams += (long) moeModel.getMaxSeqLength() * moeModel.getDModel();

        // 每层的参数（不使用MoE）
        int dModel = moeModel.getDModel();
        int dFF = moeModel.getExpertHiddenDim(); // 假设传统FFN使用相同的隐藏维度

        for (int i = 0; i < moeModel.getNumLayers(); i++) {
            // LayerNorm参数
            traditionalParams += 2L * 2 * dModel;

            // MultiHeadAttention参数
            traditionalParams += 4L * (dModel * dModel + dModel);

            // 传统FeedForward参数（单个FFN，不是多专家）
            traditionalParams += 2L * dModel * dFF + dFF + dModel;
        }

        // 最终LayerNorm和输出头
        traditionalParams += 2L * dModel;
        traditionalParams += (long) dModel * moeModel.getVocabSize();

        return traditionalParams;
    }
}