package io.leavesfly.tinyai.example.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.nlp.GPT2Model;
import io.leavesfly.tinyai.nlp.GPT2TextDataset;
import io.leavesfly.tinyai.nlp.SimpleTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2语言模型训练和生成示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 本示例展示了如何：
 * 1. 创建和配置GPT-2模型
 * 2. 准备训练数据
 * 3. 训练语言模型
 * 4. 生成文本
 */
public class GPT2Example {

    /**
     * 主函数，执行GPT-2语言模型示例
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== GPT-2 Language Model Example ===");

        try {
            // 1. 创建分词器并构建词汇表
            SimpleTokenizer tokenizer = createTokenizer();

            // 2. 创建训练数据集
            GPT2TextDataset dataset = createTrainingDataset(tokenizer);

            // 3. 创建GPT-2模型
            GPT2Model gpt2Model = GPT2Model.createTinyModel("gpt2_tiny", tokenizer.getVocabSize());
            gpt2Model.printModelInfo();

            // 4. 训练模型
            trainModel(gpt2Model, dataset);

            // 5. 生成文本
            generateText(gpt2Model, tokenizer);

            System.out.println("GPT-2 Example completed successfully!");

        } catch (Exception e) {
            System.err.println("Error in GPT-2 example: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 创建分词器并构建词汇表
     *
     * @return 创建的分词器实例
     */
    private static SimpleTokenizer createTokenizer() {
        System.out.println("\n--- Creating Tokenizer ---");

        SimpleTokenizer tokenizer = new SimpleTokenizer();

        // 准备训练文本用于构建词汇表
        List<String> vocabTexts = getVocabularyTexts();

        // 构建词汇表
        tokenizer.buildVocab(vocabTexts, 1, 1000);  // 最小频率1，最大词汇表1000
        tokenizer.printVocabInfo();

        return tokenizer;
    }

    /**
     * 获取用于构建词汇表的文本
     *
     * @return 用于构建词汇表的文本列表
     */
    private static List<String> getVocabularyTexts() {
        List<String> texts = new ArrayList<>();

        // 添加一些示例文本
        texts.add("The quick brown fox jumps over the lazy dog.");
        texts.add("Machine learning is a subset of artificial intelligence.");
        texts.add("Deep learning models can learn complex patterns from data.");
        texts.add("Natural language processing helps computers understand human language.");
        texts.add("Transformers have revolutionized the field of NLP.");
        texts.add("GPT models are autoregressive language models.");
        texts.add("Attention mechanisms allow models to focus on relevant information.");
        texts.add("Large language models can generate coherent text.");
        texts.add("Training neural networks requires careful tuning of hyperparameters.");
        texts.add("Artificial intelligence will transform many industries.");
        texts.add("Python is a popular programming language for machine learning.");
        texts.add("Data preprocessing is crucial for model performance.");
        texts.add("Gradient descent is used to optimize neural network parameters.");
        texts.add("Cross-validation helps evaluate model generalization.");
        texts.add("Overfitting occurs when models memorize training data.");
        texts.add("Regularization techniques help prevent overfitting.");
        texts.add("Feature engineering can improve model accuracy.");
        texts.add("Deep neural networks have multiple hidden layers.");
        texts.add("Convolutional networks are effective for image processing.");
        texts.add("Recurrent networks can handle sequential data.");

        return texts;
    }

    /**
     * 创建训练数据集
     *
     * @param tokenizer 分词器实例
     * @return 创建的训练数据集
     */
    private static GPT2TextDataset createTrainingDataset(SimpleTokenizer tokenizer) {
        System.out.println("\n--- Creating Training Dataset ---");

        // 使用相同的文本作为训练数据
        List<String> trainingTexts = getVocabularyTexts();

        // 创建数据集
        GPT2TextDataset dataset = new GPT2TextDataset(
                "gpt2_training",
                trainingTexts,
                tokenizer,
                64,    // 最大序列长度
                4,     // 批次大小
                true   // 打乱数据
        );

        System.out.println("Training dataset created with " + dataset.getSize() + " samples");
        return dataset;
    }

    /**
     * 训练模型
     *
     * @param gpt2Model GPT-2模型实例
     * @param dataset   训练数据集
     */
    private static void trainModel(GPT2Model gpt2Model, GPT2TextDataset dataset) {
        System.out.println("\n--- Training GPT-2 Model ---");

        try {
            // 创建模型包装器
            Model model = new Model("gpt2_model", gpt2Model);

            // 创建优化器
            Adam optimizer = new Adam(model, 0.001f, 0.9f, 0.999f, 1e-8f);

            // 创建损失函数
            SoftmaxCrossEntropy lossFunction = new SoftmaxCrossEntropy();

            // 训练参数
            int epochs = 3;
            System.out.println("Starting training for " + epochs + " epochs...");

            // 开始训练
            for (int epoch = 0; epoch < epochs; epoch++) {
                System.out.println("\n--- Epoch " + (epoch + 1) + "/" + epochs + " ---");

                dataset.prepare();
                List<io.leavesfly.tinyai.ml.dataset.Batch> batches = dataset.getBatches();

                float totalLoss = 0.0f;
                int batchCount = 0;

                for (Batch batch : batches) {
                    try {
                        // 前向传播
                        Variable predictions = model.forward(new Variable(batch.getX()[0]));

                        // 计算损失
                        Variable targets = new Variable(batch.getY()[0]);
                        Variable loss = lossFunction.loss(predictions, targets);

                        float lossValue = loss.getValue().get(0);
                        totalLoss += lossValue;
                        batchCount++;

                        // 反向传播
                        loss.backward();

                        // 更新参数
                        optimizer.update();

                        // 清除梯度
                        model.clearGrads();

                        if (batchCount % 2 == 0) {
                            System.out.println("  Batch " + batchCount + ", Loss: " +
                                    String.format("%.4f", lossValue));
                        }

                    } catch (Exception e) {
                        System.err.println("Error in batch " + batchCount + ": " + e.getMessage());
                        // 继续训练下一个批次
                    }
                }

                float avgLoss = totalLoss / Math.max(batchCount, 1);
                System.out.println("Epoch " + (epoch + 1) + " completed. Average Loss: " +
                        String.format("%.4f", avgLoss));
            }

            System.out.println("Training completed!");

        } catch (Exception e) {
            System.err.println("Error during training: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 生成文本
     *
     * @param gpt2Model GPT-2模型实例
     * @param tokenizer 分词器实例
     */
    private static void generateText(GPT2Model gpt2Model, SimpleTokenizer tokenizer) {
        System.out.println("\n--- Text Generation ---");

        try {
            String[] prompts = {
                    "Machine learning",
                    "The quick brown",
                    "Deep learning models",
                    "Natural language"
            };

            for (String prompt : prompts) {
                System.out.println("\nPrompt: \"" + prompt + "\"");
                String generatedText = generateFromPrompt(gpt2Model, tokenizer, prompt, 20);
                System.out.println("Generated: \"" + generatedText + "\"");
            }

        } catch (Exception e) {
            System.err.println("Error during text generation: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 从提示生成文本
     *
     * @param gpt2Model    GPT-2模型实例
     * @param tokenizer    分词器实例
     * @param prompt       提示文本
     * @param maxNewTokens 最大新生成token数
     * @return 生成的文本
     */
    private static String generateFromPrompt(GPT2Model gpt2Model, SimpleTokenizer tokenizer,
                                             String prompt, int maxNewTokens) {
        // 编码提示
        int[] promptTokens = tokenizer.encode(prompt, false);  // 不添加特殊token

        List<Integer> generatedTokens = new ArrayList<>();
        for (int token : promptTokens) {
            generatedTokens.add(token);
        }

        // 生成新token
        for (int i = 0; i < maxNewTokens; i++) {
            try {
                // 准备输入序列
                int[] inputTokens = new int[Math.min(generatedTokens.size(), gpt2Model.getMaxSeqLength())];
                int startIdx = Math.max(0, generatedTokens.size() - gpt2Model.getMaxSeqLength());

                for (int j = 0; j < inputTokens.length; j++) {
                    inputTokens[j] = generatedTokens.get(startIdx + j);
                }

                // 填充到最大长度
                int[] paddedTokens = tokenizer.pad(inputTokens, gpt2Model.getMaxSeqLength(), "pre");

                // 创建输入张量
                NdArray input = NdArray.of(Shape.of(1, gpt2Model.getMaxSeqLength()));
                for (int j = 0; j < gpt2Model.getMaxSeqLength(); j++) {
                    input.set(paddedTokens[j], 0, j);
                }

                // 预测下一个token
                int nextToken = gpt2Model.predictNextToken(input);

                // 检查是否是结束token
                if (nextToken == SimpleTokenizer.EOS_ID || nextToken == SimpleTokenizer.PAD_ID) {
                    break;
                }

                generatedTokens.add(nextToken);

            } catch (Exception e) {
                System.err.println("Error generating token " + i + ": " + e.getMessage());
                break;
            }
        }

        // 解码生成的文本
        int[] allTokens = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        return tokenizer.decode(allTokens, true);
    }
}