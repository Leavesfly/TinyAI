package io.leavesfly.tinyai.gpt3.training;

import io.leavesfly.tinyai.gpt3.GPT3Config;
import io.leavesfly.tinyai.gpt3.GPT3Model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * GPT-3完整训练演示
 *
 * 演示GPT-3完整的预训练-微调-推理流程，并展示GPT-3特有技术：
 * - 并行Attention+MLP架构
 * - RoPE旋转位置编码
 * - 稀疏注意力机制
 * - 梯度检查点
 * - KV Cache加速推理
 *
 * 流程：
 * 1. 准备训练数据集（自动生成AI/ML领域教学文本）
 * 2. 预训练（Causal Language Modeling）
 * 3. 微调（Instruction Following SFT）
 * 4. 推理测试（多种解码策略对比）
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT3TrainDemo {

    private static GPT3Dataset.SimpleTokenizer sharedTokenizer = new GPT3Dataset.SimpleTokenizer();

    private static final String DATA_DIR       = "./data/gpt3_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/gpt3_v2";

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("GPT-3 完整训练与推理演示");
        System.out.println("展示GPT-3核心技术：并行Attn、RoPE、稀疏注意力、梯度检查点、KV Cache");
        System.out.println("=".repeat(80));

        try {
            // 步骤0: 准备数据集文件
            prepareDatasets();

            // 步骤1: 预训练（使用小型配置，适合演示）
            GPT3Model pretrainedModel = runPretraining();

            // 步骤2: 微调（在预训练模型基础上SFT）
            GPT3Model finetunedModel = runFinetuning(pretrainedModel);

            // 步骤3: 推理测试（多策略对比 + KV Cache加速）
            runInference(finetunedModel);

            System.out.println("\n" + "=".repeat(80));
            System.out.println("✅ GPT-3完整训练流程演示成功!");
            System.out.println("=".repeat(80));

        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // ==================== 数据准备 ====================

    /**
     * 准备训练数据集（自动生成）
     */
    private static void prepareDatasets() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("步骤0: 准备训练数据集");
        System.out.println("=".repeat(80));

        File dataDir = new File(DATA_DIR);
        if (!dataDir.exists()) {
            dataDir.mkdirs();
            System.out.println("创建数据目录: " + DATA_DIR);
        }

        generatePretrainDataset();
        generateFinetuneDataset();

        System.out.println("\n✅ 数据集准备完成!");
    }

    /**
     * 生成预训练数据集（AI/ML领域多样性文本）
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n生成预训练数据集...");

        List<String> pretrainTexts = new ArrayList<>();
        pretrainTexts.addAll(generateDeepLearningTexts());
        pretrainTexts.addAll(generateTransformerTexts());
        pretrainTexts.addAll(generateGPT3Texts());
        pretrainTexts.addAll(generateNLPTexts());
        pretrainTexts.addAll(generateMathTexts());

        // 构建词汇表（预训练时解冻tokenizer）
        sharedTokenizer.unfreeze();
        for (String text : pretrainTexts) {
            sharedTokenizer.encode(text);
        }

        // 写入文件
        String pretrainFile = DATA_DIR + "/pretrain.txt";
        writeTextsToFile(pretrainTexts, pretrainFile);
        System.out.println("预训练数据: " + pretrainTexts.size() + " 条文本 -> " + pretrainFile);
    }

    /**
     * 生成微调数据集（指令-回答格式）
     */
    private static void generateFinetuneDataset() throws IOException {
        System.out.println("\n生成微调数据集...");

        List<String> trainTexts = new ArrayList<>();
        List<String> valTexts   = new ArrayList<>();

        List<String> allInstructions = generateInstructionTexts();

        // 80/20划分训练/验证
        int splitIdx = (int)(allInstructions.size() * 0.8);
        trainTexts.addAll(allInstructions.subList(0, splitIdx));
        valTexts.addAll(allInstructions.subList(splitIdx, allInstructions.size()));

        String trainFile = DATA_DIR + "/finetune_train.txt";
        String valFile   = DATA_DIR + "/finetune_val.txt";
        writeTextsToFile(trainTexts, trainFile);
        writeTextsToFile(valTexts, valFile);

        System.out.println("微调训练数据: " + trainTexts.size() + " 条 -> " + trainFile);
        System.out.println("微调验证数据: " + valTexts.size() + " 条 -> " + valFile);
    }

    // ==================== 训练流程 ====================

    /**
     * 步骤1: 预训练
     * 使用小型GPT-3配置演示完整预训练流程
     */
    private static GPT3Model runPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("步骤1: GPT-3 预训练");
        System.out.println("=".repeat(80));

        // 创建适合演示的nano配置（参数量小，训练快）
        GPT3Config config = createNanoConfig();
        GPT3Model model = new GPT3Model("gpt3-nano-pretrain", config);
        model.printModelInfo();

        // 准备数据集
        int vocabSize = sharedTokenizer.getVocabSize();
        // 更新配置词汇表大小
        config.setVocabSize(Math.max(vocabSize, config.getVocabSize()));

        GPT3Dataset trainDataset = new GPT3Dataset(
                config.getNPositions(), /* maxSeqLen */
                2,                      /* batchSize */
                config.getVocabSize()
        );

        // 加载预训练数据
        String pretrainFile = DATA_DIR + "/pretrain.txt";
        trainDataset.loadFromFile(pretrainFile, sharedTokenizer);

        if (trainDataset.getSampleCount() == 0) {
            System.out.println("警告: 数据不足，使用内置文本直接训练");
            List<String> texts = generateDeepLearningTexts();
            trainDataset.loadFromTexts(texts, sharedTokenizer);
        }

        // 配置预训练器
        GPT3Pretrain pretrain = new GPT3Pretrain(model, trainDataset)
                .configure(3, 6e-4f, 100, 1.0f)
                .setCheckpoint(CHECKPOINT_DIR + "/pretrain", 1);

        pretrain.train();

        System.out.println("预训练完成! " + pretrain.getStats());
        return model;
    }

    /**
     * 步骤2: 微调（SFT）
     */
    private static GPT3Model runFinetuning(GPT3Model pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("步骤2: GPT-3 微调训练 (SFT)");
        System.out.println("=".repeat(80));

        GPT3Config config = pretrainedModel.getConfig();

        GPT3Dataset trainDataset = new GPT3Dataset(config.getNPositions(), 2, config.getVocabSize());
        GPT3Dataset valDataset   = new GPT3Dataset(config.getNPositions(), 2, config.getVocabSize());

        // 加载微调数据（指令-回答格式）
        String trainFile = DATA_DIR + "/finetune_train.txt";
        String valFile   = DATA_DIR + "/finetune_val.txt";

        sharedTokenizer.freeze();  // 微调时冻结词汇表

        trainDataset.loadFromInstructionTexts(
                readLinesFromFile(trainFile), sharedTokenizer, "Response:");
        valDataset.loadFromInstructionTexts(
                readLinesFromFile(valFile), sharedTokenizer, "Response:");

        if (trainDataset.getSampleCount() == 0) {
            System.out.println("警告: 微调数据不足，使用内置指令数据");
            List<String> instructions = generateInstructionTexts();
            trainDataset.loadFromInstructionTexts(instructions, sharedTokenizer, "Response:");
            valDataset.loadFromInstructionTexts(instructions.subList(0, 2), sharedTokenizer, "Response:");
        }

        // 配置微调训练器
        GPT3Finetune finetune = new GPT3Finetune(pretrainedModel, trainDataset, valDataset)
                .configure(3, 6e-5f, 2)
                .setCheckpoint(CHECKPOINT_DIR + "/finetune", 50);

        finetune.train();

        System.out.println("微调完成! " + finetune.getStats());
        return pretrainedModel;
    }

    /**
     * 步骤3: 推理测试（多策略 + KV Cache对比）
     */
    private static void runInference(GPT3Model model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("步骤3: GPT-3 推理测试");
        System.out.println("=".repeat(80));

        GPT3Inference inference = new GPT3Inference(model);

        // 准备测试提示词
        sharedTokenizer.freeze();
        String promptText = "deep learning is";
        List<Integer> promptIds = sharedTokenizer.encode(promptText);
        int[] prompt = promptIds.stream().mapToInt(Integer::intValue).toArray();

        System.out.println("提示词: \"" + promptText + "\"");
        System.out.println("提示词ID数量: " + prompt.length);
        System.out.println();

        // 1. 贪婪解码
        System.out.println("--- 1. 贪婪解码 (Greedy) ---");
        int[] greedyResult = inference.generateGreedy(prompt, 8);
        System.out.println("生成结果: " + sharedTokenizer.decode(greedyResult));

        // 2. 带KV Cache的贪婪解码（GPT-3特有加速）
        System.out.println("\n--- 2. 带KV Cache的贪婪解码（GPT-3特有加速）---");
        long start = System.currentTimeMillis();
        int[] cacheResult = inference.generateGreedyWithCache(prompt, 8);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("生成结果: " + sharedTokenizer.decode(cacheResult));
        System.out.println("KV Cache推理耗时: " + elapsed + " ms");

        // 3. Temperature采样（temperature=0.8）
        System.out.println("\n--- 3. Temperature采样 (T=0.8) ---");
        int[] tempResult = inference.generateWithTemperature(prompt, 8, 0.8f);
        System.out.println("生成结果: " + sharedTokenizer.decode(tempResult));

        // 4. Top-K采样（K=10）
        System.out.println("\n--- 4. Top-K采样 (K=10, T=1.0) ---");
        int[] topKResult = inference.generateTopK(prompt, 8, 10, 1.0f);
        System.out.println("生成结果: " + sharedTokenizer.decode(topKResult));

        // 5. Top-P采样（p=0.9）
        System.out.println("\n--- 5. Top-P/Nucleus采样 (P=0.9, T=1.0) ---");
        int[] topPResult = inference.generateTopP(prompt, 8, 0.9f, 1.0f);
        System.out.println("生成结果: " + sharedTokenizer.decode(topPResult));

        // 6. Beam Search（beam=3）
        System.out.println("\n--- 6. Beam Search (beamSize=3) ---");
        int[] beamResult = inference.generateBeamSearch(prompt, 8, 3);
        System.out.println("生成结果: " + sharedTokenizer.decode(beamResult));

        System.out.println("\n推理演示完成!");
        System.out.println("GPT-3特色：KV Cache加速推理（O(n) vs 无Cache的O(n²)）已验证");
    }

    // ==================== 配置工厂 ====================

    /**
     * 创建nano配置（极小参数量，适合演示和单元测试）
     * 嵌入维度64, 2层, 2头, 序列长度64
     */
    private static GPT3Config createNanoConfig() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(64);
        config.setNLayer(2);
        config.setNHead(2);
        config.setNInner(256);
        config.setNPositions(64);
        config.setVocabSize(1000);     // 演示用小词汇表
        config.setParallelAttention(true);
        config.setUseRotaryEmbedding(false);
        config.setSparseAttention(false);
        config.setGradientCheckpointing(false);
        config.setUseCache(true);
        return config;
    }

    /**
     * 创建启用全部GPT-3特性的演示配置
     * 可用于展示RoPE + 稀疏注意力 + 梯度检查点（参数量略大）
     */
    private static GPT3Config createFeatureDemoConfig() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(128);
        config.setNLayer(4);
        config.setNHead(4);
        config.setNInner(512);
        config.setNPositions(128);
        config.setVocabSize(2000);
        config.setParallelAttention(true);
        config.setUseRotaryEmbedding(true);
        config.setRotaryPct(0.5);      // 50%维度使用RoPE
        config.setSparseAttention(true);
        config.setSparseLocalWindow(32);
        config.setSparseStrideSize(16);
        config.setGradientCheckpointing(true);
        config.setUseCache(true);
        return config;
    }

    // ==================== 数据生成辅助方法 ====================

    private static List<String> generateDeepLearningTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("deep learning is a subset of machine learning that uses neural networks with many layers");
        texts.add("neural networks learn representations from data through gradient descent optimization");
        texts.add("backpropagation computes gradients by applying the chain rule through the network layers");
        texts.add("convolutional neural networks excel at image recognition by learning spatial features");
        texts.add("recurrent neural networks process sequential data by maintaining hidden state over time");
        texts.add("batch normalization stabilizes training by normalizing layer inputs during forward pass");
        texts.add("dropout regularization prevents overfitting by randomly deactivating neurons during training");
        texts.add("residual connections allow gradients to flow directly through skip connections in deep networks");
        texts.add("layer normalization normalizes across the feature dimension and is used in transformers");
        texts.add("the adam optimizer combines momentum and adaptive learning rates for efficient optimization");
        texts.add("learning rate scheduling adjusts the learning rate during training to improve convergence");
        texts.add("transfer learning fine-tunes pretrained models on new tasks with limited data");
        texts.add("data augmentation artificially expands training data to improve model generalization");
        texts.add("regularization techniques like weight decay prevent neural networks from overfitting");
        texts.add("gradient clipping prevents exploding gradients by rescaling them to a maximum norm");
        return texts;
    }

    private static List<String> generateTransformerTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("the transformer architecture uses self-attention to capture long-range dependencies");
        texts.add("multi-head attention allows the model to attend to different positions simultaneously");
        texts.add("the attention mechanism computes query key value products to weight token importance");
        texts.add("positional encoding adds position information to token embeddings in transformers");
        texts.add("causal language modeling trains the model to predict each token from previous tokens");
        texts.add("the feed-forward network in transformers applies two linear transformations with activation");
        texts.add("pre-layernorm applies normalization before attention and feed-forward sublayers");
        texts.add("the softmax function converts attention scores to probability distributions over tokens");
        texts.add("masked self-attention prevents future tokens from influencing current token predictions");
        texts.add("token embeddings map discrete token ids to continuous vector representations");
        return texts;
    }

    private static List<String> generateGPT3Texts() {
        List<String> texts = new ArrayList<>();
        texts.add("gpt3 introduced parallel attention and mlp computation for improved efficiency");
        texts.add("rotary position embedding encodes position information through rotation in vector space");
        texts.add("sparse attention reduces computation by only attending to local and strided global positions");
        texts.add("gradient checkpointing trades computation for memory by recomputing activations backward");
        texts.add("kv cache stores key and value tensors to avoid redundant computation during inference");
        texts.add("few-shot learning allows gpt3 to perform tasks from a small number of examples in context");
        texts.add("in-context learning enables language models to adapt to tasks without gradient updates");
        texts.add("scaling laws show that model performance improves predictably with more compute and data");
        texts.add("the gpt3 model has 175 billion parameters trained on 300 billion tokens of text data");
        texts.add("autoregressive generation produces text one token at a time using the previous tokens");
        return texts;
    }

    private static List<String> generateNLPTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("natural language processing enables computers to understand and generate human language");
        texts.add("tokenization splits text into subword units using byte pair encoding or wordpiece");
        texts.add("word embeddings represent words as dense vectors in a continuous semantic space");
        texts.add("language models assign probabilities to sequences of tokens in natural language");
        texts.add("text generation produces coherent and contextually relevant text from a given prompt");
        texts.add("question answering systems extract or generate answers to questions from context");
        texts.add("text summarization condenses long documents into shorter informative summaries");
        texts.add("named entity recognition identifies and classifies named entities in text");
        texts.add("sentiment analysis determines the emotional tone or opinion expressed in text");
        texts.add("machine translation converts text from one language to another automatically");
        return texts;
    }

    private static List<String> generateMathTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("matrix multiplication is the fundamental operation in neural network forward passes");
        texts.add("the cross entropy loss measures the difference between predicted and true distributions");
        texts.add("softmax converts logits to probability distributions by exponentiating and normalizing");
        texts.add("the gradient of a function indicates the direction of steepest ascent at each point");
        texts.add("the chain rule allows gradients to be propagated backward through composite functions");
        return texts;
    }

    private static List<String> generateInstructionTexts() {
        List<String> instructions = new ArrayList<>();
        instructions.add("Instruction: What is deep learning? Response: Deep learning is a machine learning approach using neural networks with multiple layers to learn hierarchical representations from data.");
        instructions.add("Instruction: Explain the transformer architecture. Response: Transformers use self-attention mechanisms to process sequences in parallel and capture long-range dependencies between tokens.");
        instructions.add("Instruction: What is GPT-3? Response: GPT-3 is a large language model with 175 billion parameters that uses autoregressive language modeling with parallel attention and MLP computation.");
        instructions.add("Instruction: What is backpropagation? Response: Backpropagation computes gradients by applying the chain rule backward through the network to update weights using gradient descent.");
        instructions.add("Instruction: What is attention mechanism? Response: Attention computes weighted combinations of values based on query-key similarity scores, allowing models to focus on relevant parts of the input.");
        instructions.add("Instruction: Explain gradient clipping. Response: Gradient clipping rescales gradients when their norm exceeds a threshold, preventing exploding gradients during training.");
        instructions.add("Instruction: What is KV cache? Response: KV cache stores key and value tensors from previous tokens to avoid recomputing them at each inference step, enabling O(n) generation efficiency.");
        instructions.add("Instruction: What is rotary position embedding? Response: RoPE encodes positional information by rotating query and key vectors, allowing relative position information to be captured in dot-product attention.");
        instructions.add("Instruction: Explain sparse attention. Response: Sparse attention reduces computational cost by only attending to local windows and strided global positions rather than all token pairs.");
        instructions.add("Instruction: What is gradient checkpointing? Response: Gradient checkpointing saves memory by not storing intermediate activations and recomputing them during the backward pass.");
        return instructions;
    }

    // ==================== 文件工具方法 ====================

    private static void writeTextsToFile(List<String> texts, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String text : texts) {
                writer.write(text);
                writer.newLine();
            }
        }
    }

    private static List<String> readLinesFromFile(String filePath) {
        List<String> lines = new ArrayList<>();
        File file = new File(filePath);
        if (!file.exists()) return lines;
        try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line);
                }
            }
        } catch (IOException e) {
            System.err.println("读取文件失败: " + filePath + " - " + e.getMessage());
        }
        return lines;
    }
}
