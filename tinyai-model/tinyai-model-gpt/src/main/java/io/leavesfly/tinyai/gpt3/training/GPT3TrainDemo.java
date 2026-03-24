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
    private static final String CHECKPOINT_DIR = "./checkpoints/gpt3";

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

        // 先确定实际词汇表大小，再创建模型（确保输出层维度与词汇表一致，避免生成越界token ID）
        // 注意：必须直接使用实际词汇量，不能取 max，否则 config 默认值(1000)会导致输出层偏大
        int vocabSize = sharedTokenizer.getVocabSize();
        config.setVocabSize(vocabSize);

        GPT3Model model = new GPT3Model("gpt3-nano-pretrain", config);
        model.printModelInfo();

        GPT3Dataset trainDataset = new GPT3Dataset(
                config.getNPositions(), /* maxSeqLen */
                4,                      /* batchSize */
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
                .configure(30, 6e-4f, 25, 1.0f)
                .setCheckpoint(CHECKPOINT_DIR + "/pretrain", 5);

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
                .configure(10, 6e-5f, 2)
                .setCheckpoint(CHECKPOINT_DIR + "/finetune", 500);

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
        int[] greedyResult = inference.generateGreedy(prompt, model.getConfig().getNPositions());
        System.out.println("生成结果: " + sharedTokenizer.decode(greedyResult));

        // 2. 带KV Cache的贪婪解码（GPT-3特有加速）
        System.out.println("\n--- 2. 带KV Cache的贪婪解码（GPT-3特有加速）---");
        long start = System.currentTimeMillis();
        int[] cacheResult = inference.generateGreedyWithCache(prompt, model.getConfig().getNPositions());
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("生成结果: " + sharedTokenizer.decode(cacheResult));
        System.out.println("KV Cache推理耗时: " + elapsed + " ms");

        // 3. Temperature采样（temperature=0.8）
        System.out.println("\n--- 3. Temperature采样 (T=0.8) ---");
        int[] tempResult = inference.generateWithTemperature(prompt, model.getConfig().getNPositions(), 0.8f);
        System.out.println("生成结果: " + sharedTokenizer.decode(tempResult));

        // 4. Top-K采样（K=10）
        System.out.println("\n--- 4. Top-K采样 (K=10, T=1.0) ---");
        int[] topKResult = inference.generateTopK(prompt, model.getConfig().getNPositions(), 10, 1.0f);
        System.out.println("生成结果: " + sharedTokenizer.decode(topKResult));

        // 5. Top-P采样（p=0.9）
        System.out.println("\n--- 5. Top-P/Nucleus采样 (P=0.9, T=1.0) ---");
        int[] topPResult = inference.generateTopP(prompt, model.getConfig().getNPositions(), 0.9f, 1.0f);
        System.out.println("生成结果: " + sharedTokenizer.decode(topPResult));

        // 6. Beam Search（beam=3）
        System.out.println("\n--- 6. Beam Search (beamSize=3) ---");
        int[] beamResult = inference.generateBeamSearch(prompt, model.getConfig().getNPositions(), 3);
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
        config.setNPositions(32);
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
        texts.add("deep learning is a powerful subset of machine learning that uses neural networks with many layers to progressively extract higher-level features from raw input data, enabling remarkable breakthroughs in computer vision, natural language processing, and speech recognition.");
        texts.add("neural networks learn representations from data through gradient descent optimization by iteratively adjusting connection weights to minimize prediction error, automatically discovering hierarchical features without manual feature engineering.");
        texts.add("backpropagation computes gradients by applying the chain rule through the network layers, efficiently propagating error signals from the output back to earlier layers, forming the computational foundation for training all modern deep neural networks.");
        texts.add("convolutional neural networks excel at image recognition by learning spatial features through convolutional filters that detect edges, textures, and complex patterns hierarchically, revolutionizing computer vision applications from autonomous driving to medical diagnosis.");
        texts.add("recurrent neural networks process sequential data by maintaining hidden state over time, enabling the capture of temporal dependencies in text, speech, and time series, with LSTM and GRU variants solving the vanishing gradient problem for long sequences.");
        texts.add("batch normalization stabilizes training by normalizing layer inputs during forward pass to zero mean and unit variance, reducing internal covariate shift, allowing higher learning rates, and acting as a regularizer that improves generalization.");
        texts.add("dropout regularization prevents overfitting by randomly deactivating neurons during training with probability p, forcing the network to learn redundant representations and effectively training an ensemble of thinned networks that improves robustness.");
        texts.add("residual connections allow gradients to flow directly through skip connections in deep networks by adding the input of a layer to its output, enabling training of architectures with hundreds of layers that would otherwise suffer from degradation.");
        texts.add("layer normalization normalizes across the feature dimension and is used in transformers, providing stable training dynamics regardless of batch size by computing statistics per sample rather than across the batch dimension.");
        texts.add("the adam optimizer combines momentum and adaptive learning rates for efficient optimization by maintaining estimates of first and second moments of gradients, providing bias-corrected updates that work well across diverse architectures and tasks.");
        texts.add("learning rate scheduling adjusts the learning rate during training to improve convergence, with warmup phases for stability, constant phases for learning, and decay phases for fine-tuning, enabling better final performance than fixed learning rates.");
        texts.add("transfer learning fine-tunes pretrained models on new tasks with limited data by leveraging features learned on large source datasets, dramatically reducing the need for task-specific training data and accelerating development cycles.");
        texts.add("data augmentation artificially expands training data to improve model generalization by applying transformations like rotation, cropping, flipping, and color jittering, exposing the model to variations that improve robustness to real-world conditions.");
        texts.add("regularization techniques like weight decay prevent neural networks from overfitting by adding penalty terms to the loss that encourage smaller weights, improving generalization to unseen data by reducing model complexity.");
        texts.add("gradient clipping prevents exploding gradients by rescaling them to a maximum norm, stabilizing training in recurrent networks and transformers where gradients can grow exponentially through many layers during backpropagation.");
        texts.add("activation functions introduce non-linearity into neural networks enabling them to learn complex patterns that linear transformations alone cannot represent, with ReLU, GELU, and swish being popular choices for different architectures.");
        texts.add("the vanishing gradient problem occurs when gradients become too small in deep networks as they propagate backward through many layers, preventing earlier layers from learning effectively, addressed by residual connections and careful activation design.");
        texts.add("weight initialization strategies like xavier and he initialization help with training stability by setting initial weights to appropriate scales based on layer dimensions, preventing early training instability from too large or too small activations.");
        texts.add("early stopping prevents overfitting by halting training when validation loss increases, avoiding unnecessary computation while ensuring the model does not continue training on the training set beyond the point of best generalization.");
        texts.add("mini-batch gradient descent computes gradients on small subsets of data for efficiency, balancing the speed of stochastic updates with the stability of batch updates, enabling training on datasets too large to fit in memory.");
        texts.add("momentum accelerates gradient descent by accumulating velocity in consistent directions, damping oscillations and enabling faster convergence through ravines in the loss landscape where gradients point in different directions.");
        texts.add("the learning rate controls the step size during gradient descent optimization, with too large values causing divergence and too small values leading to slow convergence, making it the most important hyperparameter for training.");
        texts.add("batch size affects the noise in gradient estimates and training dynamics, with smaller batches providing regularization through noise but slower convergence, while larger batches enable efficient parallel computation but may generalize worse.");
        texts.add("epoch refers to one complete pass through the entire training dataset during training, with multiple epochs typically required for convergence as the model sees each example multiple times to refine its parameters.");
        texts.add("validation data is used to tune hyperparameters and monitor model performance during training, providing an unbiased estimate of generalization that guides decisions about learning rate, architecture, and regularization.");
        texts.add("test data provides final evaluation of model performance on unseen examples, held out from all training decisions to give an unbiased estimate of how the model will perform in production on new data.");
        texts.add("overfitting occurs when a model memorizes training data instead of learning patterns, resulting in excellent training performance but poor test performance, addressed through regularization, data augmentation, and early stopping.");
        texts.add("underfitting happens when a model is too simple to capture underlying patterns in the data, resulting in poor performance on both training and test sets, addressed by increasing model capacity or training longer.");
        texts.add("the bias-variance tradeoff balances model complexity with generalization ability, where high bias leads to underfitting simple models and high variance leads to overfitting complex models, requiring careful model selection.");
        texts.add("cross-validation estimates model performance by partitioning data into multiple folds, training on subsets while validating on held-out portions, providing robust performance estimates that account for data variability.");
        return texts;
    }

    private static List<String> generateTransformerTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("the transformer architecture uses self-attention to capture long-range dependencies without recurrence, enabling parallel processing of sequences and scaling to models with hundreds of billions of parameters across language and vision tasks.");
        texts.add("multi-head attention allows the model to attend to different positions simultaneously by running multiple attention operations in parallel, capturing diverse relationships between tokens through different learned projection subspaces.");
        texts.add("the attention mechanism computes query key value products to weight token importance, where queries match against keys to determine attention weights that control how values are combined for each output position.");
        texts.add("positional encoding adds position information to token embeddings in transformers through sinusoidal functions or learned embeddings, enabling the model to distinguish token order without the sequential processing of recurrent networks.");
        texts.add("causal language modeling trains the model to predict each token from previous tokens in an autoregressive manner, enabling text generation by learning the probability distribution over sequences during pre-training on large corpora.");
        texts.add("the feed-forward network in transformers applies two linear transformations with activation between them, processing each position independently to provide additional representational power beyond the attention mechanism.");
        texts.add("pre-layernorm applies normalization before attention and feed-forward sublayers, providing more stable training dynamics than post-layernorm and enabling faster convergence in deep transformer architectures.");
        texts.add("the softmax function converts attention scores to probability distributions over tokens by exponentiating and normalizing, ensuring attention weights sum to one and can be interpreted as importance weights for weighted combination.");
        texts.add("masked self-attention prevents future tokens from influencing current token predictions by setting attention weights for future positions to negative infinity before softmax, essential for autoregressive language modeling.");
        texts.add("token embeddings map discrete token ids to continuous vector representations, learning dense embeddings where semantically similar tokens are close in the embedding space, forming the input representation for transformer models.");
        texts.add("the encoder-decoder architecture processes input sequences and generates output sequences, with the encoder producing contextualized representations and the decoder generating outputs using cross-attention to encoder states.");
        texts.add("attention heads learn different patterns and relationships in the input data, with some heads specializing in syntactic dependencies while others capture semantic relationships or positional patterns across different layers.");
        texts.add("the residual connection adds the input of a sublayer to its output, creating identity shortcuts that allow gradients to flow directly through the network and enabling training of very deep transformer architectures.");
        texts.add("transformer models can be parallelized across sequence positions during training unlike recurrent networks, enabling efficient GPU utilization and dramatically reducing training time for large language models.");
        texts.add("the key query and value projections transform embeddings for attention computation, with learned linear projections creating different views of the input that enable flexible information retrieval through dot-product attention.");
        texts.add("scaled dot-product attention divides attention scores by the square root of dimension to prevent the dot products from growing too large, ensuring stable gradients during training and more balanced attention distributions.");
        texts.add("the decoder uses cross-attention to attend to encoder outputs, allowing the generation process to incorporate relevant information from the source sequence while producing target tokens one at a time.");
        texts.add("attention masks prevent attending to padding tokens or future positions by adding large negative values before softmax, ensuring the model only attends to valid positions during attention computation.");
        texts.add("the transformer was introduced in the attention is all you need paper in 2017, revolutionizing natural language processing by replacing recurrence with self-attention and enabling the development of modern large language models.");
        texts.add("bert uses bidirectional attention while gpt uses unidirectional causal attention, with bert excelling at understanding tasks through masked language modeling and gpt excelling at generation through autoregressive pre-training.");
        return texts;
    }

    private static List<String> generateGPT3Texts() {
        List<String> texts = new ArrayList<>();
        texts.add("gpt3 introduced parallel attention and mlp computation for improved efficiency by computing attention and feed-forward layers simultaneously rather than sequentially, reducing training time while maintaining model quality.");
        texts.add("rotary position embedding encodes position information through rotation in vector space by applying rotation matrices to query and key vectors, providing better extrapolation to longer sequences than learned positional embeddings.");
        texts.add("sparse attention reduces computation by only attending to local and strided global positions, lowering the quadratic complexity of full attention to near-linear for long sequences while preserving the ability to capture long-range dependencies.");
        texts.add("gradient checkpointing trades computation for memory by recomputing activations during the backward pass rather than storing them, enabling training of larger models or longer sequences within memory constraints at the cost of additional computation.");
        texts.add("kv cache stores key and value tensors to avoid redundant computation during inference by caching previously computed key-value pairs, dramatically speeding up autoregressive generation by eliminating repeated attention computation for earlier tokens.");
        texts.add("few-shot learning allows gpt3 to perform tasks from a small number of examples in context by conditioning on demonstrations provided in the prompt, leveraging the model's pre-trained knowledge without updating parameters through gradient descent.");
        texts.add("in-context learning enables language models to adapt to tasks without gradient updates by recognizing patterns from examples in the prompt, demonstrating emergent capabilities that scale with model size and training data diversity.");
        texts.add("scaling laws show that model performance improves predictably with more compute and data following power-law relationships, enabling accurate prediction of larger model performance from smaller experiments and guiding efficient allocation of training resources.");
        texts.add("the gpt3 model has 175 billion parameters trained on 300 billion tokens of text data, demonstrating that scaling up model size and training data leads to emergent few-shot capabilities across diverse language tasks without task-specific fine-tuning.");
        texts.add("autoregressive generation produces text one token at a time using the previous tokens as context, sampling from the predicted probability distribution at each step while maintaining coherence through the model's learned understanding of language structure.");
        texts.add("the context window limits how many tokens a model can attend to at once, determining the maximum sequence length the model can process effectively and requiring techniques like chunking or summarization for longer documents.");
        texts.add("temperature controls randomness in text generation with higher values increasing diversity by flattening the probability distribution before sampling, while lower values make generation more deterministic and focused on high-probability tokens.");
        texts.add("top-k sampling restricts generation to the k most likely next tokens by zeroing out probabilities for tokens outside the top k, preventing selection of very unlikely tokens while maintaining controlled randomness in generation.");
        texts.add("nucleus sampling selects from the smallest set of tokens whose cumulative probability exceeds threshold p, dynamically adapting the candidate pool based on the shape of the probability distribution for more natural and coherent generation.");
        texts.add("beam search explores multiple generation paths to find the most likely sequence by maintaining the k most probable partial sequences at each step, often producing higher quality outputs than greedy decoding at the cost of additional computation.");
        texts.add("the vocabulary size determines how many unique tokens the model can represent, balancing between larger vocabularies that reduce sequence length and smaller vocabularies that provide better coverage of rare words through subword tokenization.");
        texts.add("tokenization algorithms like bpe split words into frequent subword units by iteratively merging the most common character pairs, enabling efficient handling of rare words and morphological variants while keeping vocabulary size manageable.");
        texts.add("the embedding dimension determines the size of token representation vectors, with larger dimensions providing greater representational capacity at the cost of increased computation and memory requirements throughout the network.");
        texts.add("larger models generally perform better but require more compute and memory for both training and inference, creating trade-offs between capability and efficiency that drive research into efficient architectures and compression techniques.");
        texts.add("zero-shot learning tests model ability to perform tasks without any examples by following natural language instructions, demonstrating the generalization capabilities that emerge from pre-training on diverse text corpora at scale.");
        return texts;
    }

    private static List<String> generateNLPTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("natural language processing enables computers to understand and generate human language by applying computational techniques to analyze, interpret, and produce text and speech, powering applications from search engines to virtual assistants.");
        texts.add("tokenization splits text into subword units using byte pair encoding or wordpiece algorithms, balancing vocabulary size with coverage of rare words and morphological variants for efficient neural network processing.");
        texts.add("word embeddings represent words as dense vectors in a continuous semantic space where similar words are close together, enabling neural networks to process text by mapping discrete vocabulary to learnable continuous representations.");
        texts.add("language models assign probabilities to sequences of tokens in natural language by learning distributions over sequences from large corpora, enabling text generation, completion, and evaluation through likelihood computation.");
        texts.add("text generation produces coherent and contextually relevant text from a given prompt by predicting subsequent tokens based on learned language patterns, powering applications from creative writing to code completion.");
        texts.add("question answering systems extract or generate answers to questions from context passages by identifying relevant spans or synthesizing information, used in search engines, virtual assistants, and knowledge management.");
        texts.add("text summarization condenses long documents into shorter informative summaries while preserving key information, using extractive methods to select important sentences or abstractive methods to generate novel summaries.");
        texts.add("named entity recognition identifies and classifies named entities in text into categories such as person, organization, and location, serving as a crucial component in information extraction and knowledge graph construction.");
        texts.add("sentiment analysis determines the emotional tone or opinion expressed in text by classifying text as positive, negative, or neutral, used extensively in social media monitoring, customer feedback analysis, and market research.");
        texts.add("machine translation converts text from one language to another automatically by learning mappings from source to target languages using parallel corpora, with neural models achieving human-level quality on many language pairs.");
        texts.add("part-of-speech tagging assigns grammatical categories to words in a sentence by labeling each word with its syntactic role such as noun, verb, adjective, or adverb, providing essential information for downstream parsing tasks.");
        texts.add("dependency parsing analyzes grammatical structure by identifying relationships between words and building trees that represent which words depend on which others, useful for understanding sentence structure and extracting semantic relationships.");
        texts.add("coreference resolution determines which expressions refer to the same entity by linking pronouns and noun phrases to their antecedents, essential for building coherent representations of documents and understanding discourse structure.");
        texts.add("text classification assigns predefined categories to documents or sentences by training models to predict labels from text content, used for spam detection, topic labeling, intent recognition, and sentiment analysis.");
        texts.add("information extraction identifies structured information from unstructured text by detecting entities, relationships, and events, enabling population of knowledge bases and databases from text sources like news articles and scientific papers.");
        texts.add("semantic similarity measures how close two pieces of text are in meaning using embedding-based methods or knowledge-based approaches, enabling tasks like duplicate detection, recommendation, and semantic search.");
        texts.add("language identification determines the language of a given text sample by analyzing character patterns and vocabulary, serving as a preprocessing step for multilingual systems and content categorization.");
        texts.add("keyword extraction identifies the most important terms in a document using statistical methods like TF-IDF or graph-based algorithms, enabling document summarization, indexing, and topic discovery.");
        texts.add("text segmentation divides documents into coherent sections or topics by detecting boundaries between different themes or discourse units, useful for document organization and information retrieval.");
        texts.add("speech recognition converts spoken language into written text by processing audio signals through acoustic and language models, enabling voice-based interfaces, transcription services, and accessibility tools.");
        return texts;
    }

    private static List<String> generateMathTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("matrix multiplication is the fundamental operation in neural network forward passes, computing weighted sums of inputs by multiplying weight matrices with activation vectors, enabling efficient GPU parallelization through optimized linear algebra routines.");
        texts.add("the cross entropy loss measures the difference between predicted and true distributions by computing the negative log-likelihood of the correct class, heavily penalizing confident wrong predictions while encouraging calibrated probability estimates.");
        texts.add("softmax converts logits to probability distributions by exponentiating and normalizing to sum to one, enabling multi-class classification by producing interpretable probabilities from the raw output scores of neural networks.");
        texts.add("the gradient of a function indicates the direction of steepest ascent at each point, pointing toward the direction of maximum increase and enabling optimization by following the negative gradient to minimize loss functions.");
        texts.add("the chain rule allows gradients to be propagated backward through composite functions by multiplying partial derivatives at each step, forming the mathematical foundation for backpropagation that enables training of deep neural networks.");
        texts.add("the dot product measures similarity between two vectors in vector space by computing the sum of element-wise products, used extensively in attention mechanisms to compute query-key similarity scores for weighting information.");
        texts.add("the relu activation function outputs zero for negative inputs and passes positive values unchanged, providing sparse activations and efficient gradient flow while avoiding the vanishing gradient problems of sigmoid and tanh.");
        texts.add("the sigmoid function maps inputs to values between zero and one through the logistic formula, historically used in neural networks and still valuable for binary classification output layers and gating mechanisms.");
        texts.add("the tanh activation function maps inputs to values between negative one and one, providing zero-centered outputs that can speed up convergence compared to sigmoid while still suffering from vanishing gradients for large inputs.");
        texts.add("normalization scales values to have zero mean and unit variance, stabilizing neural network training by ensuring consistent activation distributions across layers and enabling the use of higher learning rates.");
        return texts;
    }

    private static List<String> generateInstructionTexts() {
        List<String> instructions = new ArrayList<>();
        instructions.add("Instruction: What is deep learning? Response: Deep learning is a machine learning approach using neural networks with multiple layers to progressively learn hierarchical representations from raw data, enabling breakthroughs in vision, language, and speech.");
        instructions.add("Instruction: Explain the transformer architecture. Response: Transformers use self-attention mechanisms to process sequences in parallel without recurrence, capturing long-range dependencies through attention weights that allow each token to attend to all other tokens directly.");
        instructions.add("Instruction: What is GPT-3? Response: GPT-3 is a large language model with 175 billion parameters trained on 300 billion tokens, demonstrating that scale enables emergent few-shot capabilities across diverse language tasks through autoregressive pre-training.");
        instructions.add("Instruction: What is backpropagation? Response: Backpropagation computes gradients by applying the chain rule backward through the network from loss to input, enabling gradient descent optimization of all weights by propagating error signals through each layer.");
        instructions.add("Instruction: What is attention mechanism? Response: Attention computes weighted combinations of values based on query-key similarity scores computed through dot products, allowing models to dynamically focus on relevant parts of the input for each output.");
        instructions.add("Instruction: Explain gradient clipping. Response: Gradient clipping rescales gradients when their norm exceeds a threshold to prevent exploding gradients that would destabilize training, commonly used in recurrent networks and transformer training.");
        instructions.add("Instruction: What is KV cache? Response: KV cache stores key and value tensors from previous tokens during autoregressive generation, avoiding recomputation at each step and reducing generation complexity from quadratic to linear in sequence length.");
        instructions.add("Instruction: What is rotary position embedding? Response: RoPE encodes positional information by rotating query and key vectors according to their position, naturally capturing relative position information through rotation angles in the attention computation.");
        instructions.add("Instruction: Explain sparse attention. Response: Sparse attention reduces computational cost from quadratic to near-linear by attending only to local windows around each position plus strided global positions, preserving long-range modeling while improving efficiency.");
        instructions.add("Instruction: What is gradient checkpointing? Response: Gradient checkpointing saves memory during training by not storing intermediate activations, instead recomputing them during the backward pass, trading computation for reduced memory usage.");
        instructions.add("Instruction: What is batch normalization? Response: Batch normalization normalizes layer inputs across the batch dimension during training to zero mean and unit variance, reducing internal covariate shift and enabling higher learning rates for faster convergence.");
        instructions.add("Instruction: Explain dropout regularization. Response: Dropout randomly deactivates neurons with probability p during training, forcing the network to learn redundant representations and effectively training an ensemble of thinned networks that improves generalization.");
        instructions.add("Instruction: What is the learning rate? Response: The learning rate controls the step size during gradient descent optimization, with too large values causing divergence and too small values leading to slow convergence, making it the most critical hyperparameter.");
        instructions.add("Instruction: What is transfer learning? Response: Transfer learning fine-tunes pretrained models on new tasks by initializing from weights learned on large source datasets, dramatically reducing data requirements and enabling rapid adaptation to specialized domains.");
        instructions.add("Instruction: Explain multi-head attention. Response: Multi-head attention runs multiple attention operations in parallel with different learned projections, allowing the model to jointly attend to information from different representation subspaces and capture diverse relationships.");
        instructions.add("Instruction: What is causal language modeling? Response: Causal language modeling trains models to predict each token conditioned only on previous tokens in the sequence, enabling autoregressive text generation through next-token prediction during pre-training.");
        instructions.add("Instruction: What is tokenization? Response: Tokenization splits text into subword units using algorithms like BPE or WordPiece that iteratively merge frequent character sequences, balancing vocabulary size with coverage of rare words and morphological variants.");
        instructions.add("Instruction: Explain softmax function. Response: Softmax converts logits to probability distributions by exponentiating each value and normalizing to sum to one, enabling multi-class classification with interpretable probability outputs from neural networks.");
        instructions.add("Instruction: What is cross-entropy loss? Response: Cross-entropy measures the difference between predicted probability distributions and true labels as the negative log-likelihood of the correct class, heavily penalizing confident wrong predictions for classification tasks.");
        instructions.add("Instruction: What is the Adam optimizer? Response: Adam combines momentum-based acceleration with adaptive per-parameter learning rates by maintaining estimates of first and second gradient moments, providing efficient optimization across diverse architectures and tasks.");
        instructions.add("Instruction: Explain residual connections. Response: Residual connections add layer inputs to outputs through skip connections, creating identity shortcuts that allow gradients to flow directly through deep networks and enabling training of architectures with hundreds of layers.");
        instructions.add("Instruction: What is layer normalization? Response: Layer normalization normalizes across the feature dimension for each sample independently, providing stable training dynamics regardless of batch size and commonly used in transformer and recurrent architectures.");
        instructions.add("Instruction: What is few-shot learning? Response: Few-shot learning enables models to perform tasks from just a few examples provided in the prompt context, leveraging pre-trained knowledge without gradient updates through pattern recognition from demonstrations.");
        instructions.add("Instruction: Explain temperature in generation. Response: Temperature controls randomness in text generation by scaling logits before softmax, with higher values producing more diverse and creative outputs while lower values make generation more deterministic.");
        instructions.add("Instruction: What is top-k sampling? Response: Top-k sampling restricts generation to the k most likely next tokens by zeroing probabilities outside the top k, preventing selection of very unlikely tokens while maintaining controlled randomness in generation.");
        instructions.add("Instruction: What is nucleus sampling? Response: Nucleus sampling selects from the smallest token set whose cumulative probability exceeds threshold p, dynamically adapting the candidate pool based on distribution shape for natural and coherent generation.");
        instructions.add("Instruction: Explain beam search. Response: Beam search explores multiple generation paths simultaneously by maintaining the k most probable partial sequences at each step, often producing higher quality outputs than greedy decoding at increased computational cost.");
        instructions.add("Instruction: What is in-context learning? Response: In-context learning allows models to adapt to new tasks from examples in the prompt without updating model weights, demonstrating emergent capabilities that scale with model size and enable few-shot task performance.");
        instructions.add("Instruction: What is positional encoding? Response: Positional encoding adds position information to token embeddings through sinusoidal functions or learned embeddings, enabling transformers to distinguish token order since attention has no inherent notion of sequence position.");
        instructions.add("Instruction: Explain the feed-forward network. Response: The feed-forward network in transformers applies two linear transformations with non-linear activation between them independently to each position, providing additional representational capacity beyond the attention mechanism.");
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
