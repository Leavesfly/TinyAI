package io.leavesfly.tinyai.deepseek.v3.training.demo;

import java.io.*;
import java.util.*;

/**
 * DeepSeek-V3 训练数据集生成器
 *
 * 负责生成用于教学演示的训练数据集，包含：
 * - 预训练数据：MoE、DeepSeek、推理、编程、Transformer、深度学习等主题的长文本段落
 * - 后训练数据：采用 ChatML 对话模板的任务感知问答对（含代码专项数据集）
 * - RLHF 数据：带奖励标注（高/中/低三档）的 ChatML 对话数据
 *
 * 数据格式说明：
 * - 预训练：纯文本句子 → 全序列因果语言建模(CLM) loss
 * - 后训练：[TASK] &lt;|im_start|&gt;user\n{问题}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n{回答}&lt;|im_end|&gt;
 * - RLHF：[REWARD:0.9] &lt;|im_start|&gt;user\n{问题}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n{回答}&lt;|im_end|&gt;
 *
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3DatasetGenerator {

    public static final String DATA_DIR = "./data/deepseek_v3_training";

    // ========== 数据集准备入口 ==========

    /**
     * 准备全部训练数据集（预训练 + 后训练 + RLHF）
     */
    public static void prepareAllDatasets() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📦 步骤0: 准备训练数据集");
        System.out.println("=".repeat(80));

        File dataDir = new File(DATA_DIR);
        if (!dataDir.exists()) {
            dataDir.mkdirs();
            System.out.println("✓ 创建数据目录: " + DATA_DIR);
        }

        generatePretrainDataset();
        generatePosttrainDataset();
        generateRLHFDataset();

        System.out.println("\n✅ 数据集准备完成!");
    }

    // ========== 预训练数据生成 ==========

    /**
     * 生成预训练数据集（深度学习、MoE、Transformer 等领域的教学长文本）
     */
    static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");

        List<String> pretrainTexts = new ArrayList<>();
        pretrainTexts.addAll(generateMoETexts());
        pretrainTexts.addAll(generateDeepSeekTexts());
        pretrainTexts.addAll(generateReasoningTexts());
        pretrainTexts.addAll(generateCodingTexts());
        pretrainTexts.addAll(generateTransformerTexts());
        pretrainTexts.addAll(generateDeepLearningTexts());

        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);

        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    // ========== 后训练数据生成 ==========

    /**
     * 生成后训练数据集（通用任务感知 + 代码专项）
     */
    static void generatePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成后训练数据集...");

        List<String> trainTexts = new ArrayList<>();
        List<String> valTexts = new ArrayList<>();

        trainTexts.addAll(generateTaskAwareQA());

        // 验证集取训练集前15条
        for (int i = 0; i < 15 && i < trainTexts.size(); i++) {
            valTexts.add(trainTexts.get(i));
        }

        String trainPath = DATA_DIR + "/posttrain_train.txt";
        writeToFile(trainTexts, trainPath);
        System.out.println("  ✓ 后训练训练集: " + trainTexts.size() + " 条，路径: " + trainPath);

        String valPath = DATA_DIR + "/posttrain_val.txt";
        writeToFile(valTexts, valPath);
        System.out.println("  ✓ 后训练验证集: " + valTexts.size() + " 条，路径: " + valPath);

        generateCodePosttrainDataset();
    }

    /**
     * 生成代码生成专项后训练数据集（纯 CODING 任务，强化 MoE 专家特化）
     */
    static void generateCodePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成代码专项后训练数据集...");

        List<String> codeTrainTexts = new ArrayList<>();
        List<String> codeValTexts = new ArrayList<>();

        codeTrainTexts.addAll(generateCodeQA());

        for (int i = 0; i < 10 && i < codeTrainTexts.size(); i++) {
            codeValTexts.add(codeTrainTexts.get(i));
        }

        String codeTrainPath = DATA_DIR + "/code_posttrain_train.txt";
        writeToFile(codeTrainTexts, codeTrainPath);
        System.out.println("  ✓ 代码专项训练集: " + codeTrainTexts.size() + " 条，路径: " + codeTrainPath);

        String codeValPath = DATA_DIR + "/code_posttrain_val.txt";
        writeToFile(codeValTexts, codeValPath);
        System.out.println("  ✓ 代码专项验证集: " + codeValTexts.size() + " 条，路径: " + codeValPath);
    }

    // ========== RLHF 数据生成 ==========

    /**
     * 生成 RLHF 数据集（带奖励标注的 ChatML 对话，高/中/低三档）
     */
    static void generateRLHFDataset() throws IOException {
        System.out.println("\n📝 生成RLHF强化学习数据集...");

        List<String> rlhfTexts = new ArrayList<>();
        rlhfTexts.addAll(generateHighRewardRLHF());
        rlhfTexts.addAll(generateMediumRewardRLHF());
        rlhfTexts.addAll(generateLowRewardRLHF());

        String filePath = DATA_DIR + "/rlhf_train.txt";
        writeToFile(rlhfTexts, filePath);

        System.out.println("  ✓ RLHF训练数据: " + rlhfTexts.size() + " 条（高奖励20条/中奖励20条/低奖励10条）");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    /**
     * 生成高奖励 RLHF 样本 (0.8-1.0)
     */
    static List<String> generateHighRewardRLHF() {
        List<String> samples = new ArrayList<>();

        samples.add("[REWARD:0.95] <|im_start|>user\nWhat is Mixture of Experts and how does it work?<|im_end|>\n<|im_start|>assistant\nMixture of Experts (MoE) is a neural network architecture that uses multiple specialized expert networks. A gating network learns to route each input token to the most relevant experts. Only the selected experts are activated, enabling sparse computation. For example, DeepSeek V3 uses 8 experts with top-2 routing, activating only 25% of parameters per token, achieving high efficiency while maintaining performance.<|im_end|>");
        samples.add("[REWARD:0.92] <|im_start|>user\nExplain load balancing in MoE systems.<|im_end|>\n<|im_start|>assistant\nLoad balancing in MoE ensures all experts are utilized evenly during training. Without load balancing, the gating network might always route to the same experts, causing expert collapse. An auxiliary loss term penalizes uneven expert utilization, encouraging the model to distribute tokens across all experts. This improves model capacity and prevents some experts from becoming underutilized.<|im_end|>");
        samples.add("[REWARD:0.90] <|im_start|>user\nWhat are the advantages of sparse activation?<|im_end|>\n<|im_start|>assistant\nSparse activation in MoE models provides three key advantages: 1) Computational efficiency - only a fraction of parameters are active per forward pass, reducing FLOPs; 2) Scalability - models can have billions of parameters while maintaining fast inference; 3) Specialization - different experts can specialize in different types of content, improving overall model capacity and performance on diverse tasks.<|im_end|>");
        samples.add("[REWARD:0.93] <|im_start|>user\nWhat makes DeepSeek V3 special compared to other models?<|im_end|>\n<|im_start|>assistant\nDeepSeek V3 combines three key innovations: 1) Mixture of Experts architecture with 8 experts and top-2 routing for 25% parameter activation; 2) Task-aware routing that adapts expert selection based on input type (reasoning, coding, math, general); 3) Multi-head latent attention for efficient processing. This combination enables state-of-the-art performance on code generation benchmarks while being 4x faster than equivalent dense models.<|im_end|>");
        samples.add("[REWARD:0.91] <|im_start|>user\nHow does task-aware architecture improve performance?<|im_end|>\n<|im_start|>assistant\nTask-aware architecture in DeepSeek V3 improves performance by dynamically selecting specialized experts based on the input task type. For reasoning tasks, it activates experts trained on logical inference; for coding tasks, it activates programming-specialized experts; for math tasks, it activates mathematical reasoning experts. This task-specific expert selection allows the model to leverage specialized knowledge more effectively than a one-size-fits-all approach.<|im_end|>");
        samples.add("[REWARD:0.94] <|im_start|>user\nWrite a Python function to implement binary search.<|im_end|>\n<|im_start|>assistant\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\nThis implementation has O(log n) time complexity and O(1) space complexity. The array must be sorted for binary search to work correctly.<|im_end|>");
        samples.add("[REWARD:0.89] <|im_start|>user\nExplain the difference between list and tuple in Python.<|im_end|>\n<|im_start|>assistant\nIn Python, lists and tuples are both sequence types, but they have key differences: 1) Mutability - lists are mutable (can be modified), tuples are immutable (cannot be changed after creation); 2) Syntax - lists use square brackets [], tuples use parentheses (); 3) Performance - tuples are slightly faster and use less memory; 4) Use case - lists for collections that may change, tuples for fixed collections like coordinates or database records.<|im_end|>");
        samples.add("[REWARD:0.90] <|im_start|>user\nWhat is self-attention in Transformers?<|im_end|>\n<|im_start|>assistant\nSelf-attention is a mechanism that allows each position in a sequence to attend to all other positions, computing weighted representations. For each token, it computes query, key, and value vectors. The attention weight between positions i and j is the softmax of the dot product between query_i and key_j. This allows the model to capture long-range dependencies and contextual relationships regardless of distance in the sequence.<|im_end|>");
        samples.add("[REWARD:0.93] <|im_start|>user\nSolve: If f(x) = 2x + 3, what is f(f(2))?<|im_end|>\n<|im_start|>assistant\nTo solve f(f(2)), we first compute f(2): f(2) = 2*2 + 3 = 7. Then we compute f(7): f(7) = 2*7 + 3 = 17. Therefore, f(f(2)) = 17.<|im_end|>");
        samples.add("[REWARD:0.91] <|im_start|>user\nWhat is the derivative of sin(x²)?<|im_end|>\n<|im_start|>assistant\nUsing the chain rule: d/dx[sin(x²)] = cos(x²) * d/dx[x²] = cos(x²) * 2x = 2x*cos(x²).<|im_end|>");
        samples.add("[REWARD:0.88] <|im_start|>user\nExplain backpropagation in neural networks.<|im_end|>\n<|im_start|>assistant\nBackpropagation is an algorithm for training neural networks by computing gradients efficiently. It works by: 1) Forward pass - compute the output and loss; 2) Backward pass - compute the gradient of the loss with respect to each parameter using the chain rule; 3) Update parameters using gradient descent. The key insight is that gradients can be computed recursively from the output layer back to the input layer, reusing intermediate computations.<|im_end|>");
        samples.add("[REWARD:0.87] <|im_start|>user\nWhat is overfitting and how to prevent it?<|im_end|>\n<|im_start|>assistant\nOverfitting occurs when a model learns the training data too well, including noise and patterns that don't generalize. Prevention techniques include: 1) Regularization (L1/L2) to penalize complex models; 2) Dropout to randomly disable neurons during training; 3) Early stopping to halt training when validation performance degrades; 4) Data augmentation to increase training diversity; 5) Using more training data; 6) Reducing model complexity.<|im_end|>");
        samples.add("[REWARD:0.86] <|im_start|>user\nWhat is the difference between supervised and unsupervised learning?<|im_end|>\n<|im_start|>assistant\nSupervised learning uses labeled data where the model learns to map inputs to known outputs (e.g., classification, regression). Unsupervised learning uses unlabeled data to discover hidden patterns (e.g., clustering, dimensionality reduction). Supervised learning typically achieves higher accuracy on specific tasks but requires labeled data, while unsupervised learning can work with raw data but is more exploratory.<|im_end|>");
        samples.add("[REWARD:0.85] <|im_start|>user\nExplain the concept of gradient descent.<|im_end|>\n<|im_start|>assistant\nGradient descent is an optimization algorithm used to minimize a loss function. It works by: 1) Computing the gradient of the loss with respect to parameters; 2) Moving parameters in the opposite direction of the gradient (downhill) by a small step size (learning rate); 3) Repeating until convergence. The gradient points in the direction of steepest ascent, so moving opposite reduces the loss. Variants include SGD, Adam, and RMSprop for faster convergence.<|im_end|>");
        samples.add("[REWARD:0.84] <|im_start|>user\nWhat is a neural network?<|im_end|>\n<|im_start|>assistant\nA neural network is a computational model inspired by biological neurons. It consists of layers of interconnected nodes (neurons). Each connection has a weight that is learned during training. Input data flows through the network, with each layer transforming the representation. Deep learning uses networks with many layers (deep networks) to learn hierarchical feature representations automatically from data.<|im_end|>");
        samples.add("[REWARD:0.83] <|im_start|>user\nExplain the bias-variance tradeoff.<|im_end|>\n<|im_start|>assistant\nThe bias-variance tradeoff describes the balance between underfitting (high bias) and overfitting (high variance). High bias models are too simple and underfit the training data. High variance models are too complex and overfit noise. The goal is to find the sweet spot with low bias and low variance. Techniques like regularization, cross-validation, and ensemble methods help manage this tradeoff.<|im_end|>");
        samples.add("[REWARD:0.82] <|im_start|>user\nWhat is cross-validation?<|im_end|>\n<|im_start|>assistant\nCross-validation is a technique for evaluating model performance by partitioning data into multiple subsets. The most common is k-fold cross-validation: data is split into k folds, the model is trained on k-1 folds and tested on the remaining fold, repeated k times. This provides a more robust estimate of model performance than a single train-test split, especially with limited data.<|im_end|>");
        samples.add("[REWARD:0.81] <|im_start|>user\nExplain the concept of embeddings.<|im_end|>\n<|im_start|>assistant\nEmbeddings are dense vector representations of discrete items (words, users, products) that capture semantic relationships. Unlike one-hot encoding (sparse and high-dimensional), embeddings are learned continuous vectors where similar items are close in vector space. For example, word embeddings capture that 'king' - 'man' + 'woman' ≈ 'queen'. They enable neural networks to process categorical data efficiently and discover meaningful patterns.<|im_end|>");

        return samples;
    }

    /**
     * 生成中奖励 RLHF 样本 (0.4-0.7)
     */
    static List<String> generateMediumRewardRLHF() {
        List<String> samples = new ArrayList<>();

        samples.add("[REWARD:0.65] <|im_start|>user\nWhat is MoE?<|im_end|>\n<|im_start|>assistant\nMoE stands for Mixture of Experts. It's a type of neural network that uses multiple expert networks and routes inputs to them. Only some experts are used for each input, which makes it efficient.<|im_end|>");
        samples.add("[REWARD:0.60] <|im_start|>user\nHow many experts does DeepSeek V3 use?<|im_end|>\n<|im_start|>assistant\nDeepSeek V3 uses 8 expert networks in its MoE architecture.<|im_end|>");
        samples.add("[REWARD:0.55] <|im_start|>user\nWhat is load balancing?<|im_end|>\n<|im_start|>assistant\nLoad balancing in MoE is about making sure all experts are used evenly during training so none are idle or overused.<|im_end|>");
        samples.add("[REWARD:0.63] <|im_start|>user\nWhat is DeepSeek V3?<|im_end|>\n<|im_start|>assistant\nDeepSeek V3 is a language model that uses MoE architecture and task-aware routing to be efficient and perform well on various tasks like coding and reasoning.<|im_end|>");
        samples.add("[REWARD:0.58] <|im_start|>user\nWhat programming languages does DeepSeek support?<|im_end|>\n<|im_start|>assistant\nDeepSeek supports multiple programming languages including Python, Java, C++, JavaScript, and others.<|im_end|>");
        samples.add("[REWARD:0.67] <|im_start|>user\nWrite a function to reverse a string in Python.<|im_end|>\n<|im_start|>assistant\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\nThis uses Python slicing to reverse the string.<|im_end|>");
        samples.add("[REWARD:0.62] <|im_start|>user\nWhat is a list in Python?<|im_end|>\n<|im_start|>assistant\nA list in Python is a mutable sequence that can hold multiple items. You can create one with square brackets like [1, 2, 3].<|im_end|>");
        samples.add("[REWARD:0.57] <|im_start|>user\nHow do you define a function in Python?<|im_end|>\n<|im_start|>assistant\nYou use the def keyword followed by the function name and parameters. For example: def my_function(): pass<|im_end|>");
        samples.add("[REWARD:0.64] <|im_start|>user\nWhat is attention in Transformers?<|im_end|>\n<|im_start|>assistant\nAttention is a mechanism that lets the model focus on relevant parts of the input. Self-attention lets each position look at all other positions.<|im_end|>");
        samples.add("[REWARD:0.59] <|im_start|>user\nWhat is multi-head attention?<|im_end|>\n<|im_start|>assistant\nMulti-head attention runs multiple attention mechanisms in parallel, each learning different aspects of the relationships between tokens.<|im_end|>");
        samples.add("[REWARD:0.66] <|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n2 + 2 = 4<|im_end|>");
        samples.add("[REWARD:0.61] <|im_start|>user\nWhat is the square root of 16?<|im_end|>\n<|im_start|>assistant\nThe square root of 16 is 4.<|im_end|>");
        samples.add("[REWARD:0.56] <|im_start|>user\nWhat is a derivative?<|im_end|>\n<|im_start|>assistant\nA derivative measures how a function changes as its input changes. It's the slope of the function at a point.<|im_end|>");
        samples.add("[REWARD:0.54] <|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\nMachine learning is a field where computers learn from data to make predictions or decisions without being explicitly programmed.<|im_end|>");
        samples.add("[REWARD:0.53] <|im_start|>user\nWhat is a neural network layer?<|im_end|>\n<|im_start|>assistant\nA neural network layer is a set of neurons that processes inputs and produces outputs. Each layer transforms the data in some way.<|im_end|>");
        samples.add("[REWARD:0.52] <|im_start|>user\nWhat is activation function?<|im_end|>\n<|im_start|>assistant\nAn activation function introduces non-linearity into a neural network. Common ones include ReLU, sigmoid, and tanh.<|im_end|>");
        samples.add("[REWARD:0.51] <|im_start|>user\nWhat is regularization?<|im_end|>\n<|im_start|>assistant\nRegularization is a technique to prevent overfitting by adding a penalty term to the loss function. L1 and L2 are common types.<|im_end|>");
        samples.add("[REWARD:0.50] <|im_start|>user\nWhat is batch normalization?<|im_end|>\n<|im_start|>assistant\nBatch normalization normalizes the inputs of each layer to improve training stability and speed.<|im_end|>");
        samples.add("[REWARD:0.49] <|im_start|>user\nWhat is dropout?<|im_end|>\n<|im_start|>assistant\nDropout randomly disables some neurons during training to prevent overfitting and improve generalization.<|im_end|>");
        samples.add("[REWARD:0.47] <|im_start|>user\nWhat is learning rate?<|im_end|>\n<|im_start|>assistant\nLearning rate controls how much the model parameters are updated during training. Too high can cause divergence, too slow can cause slow convergence.<|im_end|>");

        return samples;
    }

    /**
     * 生成低奖励 RLHF 样本 (0.1-0.3)
     */
    static List<String> generateLowRewardRLHF() {
        List<String> samples = new ArrayList<>();

        samples.add("[REWARD:0.25] <|im_start|>user\nWhat is Mixture of Experts?<|im_end|>\n<|im_start|>assistant\nIt's a type of neural network thing.<|im_end|>");
        samples.add("[REWARD:0.20] <|im_start|>user\nHow does MoE routing work?<|im_end|>\n<|im_start|>assistant\nI don't know.<|im_end|>");
        samples.add("[REWARD:0.22] <|im_start|>user\nWhat is DeepSeek V3?<|im_end|>\n<|im_start|>assistant\nIt's a model.<|im_end|>");
        samples.add("[REWARD:0.18] <|im_start|>user\nWrite a Python function to sort a list.<|im_end|>\n<|im_start|>assistant\n```python\ndef sort_list(lst):\n    # This is wrong\n    return lst\n```<|im_end|>");
        samples.add("[REWARD:0.15] <|im_start|>user\nWhat is self-attention?<|im_end|>\n<|im_start|>assistant\nIt's attention on itself.<|im_end|>");
        samples.add("[REWARD:0.12] <|im_start|>user\nWhat is backpropagation?<|im_end|>\n<|im_start|>assistant\nBackpropagation is when you go backwards.<|im_end|>");
        samples.add("[REWARD:0.10] <|im_start|>user\nExplain overfitting.<|im_end|>\n<|im_start|>assistant\nOverfitting is bad.<|im_end|>");
        samples.add("[REWARD:0.28] <|im_start|>user\nWhat is the derivative of x²?<|im_end|>\n<|im_start|>assistant\n2x is the derivative.<|im_end|>");
        samples.add("[REWARD:0.23] <|im_start|>user\nWhat is a neural network?<|im_end|>\n<|im_start|>assistant\nIt's like a brain but with math.<|im_end|>");
        samples.add("[REWARD:0.30] <|im_start|>user\nHow do you prevent overfitting?<|im_end|>\n<|im_start|>assistant\nUse more data maybe.<|im_end|>");

        return samples;
    }

    // ========== 预训练文本素材生成 ==========

    /** 生成 MoE 相关长文本段落（适合预训练） */
    static List<String> generateMoETexts() {
        return Arrays.asList(
            "Mixture of Experts, commonly abbreviated as MoE, is a neural network architecture that uses multiple expert networks to process different parts of the input data. Each expert network specializes in handling specific types of patterns or features, allowing the overall model to achieve better performance on diverse tasks. The key innovation of MoE is the sparse activation mechanism, where only a subset of experts is activated for each input, dramatically reducing computational cost while maintaining model capacity.",
            "MoE models route inputs to different expert networks based on learned gating mechanisms that act as intelligent routers. The gating network analyzes each input token and computes routing scores for all available experts. These scores determine which experts are most relevant for processing the current input. This dynamic routing enables the model to leverage specialized knowledge from different experts, similar to how a team of specialists collaborates to solve complex problems in various domains.",
            "The gating network in MoE architecture plays a crucial role in determining which experts should process each input token. It typically uses a softmax function to produce a probability distribution over all experts, and then selects the top K experts with the highest probabilities. This selection process must balance between utilizing the most relevant experts and maintaining diversity in expert usage to prevent certain experts from being overused while others remain underutilized during the training process.",
            "MoE achieves sparse activation by only using a subset of experts for each input, which is one of its most significant advantages over traditional dense neural networks. In a dense network, every parameter participates in processing every input, leading to high computational costs. In contrast, MoE models can have hundreds of billions of parameters while only activating a small fraction for each forward pass, making it possible to train and deploy extremely large models with reasonable computational resources.",
            "Top K routing is a popular strategy in MoE architectures that selects the K most relevant experts for each input token based on routing scores. Common choices include Top 1 routing, where only the single best expert is used, and Top 2 routing, where two experts contribute to the output. The choice of K involves a tradeoff between computational efficiency and model performance, with larger K values providing more capacity at the cost of increased computation.",
            "Load balancing in MoE is essential for ensuring that all experts are utilized evenly during training, preventing the collapse problem where the model relies heavily on only a few experts. Auxiliary load balancing losses are added to the training objective to encourage uniform expert utilization. These losses typically include terms that penalize imbalanced expert selection frequencies and encourage each expert to receive a roughly equal number of tokens throughout the training process.",
            "MoE models can scale to billions of parameters while maintaining efficient inference due to their sparse activation property. Models like DeepSeek V3, Mixtral, and Switch Transformer demonstrate that it is possible to achieve state of the art performance with massive parameter counts while keeping inference costs manageable. This scalability makes MoE an attractive architecture for building large language models that can be deployed in production environments with limited computational resources.",
            "DeepSeek V3 uses eight expert networks with top two routing strategy, where each token is processed by exactly two experts selected based on routing scores. This architecture enables the model to achieve 25 percent parameter activation during inference, meaning only a quarter of the total parameters are used for any given input. The combination of expert specialization and efficient routing allows DeepSeek V3 to deliver strong performance across diverse tasks including reasoning, coding, and mathematical problem solving.",
            "Expert networks in MoE architectures specialize in different aspects of the input distribution, learning to handle specific types of patterns, topics, or tasks. Through training, some experts may become specialized in processing mathematical expressions, while others excel at natural language understanding or code generation. This emergent specialization occurs naturally without explicit supervision, as the routing mechanism learns to direct different inputs to the most suitable experts.",
            "MoE reduces computational cost by activating only a fraction of total parameters during both training and inference. In traditional dense models, computational cost scales linearly with model size, making very large models impractical. MoE breaks this linear relationship by allowing the model to have many parameters while only using a small, fixed percentage for each input. This enables researchers to build models with hundreds of billions of parameters that can run on commodity hardware.",
            "Auxiliary loss in MoE encourages balanced expert utilization by adding penalty terms to the training objective. Common approaches include the load balancing loss, which penalizes uneven token distribution across experts, and the router z loss, which stabilizes training by penalizing large router logit values. These auxiliary losses work together with the primary language modeling loss to ensure that all experts learn meaningful representations and contribute to the model predictions.",
            "Sparse MoE models achieve better performance than dense models with similar computational budgets by efficiently allocating model capacity. Research has shown that MoE models can match or exceed the performance of dense models that use significantly more FLOPs per token. This efficiency advantage makes MoE particularly valuable in scenarios where computational resources are limited but high model capacity is required for achieving strong task performance.",
            "Expert capacity limits the number of tokens each expert can process per batch in MoE architectures. When an expert receives more tokens than its capacity allows, the excess tokens are either routed to alternative experts or processed by identity mappings. Capacity factors are typically set between one and two to balance between computational efficiency and ensuring that most tokens can be processed by their preferred experts without overflow.",
            "MoE routing can be learned jointly with the model during training through end to end backpropagation. The router parameters are updated along with the expert parameters to optimize the overall model performance. Differentiable routing strategies have been developed to allow gradient flow through the routing decisions, enabling the model to learn optimal routing policies that maximize task performance while maintaining computational efficiency.",
            "Switch Transformer uses one expert per token for extreme sparsity, representing a simplified variant of MoE architecture. By selecting only the single best expert for each token, Switch Transformer achieves maximum sparsity while still benefiting from conditional computation. This design simplifies the routing mechanism and reduces communication overhead in distributed training, making it easier to scale MoE models to extremely large sizes.",
            "MoE enables training of very large models on limited hardware resources by distributing the computational load across multiple experts. The sparse activation pattern means that each GPU only needs to hold and compute a subset of the model parameters at any time. This property has enabled researchers to train models with over one trillion parameters using distributed computing infrastructure that would be insufficient for training equivalent dense models.",
            "Expert parallelism allows MoE layers to scale across multiple devices by assigning different experts to different GPUs or TPUs. When processing a batch of tokens, each device computes the output for its assigned experts and communicates the results to other devices as needed. Expert parallelism is often combined with data parallelism and tensor parallelism to achieve efficient training of very large MoE models across hundreds or thousands of devices.",
            "Dynamic routing in MoE adapts to different input patterns automatically, allowing the model to leverage specialized expertise based on the content being processed. Unlike static routing rules, learned routing can discover complex patterns in the data and make nuanced decisions about which experts to engage. This adaptability enables MoE models to perform well across diverse tasks without requiring task specific architectural modifications or manual routing configurations.",
            "MoE models show strong few shot learning capabilities, demonstrating the ability to generalize to new tasks with minimal examples. The expert specialization within MoE architectures may contribute to this capability by providing diverse knowledge representations that can be combined flexibly. Research has shown that large MoE models can achieve competitive few shot performance on benchmarks while using significantly less computational resources than equivalent dense models during inference.",
            "Load balancing loss prevents expert collapse where some experts are never used during training. Without proper regularization, MoE models may converge to solutions where only a subset of experts handles all inputs while others remain dormant. Load balancing losses add a penalty proportional to the variance in expert utilization, encouraging the router to distribute tokens more evenly and ensuring that all experts participate in learning useful representations."
        );
    }

    /** 生成 DeepSeek 相关长文本段落（适合预训练） */
    static List<String> generateDeepSeekTexts() {
        return Arrays.asList(
            "DeepSeek is a series of advanced language models developed with innovative architectures that push the boundaries of what is possible in natural language processing. These models incorporate cutting edge techniques including Mixture of Experts, multi head latent attention, and task aware routing mechanisms. The DeepSeek family includes models optimized for various applications including general purpose language understanding, code generation, mathematical reasoning, and multimodal processing.",
            "DeepSeek V3 combines Mixture of Experts architecture with task aware routing for improved performance across diverse applications. The model uses eight expert networks that specialize in different domains such as reasoning, coding, mathematics, and general knowledge. The task aware routing mechanism automatically classifies input tokens and routes them to the most appropriate experts, enabling efficient utilization of model capacity while maintaining strong performance on specialized tasks.",
            "Task aware architecture in DeepSeek adapts to different downstream tasks through an intelligent routing system that recognizes the nature of the input and activates specialized experts accordingly. This approach differs from traditional MoE models that use purely learned routing without explicit task awareness. By incorporating task classification into the routing process, DeepSeek can leverage domain specific expertise more effectively, leading to improved performance on tasks that require specialized knowledge.",
            "DeepSeek models support reasoning, coding, mathematics, and multimodal tasks through a unified architecture that combines specialized expert networks with shared foundational components. The reasoning capabilities enable complex logical inference and chain of thought problem solving. The coding abilities cover multiple programming languages with high accuracy on code generation, completion, and debugging tasks. Mathematical capabilities include arithmetic, algebra, calculus, and proof verification.",
            "DeepSeek V3 achieves state of the art results on code generation benchmarks including HumanEval, MBPP, and CodeContests. The model demonstrates strong performance across multiple programming languages including Python, JavaScript, Java, C plus plus, Go, Rust, TypeScript, PHP, Ruby, and Swift. Code generation quality is evaluated on multiple dimensions including functional correctness, code readability, algorithmic efficiency, and adherence to language specific style conventions.",
            "The reasoning module in DeepSeek enhances logical inference capabilities through specialized attention mechanisms and training on reasoning intensive datasets. This module enables the model to perform multi step reasoning, follow logical chains, and verify the correctness of intermediate conclusions. The architecture incorporates techniques from chain of thought prompting and self consistency to improve the reliability of reasoning outputs.",
            "DeepSeek supports ten programming languages including Python, Java, C plus plus, JavaScript, Go, Rust, TypeScript, Ruby, PHP, and Swift. For each language, the model has been trained on large code corpora and can perform tasks including code generation, code completion, bug fixing, code explanation, and test case generation. The language coverage enables DeepSeek to serve as a versatile programming assistant for developers working in diverse technology stacks.",
            "Code quality in DeepSeek is evaluated on correctness, readability, efficiency, and style, reflecting the multidimensional nature of high quality software development. Correctness ensures the generated code fulfills the specified requirements without bugs. Readability measures how easily other developers can understand the code. Efficiency considers algorithmic complexity and resource usage. Style adherence ensures the code follows language specific conventions and best practices.",
            "DeepSeek V3 uses multi head latent attention for efficient processing of long sequences while maintaining high quality attention computations. This attention variant reduces the memory and computational requirements compared to standard multi head attention by projecting queries, keys, and values into a latent space before computing attention weights. The latent attention mechanism enables the model to handle longer context windows without the quadratic scaling of standard attention.",
            "Task type classification helps DeepSeek route inputs to specialized experts by first analyzing the input to determine its domain, whether it is reasoning, coding, mathematics, or general knowledge. This classification happens implicitly through the routing network without requiring explicit task labels. The routing scores indicate which experts are most relevant for the current input, enabling efficient allocation of computational resources.",
            "DeepSeek models can handle sequences up to thousands of tokens in length through efficient attention mechanisms and memory optimization techniques. The context window size determines how much previous context the model can consider when generating predictions. Large context windows are particularly important for tasks involving long documents, extended conversations, or complex code files where understanding the full context is essential for accurate outputs.",
            "The architecture of DeepSeek V3 enables twenty five percent parameter activation during inference through the Top 2 routing strategy with eight experts. This means that for any given input token, only two out of eight experts are activated, resulting in efficient computation while maintaining the benefits of a large model. The total parameter count can be very large, but the active parameters for any forward pass remain manageable.",
            "DeepSeek V3 training uses load balanced MoE loss for expert utilization, combining the primary language modeling objective with auxiliary losses that encourage even distribution of tokens across experts. The load balancing loss penalizes situations where some experts receive significantly more tokens than others. Additional auxiliary losses may include router entropy regularization and expert diversity penalties to ensure robust expert specialization.",
            "Inference in DeepSeek V3 is four times faster than equivalent dense models with similar total parameter counts, demonstrating the computational efficiency benefits of sparse MoE architecture. This speedup comes from activating only a fraction of parameters for each token while maintaining competitive model quality. The efficiency gains are most pronounced for longer sequences and larger batch sizes where the savings in memory and computation accumulate.",
            "DeepSeek supports both causal language modeling and instruction tuning through a flexible training pipeline. Causal language modeling involves predicting the next token given previous tokens, providing foundational language understanding capabilities. Instruction tuning trains the model to follow user instructions and engage in conversational interactions. The combination enables DeepSeek to serve as both a completion engine and an interactive assistant.",
            "DeepSeek models show strong performance on mathematical problem solving including arithmetic operations, algebraic manipulations, calculus problems, and proof verification. The mathematical capabilities are enhanced through training on mathematical corpora and specialized expert networks that focus on quantitative reasoning. The models can interpret mathematical notation, apply relevant formulas, and show step by step solutions to complex problems.",
            "Multi task learning in DeepSeek improves generalization across domains by training on diverse datasets that span reasoning, coding, mathematics, and general language tasks. This joint training enables knowledge transfer between domains, where skills learned in one area can benefit performance in related areas. The MoE architecture facilitates multi task learning by allowing different experts to specialize in different domains while sharing common foundational knowledge.",
            "DeepSeek V3 pre training uses large scale diverse text corpora covering web documents, books, academic papers, code repositories, and structured knowledge bases. The pre training data is carefully filtered and balanced to ensure high quality learning across different domains and languages. The scale of pre training data correlates strongly with model capabilities, and DeepSeek V3 benefits from training on trillions of tokens from diverse sources.",
            "Post training in DeepSeek fine tunes the model for specific applications through supervised fine tuning and reinforcement learning from human feedback. Supervised fine tuning uses curated instruction response pairs to teach the model how to follow user instructions and engage in helpful conversations. Reinforcement learning further aligns the model with human preferences for safety, helpfulness, and honesty through reward model optimization.",
            "DeepSeek achieves competitive results while using fewer active parameters than equivalent dense models, demonstrating the efficiency benefits of sparse MoE architecture. The total parameter count can be very large, enabling high model capacity for learning complex patterns. However, during inference, only a fraction of these parameters are used for each prediction, reducing computational cost while maintaining output quality."
        );
    }

    /** 生成推理相关长文本段落（适合预训练） */
    static List<String> generateReasoningTexts() {
        return Arrays.asList(
            "Reasoning in AI involves drawing logical conclusions from available information through systematic processes that mirror human thought patterns. Modern language models develop reasoning capabilities through exposure to large amounts of text containing logical arguments, problem solutions, and explanatory content. These models learn to recognize patterns in reasoning, such as cause and effect relationships, logical implications, and the structure of valid arguments, enabling them to perform increasingly sophisticated reasoning tasks.",
            "Chain of thought prompting improves reasoning by showing intermediate steps between the problem statement and the final answer. This technique encourages the model to break down complex problems into simpler subproblems and solve them sequentially. Research has demonstrated that chain of thought prompting significantly improves performance on mathematical word problems, logical puzzles, and multi step reasoning tasks. The approach mimics how humans solve problems by thinking through each step explicitly before arriving at a conclusion.",
            "Task aware models adapt their reasoning strategy based on problem type by first classifying the nature of the problem and then selecting appropriate reasoning methods. For mathematical problems, the model may rely more heavily on arithmetic and algebraic manipulation. For logical problems, it may apply deductive reasoning principles. For commonsense questions, it may draw on world knowledge and typical human behavior patterns. This adaptive approach enables more efficient and accurate reasoning across diverse problem types.",
            "Mathematical reasoning requires understanding of numerical relationships, arithmetic operations, algebraic manipulations, and mathematical proof techniques. Language models develop mathematical reasoning capabilities through training on mathematical texts, problem solutions, and formal proofs. Advanced models can solve equations, perform multi step calculations, verify mathematical statements, and even generate proofs for simple theorems. The key challenge is ensuring that mathematical reasoning is exact and consistent, as language models are inherently probabilistic.",
            "Logical inference applies rules to derive new facts from existing knowledge through valid reasoning patterns such as modus ponens, modus tollens, and hypothetical syllogism. Language models learn these inference patterns from training data containing logical arguments and their conclusions. The ability to perform logical inference is fundamental to many reasoning tasks, including question answering, fact checking, and automated theorem proving. Models must distinguish valid inferences from fallacious reasoning patterns that may appear in training data.",
            "Multi step reasoning breaks complex problems into manageable sub problems that can be solved independently before combining their solutions. This decomposition strategy is essential for tackling challenging tasks like mathematical proofs, complex decision making, and long horizon planning. Language models develop multi step reasoning abilities through exposure to worked examples that demonstrate how to decompose and solve complex problems step by step.",
            "Commonsense reasoning requires understanding of everyday knowledge and context that humans take for granted but machines must learn from data. This includes knowledge about physical objects and their properties, human behaviors and motivations, social conventions, and cause and effect relationships in daily life. Commonsense reasoning benchmarks like CommonsenseQA and PIQA test whether models have acquired this foundational knowledge and can apply it to answer questions about everyday situations.",
            "Analogical reasoning transfers knowledge from familiar to novel situations by identifying structural similarities between domains. This capability enables learning from limited examples and generalizing to new contexts. Language models develop analogical reasoning through exposure to texts containing analogies, metaphors, and comparative analyses. Strong analogical reasoning is associated with creative problem solving and the ability to recognize deep patterns that transcend surface differences between problems.",
            "Causal reasoning identifies cause and effect relationships between events, going beyond mere correlation to understand the mechanisms that connect causes to their effects. This type of reasoning is crucial for answering why questions, making predictions about interventions, and planning actions that achieve desired outcomes. Language models learn causal patterns from text, though distinguishing genuine causation from spurious correlation remains challenging.",
            "Deductive reasoning applies general principles to specific cases to derive logically certain conclusions. If all humans are mortal and Socrates is human, then Socrates must be mortal. This form of reasoning guarantees truth preserving inference when the premises are true. Language models develop deductive reasoning abilities through training on texts containing logical arguments and by learning to follow explicit reasoning instructions that specify the deductive steps to take.",
            "Inductive reasoning generalizes from specific observations to broad patterns or rules, producing conclusions that are probable rather than certain. After observing many white swans, one might inductively conclude that all swans are white, though this conclusion can be falsified by counterexamples. Language models use inductive reasoning when they learn patterns from training data and apply them to new inputs, a fundamental capability underlying their generalization abilities.",
            "Self correction in reasoning improves accuracy through iterative refinement, where the model identifies and fixes errors in its initial reasoning attempts. This process can involve re reading the problem statement, checking intermediate steps for consistency, and verifying that the final answer addresses the original question. Self correction is particularly valuable for complex reasoning tasks where initial attempts often contain subtle errors.",
            "Reasoning traces provide interpretability by showing the thought process that leads from problem to solution. These traces help users understand how the model arrived at its conclusion, making it easier to verify correctness and identify potential errors. Reasoning traces are especially important in high stakes applications where users need to trust and validate model outputs before acting on them.",
            "Meta reasoning involves thinking about thinking and strategy selection, enabling the model to choose the most appropriate reasoning approach for a given problem. This includes recognizing when to use chain of thought versus direct answering, when to break down problems versus solve them holistically, and when to seek additional information versus working with available knowledge. Meta reasoning capabilities improve the efficiency and effectiveness of reasoning processes.",
            "Spatial reasoning involves understanding geometric relationships and transformations, including concepts like distance, direction, shape, and spatial configurations. This capability is important for tasks involving physical reasoning, navigation, and visual imagination. Language models develop spatial reasoning through training on texts describing spatial relationships and through instruction tuning that explicitly teaches spatial concepts and operations.",
            "Temporal reasoning tracks changes and sequences over time, understanding concepts like duration, ordering, causation across time, and temporal constraints. This capability is essential for understanding narratives, planning actions, and reasoning about processes that unfold over time. Language models learn temporal reasoning from texts that describe events in temporal order, use temporal language like before and after, and discuss changes in state over time.",
            "Reasoning modules can be trained to verify their own conclusions through self consistency checks and verification procedures. A model might generate multiple reasoning paths to the same problem and check whether they converge to the same answer. Alternatively, it might generate a candidate solution and then critically evaluate it for errors or inconsistencies. This self verification capability improves the reliability of reasoning outputs.",
            "Abductive reasoning infers the most likely explanation for observations, producing hypotheses that best account for available evidence. This form of reasoning is essential for diagnosis, scientific discovery, and everyday understanding of events. When you see wet ground and infer that it probably rained, you are using abductive reasoning. Language models develop this capability through training on explanatory texts and by learning patterns of what kinds of explanations are appropriate for different observations.",
            "Abstract reasoning manipulates concepts without concrete examples, operating at a level of generality that transcends specific instances. This capability enables solving problems expressed in symbolic form, understanding mathematical abstractions, and engaging in philosophical reasoning. Abstract reasoning is tested through tasks like Raven Progressive Matrices and pattern completion problems that require identifying rules governing abstract visual or symbolic sequences.",
            "Reasoning confidence indicates the model certainty in its conclusions and can be estimated through techniques like ensemble methods, Bayesian approaches, or calibration training. High confidence in incorrect answers is a common failure mode that reasoning systems must address. Confidence calibration helps users understand when they can trust model outputs and when additional verification is needed, which is crucial for deploying reasoning systems in high stakes applications."
        );
    }

    /** 生成编程相关长文本段落（适合预训练） */
    static List<String> generateCodingTexts() {
        return Arrays.asList(
            "Programming languages provide formal systems for instructing computers to perform specific tasks through precisely defined syntax and semantics. Each programming language offers a unique set of abstractions, type systems, and paradigms that shape how developers think about and solve problems. The choice of programming language can significantly impact development speed, code maintainability, runtime performance, and the range of problems that can be effectively addressed.",
            "Python is widely used for machine learning and data science applications due to its clear syntax, extensive library ecosystem, and strong community support. Popular frameworks like TensorFlow, PyTorch, and scikit learn make Python the language of choice for deep learning research and production deployments. The language emphasis on readability and simplicity enables rapid prototyping and experimentation, while libraries like NumPy and Pandas provide efficient tools for numerical computing and data manipulation.",
            "Java is a statically typed object oriented programming language designed for portability and reliability across diverse computing environments. The write once run anywhere philosophy enables Java applications to run on any platform with a Java Virtual Machine. Java strong type system, garbage collection, and exception handling make it suitable for large scale enterprise applications where reliability and maintainability are critical. The extensive standard library and mature ecosystem support a wide range of application domains.",
            "JavaScript enables interactive web applications in browsers by manipulating the Document Object Model and responding to user events. Originally a simple scripting language for web pages, JavaScript has evolved into a full featured programming language used for both client side and server side development. Modern frameworks like React, Vue, and Angular enable building sophisticated single page applications, while Node.js extends JavaScript to server side programming and command line tools.",
            "C plus plus offers low level control with high level abstractions, making it suitable for systems programming, game development, and performance critical applications. The language provides direct memory manipulation through pointers while also supporting object oriented and generic programming paradigms. C plus plus is widely used in operating systems, embedded systems, game engines, and applications where fine grained control over resources and maximum performance are essential.",
            "Code generation models translate natural language to executable programs by learning patterns from large code corpora and understanding the mapping between descriptions and implementations. These models can generate code from problem descriptions, complete partial code snippets, and suggest implementations based on function signatures and comments. Advanced code generation capabilities include handling multiple programming languages, following coding conventions, and producing code that passes test cases.",
            "Syntax correctness ensures code follows language grammar rules, preventing errors that would prevent compilation or interpretation. Syntax errors are typically caught early in the development process by compilers, interpreters, or integrated development environments. Modern development tools provide real time syntax checking, automatic formatting, and suggestions for fixing syntax errors, significantly reducing the time developers spend on syntax related debugging.",
            "Code readability makes programs easier to understand and maintain, reducing the cognitive load required for developers to comprehend existing codebases. Readable code uses meaningful variable and function names, consistent formatting, appropriate comments, and clear control flow. The importance of readability increases with codebase size and team size, as code that is difficult to understand leads to bugs, slow development, and difficulty onboarding new team members.",
            "Algorithm efficiency measures computational complexity and resource usage, typically expressed using big O notation to describe how runtime or memory requirements scale with input size. Understanding algorithm efficiency is crucial for writing code that performs well on large inputs or in resource constrained environments. Common complexity classes include constant time O(1), logarithmic O(log n), linear O(n), and quadratic O(n squared), each with different implications for scalability.",
            "Code style guidelines promote consistency across programming projects, making code easier to read, review, and maintain. Style guidelines cover aspects like naming conventions, indentation, line length, and code organization. Tools like linters and formatters can automatically enforce style guidelines, catching violations before code review. Consistent style reduces cognitive overhead when switching between files or projects and helps teams collaborate more effectively.",
            "Debugging identifies and fixes errors in program logic through systematic investigation of program behavior. Effective debugging involves reproducing the error, understanding the expected versus actual behavior, isolating the problematic code section, and verifying that the fix resolves the issue without introducing new problems. Debugging tools like debuggers, loggers, and profilers help developers understand program execution and locate the source of errors.",
            "Unit testing verifies individual components function correctly in isolation, providing confidence that each piece of the system works as expected. Well written unit tests serve as executable documentation of component behavior and catch regressions when code is modified. Test driven development takes this further by writing tests before implementation, ensuring that code is designed with testability in mind and that requirements are clearly specified.",
            "Version control tracks changes and enables collaboration on code by maintaining a complete history of modifications and providing tools for merging concurrent changes. Git has become the dominant version control system, enabling distributed development workflows, branching for feature development, and pull requests for code review. Version control is essential for coordinating team development, maintaining multiple release versions, and recovering from mistakes.",
            "Code refactoring improves structure without changing behavior, making code more maintainable, readable, or efficient while preserving its functionality. Common refactoring patterns include extracting methods, renaming variables, moving code between classes, and simplifying conditional expressions. Refactoring is most effective when done incrementally alongside a comprehensive test suite that verifies behavior preservation.",
            "Design patterns provide reusable solutions to common problems in software design, capturing best practices that have evolved over time. Creational patterns like Factory and Singleton address object creation, structural patterns like Adapter and Decorator address composition, and behavioral patterns like Observer and Strategy address interaction between objects. Understanding design patterns helps developers communicate about design decisions and avoid reinventing solutions to common problems.",
            "Type systems prevent errors by checking data type compatibility at compile time or runtime, catching type mismatches before they cause failures. Static type systems perform checking at compile time, enabling early error detection and better tooling support. Dynamic type systems defer checking to runtime, offering more flexibility but requiring more testing to catch type errors. Type inference in modern languages provides the benefits of static typing with less syntactic overhead.",
            "API design defines interfaces for software components, specifying how different parts of a system interact with each other. Well designed APIs are intuitive to use, consistent in their conventions, and stable across versions. API design considerations include naming conventions, parameter types, error handling, and documentation. Good API design reduces integration errors and makes it easier for developers to accomplish their tasks.",
            "Error handling manages exceptional conditions gracefully, preventing crashes and providing useful feedback when things go wrong. Robust error handling includes validating inputs, checking for error conditions, and providing informative error messages. Different languages offer different error handling mechanisms, including exceptions, error codes, and result types, each with tradeoffs in terms of explicitness and ease of use.",
            "Documentation explains code purpose, usage, and implementation, serving as a guide for developers who need to understand, use, or modify the code. Effective documentation includes high level descriptions of module purpose, API documentation for public interfaces, and inline comments for complex implementation details. Documentation should be maintained alongside code changes to prevent it from becoming outdated and misleading.",
            "Code optimization improves performance and resource efficiency, making programs run faster or use less memory. Optimization techniques include choosing efficient algorithms and data structures, reducing unnecessary computation, and leveraging hardware capabilities. Premature optimization should be avoided in favor of writing clear code first and then optimizing based on profiling data that identifies actual bottlenecks."
        );
    }

    /** 生成 Transformer 相关长文本段落（适合预训练） */
    static List<String> generateTransformerTexts() {
        return Arrays.asList(
            "Transformer architecture revolutionized natural language processing when it was introduced in the seminal 2017 paper Attention Is All You Need. Prior to transformers, recurrent neural networks and convolutional networks were the dominant architectures for sequence processing. Transformers replaced recurrence with self attention, enabling parallel processing of sequence positions and eliminating the vanishing gradient problem that plagued training of deep recurrent networks.",
            "Self attention computes relationships between all positions in parallel, allowing each position in a sequence to attend to all other positions directly. For each position, the attention mechanism computes weighted sums of values from all positions, where weights are determined by the compatibility between queries and keys. This enables capturing long range dependencies without the sequential bottleneck of recurrent architectures, as information can flow directly between any two positions.",
            "Multi head attention captures different types of dependencies simultaneously by running multiple attention operations in parallel with different learned projection matrices. Each attention head can learn to focus on different aspects of the relationships between positions, such as syntactic dependencies, semantic relationships, or positional patterns. The outputs of all heads are concatenated and projected to produce the final attention output, combining diverse attention patterns into a unified representation.",
            "Positional encoding adds sequence order information to token embeddings, as self attention is permutation invariant and has no inherent notion of position. The original transformer used sinusoidal positional encodings where each dimension corresponds to a sinusoid of different frequency, allowing the model to learn relative positions. Later variants introduced learned positional embeddings, relative positional encodings, and rotary positional embeddings that have become standard in modern language models.",
            "Query key value projections enable flexible attention computation by linearly projecting the input into three different representation spaces. The query projection represents what each position is looking for, the key projection represents what each position offers, and the value projection represents the information to be aggregated. The attention score between two positions is computed as the dot product of their query and key, and the output is the weighted sum of values.",
            "Scaled dot product attention prevents gradient issues with large dimensions by dividing the attention scores by the square root of the key dimension. Without this scaling, the dot products can grow large in magnitude, pushing the softmax function into regions of extremely small gradients. This would make training difficult as gradients would become too small to propagate effectively. The scaling ensures that the attention weights remain in a regime with useful gradients.",
            "Layer normalization stabilizes training in deep transformer networks by normalizing activations across the feature dimension for each position independently. Unlike batch normalization, layer normalization does not depend on batch statistics and works well with small batch sizes and variable length sequences. Pre normalization, where layer normalization is applied before the attention and feed forward layers rather than after, has become the preferred approach for training very deep transformers.",
            "Feed forward networks process each position independently in transformers, applying the same learned transformation to every position. Each feed forward layer typically consists of two linear transformations with a nonlinearity in between, often with a larger hidden dimension than the model dimension. This position wise processing complements the position mixing done by attention, enabling the model to compute complex transformations of the information aggregated by attention.",
            "Residual connections enable training of very deep transformer models by providing gradient shortcuts that bypass each layer. Without residual connections, gradients would need to flow through many sequential transformations during backpropagation, leading to vanishing or exploding gradients. Residual connections allow gradients to flow directly to earlier layers, enabling effective training of models with hundreds of layers.",
            "Transformer decoder uses masked self attention for autoregressive generation, preventing positions from attending to future positions in the sequence. The masking ensures that the model can only use information from positions that have already been generated when predicting the next token. This causal masking is essential for language generation tasks where the model must predict tokens sequentially without peeking at future context.",
            "Cross attention connects encoder and decoder in sequence to sequence tasks, allowing the decoder to attend to encoder representations. The queries come from the decoder while keys and values come from the encoder, enabling the decoder to selectively focus on relevant parts of the source sequence when generating each target token. Cross attention is the primary mechanism for information transfer between encoder and decoder in translation and other conditional generation tasks.",
            "Pre training on large corpora gives transformers broad language understanding by exposing them to diverse linguistic patterns and world knowledge. The model learns to predict masked tokens or next tokens, forcing it to develop representations that capture syntax, semantics, and common sense reasoning. The scale of pre training data and model parameters correlates strongly with downstream task performance, driving the trend toward ever larger language models.",
            "Fine tuning adapts pre trained transformers to downstream tasks efficiently by continuing training on task specific datasets with a smaller learning rate. The pre trained weights provide a strong initialization that already encodes general language knowledge, so only modest task specific adaptation is needed. This transfer learning approach has made it practical to apply large language models to many tasks without training from scratch.",
            "Transformers eliminate recurrence enabling parallel sequence processing, which significantly speeds up both training and inference compared to recurrent architectures. In RNNs, each position must be processed sequentially because the hidden state is passed from one position to the next. Transformers process all positions in parallel during the forward pass, making it possible to leverage modern GPU and TPU hardware more efficiently.",
            "Sparse attention reduces computational cost for long sequences by limiting each position to attend to only a subset of other positions rather than all positions. Various sparsity patterns have been proposed, including local attention where positions only attend to nearby neighbors, strided attention with fixed patterns, and learned sparse attention where the model discovers useful sparse patterns. Sparse attention enables transformers to scale to sequence lengths that would be prohibitively expensive with dense attention.",
            "Relative position encoding captures position relationships more flexibly by encoding the relative distance between positions rather than absolute positions. This allows the model to generalize better to sequences longer than those seen during training and captures translation invariant patterns. Relative positional encodings have become standard in many modern transformer variants, often implemented by adding relative position biases to attention scores.",
            "Attention visualization reveals linguistic patterns learned by the model, showing how different heads attend to different types of relationships. Some heads learn to attend to the previous or next token, capturing local syntactic structure. Others attend to the start of sentences, matching punctuation, or words with specific grammatical roles. However, attention is an imperfect proxy for model reasoning, and the relationship between attention patterns and model behavior remains an active area of research.",
            "Transformer XL extends context through segment level recurrence, enabling the model to process documents longer than the fixed context window by maintaining a memory of previous segments. Hidden states from previous segments are cached and used as extended context for the current segment, allowing information to flow across segment boundaries. This approach enables processing of very long documents while maintaining the efficiency of fixed context window processing.",
            "GPT uses decoder only transformers for language generation, demonstrating that the decoder architecture alone is sufficient for strong language modeling performance. GPT models are trained with a causal language modeling objective to predict the next token given previous tokens. The decoder only architecture has become the dominant paradigm for large language models, as it is simpler than encoder decoder architectures and well suited for text generation tasks.",
            "Attention weights show which input positions influence each output, providing interpretability into how the model processes information. Visualizing attention patterns reveals that lower layers often attend to local context while higher layers capture broader dependencies. Some attention heads learn to attend to specific linguistic patterns like subject verb agreement or coreference resolution, though the extent to which attention weights reflect meaningful computation is debated."
        );
    }

    /** 生成深度学习基础长文本段落（适合预训练） */
    static List<String> generateDeepLearningTexts() {
        return Arrays.asList(
            "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. Each layer transforms the input into increasingly abstract features, with early layers capturing simple patterns and deeper layers capturing complex concepts. This hierarchical learning is inspired by the visual cortex, where neurons detect increasingly complex features from edges to shapes to objects. Deep learning has achieved remarkable success across computer vision, natural language processing, speech recognition, and many other domains.",
            "Backpropagation computes gradients for training neural networks by applying the chain rule of calculus to propagate error signals backward through the network. Starting from the loss at the output, gradients are computed layer by layer by multiplying local gradients with upstream gradients. This efficient gradient computation makes it practical to train networks with millions or billions of parameters. Understanding backpropagation is fundamental to understanding how neural networks learn from data.",
            "Gradient descent optimizes network parameters iteratively by moving in the direction that reduces the loss function. The learning rate controls the step size, with too large a rate causing divergence and too small a rate causing slow convergence. Modern optimizers extend basic gradient descent with momentum, adaptive learning rates, and other techniques to achieve faster and more reliable convergence across diverse optimization landscapes.",
            "Loss functions measure prediction errors during training, providing the signal that drives learning. Common loss functions include mean squared error for regression, cross entropy for classification, and various specialized losses for tasks like object detection and sequence labeling. The choice of loss function encodes assumptions about what constitutes good predictions and significantly impacts model behavior and performance.",
            "Activation functions introduce non linearity into networks, enabling them to approximate complex functions that linear transformations alone cannot represent. Popular activation functions include ReLU, which outputs the input if positive and zero otherwise, sigmoid and tanh for bounded outputs, and newer variants like GELU and Swish that provide smoother gradients. The choice of activation function affects training dynamics, gradient flow, and model performance.",
            "Batch normalization accelerates training and improves stability by normalizing activations within each layer. For each mini batch, the mean and variance are computed and used to normalize activations, followed by learnable scale and shift parameters. Batch normalization reduces internal covariate shift, allows higher learning rates, and provides a regularizing effect. It has become a standard component in many neural network architectures.",
            "Dropout prevents overfitting by randomly disabling neurons during training, forcing the network to learn robust features that do not depend on any specific neuron. At each training step, each neuron is retained with probability p and set to zero otherwise. During inference, all neurons are active but outputs are scaled by p to maintain expected values. Dropout can be interpreted as training an ensemble of networks that share parameters.",
            "Adam optimizer adapts learning rates for each parameter individually by maintaining estimates of first and second moments of gradients. Adam combines the benefits of momentum, which accelerates convergence by smoothing gradient directions, and adaptive learning rates, which adjust step sizes based on gradient magnitudes. Adam has become the default optimizer for many deep learning tasks due to its robustness and efficiency.",
            "Regularization techniques prevent models from overfitting by adding constraints or penalties that encourage simpler solutions. Common regularization approaches include weight decay, which adds a penalty proportional to weight magnitudes, dropout, which randomly omits units during training, and data augmentation, which artificially expands the training set. Regularization is essential for achieving good generalization when training large models on limited data.",
            "Transfer learning reuses knowledge from pre trained models by fine tuning them on new tasks, dramatically reducing the amount of data and computation needed to achieve good performance. Pre trained models learn general features that are useful across many tasks, and fine tuning adapts these features to specific applications. Transfer learning has enabled rapid progress in computer vision and natural language processing.",
            "Neural networks can approximate any continuous function given sufficient capacity, as established by the universal approximation theorem. This theoretical result shows that even a single hidden layer neural network can approximate arbitrary continuous functions to any desired accuracy. However, the theorem does not address learnability, and in practice, deeper networks with appropriate architectures learn more efficiently.",
            "Deep architectures learn hierarchical feature representations where each layer captures increasingly abstract concepts. In image recognition, early layers detect edges and textures, middle layers detect shapes and parts, and later layers detect objects and scenes. This hierarchical decomposition mirrors how humans process visual information and enables learning complex concepts from simpler building blocks.",
            "Convolutional networks excel at processing grid structured data like images by using convolutional layers that share weights across spatial positions. This weight sharing provides translation invariance and dramatically reduces the number of parameters compared to fully connected networks. Convolutional networks have achieved superhuman performance on image classification and are used throughout computer vision applications.",
            "Recurrent networks handle sequential and temporal data by maintaining hidden state that evolves over time. At each time step, the network processes the current input and previous hidden state to produce an output and updated hidden state. Long Short Term Memory and Gated Recurrent Unit architectures address the vanishing gradient problem in simple recurrent networks, enabling learning of longer range dependencies.",
            "Attention mechanisms focus on relevant input parts by computing weighted combinations of input representations, where weights are determined by learned compatibility functions. Attention provides a flexible way to aggregate information without imposing rigid sequential processing. Self attention, where positions attend to each other within a sequence, has become the dominant approach for natural language processing.",
            "Skip connections help gradients flow in deep networks by providing direct paths from earlier layers to later layers. Popularized by ResNet, skip connections add the input of a layer block to its output, allowing gradients to bypass the block during backpropagation. Skip connections enable training of networks with hundreds or thousands of layers and have become standard in many architectures.",
            "Embedding layers map discrete tokens to continuous vectors, enabling neural networks to process categorical data like words or items. The embedding vectors are learned during training and capture semantic relationships between tokens, with similar tokens having similar embeddings. Word embeddings have revolutionized natural language processing by providing dense representations that capture linguistic regularities.",
            "Early stopping monitors validation performance to prevent overfitting by halting training when validation performance stops improving. Rather than training for a fixed number of epochs, early stopping uses the validation set to detect when the model begins to overfit and restores the best checkpoint. This simple technique is highly effective and should be used in most training pipelines.",
            "Data augmentation increases training data diversity by applying transformations to existing examples. For images, common augmentations include random cropping, flipping, rotation, and color adjustments. For text, augmentations include synonym replacement, random insertion, and back translation. Data augmentation acts as a regularizer and can significantly improve model generalization when labeled data is limited.",
            "Learning rate controls optimization step size and is one of the most important hyperparameters for training neural networks. Finding the right learning rate often requires experimentation, with techniques like learning rate schedules, warmup, and cyclical learning rates helping to achieve good results. Learning rate is typically set higher at the start of training and decreased over time to fine tune parameters."
        );
    }

    // ========== 后训练问答数据生成 ==========

    /**
     * 生成任务感知的指令问答对（后训练通用数据集）
     *
     * 采用 ChatML 对话模板：[TASK_TYPE] &lt;|im_start|&gt;user\n{问题}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n{回答}&lt;|im_end|&gt;
     * 训练时只对 assistant 回答部分计算 loss（answer-only loss masking）
     */
    static List<String> generateTaskAwareQA() {
        List<String> qa = new ArrayList<>();

        // MoE 相关问答 (5条)
        qa.add("[REASONING] <|im_start|>user\nWhat is Mixture of Experts?<|im_end|>\n<|im_start|>assistant\nMixture of Experts is an architecture that uses multiple specialized expert networks with a gating mechanism to route inputs efficiently<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nHow does MoE routing work?<|im_end|>\n<|im_start|>assistant\nMoE routing uses a gating network to compute scores for each expert and selects the top K experts to process each input token<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nWhy is load balancing important in MoE?<|im_end|>\n<|im_start|>assistant\nLoad balancing ensures all experts are utilized evenly preventing some experts from being overused while others remain idle<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nWhat is sparse activation?<|im_end|>\n<|im_start|>assistant\nSparse activation means only a subset of model parameters are active for each input reducing computational cost significantly<|im_end|>");
        qa.add("[MATH] <|im_start|>user\nIf MoE has 8 experts and uses top 2 routing what is the activation ratio?<|im_end|>\n<|im_start|>assistant\nThe activation ratio is 2 divided by 8 which equals 0.25 or 25 percent<|im_end|>");

        // DeepSeek 相关问答 (5条)
        qa.add("[GENERAL] <|im_start|>user\nWhat is DeepSeek V3?<|im_end|>\n<|im_start|>assistant\nDeepSeek V3 is an advanced language model combining MoE architecture with task aware routing for efficient and high quality text generation<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nHow does task aware architecture help?<|im_end|>\n<|im_start|>assistant\nTask aware architecture adapts model behavior based on task type routing inputs to experts specialized for reasoning coding math or other domains<|im_end|>");
        qa.add("[CODING] <|im_start|>user\nWhat languages does DeepSeek support?<|im_end|>\n<|im_start|>assistant\nDeepSeek supports ten programming languages including Python Java C plus plus JavaScript Go Rust TypeScript Ruby PHP and Swift<|im_end|>");
        qa.add("[CODING] <|im_start|>user\nHow is code quality evaluated?<|im_end|>\n<|im_start|>assistant\nCode quality is evaluated on four dimensions correctness readability efficiency and adherence to style guidelines<|im_end|>");
        qa.add("[GENERAL] <|im_start|>user\nWhat are DeepSeek advantages?<|im_end|>\n<|im_start|>assistant\nDeepSeek advantages include efficient sparse computation task adaptive routing strong code generation and fast inference speed<|im_end|>");

        // 编程相关问答 (5条)
        qa.add("[CODING] <|im_start|>user\nWhat is Python used for?<|im_end|>\n<|im_start|>assistant\nPython is used for machine learning data science web development automation scientific computing and general purpose programming<|im_end|>");
        qa.add("[CODING] <|im_start|>user\nExplain object oriented programming.<|im_end|>\n<|im_start|>assistant\nObject oriented programming organizes code into objects that combine data and methods providing encapsulation inheritance and polymorphism<|im_end|>");
        qa.add("[CODING] <|im_start|>user\nWhat is algorithm complexity?<|im_end|>\n<|im_start|>assistant\nAlgorithm complexity measures computational resources required typically expressed as time complexity and space complexity using big O notation<|im_end|>");
        qa.add("[CODING] <|im_start|>user\nWhy is code readability important?<|im_end|>\n<|im_start|>assistant\nCode readability makes programs easier to understand maintain debug and extend by other developers or future self<|im_end|>");
        qa.add("[CODING] <|im_start|>user\nWhat are design patterns?<|im_end|>\n<|im_start|>assistant\nDesign patterns are reusable solutions to common software design problems providing tested templates for solving recurring challenges<|im_end|>");

        // Transformer 相关问答 (5条)
        qa.add("[REASONING] <|im_start|>user\nWhat is self attention?<|im_end|>\n<|im_start|>assistant\nSelf attention computes relationships between all positions in a sequence allowing each position to attend to all other positions in parallel<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nWhy use multi head attention?<|im_end|>\n<|im_start|>assistant\nMulti head attention allows the model to attend to different representation subspaces simultaneously capturing diverse types of relationships<|im_end|>");
        qa.add("[GENERAL] <|im_start|>user\nWhat is positional encoding?<|im_end|>\n<|im_start|>assistant\nPositional encoding injects information about token positions into embeddings since transformers have no inherent notion of sequence order<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nHow does attention scaling work?<|im_end|>\n<|im_start|>assistant\nAttention scores are scaled by the square root of dimension to prevent extremely small gradients when dot products grow large<|im_end|>");
        qa.add("[GENERAL] <|im_start|>user\nWhat is the transformer advantage?<|im_end|>\n<|im_start|>assistant\nTransformers enable parallel processing of sequences eliminate vanishing gradients and capture long range dependencies effectively<|im_end|>");

        // 深度学习基础问答 (5条)
        qa.add("[REASONING] <|im_start|>user\nWhat is backpropagation?<|im_end|>\n<|im_start|>assistant\nBackpropagation computes gradients of loss with respect to parameters by applying chain rule backwards through the computational graph<|im_end|>");
        qa.add("[MATH] <|im_start|>user\nHow does gradient descent work?<|im_end|>\n<|im_start|>assistant\nGradient descent updates parameters by moving in the direction opposite to the gradient scaled by learning rate to minimize loss<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nWhy use activation functions?<|im_end|>\n<|im_start|>assistant\nActivation functions introduce non linearity enabling neural networks to learn complex patterns beyond linear relationships<|im_end|>");
        qa.add("[GENERAL] <|im_start|>user\nWhat is overfitting?<|im_end|>\n<|im_start|>assistant\nOverfitting occurs when a model learns training data too well including noise and fails to generalize to new unseen data<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nHow does dropout prevent overfitting?<|im_end|>\n<|im_start|>assistant\nDropout randomly disables neurons during training forcing the network to learn robust features that do not rely on specific neurons<|im_end|>");

        return qa;
    }

    /**
     * 生成代码专项问答数据（纯 CODING 任务，强化 MoE 专家特化）
     *
     * 覆盖 Python/Java/JavaScript/C++ 及通用编程概念，均使用 ChatML 格式
     */
    static List<String> generateCodeQA() {
        List<String> codeQA = new ArrayList<>();

        // Python 相关问答 (5条)
        codeQA.add("[CODING] <|im_start|>user\nHow to define a function in Python?<|im_end|>\n<|im_start|>assistant\nUse def keyword followed by function name parentheses for parameters and colon then indent the function body<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is list comprehension in Python?<|im_end|>\n<|im_start|>assistant\nList comprehension provides concise syntax to create lists using bracket notation with for loop and optional if condition<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to handle exceptions in Python?<|im_end|>\n<|im_start|>assistant\nUse try block for code that may raise exceptions except block to catch and handle specific exception types and finally for cleanup<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat are Python decorators?<|im_end|>\n<|im_start|>assistant\nDecorators are functions that modify behavior of other functions using at symbol syntax wrapping the original function with additional functionality<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to read files in Python?<|im_end|>\n<|im_start|>assistant\nUse open function with file path and mode then read content with read or readlines method always close file or use with statement<|im_end|>");

        // Java 相关问答 (5条)
        codeQA.add("[CODING] <|im_start|>user\nHow to declare a class in Java?<|im_end|>\n<|im_start|>assistant\nUse public class keyword followed by class name then curly braces containing fields methods and constructors<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is inheritance in Java?<|im_end|>\n<|im_start|>assistant\nInheritance allows a class to inherit fields and methods from parent class using extends keyword promoting code reuse<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to create an interface in Java?<|im_end|>\n<|im_start|>assistant\nUse interface keyword followed by name declare method signatures without implementation classes implement interface with implements keyword<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is Java generics?<|im_end|>\n<|im_start|>assistant\nGenerics enable types to be parameters when defining classes interfaces and methods providing compile time type safety<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to handle null in Java?<|im_end|>\n<|im_start|>assistant\nCheck for null before dereferencing use Optional class or annotations like NonNull to prevent null pointer exceptions<|im_end|>");

        // JavaScript 相关问答 (5条)
        codeQA.add("[CODING] <|im_start|>user\nWhat are arrow functions in JavaScript?<|im_end|>\n<|im_start|>assistant\nArrow functions provide shorter syntax using arrow notation capture this from enclosing scope unlike regular functions<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to handle async operations in JavaScript?<|im_end|>\n<|im_start|>assistant\nUse promises with then and catch or async await syntax for cleaner asynchronous code handling<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is closure in JavaScript?<|im_end|>\n<|im_start|>assistant\nClosure is a function that remembers variables from its outer scope even after outer function has finished executing<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to iterate arrays in JavaScript?<|im_end|>\n<|im_start|>assistant\nUse for loop forEach map filter or for of loop depending on whether you need transformation filtering or simple iteration<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is destructuring in JavaScript?<|im_end|>\n<|im_start|>assistant\nDestructuring extracts values from arrays or objects into distinct variables using bracket or curly brace syntax<|im_end|>");

        // C++ 相关问答 (5条)
        codeQA.add("[CODING] <|im_start|>user\nWhat are pointers in C plus plus?<|im_end|>\n<|im_start|>assistant\nPointers store memory addresses enabling dynamic memory allocation direct memory access and efficient data structure implementation<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to use templates in C plus plus?<|im_end|>\n<|im_start|>assistant\nTemplates enable generic programming using template keyword with type parameters allowing code to work with different data types<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is RAII in C plus plus?<|im_end|>\n<|im_start|>assistant\nResource Acquisition Is Initialization ties resource lifetime to object lifetime using constructors and destructors for automatic resource management<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to handle memory in C plus plus?<|im_end|>\n<|im_start|>assistant\nUse new for dynamic allocation delete to free memory or prefer smart pointers like unique ptr and shared ptr<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat are virtual functions in C plus plus?<|im_end|>\n<|im_start|>assistant\nVirtual functions enable polymorphism allowing derived classes to override base class methods using virtual keyword<|im_end|>");

        // 通用编程概念 (5条)
        codeQA.add("[CODING] <|im_start|>user\nWhat is time complexity?<|im_end|>\n<|im_start|>assistant\nTime complexity measures how algorithm runtime grows with input size using big O notation like O of n or O of n squared<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to optimize code performance?<|im_end|>\n<|im_start|>assistant\nProfile code identify bottlenecks use efficient algorithms and data structures minimize memory allocations and avoid premature optimization<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is recursion?<|im_end|>\n<|im_start|>assistant\nRecursion is when function calls itself to solve problem by breaking it into smaller subproblems requires base case to terminate<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to debug code effectively?<|im_end|>\n<|im_start|>assistant\nUse debugger set breakpoints inspect variables trace execution flow write unit tests and use logging strategically<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is code refactoring?<|im_end|>\n<|im_start|>assistant\nRefactoring improves code structure readability and maintainability without changing external behavior through small incremental changes<|im_end|>");

        return codeQA;
    }

    // ========== 文件读写工具 ==========

    /**
     * 将文本列表写入文件（每行一条记录）
     */
    public static void writeToFile(List<String> texts, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String text : texts) {
                writer.write(text);
                writer.newLine();
            }
        }
    }

    /**
     * 从文件读取文本列表（跳过空行）
     */
    public static List<String> readFromFile(String filePath) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line);
                }
            }
        }
        return lines;
    }
}
