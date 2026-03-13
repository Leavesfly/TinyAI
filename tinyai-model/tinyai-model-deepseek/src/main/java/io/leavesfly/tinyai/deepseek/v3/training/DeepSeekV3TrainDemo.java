package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.deepseek.base.TaskType;

import java.io.*;
import java.util.*;

/**
 * DeepSeek-V3完整训练演示 V2版本
 * 
 * 参考GPT1TrainDemoV2的实现方式，提供完整的训练流程：
 * 1. 准备真实的教学数据集（适用于教育学习）
 * 2. 预训练阶段 - 基础语言建模训练
 * 3. 后训练阶段 - 任务特定微调
 * 4. RLHF阶段 - 人类反馈强化学习训练
 * 5. 推理阶段 - 多种生成策略演示
 * 
 * 改进点：
 * - 使用真实文本数据而非随机数据
 * - 支持从文件加载数据集
 * - 包含数据集自动生成功能
 * - 详细的训练过程说明和日志
 * - 完整的预训练-后训练-RLHF-推理流程
 * - 后训练采用行业主流的 ChatML 对话模板格式
 * - 实现 Answer-Only Loss Masking（仅对回答部分计算损失）
 * 
 * V3特色：
 * - MoE架构的负载均衡训练
 * - 任务感知的数据标注和训练
 * - 代码生成任务的专门支持
 * - RLHF对齐人类偏好
 * 
 * 预训练 vs 后训练 vs RLHF数据格式对比：
 * - 预训练: 纯文本句子 → 全序列因果语言建模(CLM) loss
 * - 后训练: ChatML 对话模板 → Answer-Only loss（只对 assistant 回答计算损失）
 *   格式: [TASK] &lt;|im_start|&gt;user\n{问题}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n{回答}&lt;|im_end|&gt;
 * - RLHF: ChatML 对话模板 + 奖励标注 → Reward-weighted loss（奖励加权回归）
 *   格式: [REWARD:0.9] &lt;|im_start|&gt;user\n{问题}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n{回答}&lt;|im_end|&gt;
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3TrainDemo {
    
    private static SimpleTokenizer sharedTokenizer = new SimpleTokenizer();
    
    private static final String DATA_DIR = "./data/deepseek_v3_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/deepseek_v3_v2";
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 完整训练与推理演示 V2");
        System.out.println("适用于教学和学习的小型数据集训练方案");
        System.out.println("=".repeat(80));
        
        try {
            // 步骤0: 准备数据集文件
            prepareDatasets();
            
            // 步骤1: 预训练
            DeepSeekV3Model pretrainedModel = runPretraining();
            
            // 步骤2: 通用后训练（任务感知微调）
            DeepSeekV3Model finetunedModel = runPosttraining(pretrainedModel);
            
            // 步骤2B (可选): 代码生成专项后训练
            // 说明：此步骤强化MoE专家对代码任务的特化能力
            DeepSeekV3Model codeSpecializedModel = runCodePosttraining(finetunedModel);
            
            // 步骤3: RLHF强化学习训练（人类反馈强化学习）
            // 说明：RLHF 是 V3 训练流程的标准步骤，用于对齐人类偏好
            DeepSeekV3Model rlhfModel = runRLHFTraining(codeSpecializedModel);
            
            // 步骤4: 推理测试
            runInference(rlhfModel);
            
            System.out.println("\n" + "=".repeat(80));
            System.out.println("✅ 完整训练流程演示成功!");
            System.out.println("=".repeat(80));
            
        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 准备训练数据集
     * 生成pretrain和posttrain数据文件
     */
    private static void prepareDatasets() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📦 步骤0: 准备训练数据集");
        System.out.println("=".repeat(80));
        
        File dataDir = new File(DATA_DIR);
        if (!dataDir.exists()) {
            dataDir.mkdirs();
            System.out.println("✓ 创建数据目录: " + DATA_DIR);
        }
        
        // 生成预训练数据集
        generatePretrainDataset();
        
        // 生成后训练数据集
        generatePosttrainDataset();
        
        // 生成RLHF数据集
        generateRLHFDataset();
        
        System.out.println("\n✅ 数据集准备完成!");
    }
    
    /**
     * 生成预训练数据集
     * 包含深度学习、MoE、Transformer等领域的教学文本
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");
        
        List<String> pretrainTexts = new ArrayList<>();
        
        // 1. MoE和专家系统相关知识 (50条)
        pretrainTexts.addAll(generateMoETexts());
        
        // 2. DeepSeek和大模型相关知识 (50条)
        pretrainTexts.addAll(generateDeepSeekTexts());
        
        // 3. 任务感知和推理相关知识 (40条)
        pretrainTexts.addAll(generateReasoningTexts());
        
        // 4. 代码生成和编程相关知识 (40条)
        pretrainTexts.addAll(generateCodingTexts());
        
        // 5. Transformer和注意力机制 (40条)
        pretrainTexts.addAll(generateTransformerTexts());
        
        // 6. 深度学习基础 (30条)
        pretrainTexts.addAll(generateDeepLearningTexts());
        
        // 写入文件
        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);
        
        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }
    
    /**
     * 生成后训练数据集
     * 包含任务特定的指令-回答对
     */
    private static void generatePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成后训练数据集...");
        
        List<String> trainTexts = new ArrayList<>();
        List<String> valTexts = new ArrayList<>();
        
        // 训练集: 100条任务感知的指令-回答对
        trainTexts.addAll(generateTaskAwareQA());
        
        // 验证集: 从训练集中抽取15条
        for (int i = 0; i < 15 && i < trainTexts.size(); i++) {
            valTexts.add(trainTexts.get(i));
        }
        
        // 写入训练集
        String trainPath = DATA_DIR + "/posttrain_train.txt";
        writeToFile(trainTexts, trainPath);
        System.out.println("  ✓ 后训练训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + trainPath);
        
        // 写入验证集
        String valPath = DATA_DIR + "/posttrain_val.txt";
        writeToFile(valTexts, valPath);
        System.out.println("  ✓ 后训练验证集: " + valTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + valPath);
        
        // 生成代码专项训练数据集
        generateCodePosttrainDataset();
    }
    
    /**
     * 生成代码生成专项后训练数据集
     * 纯代码任务数据，用于强化MoE专家特化能力
     */
    private static void generateCodePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成代码专项后训练数据集...");
        
        List<String> codeTrainTexts = new ArrayList<>();
        List<String> codeValTexts = new ArrayList<>();
        
        // 训练集: 60条纯代码任务问答
        codeTrainTexts.addAll(generateCodeQA());
        
        // 验证集: 从训练集中抽取10条
        for (int i = 0; i < 10 && i < codeTrainTexts.size(); i++) {
            codeValTexts.add(codeTrainTexts.get(i));
        }
        
        // 写入训练集
        String codeTrainPath = DATA_DIR + "/code_posttrain_train.txt";
        writeToFile(codeTrainTexts, codeTrainPath);
        System.out.println("  ✓ 代码专项训练集: " + codeTrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + codeTrainPath);
        
        // 写入验证集
        String codeValPath = DATA_DIR + "/code_posttrain_val.txt";
        writeToFile(codeValTexts, codeValPath);
        System.out.println("  ✓ 代码专项验证集: " + codeValTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + codeValPath);
    }
    
    /**
     * 生成RLHF数据集
     * 包含带奖励标注的ChatML对话数据，用于人类反馈强化学习
     */
    private static void generateRLHFDataset() throws IOException {
        System.out.println("\n📝 生成RLHF强化学习数据集...");
        
        List<String> rlhfTexts = new ArrayList<>();
        
        // 高奖励样本 (0.8-1.0) - 20条
        rlhfTexts.addAll(generateHighRewardRLHF());
        
        // 中奖励样本 (0.4-0.7) - 20条
        rlhfTexts.addAll(generateMediumRewardRLHF());
        
        // 低奖励样本 (0.1-0.3) - 10条
        rlhfTexts.addAll(generateLowRewardRLHF());
        
        // 写入文件
        String filePath = DATA_DIR + "/rlhf_train.txt";
        writeToFile(rlhfTexts, filePath);
        
        System.out.println("  ✓ RLHF训练数据: " + rlhfTexts.size() + " 条");
        System.out.println("  ✓ 高奖励样本: 20 条");
        System.out.println("  ✓ 中奖励样本: 20 条");
        System.out.println("  ✓ 低奖励样本: 10 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }
    
    /**
     * 生成高奖励RLHF样本 (0.8-1.0)
     */
    private static List<String> generateHighRewardRLHF() {
        List<String> samples = new ArrayList<>();
        
        // MoE相关 - 高质量回答
        samples.add("[REWARD:0.95] <|im_start|>user\nWhat is Mixture of Experts and how does it work?<|im_end|>\n<|im_start|>assistant\nMixture of Experts (MoE) is a neural network architecture that uses multiple specialized expert networks. A gating network learns to route each input token to the most relevant experts. Only the selected experts are activated, enabling sparse computation. For example, DeepSeek V3 uses 8 experts with top-2 routing, activating only 25% of parameters per token, achieving high efficiency while maintaining performance.<|im_end|>");
        samples.add("[REWARD:0.92] <|im_start|>user\nExplain load balancing in MoE systems.<|im_end|>\n<|im_start|>assistant\nLoad balancing in MoE ensures all experts are utilized evenly during training. Without load balancing, the gating network might always route to the same experts, causing expert collapse. An auxiliary loss term penalizes uneven expert utilization, encouraging the model to distribute tokens across all experts. This improves model capacity and prevents some experts from becoming underutilized.<|im_end|>");
        samples.add("[REWARD:0.90] <|im_start|>user\nWhat are the advantages of sparse activation?<|im_end|>\n<|im_start|>assistant\nSparse activation in MoE models provides three key advantages: 1) Computational efficiency - only a fraction of parameters are active per forward pass, reducing FLOPs; 2) Scalability - models can have billions of parameters while maintaining fast inference; 3) Specialization - different experts can specialize in different types of content, improving overall model capacity and performance on diverse tasks.<|im_end|>");
        
        // DeepSeek相关 - 高质量回答
        samples.add("[REWARD:0.93] <|im_start|>user\nWhat makes DeepSeek V3 special compared to other models?<|im_end|>\n<|im_start|>assistant\nDeepSeek V3 combines three key innovations: 1) Mixture of Experts architecture with 8 experts and top-2 routing for 25% parameter activation; 2) Task-aware routing that adapts expert selection based on input type (reasoning, coding, math, general); 3) Multi-head latent attention for efficient processing. This combination enables state-of-the-art performance on code generation benchmarks while being 4x faster than equivalent dense models.<|im_end|>");
        samples.add("[REWARD:0.91] <|im_start|>user\nHow does task-aware architecture improve performance?<|im_end|>\n<|im_start|>assistant\nTask-aware architecture in DeepSeek V3 improves performance by dynamically selecting specialized experts based on the input task type. For reasoning tasks, it activates experts trained on logical inference; for coding tasks, it activates programming-specialized experts; for math tasks, it activates mathematical reasoning experts. This task-specific expert selection allows the model to leverage specialized knowledge more effectively than a one-size-fits-all approach.<|im_end|>");
        
        // 代码相关 - 高质量回答
        samples.add("[REWARD:0.94] <|im_start|>user\nWrite a Python function to implement binary search.<|im_end|>\n<|im_start|>assistant\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\nThis implementation has O(log n) time complexity and O(1) space complexity. The array must be sorted for binary search to work correctly.<|im_end|>");
        samples.add("[REWARD:0.89] <|im_start|>user\nExplain the difference between list and tuple in Python.<|im_end|>\n<|im_start|>assistant\nIn Python, lists and tuples are both sequence types, but they have key differences: 1) Mutability - lists are mutable (can be modified), tuples are immutable (cannot be changed after creation); 2) Syntax - lists use square brackets [], tuples use parentheses (); 3) Performance - tuples are slightly faster and use less memory; 4) Use case - lists for collections that may change, tuples for fixed collections like coordinates or database records.<|im_end|>");
        
        // Transformer相关 - 高质量回答
        samples.add("[REWARD:0.90] <|im_start|>user\nWhat is self-attention in Transformers?<|im_end|>\n<|im_start|>assistant\nSelf-attention is a mechanism that allows each position in a sequence to attend to all other positions, computing weighted representations. For each token, it computes query, key, and value vectors. The attention weight between positions i and j is the softmax of the dot product between query_i and key_j. This allows the model to capture long-range dependencies and contextual relationships regardless of distance in the sequence.<|im_end|>");
        
        // 数学相关 - 高质量回答
        samples.add("[REWARD:0.93] <|im_start|>user\nSolve: If f(x) = 2x + 3, what is f(f(2))?<|im_end|>\n<|im_start|>assistant\nTo solve f(f(2)), we first compute f(2): f(2) = 2*2 + 3 = 7. Then we compute f(7): f(7) = 2*7 + 3 = 17. Therefore, f(f(2)) = 17.<|im_end|>");
        samples.add("[REWARD:0.91] <|im_start|>user\nWhat is the derivative of sin(x²)?<|im_end|>\n<|im_start|>assistant\nUsing the chain rule: d/dx[sin(x²)] = cos(x²) * d/dx[x²] = cos(x²) * 2x = 2x*cos(x²).<|im_end|>");
        
        // 更多高质量样本
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
     * 生成中奖励RLHF样本 (0.4-0.7)
     */
    private static List<String> generateMediumRewardRLHF() {
        List<String> samples = new ArrayList<>();
        
        // MoE相关 - 中等质量回答
        samples.add("[REWARD:0.65] <|im_start|>user\nWhat is MoE?<|im_end|>\n<|im_start|>assistant\nMoE stands for Mixture of Experts. It's a type of neural network that uses multiple expert networks and routes inputs to them. Only some experts are used for each input, which makes it efficient.<|im_end|>");
        samples.add("[REWARD:0.60] <|im_start|>user\nHow many experts does DeepSeek V3 use?<|im_end|>\n<|im_start|>assistant\nDeepSeek V3 uses 8 expert networks in its MoE architecture.<|im_end|>");
        samples.add("[REWARD:0.55] <|im_start|>user\nWhat is load balancing?<|im_end|>\n<|im_start|>assistant\nLoad balancing in MoE is about making sure all experts are used evenly during training so none are idle or overused.<|im_end|>");
        
        // DeepSeek相关 - 中等质量回答
        samples.add("[REWARD:0.63] <|im_start|>user\nWhat is DeepSeek V3?<|im_end|>\n<|im_start|>assistant\nDeepSeek V3 is a language model that uses MoE architecture and task-aware routing to be efficient and perform well on various tasks like coding and reasoning.<|im_end|>");
        samples.add("[REWARD:0.58] <|im_start|>user\nWhat programming languages does DeepSeek support?<|im_end|>\n<|im_start|>assistant\nDeepSeek supports multiple programming languages including Python, Java, C++, JavaScript, and others.<|im_end|>");
        
        // 代码相关 - 中等质量回答
        samples.add("[REWARD:0.67] <|im_start|>user\nWrite a function to reverse a string in Python.<|im_end|>\n<|im_start|>assistant\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\nThis uses Python slicing to reverse the string.<|im_end|>");
        samples.add("[REWARD:0.62] <|im_start|>user\nWhat is a list in Python?<|im_end|>\n<|im_start|>assistant\nA list in Python is a mutable sequence that can hold multiple items. You can create one with square brackets like [1, 2, 3].<|im_end|>");
        samples.add("[REWARD:0.57] <|im_start|>user\nHow do you define a function in Python?<|im_end|>\n<|im_start|>assistant\nYou use the def keyword followed by the function name and parameters. For example: def my_function(): pass<|im_end|>");
        
        // Transformer相关 - 中等质量回答
        samples.add("[REWARD:0.64] <|im_start|>user\nWhat is attention in Transformers?<|im_end|>\n<|im_start|>assistant\nAttention is a mechanism that lets the model focus on relevant parts of the input. Self-attention lets each position look at all other positions.<|im_end|>");
        samples.add("[REWARD:0.59] <|im_start|>user\nWhat is multi-head attention?<|im_end|>\n<|im_start|>assistant\nMulti-head attention runs multiple attention mechanisms in parallel, each learning different aspects of the relationships between tokens.<|im_end|>");
        
        // 数学相关 - 中等质量回答
        samples.add("[REWARD:0.66] <|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n2 + 2 = 4<|im_end|>");
        samples.add("[REWARD:0.61] <|im_start|>user\nWhat is the square root of 16?<|im_end|>\n<|im_start|>assistant\nThe square root of 16 is 4.<|im_end|>");
        samples.add("[REWARD:0.56] <|im_start|>user\nWhat is a derivative?<|im_end|>\n<|im_start|>assistant\nA derivative measures how a function changes as its input changes. It's the slope of the function at a point.<|im_end|>");
        
        // 更多中等质量样本
        samples.add("[REWARD:0.54] <|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\nMachine learning is a field where computers learn from data to make predictions or decisions without being explicitly programmed.<|im_end|>");
        samples.add("[REWARD:0.53] <|im_start|>user\nWhat is a neural network layer?<|im_end|>\n<|im_start|>assistant\nA neural network layer is a set of neurons that processes inputs and produces outputs. Each layer transforms the data in some way.<|im_end|>");
        samples.add("[REWARD:0.52] <|im_start|>user\nWhat is activation function?<|im_end|>\n<|im_start|>assistant\nAn activation function introduces non-linearity into a neural network. Common ones include ReLU, sigmoid, and tanh.<|im_end|>");
        samples.add("[REWARD:0.51] <|im_start|>user\nWhat is regularization?<|im_end|>\n<|im_start|>assistant\nRegularization is a technique to prevent overfitting by adding a penalty term to the loss function. L1 and L2 are common types.<|im_end|>");
        samples.add("[REWARD:0.50] <|im_start|>user\nWhat is batch normalization?<|im_end|>\n<|im_start|>assistant\nBatch normalization normalizes the inputs of each layer to improve training stability and speed.<|im_end|>");
        samples.add("[REWARD:0.49] <|im_start|>user\nWhat is dropout?<|im_end|>\n<|im_start|>assistant\nDropout randomly disables some neurons during training to prevent overfitting and improve generalization.<|im_end|>");
        samples.add("[REWARD:0.48] <|im_start|>user\nWhat is gradient clipping?<|im_end|>\n<|im_start|>assistant\nGradient clipping limits the maximum value of gradients during training to prevent exploding gradients.<|im_end|>");
        samples.add("[REWARD:0.47] <|im_start|>user\nWhat is learning rate?<|im_end|>\n<|im_start|>assistant\nLearning rate controls how much the model parameters are updated during training. Too high can cause divergence, too slow can cause slow convergence.<|im_end|>");
        samples.add("[REWARD:0.46] <|im_start|>user\nWhat is an epoch?<|im_end|>\n<|im_start|>assistant\nAn epoch is one complete pass through the entire training dataset during model training.<|im_end|>");
        samples.add("[REWARD:0.45] <|im_start|>user\nWhat is a batch?<|im_end|>\n<|im_start|>assistant\nA batch is a subset of the training data used in one iteration of training. Batch size affects memory usage and training stability.<|im_end|>");
        
        return samples;
    }
    
    /**
     * 生成低奖励RLHF样本 (0.1-0.3)
     */
    private static List<String> generateLowRewardRLHF() {
        List<String> samples = new ArrayList<>();
        
        // 低质量回答 - 错误、不完整或不相关
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
    
    /**
     * 生成MoE相关文本
     */
    private static List<String> generateMoETexts() {
        return Arrays.asList(
            "Mixture of Experts is a neural network architecture that uses multiple expert networks",
            "MoE models route inputs to different expert networks based on learned gating mechanisms",
            "The gating network in MoE decides which experts should process each input",
            "MoE achieves sparse activation by only using a subset of experts for each input",
            "Top K routing selects the K most relevant experts for each input token",
            "Load balancing in MoE ensures all experts are utilized evenly during training",
            "MoE models can scale to billions of parameters while maintaining efficient inference",
            "DeepSeek V3 uses eight expert networks with top two routing strategy",
            "Expert networks in MoE specialize in different aspects of the input distribution",
            "MoE reduces computational cost by activating only a fraction of total parameters",
            "Auxiliary loss in MoE encourages balanced expert utilization",
            "Sparse MoE models achieve better performance than dense models with similar compute",
            "Expert capacity limits the number of tokens each expert can process per batch",
            "MoE routing can be learned jointly with the model during training",
            "Switch Transformer uses one expert per token for extreme sparsity",
            "MoE enables training of very large models on limited hardware resources",
            "Expert parallelism allows MoE layers to scale across multiple devices",
            "Dynamic routing in MoE adapts to different input patterns automatically",
            "MoE models show strong few shot learning capabilities",
            "Load balancing loss prevents expert collapse where some experts are never used",
            "MoE gating uses softmax to produce probability distribution over experts",
            "Hard routing selects top K experts while soft routing uses weighted combination",
            "MoE architecture is particularly effective for multi task learning",
            "Expert dropout can improve MoE model robustness and generalization",
            "MoE models can learn hierarchical specialization across expert networks"
        );
    }
    
    /**
     * 生成DeepSeek相关文本
     */
    private static List<String> generateDeepSeekTexts() {
        return Arrays.asList(
            "DeepSeek is a series of advanced language models with innovative architectures",
            "DeepSeek V3 combines MoE with task aware routing for improved performance",
            "Task aware architecture in DeepSeek adapts to different downstream tasks",
            "DeepSeek models support reasoning coding math and multimodal tasks",
            "DeepSeek V3 achieves state of the art results on code generation benchmarks",
            "The reasoning module in DeepSeek enhances logical inference capabilities",
            "DeepSeek supports ten programming languages including Python Java and C plus plus",
            "Code quality in DeepSeek is evaluated on correctness readability efficiency and style",
            "DeepSeek V3 uses multi head latent attention for efficient processing",
            "Task type classification helps DeepSeek route inputs to specialized experts",
            "DeepSeek models can handle sequences up to thousands of tokens",
            "The architecture of DeepSeek V3 enables twenty five percent parameter activation",
            "DeepSeek V3 training uses load balanced MoE loss for expert utilization",
            "Inference in DeepSeek V3 is four times faster than equivalent dense models",
            "DeepSeek supports both causal language modeling and instruction tuning",
            "The reflection module in DeepSeek R1 enables self correction during reasoning",
            "DeepSeek models show strong performance on mathematical problem solving",
            "Multi task learning in DeepSeek improves generalization across domains",
            "DeepSeek V3 pre training uses large scale diverse text corpora",
            "Post training in DeepSeek fine tunes the model for specific applications",
            "DeepSeek achieves competitive results while using fewer active parameters",
            "The gating network in DeepSeek learns task specific expert selection",
            "DeepSeek models support both English and Chinese languages",
            "Code synthesis in DeepSeek generates functionally correct programs",
            "DeepSeek V3 architecture enables efficient deployment on edge devices"
        );
    }
    
    /**
     * 生成推理相关文本
     */
    private static List<String> generateReasoningTexts() {
        return Arrays.asList(
            "Reasoning in AI involves drawing logical conclusions from available information",
            "Chain of thought prompting improves reasoning by showing intermediate steps",
            "Task aware models adapt their reasoning strategy based on problem type",
            "Mathematical reasoning requires understanding of numerical relationships and operations",
            "Logical inference applies rules to derive new facts from existing knowledge",
            "Multi step reasoning breaks complex problems into manageable sub problems",
            "Reasoning confidence indicates the model certainty in its conclusions",
            "Commonsense reasoning requires understanding of everyday knowledge and context",
            "Analogical reasoning transfers knowledge from familiar to novel situations",
            "Causal reasoning identifies cause and effect relationships between events",
            "Abstract reasoning manipulates concepts without concrete examples",
            "Deductive reasoning applies general principles to specific cases",
            "Inductive reasoning generalizes from specific observations to broad patterns",
            "Abductive reasoning infers the most likely explanation for observations",
            "Spatial reasoning involves understanding geometric relationships and transformations",
            "Temporal reasoning tracks changes and sequences over time",
            "Reasoning modules can be trained to verify their own conclusions",
            "Self correction in reasoning improves accuracy through iterative refinement",
            "Reasoning traces provide interpretability by showing thought process",
            "Meta reasoning involves thinking about thinking and strategy selection"
        );
    }
    
    /**
     * 生成编程相关文本
     */
    private static List<String> generateCodingTexts() {
        return Arrays.asList(
            "Programming languages provide formal systems for instructing computers",
            "Python is widely used for machine learning and data science applications",
            "Java is a statically typed object oriented programming language",
            "JavaScript enables interactive web applications in browsers",
            "C plus plus offers low level control with high level abstractions",
            "Code generation models translate natural language to executable programs",
            "Syntax correctness ensures code follows language grammar rules",
            "Code readability makes programs easier to understand and maintain",
            "Algorithm efficiency measures computational complexity and resource usage",
            "Code style guidelines promote consistency across programming projects",
            "Debugging identifies and fixes errors in program logic",
            "Unit testing verifies individual components function correctly",
            "Version control tracks changes and enables collaboration on code",
            "Code refactoring improves structure without changing behavior",
            "Documentation explains code purpose usage and implementation",
            "API design defines interfaces for software components",
            "Error handling manages exceptional conditions gracefully",
            "Code optimization improves performance and resource efficiency",
            "Design patterns provide reusable solutions to common problems",
            "Type systems prevent errors by checking data type compatibility"
        );
    }
    
    /**
     * 生成Transformer相关文本
     */
    private static List<String> generateTransformerTexts() {
        return Arrays.asList(
            "Transformer architecture revolutionized natural language processing",
            "Self attention computes relationships between all positions in parallel",
            "Multi head attention captures different types of dependencies simultaneously",
            "Positional encoding adds sequence order information to token embeddings",
            "Query key value projections enable flexible attention computation",
            "Scaled dot product attention prevents gradient issues with large dimensions",
            "Attention weights show which input positions influence each output",
            "Layer normalization stabilizes training in deep transformer networks",
            "Feed forward networks process each position independently in transformers",
            "Residual connections enable training of very deep transformer models",
            "Transformer decoder uses masked self attention for autoregressive generation",
            "Cross attention connects encoder and decoder in sequence to sequence tasks",
            "Pre training on large corpora gives transformers broad language understanding",
            "Fine tuning adapts pre trained transformers to downstream tasks efficiently",
            "Transformers eliminate recurrence enabling parallel sequence processing",
            "Attention visualization reveals linguistic patterns learned by the model",
            "Sparse attention reduces computational cost for long sequences",
            "Relative position encoding captures position relationships more flexibly",
            "Transformer XL extends context through segment level recurrence",
            "GPT uses decoder only transformers for language generation"
        );
    }
    
    /**
     * 生成深度学习基础文本
     */
    private static List<String> generateDeepLearningTexts() {
        return Arrays.asList(
            "Deep learning uses neural networks with multiple layers",
            "Backpropagation computes gradients for training neural networks",
            "Gradient descent optimizes network parameters iteratively",
            "Loss functions measure prediction errors during training",
            "Activation functions introduce non linearity into networks",
            "Batch normalization accelerates training and improves stability",
            "Dropout prevents overfitting by randomly disabling neurons",
            "Learning rate controls optimization step size",
            "Adam optimizer adapts learning rates for each parameter",
            "Regularization techniques prevent models from overfitting",
            "Early stopping monitors validation performance to prevent overfitting",
            "Data augmentation increases training data diversity",
            "Transfer learning reuses knowledge from pre trained models",
            "Neural networks can approximate any continuous function",
            "Deep architectures learn hierarchical feature representations",
            "Convolutional networks excel at processing grid structured data",
            "Recurrent networks handle sequential and temporal data",
            "Attention mechanisms focus on relevant input parts",
            "Skip connections help gradients flow in deep networks",
            "Embedding layers map discrete tokens to continuous vectors"
        );
    }
    
    /**
     * 生成任务感知的问答对（后训练数据）
     * 
     * 采用行业主流的 ChatML 对话模板格式：
     *   [TASK_TYPE] <|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n{回答}<|im_end|>
     * 
     * 与预训练纯文本格式的关键区别：
     * 1. 使用 <|im_start|>/<|im_end|> 特殊标记区分对话角色边界
     * 2. 明确的 user/assistant 角色标注
     * 3. 训练时只对 assistant 回答部分计算 loss（answer-only loss masking）
     */
    private static List<String> generateTaskAwareQA() {
        List<String> qa = new ArrayList<>();
        
        // MoE相关问答 (5条)
        qa.add("[REASONING] <|im_start|>user\nWhat is Mixture of Experts?<|im_end|>\n<|im_start|>assistant\nMixture of Experts is an architecture that uses multiple specialized expert networks with a gating mechanism to route inputs efficiently<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nHow does MoE routing work?<|im_end|>\n<|im_start|>assistant\nMoE routing uses a gating network to compute scores for each expert and selects the top K experts to process each input token<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nWhy is load balancing important in MoE?<|im_end|>\n<|im_start|>assistant\nLoad balancing ensures all experts are utilized evenly preventing some experts from being overused while others remain idle<|im_end|>");
        qa.add("[REASONING] <|im_start|>user\nWhat is sparse activation?<|im_end|>\n<|im_start|>assistant\nSparse activation means only a subset of model parameters are active for each input reducing computational cost significantly<|im_end|>");
        qa.add("[MATH] <|im_start|>user\nIf MoE has 8 experts and uses top 2 routing what is the activation ratio?<|im_end|>\n<|im_start|>assistant\nThe activation ratio is 2 divided by 8 which equals 0.25 or 25 percent<|im_end|>");
        
        // DeepSeek相关问答 (5条)
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
        
        // Transformer相关问答 (5条)
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
     * 生成代码专项问答数据（纯CODING任务）
     * 用于强化MoE专家对代码任务的特化能力
     * 
     * 同样采用 ChatML 对话模板格式，所有数据的任务类型均为 CODING
     */
    private static List<String> generateCodeQA() {
        List<String> codeQA = new ArrayList<>();
        
        // Python相关问答 (5条)
        codeQA.add("[CODING] <|im_start|>user\nHow to define a function in Python?<|im_end|>\n<|im_start|>assistant\nUse def keyword followed by function name parentheses for parameters and colon then indent the function body<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is list comprehension in Python?<|im_end|>\n<|im_start|>assistant\nList comprehension provides concise syntax to create lists using bracket notation with for loop and optional if condition<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to handle exceptions in Python?<|im_end|>\n<|im_start|>assistant\nUse try block for code that may raise exceptions except block to catch and handle specific exception types and finally for cleanup<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat are Python decorators?<|im_end|>\n<|im_start|>assistant\nDecorators are functions that modify behavior of other functions using at symbol syntax wrapping the original function with additional functionality<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to read files in Python?<|im_end|>\n<|im_start|>assistant\nUse open function with file path and mode then read content with read or readlines method always close file or use with statement<|im_end|>");
        
        // Java相关问答 (5条)
        codeQA.add("[CODING] <|im_start|>user\nHow to declare a class in Java?<|im_end|>\n<|im_start|>assistant\nUse public class keyword followed by class name then curly braces containing fields methods and constructors<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is inheritance in Java?<|im_end|>\n<|im_start|>assistant\nInheritance allows a class to inherit fields and methods from parent class using extends keyword promoting code reuse<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to create an interface in Java?<|im_end|>\n<|im_start|>assistant\nUse interface keyword followed by name declare method signatures without implementation classes implement interface with implements keyword<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is Java generics?<|im_end|>\n<|im_start|>assistant\nGenerics enable types to be parameters when defining classes interfaces and methods providing compile time type safety<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to handle null in Java?<|im_end|>\n<|im_start|>assistant\nCheck for null before dereferencing use Optional class or annotations like NonNull to prevent null pointer exceptions<|im_end|>");
        
        // JavaScript相关问答 (5条)
        codeQA.add("[CODING] <|im_start|>user\nWhat are arrow functions in JavaScript?<|im_end|>\n<|im_start|>assistant\nArrow functions provide shorter syntax using arrow notation capture this from enclosing scope unlike regular functions<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to handle async operations in JavaScript?<|im_end|>\n<|im_start|>assistant\nUse promises with then and catch or async await syntax for cleaner asynchronous code handling<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is closure in JavaScript?<|im_end|>\n<|im_start|>assistant\nClosure is a function that remembers variables from its outer scope even after outer function has finished executing<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nHow to iterate arrays in JavaScript?<|im_end|>\n<|im_start|>assistant\nUse for loop forEach map filter or for of loop depending on whether you need transformation filtering or simple iteration<|im_end|>");
        codeQA.add("[CODING] <|im_start|>user\nWhat is destructuring in JavaScript?<|im_end|>\n<|im_start|>assistant\nDestructuring extracts values from arrays or objects into distinct variables using bracket or curly brace syntax<|im_end|>");
        
        // C++相关问答 (5条)
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
    
    /**
     * 将文本列表写入文件
     */
    private static void writeToFile(List<String> texts, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String text : texts) {
                writer.write(text);
                writer.newLine();
            }
        }
    }
    
    /**
     * 执行预训练
     */
    private static DeepSeekV3Model runPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: DeepSeek-V3 预训练 (Pretrain)");
        System.out.println("=".repeat(80));
        
        // 1. 读取所有数据用于构建完整词汇表
        System.out.println("\n📝 加载所有数据以构建词汇表...");
        String pretrainPath = DATA_DIR + "/pretrain.txt";
        String posttrainTrainPath = DATA_DIR + "/posttrain_train.txt";
        String posttrainValPath = DATA_DIR + "/posttrain_val.txt";
        
        List<String> pretrainTexts = readFromFile(pretrainPath);
        List<String> posttrainTrainTexts = readFromFile(posttrainTrainPath);
        List<String> posttrainValTexts = readFromFile(posttrainValPath);
        
        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练训练数据: " + posttrainTrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练验证数据: " + posttrainValTexts.size() + " 条");
        
        // 2. 基于所有数据构建完整词汇表
        System.out.println("\n📝 构建完整词汇表...");
        List<String> allTexts = new ArrayList<>();
        allTexts.addAll(pretrainTexts);
        allTexts.addAll(posttrainTrainTexts);
        allTexts.addAll(posttrainValTexts);
        
        // 遍历所有文本构建词汇表
        for (String text : allTexts) {
            // 移除任务标签后再编码
            String cleanText = removeTaskLabel(text);
            sharedTokenizer.encode(cleanText);
        }
        int vocabSize = sharedTokenizer.getVocabSize();
        
        // 冻结词汇表
        sharedTokenizer.freeze();
        
        System.out.println("  ✓ 完整词汇表大小: " + vocabSize);
        System.out.println("  ✓ 词汇表已冻结,后续不再增加新词");
        
        // 3. 创建DeepSeek-V3模型
        System.out.println("\n📝 创建DeepSeek-V3模型...");
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        config.setVocabSize(vocabSize);
        
        DeepSeekV3Model model = new DeepSeekV3Model("deepseek-v3-pretrain-v2", config);
        
        System.out.println("  ✓ 模型配置: Micro (教学专用)");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getNEmbd());
        System.out.println("  ✓ 层数: " + config.getNLayer());
        System.out.println("  ✓ 注意力头数: " + config.getNHead());
        System.out.println("  ✓ 专家数量: " + config.getNumExperts());
        System.out.println("  ✓ Top-K选择: " + config.getTopK());
        System.out.println("  ✓ 序列长度: " + config.getNPositions());
        
        // 4. 准备数据集
        System.out.println("\n📝 准备训练数据集...");
        DeepSeekV3Dataset dataset = createDatasetFromTexts(
            pretrainTexts,
            config.getNPositions(),
            4,  // batch size
            config.getVocabSize(),
            false  // 预训练不使用任务标签
        );
        
        System.out.println("  ✓ 训练样本: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次大小: 4");
        System.out.println("  ✓ 序列长度: " + config.getNPositions());
        
        // 5. 配置训练器
        System.out.println("\n📝 配置预训练器...");
        DeepSeekV3Pretrain trainer = new DeepSeekV3Pretrain(model, dataset);
        // 修复：增加训练轮次，减少warmupSteps以适应小数据集
        trainer.configure(
            10,         // maxEpochs (增加到10轮，小数据集需要更多轮次)
            5e-4f,      // learningRate (提高学习率)
            10,         // warmupSteps (减少warmup步数，数据少时需快速进入正常训练)
            1.0f        // maxGradNorm
        ).setCheckpoint(CHECKPOINT_DIR + "/pretrain", 500);
        
        System.out.println("  ✓ 最大轮次: 10 (小数据集需要更多轮次)");
        System.out.println("  ✓ 学习率: 5e-4 (提高学习率加速收敛)");
        System.out.println("  ✓ Warmup步数: 10 (减少warmup适应小数据集)");
        System.out.println("  ✓ MoE负载均衡权重: " + config.getLoadBalanceLossWeight());
        
        // 6. 开始训练
        System.out.println("\n📝 开始预训练...");
        System.out.println("-".repeat(80));
        trainer.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 预训练完成!");
        System.out.println("\n💡 预训练阶段总结:");
        System.out.println("  - 目标: 学习语言的通用表示和MoE路由");
        System.out.println("  - 任务: 因果语言建模 + MoE负载均衡");
        System.out.println("  - 数据: 大规模无标注文本");
        System.out.println("  - 特色: 稀疏激活(25%参数) + 专家网络");
        
        return model;
    }
    
    /**
     * 执行后训练/微调
     */
    private static DeepSeekV3Model runPosttraining(DeepSeekV3Model pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: DeepSeek-V3 后训练/微调 (Posttrain)");
        System.out.println("=".repeat(80));
        
        // 1. 加载后训练数据
        System.out.println("\n📝 加载后训练数据...");
        String trainPath = DATA_DIR + "/posttrain_train.txt";
        String valPath = DATA_DIR + "/posttrain_val.txt";
        
        List<String> trainTexts = readFromFile(trainPath);
        List<String> valTexts = readFromFile(valPath);
        
        System.out.println("  ✓ 训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 验证集: " + valTexts.size() + " 条");
        
        // 2. 准备数据集（ChatML 对话模板 + answer-only loss mask）
        System.out.println("\n📝 准备后训练数据集（ChatML 格式 + Answer-Only Loss）...");
        DeepSeekV3Config config = pretrainedModel.getConfig();
        
        DeepSeekV3Dataset trainDataset = createDatasetFromTexts(
            trainTexts,
            config.getNPositions(),
            2,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签 + ChatML 格式 + loss mask
        );
        
        DeepSeekV3Dataset valDataset = createDatasetFromTexts(
            valTexts,
            config.getNPositions(),
            1,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签 + ChatML 格式 + loss mask
        );
        
        System.out.println("  ✓ 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + valDataset.getSampleCount());
        System.out.println("  ✓ 数据格式: ChatML 对话模板 (<|im_start|>user/assistant<|im_end|>)");
        System.out.println("  ✓ Loss策略: Answer-Only (仅对 assistant 回答部分计算 loss)");
        
        // 3. 配置后训练器
        System.out.println("\n📝 配置后训练器...");
        DeepSeekV3Posttrain posttrain = new DeepSeekV3Posttrain(
            pretrainedModel,
            trainDataset,
            valDataset
        );
        
        // 修复：增加训练轮次以适应小数据集
        posttrain.configure(
            5,          // maxEpochs (增加轮次)
            5e-5f,      // learningRate (适当提高)
            3           // patience (增加耐心值)
        );
        
        System.out.println("  ✓ 最大轮次: 5 (小数据集需要更多轮次)");
        System.out.println("  ✓ 学习率: 5e-5 (适当提高)");
        System.out.println("  ✓ 早停耐心值: 3");
        
        // 4. 开始后训练
        System.out.println("\n📝 开始后训练...");
        System.out.println("-".repeat(80));
        posttrain.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 后训练完成!");
        System.out.println("\n💡 后训练阶段总结:");
        System.out.println("  - 目标: 适应任务特定的推理和生成");
        System.out.println("  - 任务: 任务感知的指令跟随（ChatML 对话格式）");
        System.out.println("  - 数据: ChatML 模板 + 任务标签（行业主流 SFT 格式）");
        System.out.println("  - Loss: Answer-Only Loss Masking（仅对回答部分计算损失）");
        System.out.println("  - 技巧: 小学习率 + 早停 + 任务路由");
        System.out.println("  - 结果: 模型获得任务感知能力");
        System.out.println("\n💡 与预训练的关键区别:");
        System.out.println("  - 预训练: 纯文本 → 全序列 CLM loss");
        System.out.println("  - 后训练: ChatML 对话模板 → Answer-Only loss (只学回答)");
        
        return pretrainedModel;
    }
    
    /**
     * 执行代码生成专项后训练
     * 纯代码任务数据，强化MoE专家对代码任务的特化能力
     */
    private static DeepSeekV3Model runCodePosttraining(DeepSeekV3Model finetunedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("💻 步骤2B: DeepSeek-V3 代码生成专项后训练");
        System.out.println("=".repeat(80));
        System.out.println("💡 目标：强化MoE专家对代码任务的特化能力");
        System.out.println("💡 策略：纯代码任务数据 + 更小学习率 + 更多训练轮次");
        
        // 1. 加载代码专项数据
        System.out.println("\n📝 加载代码专项数据...");
        String codeTrainPath = DATA_DIR + "/code_posttrain_train.txt";
        String codeValPath = DATA_DIR + "/code_posttrain_val.txt";
        
        List<String> codeTrainTexts = readFromFile(codeTrainPath);
        List<String> codeValTexts = readFromFile(codeValPath);
        
        System.out.println("  ✓ 代码专项训练集: " + codeTrainTexts.size() + " 条");
        System.out.println("  ✓ 代码专项验证集: " + codeValTexts.size() + " 条");
        System.out.println("  ✓ 任务类型: 纯CODING (所有数据都是代码任务)");
        
        // 2. 准备数据集（纯CODING任务）
        System.out.println("\n📝 准备代码专项数据集...");
        DeepSeekV3Config config = finetunedModel.getConfig();
        
        DeepSeekV3Dataset codeTrainDataset = createDatasetFromTexts(
            codeTrainTexts,
            config.getNPositions(),
            2,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签（全是CODING）
        );
        
        DeepSeekV3Dataset codeValDataset = createDatasetFromTexts(
            codeValTexts,
            config.getNPositions(),
            1,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签
        );
        
        System.out.println("  ✓ 训练样本: " + codeTrainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + codeValDataset.getSampleCount());
        System.out.println("  ✓ 数据格式: ChatML 对话模板 + Answer-Only Loss");
        System.out.println("  ✓ 支持语言: Python, Java, JavaScript, C++");
        
        // 3. 配置代码专项后训练器
        System.out.println("\n📝 配置代码专项后训练器...");
        DeepSeekV3Posttrain codePosttrain = new DeepSeekV3Posttrain(
            finetunedModel,
            codeTrainDataset,
            codeValDataset
        );
        
        // 代码任务使用更小的学习率和更多的轮次
        codePosttrain.configure(
            6,          // maxEpochs (代码任务需要更多轮次)
            2e-5f,      // learningRate (适当调整)
            3           // patience
        );
        
        System.out.println("  ✓ 最大轮次: 6 (代码任务需要更多轮次)");
        System.out.println("  ✓ 学习率: 2e-5 (适当调整)");
        System.out.println("  ✓ 早停耐心值: 3");
        System.out.println("  ✓ 专家特化: 持续激活CODING专家");
        
        // 4. 开始代码专项后训练
        System.out.println("\n📝 开始代码生成专项后训练...");
        System.out.println("-".repeat(80));
        codePosttrain.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 代码专项后训练完成!");
        System.out.println("\n💡 代码专项后训练阶段总结:");
        System.out.println("  - 目标: 强化MoE专家对代码任务的特化");
        System.out.println("  - 任务: 纯CODING任务的指令跟随（ChatML 格式）");
        System.out.println("  - 数据: 25条代码生成 ChatML 问答 (Python/Java/JS/C++)");
        System.out.println("  - Loss: Answer-Only Loss Masking（仅对代码回答部分计算损失）");
        System.out.println("  - 特色: 持续激活同一批专家 -> 专家特化能力增强");
        System.out.println("  - 结果: 模型获得更强的代码生成能力");
        System.out.println("\nℹ️ MoE专家特化说明:");
        System.out.println("  - 通用后训练: CODING数据仅占~20%, 专家激活模式混合");
        System.out.println("  - 代码专项训练: CODING数据100%, 持续强化特定专家");
        System.out.println("  - 预期效果: Expert 2,5成为代码专家, CODING任务时激活概率大幅提升");
        
        return finetunedModel;
    }
    
    /**
     * 执行RLHF训练（人类反馈强化学习）
     */
    private static DeepSeekV3Model runRLHFTraining(DeepSeekV3Model model) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤3: DeepSeek-V3 RLHF训练 - 人类反馈强化学习");
        System.out.println("=".repeat(80));
        System.out.println("💡 目标：通过人类反馈对齐模型行为");
        System.out.println("💡 算法：Reward-weighted Regression（奖励加权回归）");
        
        // 1. 加载RLHF数据
        System.out.println("\n📝 加载RLHF训练数据...");
        String rlhfPath = DATA_DIR + "/rlhf_train.txt";
        List<String> rlhfTexts = readFromFile(rlhfPath);
        
        System.out.println("  ✓ RLHF样本: " + rlhfTexts.size() + " 条");
        
        // 2. 解析RLHF数据并构建数据集
        System.out.println("\n📝 准备RLHF数据集...");
        DeepSeekV3Config config = model.getConfig();
        
        DeepSeekV3Dataset rlhfDataset = createRLHFDatasetFromTexts(
            rlhfTexts,
            config.getNPositions(),
            2,  // batch size
            config.getVocabSize()
        );
        
        System.out.println("  ✓ RLHF训练样本: " + rlhfDataset.getSampleCount());
        System.out.println("  ✓ Loss Mask: 启用（Answer-only Loss）");
        System.out.println("  ✓ 数据格式: ChatML 模板 + 奖励标注");
        
        // 3. 配置RLHF训练器
        System.out.println("\n📝 配置RLHF训练器（Reward-weighted Regression）...");
        DeepSeekV3RLHFTrainer rlhfTrainer = new DeepSeekV3RLHFTrainer(
            model,
            rlhfDataset
        );
        
        rlhfTrainer.configure(
            2,          // maxEpochs
            5e-4f,      // learningRate
            1.0f,       // rewardWeight
            0.5f        // moeQualityWeight
        );
        
        System.out.println("  ✓ 最大轮次: 2");
        System.out.println("  ✓ 学习率: 5e-4");
        System.out.println("  ✓ 算法: Reward-weighted Regression");
        
        // 4. 开始RLHF训练
        System.out.println("\n📝 开始RLHF强化学习训练...");
        System.out.println("-".repeat(80));
        rlhfTrainer.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ RLHF训练完成!");
        System.out.println("\n💡 RLHF阶段总结:");
        System.out.println("  - 目标: 通过人类反馈对齐模型行为");
        System.out.println("  - 算法: Reward-weighted Regression（奖励加权回归）");
        System.out.println("  - 数据: ChatML 模板 + 奖励标注（高/中/低三档）");
        System.out.println("  - 技巧: Answer-only Loss Mask + 奖励加权梯度");
        System.out.println("  - 效果: 高奖励样本被强化，低奖励样本被弱化");
        System.out.println("  - 结果: 模型更符合人类偏好，回答更礼貌、更安全");
        
        return model;
    }
    
    /**
     * 从RLHF文本创建数据集
     * 解析奖励标注和ChatML内容，构建带loss mask和rewards的数据集
     */
    private static DeepSeekV3Dataset createRLHFDatasetFromTexts(
            List<String> texts,
            int maxSeqLength,
            int batchSize,
            int vocabSize) {
        
        List<int[]> sequences = new ArrayList<>();
        List<float[]> lossMasks = new ArrayList<>();
        List<float[]> rewards = new ArrayList<>();
        
        for (String text : texts) {
            // 解析奖励标注
            float reward = 0.5f;  // 默认奖励
            String content = text;
            
            // 提取奖励标注 [REWARD:0.x]
            int rewardStart = text.indexOf("[REWARD:");
            if (rewardStart >= 0) {
                int rewardEnd = text.indexOf("]", rewardStart);
                if (rewardEnd > rewardStart) {
                    String rewardStr = text.substring(rewardStart + 8, rewardEnd);
                    try {
                        reward = Float.parseFloat(rewardStr);
                    } catch (NumberFormatException e) {
                        reward = 0.5f;
                    }
                    content = text.substring(rewardEnd + 1).trim();
                }
            }
            
            // 解析ChatML内容并构建loss mask
            if (content.contains("<|im_start|>assistant")) {
                int assistantContentStart = content.indexOf("<|im_start|>assistant");
                String userPart = content.substring(0, assistantContentStart);
                String assistantPart = content.substring(assistantContentStart);
                
                // 分别编码
                List<Integer> userTokens = sharedTokenizer.encode(userPart);
                List<Integer> assistantTokens = sharedTokenizer.encode(assistantPart);
                
                // 合并 token 序列
                List<Integer> allTokens = new ArrayList<>(userTokens);
                allTokens.addAll(assistantTokens);
                int[] sequence = allTokens.stream().mapToInt(Integer::intValue).toArray();
                
                // 构建 loss mask：user 部分为 0.0，assistant 部分为 1.0
                float[] mask = new float[maxSeqLength];
                int userLen = userTokens.size();
                int totalLen = Math.min(allTokens.size(), maxSeqLength);
                for (int j = 0; j < totalLen; j++) {
                    mask[j] = (j >= userLen) ? 1.0f : 0.0f;
                }
                
                // Padding
                int[] paddedSeq = new int[maxSeqLength];
                Arrays.fill(paddedSeq, SimpleTokenizer.PAD_TOKEN_ID);
                int copyLen = Math.min(sequence.length, maxSeqLength);
                System.arraycopy(sequence, 0, paddedSeq, 0, copyLen);
                
                sequences.add(paddedSeq);
                lossMasks.add(mask);
                
                // 添加奖励
                float[] rewardArray = new float[1];
                rewardArray[0] = reward;
                rewards.add(rewardArray);
            }
        }
        
        // 构建任务类型列表（默认为GENERAL）
        List<TaskType> taskTypeList = new ArrayList<>();
        for (int i = 0; i < sequences.size(); i++) {
            taskTypeList.add(TaskType.GENERAL);
        }
        
        // 转换奖励列表
        List<Float> rewardList = new ArrayList<>();
        for (float[] rewardArray : rewards) {
            rewardList.add(rewardArray[0]);
        }
        
        // 创建RLHF数据集
        return new DeepSeekV3Dataset(
            sequences,
            taskTypeList,
            lossMasks,
            rewardList,
            maxSeqLength,
            batchSize,
            true  // shuffle
        );
    }
    
    /**
     * 执行推理测试
     */
    private static void runInference(DeepSeekV3Model model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤4: DeepSeek-V3 推理与文本生成");
        System.out.println("=".repeat(80));
        
        // 1. 创建推理器
        System.out.println("\n📝 创建推理器...");
        DeepSeekV3Inference inference = new DeepSeekV3Inference(model);
        inference.setSeed(42);
        System.out.println("  ✓ 推理器准备完成");
        
        // 2. 测试用例
        TestCase[] testCases = {
            new TestCase("Mixture of Experts is"),
            new TestCase("DeepSeek V3 combines"),
            new TestCase("Python is used for"),
            new TestCase("Self attention computes")
        };
        
        System.out.println("\n📝 执行文本生成测试...\n");
        
        for (int i = 0; i < testCases.length; i++) {
            TestCase testCase = testCases[i];
            System.out.println("测试 " + (i + 1) + ": \"" + testCase.prompt + "\"");
            System.out.println("-".repeat(80));
            
            try {
                List<Integer> tokens = sharedTokenizer.encode(testCase.prompt);
                int[] promptIds = tokens.stream().mapToInt(Integer::intValue).toArray();
                
                // Greedy解码
                System.out.println("  策略1 [Greedy贪婪]: ");
                var greedyResult = inference.generateGreedy(promptIds, 12);
                String greedyText = sharedTokenizer.decode(greedyResult.tokens);
                System.out.println("    → " + greedyText);
                
                // Temperature采样
                System.out.println("  策略2 [Temperature=0.8]: ");
                var tempResult = inference.generateWithTemperature(
                    promptIds, 12, 0.8f
                );
                String tempText = sharedTokenizer.decode(tempResult.tokens);
                System.out.println("    → " + tempText);
                
                // Top-K采样
                System.out.println("  策略3 [Top-K=50]: ");
                var topKResult = inference.generateTopK(promptIds, 12, 50);
                String topKText = sharedTokenizer.decode(topKResult.tokens);
                System.out.println("    → " + topKText);
                
            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }
            
            System.out.println();
        }
        
        System.out.println("✅ 推理测试完成!");
        System.out.println("\n💡 推理阶段总结:");
        System.out.println("  - 输入: 提示词");
        System.out.println("  - 处理: 自回归生成");
        System.out.println("  - 输出: 生成的完整文本");
        System.out.println("  - 策略: Greedy/Temperature/TopK/TopP");
        System.out.println("  - 特色: MoE稀疏激活");
    }
    
    /**
     * 从文本创建数据集
     * 
     * 当 useTaskLabels=true 时，解析 ChatML 对话模板格式，并生成 answer-only loss mask：
     * - user 提问部分和特殊标记的 loss mask 为 0.0（不参与 loss 计算）
     * - assistant 回答部分的 loss mask 为 1.0（参与 loss 计算）
     * 
     * 当 useTaskLabels=false 时（预训练模式），不生成 loss mask，所有位置都参与 loss 计算。
     */
    private static DeepSeekV3Dataset createDatasetFromTexts(
            List<String> texts,
            int maxSeqLength,
            int batchSize,
            int vocabSize,
            boolean useTaskLabels) {
        
        List<int[]> sequences = new ArrayList<>();
        List<float[]> lossMasks = new ArrayList<>();
        
        for (String text : texts) {
            String cleanText = text;
            
            if (useTaskLabels) {
                // 移除任务标签
                cleanText = removeTaskLabel(text);
            }
            
            if (useTaskLabels && cleanText.contains("<|im_start|>assistant")) {
                // ChatML 格式：分别编码 user 和 assistant 部分，生成 loss mask
                int assistantContentStart = cleanText.indexOf("<|im_start|>assistant");
                if (assistantContentStart < 0) {
                    // 防御性检查：如果找不到 assistant 标记，按普通文本处理
                    List<Integer> tokens = sharedTokenizer.encode(cleanText);
                    int[] sequence = tokens.stream().mapToInt(Integer::intValue).toArray();
                    int[] paddedSeq = new int[maxSeqLength];
                    Arrays.fill(paddedSeq, SimpleTokenizer.PAD_TOKEN_ID);
                    int copyLen = Math.min(sequence.length, maxSeqLength);
                    System.arraycopy(sequence, 0, paddedSeq, 0, copyLen);
                    sequences.add(paddedSeq);
                    continue;
                }
                String userPart = cleanText.substring(0, assistantContentStart);
                String assistantPart = cleanText.substring(assistantContentStart);
                
                // 分别编码
                List<Integer> userTokens = sharedTokenizer.encode(userPart);
                List<Integer> assistantTokens = sharedTokenizer.encode(assistantPart);
                
                // 合并 token 序列
                List<Integer> allTokens = new ArrayList<>(userTokens);
                allTokens.addAll(assistantTokens);
                int[] sequence = allTokens.stream().mapToInt(Integer::intValue).toArray();
                
                // 构建 loss mask：user 部分为 0.0，assistant 部分为 1.0
                float[] mask = new float[maxSeqLength];
                int userLen = userTokens.size();
                int totalLen = Math.min(allTokens.size(), maxSeqLength);
                for (int j = 0; j < totalLen; j++) {
                    mask[j] = (j >= userLen) ? 1.0f : 0.0f;
                }
                // padding 部分保持 0.0（数组默认值）
                
                // 截断或填充到 maxSeqLength
                int[] paddedSeq = new int[maxSeqLength];
                Arrays.fill(paddedSeq, SimpleTokenizer.PAD_TOKEN_ID);
                int copyLen = Math.min(sequence.length, maxSeqLength);
                System.arraycopy(sequence, 0, paddedSeq, 0, copyLen);
                
                sequences.add(paddedSeq);
                lossMasks.add(mask);
            } else {
                // 预训练模式或非 ChatML 格式：整条文本编码，无 loss mask
                List<Integer> tokens = sharedTokenizer.encode(cleanText);
                int[] sequence = tokens.stream().mapToInt(Integer::intValue).toArray();
                
                int[] paddedSeq = new int[maxSeqLength];
                Arrays.fill(paddedSeq, SimpleTokenizer.PAD_TOKEN_ID);
                int copyLen = Math.min(sequence.length, maxSeqLength);
                System.arraycopy(sequence, 0, paddedSeq, 0, copyLen);
                
                sequences.add(paddedSeq);
            }
        }
        
        // 如果有 loss mask 数据，使用带 mask 的构造函数
        if (!lossMasks.isEmpty()) {
            return new DeepSeekV3Dataset(sequences, new ArrayList<>(), lossMasks,
                maxSeqLength, batchSize, true, true);
        }
        return new DeepSeekV3Dataset(sequences, maxSeqLength, batchSize, true);
    }
    
    /**
     * 移除任务标签
     */
    private static String removeTaskLabel(String text) {
        return text.replaceFirst("^\\[\\w+\\]\\s*", "");
    }
    
    /**
     * 从文件读取文本
     */
    private static List<String> readFromFile(String filePath) throws IOException {
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
    
    /**
     * 测试用例类
     */
    private static class TestCase {
        final String prompt;
        
        TestCase(String prompt) {
            this.prompt = prompt;
        }
    }
    
    /**
     * 简单分词器（类似GPT1的SimpleTokenizer）
     * 注意：id=0保留给PAD token，避免与词汇冲突
     */
    static class SimpleTokenizer {
        private final Map<String, Integer> vocab;
        private final Map<Integer, String> reverseVocab;
        private int nextId;
        private boolean frozen;
        
        /** PAD token的ID，用于填充 */
        public static final int PAD_TOKEN_ID = 0;
        
        public SimpleTokenizer() {
            this.vocab = new HashMap<>();
            this.reverseVocab = new HashMap<>();
            // id=0保留给PAD，词汇从1开始
            this.nextId = 1;
            this.frozen = false;
            // 预注册PAD token
            this.vocab.put("<PAD>", PAD_TOKEN_ID);
            this.reverseVocab.put(PAD_TOKEN_ID, "<PAD>");
        }
        
        public List<Integer> encode(String text) {
            String[] words = text.toLowerCase()
                .replaceAll("[^a-z0-9\\s]", " ")
                .split("\\s+");
            
            List<Integer> tokens = new ArrayList<>();
            for (String word : words) {
                if (word.isEmpty()) continue;
                
                if (!vocab.containsKey(word)) {
                    if (!frozen) {
                        vocab.put(word, nextId);
                        reverseVocab.put(nextId, word);
                        nextId++;
                    } else {
                        // 冻结后使用UNK token (使用id=1作为UNK，避免与PAD冲突)
                        tokens.add(1);
                        continue;
                    }
                }
                tokens.add(vocab.get(word));
            }
            return tokens;
        }
        
        public String decode(int[] tokens) {
            StringBuilder sb = new StringBuilder();
            for (int token : tokens) {
                // 跳过PAD token
                if (token == PAD_TOKEN_ID) continue;
                if (reverseVocab.containsKey(token)) {
                    if (sb.length() > 0) sb.append(" ");
                    sb.append(reverseVocab.get(token));
                }
            }
            return sb.toString();
        }
        
        public int getVocabSize() {
            return nextId;
        }
        
        public void freeze() {
            this.frozen = true;
        }
    }
}
