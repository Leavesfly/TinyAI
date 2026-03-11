package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * GPT-1完整训练演示 V2版本
 * <p>
 * 改进点:
 * 1. 更大规模的教学数据集(pretrain和posttrain)
 * 2. 支持从文件加载数据
 * 3. 数据集自动生成功能
 * 4. 更详细的训练过程说明
 * 5. 完整的预训练-微调-推理流程
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT1TrainDemo {

    private static GPT1Dataset.SimpleTokenizer sharedTokenizer = new GPT1Dataset.SimpleTokenizer();

    private static final String DATA_DIR = "./data/gpt1_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/gpt1_v2";

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("GPT-1 完整训练与推理演示 V2");
        System.out.println("适用于教学和学习的小型数据集训练方案");
        System.out.println("=".repeat(80));

        try {
            // 步骤0: 准备数据集文件
            prepareDatasets();

            // 步骤1: 预训练
            GPT1Model pretrainedModel = runPretraining();

            // 步骤2: 微调
            GPT1Model finetunedModel = runFinetuning(pretrainedModel);

            // 步骤3: 推理测试
            runInference(finetunedModel);

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

        // 生成微调数据集
        generateFinetuneDataset();

        System.out.println("\n✅ 数据集准备完成!");
    }

    /**
     * 生成预训练数据集
     * 包含深度学习、NLP、Transformer等多个领域的教学文本
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");

        List<String> pretrainTexts = new ArrayList<>();

        // 1. 深度学习基础知识
        pretrainTexts.addAll(generateDeepLearningTexts());

        // 2. 机器学习概念
        pretrainTexts.addAll(generateMachineLearningTexts());

        // 3. 神经网络架构
        pretrainTexts.addAll(generateNeuralNetworkTexts());

        // 4. NLP和语言模型
        pretrainTexts.addAll(generateNLPTexts());

        // 5. Transformer和注意力机制
        pretrainTexts.addAll(generateTransformerTexts());

        // 6. TinyAI框架知识
        pretrainTexts.addAll(generateTinyAITexts());

        // 7. 数学基础
        pretrainTexts.addAll(generateMathematicsTexts());

        // 8. 编程概念
        pretrainTexts.addAll(generateProgrammingTexts());

        // 9. AI历史与发展
        pretrainTexts.addAll(generateAIHistoryTexts());

        // 10. 优化算法
        pretrainTexts.addAll(generateOptimizationTexts());

        // 11. 计算机视觉
        pretrainTexts.addAll(generateComputerVisionTexts());

        // 12. 强化学习
        pretrainTexts.addAll(generateReinforcementLearningTexts());

        // 13. 数据处理
        pretrainTexts.addAll(generateDataProcessingTexts());

        // 14. 模型评估
        pretrainTexts.addAll(generateModelEvaluationTexts());

        // 15. 分布式训练
        pretrainTexts.addAll(generateDistributedTrainingTexts());

        // 写入文件
        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);

        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    /**
     * 生成微调数据集
     * 包含指令-回答对,用于训练问答能力
     */
    private static void generateFinetuneDataset() throws IOException {
        System.out.println("\n📝 生成微调数据集...");

        List<String> trainTexts = new ArrayList<>();
        List<String> valTexts = new ArrayList<>();

        // 训练集: 200条指令-回答对
        trainTexts.addAll(generateInstructionQA());

        // 验证集: 从训练集中抽取20条
        for (int i = 0; i < 20 && i < trainTexts.size(); i++) {
            valTexts.add(trainTexts.get(i));
        }

        // 写入训练集
        String trainPath = DATA_DIR + "/finetune_train.txt";
        writeToFile(trainTexts, trainPath);
        System.out.println("  ✓ 微调训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + trainPath);

        // 写入验证集
        String valPath = DATA_DIR + "/finetune_val.txt";
        writeToFile(valTexts, valPath);
        System.out.println("  ✓ 微调验证集: " + valTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + valPath);
    }

    /**
     * 生成深度学习相关文本
     */
    private static List<String> generateDeepLearningTexts() {
        List<String> texts = new ArrayList<>();
        
        // 深度学习基础概念
        texts.addAll(Arrays.asList(
                "Deep learning is a subset of machine learning that uses neural networks with multiple layers",
                "Deep learning models can automatically learn hierarchical representations from raw data",
                "Deep learning has revolutionized fields like computer vision and natural language processing",
                "Convolutional neural networks are the foundation of modern computer vision systems",
                "Recurrent neural networks excel at processing sequential data like text and time series",
                "Deep learning requires large amounts of labeled data for effective training",
                "Transfer learning allows deep learning models to reuse knowledge from related tasks",
                "Backpropagation is the fundamental algorithm for training deep neural networks",
                "Gradient descent optimizes neural network parameters by minimizing the loss function",
                "Deep learning frameworks like TensorFlow and PyTorch simplify model development",
                "Activation functions introduce non linearity into neural network computations",
                "Dropout is a regularization technique that prevents overfitting in deep networks",
                "Batch normalization accelerates training and improves model generalization",
                "Deep learning models can achieve superhuman performance on specific tasks",
                "Pretrained models enable rapid development of AI applications with limited data",
                "Deep neural networks can approximate any continuous function given enough capacity",
                "Layer normalization stabilizes training in recurrent and transformer architectures",
                "Residual connections help train very deep neural networks by avoiding vanishing gradients",
                "Deep learning has enabled breakthroughs in speech recognition and synthesis",
                "Attention mechanisms allow models to focus on relevant parts of the input"
        ));
        
        // 网络架构
        texts.addAll(Arrays.asList(
                "Deep learning systems learn features automatically without manual feature engineering",
                "Convolutional layers extract spatial hierarchies from images and visual data",
                "Pooling layers reduce spatial dimensions while preserving important features",
                "Deep architectures can learn abstract concepts through multiple levels of representation",
                "Neural networks trained on large datasets show emergent abilities on new tasks",
                "Autoencoders learn compressed representations through encoder decoder architecture",
                "Variational autoencoders add probabilistic elements to learn smooth latent spaces",
                "Generative adversarial networks pit two networks against each other for data generation",
                "Discriminator networks in GANs learn to distinguish real from generated samples",
                "Generator networks learn to produce realistic samples that fool the discriminator",
                "Style transfer networks apply artistic styles to images using feature representations",
                "Image segmentation networks assign class labels to each pixel in an image",
                "Object detection networks identify and locate multiple objects in images",
                "Face recognition networks map facial images to compact identity embeddings",
                "Medical imaging networks assist in diagnosis by analyzing scans and x-rays"
        ));
        
        // 训练技术
        texts.addAll(Arrays.asList(
                "Data augmentation artificially expands training datasets through transformations",
                "Mixup augmentation creates new training samples by mixing pairs of examples",
                "Cutout augmentation randomly masks regions of input images during training",
                "Label smoothing prevents overconfidence by softening target distributions",
                "Warmup learning rate schedules gradually increase learning rate at training start",
                "Cosine annealing schedules decay learning rate following a cosine curve",
                "One cycle learning rate policy combines warmup and decay in a single cycle",
                "Gradient accumulation enables training with larger effective batch sizes",
                "Mixed precision training uses lower precision for faster computation",
                "Distributed data parallel training splits batches across multiple GPUs",
                "Model parallelism distributes large models across multiple devices",
                "Checkpoint activation trading recomputes activations to save memory",
                "Dynamic batching groups similar length sequences for efficient training",
                "Curriculum learning presents training examples in order of increasing difficulty",
                "Self supervised learning creates supervision signals from unlabeled data"
        ));
        
        // 损失函数
        texts.addAll(Arrays.asList(
                "Cross entropy loss measures the difference between predicted and true probability distributions",
                "Mean squared error loss penalizes large errors more heavily than small ones",
                "Binary cross entropy loss is used for binary classification tasks",
                "Focal loss addresses class imbalance by down weighting easy examples",
                "Triplet loss learns embeddings by comparing anchor positive and negative samples",
                "Contrastive loss learns representations by bringing similar samples closer",
                "Hinge loss is used for maximum margin classification in support vector machines",
                "Dice loss is effective for segmentation tasks with class imbalance",
                "IoU loss directly optimizes intersection over union for object detection",
                "Perceptual loss uses features from pretrained networks for image generation",
                "Adversarial loss from discriminator networks improves generator outputs",
                "Cycle consistency loss ensures transformations are reversible in cycle GANs",
                "Knowledge distillation loss transfers knowledge from teacher to student models",
                "Auxiliary loss helps train deep networks by adding supervision at intermediate layers",
                "Multi task learning shares representations across related tasks with combined losses"
        ));
        
        // 正则化技术
        texts.addAll(Arrays.asList(
                "L1 regularization adds absolute value penalty encouraging sparse weights",
                "L2 regularization adds squared penalty preventing large weight values",
                "Elastic net combines L1 and L2 regularization for balanced sparsity and stability",
                "Dropconnect randomly drops individual weights rather than entire neurons",
                "Spatial dropout drops entire feature maps for convolutional networks",
                "DropPath randomly drops entire paths in networks with skip connections",
                "Zoneout randomly preserves hidden states in recurrent networks",
                "ShakeDrop adds random perturbations during training of deep networks",
                "CutMix combines image patches and labels for regularization",
                "Stochastic depth randomly drops layers during training",
                "Label noise regularization adds random label perturbations",
                "Virtual adversarial training adds small perturbations to improve robustness",
                "Confidence penalty discourages overconfident predictions",
                "Early stopping halts training when validation performance degrades",
                "Weight decay is equivalent to L2 regularization in gradient descent"
        ));
        
        return texts;
    }

    /**
     * 生成机器学习相关文本
     */
    private static List<String> generateMachineLearningTexts() {
        List<String> texts = new ArrayList<>();
        
        // 基础概念
        texts.addAll(Arrays.asList(
                "Machine learning enables computers to learn patterns from data without explicit programming",
                "Supervised learning uses labeled examples to train predictive models",
                "Unsupervised learning discovers hidden patterns in unlabeled data",
                "Reinforcement learning agents learn optimal behaviors through trial and error",
                "Feature engineering transforms raw data into representations suitable for learning",
                "Cross validation assesses model performance and prevents overfitting",
                "Hyperparameter tuning optimizes model settings for best performance",
                "Ensemble methods combine multiple models to improve prediction accuracy",
                "Overfitting occurs when models memorize training data instead of learning patterns",
                "Regularization techniques like L1 and L2 prevent overfitting by constraining parameters",
                "The bias variance tradeoff balances model simplicity and flexibility",
                "Training data should be representative of the real world distribution",
                "Data augmentation increases dataset diversity through transformations",
                "Early stopping prevents overfitting by monitoring validation performance",
                "Learning curves visualize model performance as training data increases"
        ));
        
        // 分类算法
        texts.addAll(Arrays.asList(
                "Model evaluation requires separate test data never seen during training",
                "Classification assigns inputs to discrete categories or classes",
                "Regression predicts continuous numerical values from input features",
                "Clustering groups similar data points without predefined labels",
                "Dimensionality reduction simplifies data while preserving important information",
                "Decision trees partition feature space through recursive splits",
                "Random forests aggregate multiple decision trees for robust predictions",
                "Support vector machines find optimal hyperplanes for classification",
                "K nearest neighbors classifies based on proximity in feature space",
                "Principal component analysis reduces dimensions while retaining variance",
                "Logistic regression predicts probabilities for binary classification",
                "Naive Bayes uses probability theory with independence assumptions",
                "Linear discriminant analysis finds linear combinations for class separation",
                "Gradient boosting sequentially builds models to correct previous errors",
                "XGBoost is an optimized implementation of gradient boosting",
                "LightGBM uses histogram based algorithms for faster training",
                "CatBoost handles categorical features natively without preprocessing"
        ));
        
        // 聚类算法
        texts.addAll(Arrays.asList(
                "K means clustering partitions data into k groups by minimizing variance",
                "Hierarchical clustering builds a tree of nested clusters",
                "DBSCAN identifies clusters based on density of data points",
                "Gaussian mixture models represent clusters as probability distributions",
                "Spectral clustering uses graph theory to identify clusters",
                "Mean shift clustering finds modes in the data distribution",
                "Affinity propagation identifies exemplars for clustering",
                "Agglomerative clustering iteratively merges similar clusters",
                "Divisive clustering recursively splits clusters into smaller groups",
                "OPTICS extends DBSCAN to handle varying density clusters"
        ));
        
        // 降维技术
        texts.addAll(Arrays.asList(
                "t-SNE visualizes high dimensional data in 2D or 3D space",
                "UMAP preserves both local and global structure in embeddings",
                "Autoencoders learn compressed representations through bottleneck layers",
                "Independent component analysis separates mixed signals into components",
                "Factor analysis models observed variables as linear combinations of factors",
                "Non negative matrix factorization learns parts based representations",
                "Locally linear embedding preserves local relationships in data",
                "Isomap extends classical scaling to geodesic distances",
                "Random projection provides fast dimensionality reduction",
                "Linear discriminant analysis maximizes class separability"
        ));
        
        // 模型选择与评估
        texts.addAll(Arrays.asList(
                "Confusion matrix summarizes classification predictions versus actual labels",
                "Precision measures accuracy of positive predictions",
                "Recall measures coverage of actual positive cases",
                "F1 score balances precision and recall in a single metric",
                "ROC curves plot true positive rate against false positive rate",
                "AUC measures the area under the ROC curve for model comparison",
                "Precision recall curves are useful for imbalanced datasets",
                "Mean absolute error measures average prediction deviation",
                "Root mean squared error penalizes larger errors more heavily",
                "R squared measures the proportion of variance explained by the model",
                "Adjusted R squared accounts for the number of predictors in the model",
                "Silhouette score evaluates clustering quality without ground truth",
                "Calinski Harabasz index measures cluster separation and cohesion",
                "Davies Bouldin index quantifies cluster separation",
                "Cross validation provides robust estimates of model performance"
        ));
        
        return texts;
    }

    /**
     * 生成神经网络相关文本
     */
    private static List<String> generateNeuralNetworkTexts() {
        List<String> texts = new ArrayList<>();
        
        // 神经网络基础
        texts.addAll(Arrays.asList(
                "Neural networks consist of interconnected layers of artificial neurons",
                "Each neuron computes a weighted sum of inputs and applies an activation function",
                "The input layer receives raw features while the output layer produces predictions",
                "Hidden layers learn increasingly abstract representations of the input",
                "Feedforward networks process information in one direction from input to output",
                "Weights and biases are the learnable parameters of a neural network",
                "The sigmoid activation function maps values to a range between zero and one",
                "ReLU activation is computationally efficient and helps avoid vanishing gradients",
                "Softmax converts network outputs into a probability distribution over classes",
                "Loss functions measure the difference between predictions and true labels"
        ));
        
        // 激活函数详解
        texts.addAll(Arrays.asList(
                "Mean squared error is commonly used for regression tasks",
                "Cross entropy loss is standard for classification problems",
                "Stochastic gradient descent updates parameters using small batches of data",
                "Adam optimizer adapts learning rates for each parameter automatically",
                "Learning rate controls the step size during gradient descent optimization",
                "Mini batch training balances computational efficiency and gradient quality",
                "Vanishing gradients make it difficult to train very deep networks",
                "Exploding gradients can cause training instability and divergence",
                "Gradient clipping prevents exploding gradients by limiting their magnitude",
                "Neural networks can be viewed as universal function approximators",
                "Weight initialization affects training speed and convergence",
                "Momentum accelerates optimization by accumulating gradient history",
                "Leaky ReLU allows small gradients for negative inputs",
                "Tanh activation maps inputs to range between negative one and one",
                "Network depth enables learning of complex hierarchical features",
                "GELU activation combines properties of ReLU and sigmoid smoothly",
                "Swish activation is a self gated function that outperforms ReLU",
                "Mish activation is smooth and non monotonic for better gradient flow",
                "Parametric ReLU learns the negative slope during training",
                "ELU activation helps mitigate vanishing gradients for negative inputs"
        ));
        
        // 网络架构类型
        texts.addAll(Arrays.asList(
                "Multilayer perceptrons are the simplest form of deep neural networks",
                "Convolutional networks share weights across spatial locations",
                "Recurrent networks maintain hidden states across time steps",
                "LSTM networks use gates to control information flow in long sequences",
                "GRU simplifies LSTM with fewer parameters while maintaining performance",
                "Bidirectional RNNs process sequences in both forward and backward directions",
                "ResNet uses skip connections to train very deep networks",
                "DenseNet connects each layer to every other layer for feature reuse",
                "Wide ResNet increases width for better parallelization",
                "MobileNet uses depthwise separable convolutions for efficiency",
                "EfficientNet scales network dimensions in a principled way",
                "Attention networks dynamically weight different parts of the input",
                "Memory networks combine neural networks with external memory",
                "Neural Turing machines learn to read and write to external memory",
                "Graph neural networks process data structured as graphs"
        ));
        
        return texts;
    }

    /**
     * 生成NLP相关文本
     */
    private static List<String> generateNLPTexts() {
        List<String> texts = new ArrayList<>();
        
        // NLP基础
        texts.addAll(Arrays.asList(
                "Natural language processing enables computers to understand and generate human language",
                "Tokenization splits text into smaller units like words or subwords",
                "Word embeddings represent words as dense vectors in a continuous space",
                "Word2Vec learns embeddings by predicting context words from target words",
                "GloVe embeddings capture global statistical information from word co occurrences",
                "Language models estimate the probability distribution over sequences of words",
                "N gram models predict words based on previous n minus one words",
                "Neural language models use recurrent networks to capture long range dependencies",
                "Perplexity measures how well a language model predicts a test corpus",
                "Text classification assigns documents to predefined categories"
        ));
        
        // NLP任务
        texts.addAll(Arrays.asList(
                "Sentiment analysis determines the emotional tone of text",
                "Named entity recognition identifies and classifies entities in text",
                "Machine translation converts text from one language to another",
                "Question answering systems provide answers to natural language questions",
                "Text summarization creates concise summaries of longer documents",
                "Sequence to sequence models map input sequences to output sequences",
                "Encoder decoder architectures are fundamental to many NLP tasks",
                "Beam search finds high probability output sequences during generation",
                "Byte pair encoding creates subword vocabularies for neural models",
                "Contextualized embeddings like BERT capture word meaning based on context"
        ));
        
        // NLP高级技术
        texts.addAll(Arrays.asList(
                "Part of speech tagging identifies grammatical roles of words",
                "Dependency parsing analyzes syntactic structure of sentences",
                "Coreference resolution links pronouns to their referents",
                "Semantic role labeling identifies predicate argument structures",
                "Information extraction retrieves structured data from unstructured text",
                "Relation extraction identifies relationships between entities in text",
                "Event extraction identifies events and their participants from text",
                "Text generation produces fluent and coherent text sequences",
                "Dialogue systems enable conversational interaction with machines",
                "Reading comprehension systems answer questions based on given passages",
                "Semantic parsing maps natural language to formal meaning representations",
                "Text entailment determines if one text implies another",
                "Paraphrase detection identifies texts with equivalent meaning",
                "Text similarity measures semantic closeness between documents",
                "Keyword extraction identifies important terms in documents"
        ));
        
        // 语言模型技术
        texts.addAll(Arrays.asList(
                "GPT models generate text by predicting tokens autoregressively",
                "BERT learns bidirectional representations using masked language modeling",
                "XLNet combines autoregressive and autoencoding approaches",
                "RoBERTa improves BERT through optimized training procedures",
                "ALBERT reduces parameters through factorized embeddings",
                "T5 frames all NLP tasks as text to text transformations",
                "BART combines bidirectional encoding with autoregressive decoding",
                "ELECTRA uses replaced token detection for efficient pretraining",
                "SpanBERT extends BERT by masking contiguous spans of tokens",
                "Longformer uses sliding window attention for long documents",
                "BigBird combines random and local attention for longer sequences",
                "Reformer uses locality sensitive hashing for efficient attention",
                "Performer uses kernel approximations for linear attention",
                "Linformer projects keys and values to lower dimensions",
                "Sparse transformer uses patterns to reduce attention complexity"
        ));
        
        return texts;
    }

    /**
     * 生成Transformer相关文本
     */
    private static List<String> generateTransformerTexts() {
        List<String> texts = new ArrayList<>();
        
        // Transformer基础
        texts.addAll(Arrays.asList(
                "Transformer architecture revolutionized natural language processing in 2017",
                "Self attention allows models to weigh the importance of different input positions",
                "Multi head attention captures different aspects of relationships between tokens",
                "Positional encoding injects sequence order information into transformer models",
                "The transformer consists of an encoder and decoder with stacked layers",
                "Query key and value vectors are fundamental components of attention mechanisms",
                "Scaled dot product attention computes attention weights efficiently",
                "Feed forward networks process each position independently in transformers",
                "Layer normalization is applied before or after transformer sublayers",
                "Residual connections help train deep transformer models effectively"
        ));
        
        // Transformer变体
        texts.addAll(Arrays.asList(
                "BERT uses bidirectional transformers for language understanding tasks",
                "GPT models use decoder only transformers for text generation",
                "Masked language modeling is the pretraining objective for BERT",
                "Causal language modeling predicts the next token in autoregressive models",
                "Transformers eliminate recurrence and enable parallel processing of sequences",
                "Attention patterns can be visualized to understand model behavior",
                "The transformer model achieves state of the art results across NLP tasks",
                "Large scale pretraining of transformers requires significant computational resources",
                "Fine tuning adapts pretrained transformers to downstream tasks efficiently",
                "Transformer models scale effectively with increased data and parameters"
        ));
        
        // 注意力机制详解
        texts.addAll(Arrays.asList(
                "Cross attention connects encoder and decoder in sequence to sequence models",
                "Attention heads learn to focus on different linguistic phenomena",
                "Positional embeddings can be learned or defined using sinusoidal functions",
                "The transformer architecture is fully differentiable and trainable end to end",
                "Sparse attention patterns improve efficiency for long sequences",
                "Flash attention reduces memory access for faster attention computation",
                "Multi query attention shares keys and values across attention heads",
                "Grouped query attention balances efficiency and quality",
                "Sliding window attention limits attention to local neighborhoods",
                "Dilated attention expands receptive field with sparse patterns",
                "Memory efficient attention computes attention without storing full matrices",
                "Linear attention approximates softmax attention for efficiency",
                "Local attention restricts attention to nearby tokens",
                "Global attention allows special tokens to attend to all positions",
                "Axial attention decomposes attention along different dimensions"
        ));
        
        // 大模型技术
        texts.addAll(Arrays.asList(
                "Scaling laws predict model performance based on compute and data",
                "Emergent abilities appear in large language models unexpectedly",
                "In context learning allows models to learn from examples in prompts",
                "Chain of thought prompting improves reasoning step by step",
                "Instruction tuning trains models to follow human instructions",
                "RLHF uses human feedback to align models with human preferences",
                "Constitutional AI provides principles for model behavior",
                "Prompt engineering designs effective inputs for language models",
                "Few shot learning adapts to new tasks with minimal examples",
                "Zero shot learning generalizes to tasks without examples",
                "Retrieval augmented generation grounds model outputs in documents",
                "Tool use enables models to call external APIs and functions",
                "Agentic behavior allows models to plan and execute multi step tasks",
                "Speculative decoding speeds up generation with smaller draft models",
                "Quantization reduces model size by lowering parameter precision"
        ));
        
        return texts;
    }

    /**
     * 生成TinyAI框架相关文本
     */
    private static List<String> generateTinyAITexts() {
        List<String> texts = new ArrayList<>();
        
        // TinyAI核心组件
        texts.addAll(Arrays.asList(
                "TinyAI is a Java based deep learning framework for education and research",
                "NdArray is the core multidimensional array abstraction in TinyAI",
                "The autograd engine in TinyAI enables automatic differentiation",
                "Variable wraps NdArray and tracks computational graphs for backpropagation",
                "Module is the base class for all neural network layers in TinyAI",
                "Parameter represents learnable weights in TinyAI neural networks",
                "The forward method defines the computation performed by a module",
                "Backward propagation computes gradients through the computational graph",
                "SGD optimizer updates parameters using stochastic gradient descent",
                "Linear layer performs affine transformation of input features"
        ));
        
        // TinyAI层和组件
        texts.addAll(Arrays.asList(
                "Embedding layer maps discrete tokens to continuous vector representations",
                "MultiHeadAttention implements the attention mechanism for transformers",
                "LayerNorm normalizes activations to stabilize training",
                "Dropout randomly zeros elements during training for regularization",
                "SoftmaxCrossEntropy combines softmax and cross entropy for classification",
                "Dataset class handles data loading and batching in TinyAI",
                "Trainer orchestrates the training loop and model optimization",
                "Model class wraps modules and provides high level training interface",
                "TinyAI supports both CPU and GPU computation for neural networks",
                "The framework provides comprehensive examples for learning deep learning"
        ));
        
        // TinyAI架构设计
        texts.addAll(Arrays.asList(
                "Block abstraction allows composition of complex network architectures",
                "Loss functions quantify prediction errors during training",
                "Optimizer algorithms minimize loss by updating model parameters",
                "TinyAI implements common activation functions like ReLU and GELU",
                "The framework supports saving and loading trained model checkpoints",
                "Shape class represents tensor dimensions in TinyAI",
                "Operations like matmul and add are implemented for NdArray",
                "The computational graph tracks dependencies between operations",
                "Gradients flow backward through the graph via chain rule",
                "Parameter initialization methods include Xavier and Kaiming",
                "Learning rate schedulers adjust optimizer rates during training",
                "Regularization techniques prevent overfitting in TinyAI models",
                "The framework provides utilities for text processing and tokenization",
                "Model serialization enables saving and loading trained weights",
                "Training callbacks allow custom behavior at training events"
        ));
        
        return texts;
    }

    /**
     * 生成数学基础相关文本
     */
    private static List<String> generateMathematicsTexts() {
        List<String> texts = new ArrayList<>();
        
        // 线性代数
        texts.addAll(Arrays.asList(
                "Linear algebra is fundamental to machine learning and deep learning",
                "Vectors are one dimensional arrays of numbers used to represent data",
                "Matrices are two dimensional arrays used for linear transformations",
                "Matrix multiplication combines two matrices following specific rules",
                "The dot product measures similarity between two vectors",
                "Eigenvalues and eigenvectors characterize linear transformations",
                "Matrix decomposition breaks matrices into simpler components",
                "Singular value decomposition factorizes matrices into orthogonal components",
                "QR decomposition factorizes matrices into orthogonal and upper triangular",
                "LU decomposition factorizes matrices into lower and upper triangular"
        ));
        
        // 微积分
        texts.addAll(Arrays.asList(
                "Calculus provides tools for optimization in machine learning",
                "Derivatives measure the rate of change of functions",
                "Gradients are vectors of partial derivatives for multivariate functions",
                "The chain rule enables backpropagation through composite functions",
                "Partial derivatives measure change with respect to one variable",
                "Integration computes areas under curves and accumulations",
                "Jacobian matrices contain all first order partial derivatives",
                "Hessian matrices contain second order partial derivatives",
                "Taylor series approximate functions using polynomial expansions",
                "Gradient descent uses derivatives to find function minima"
        ));
        
        // 概率统计
        texts.addAll(Arrays.asList(
                "Probability theory provides foundations for machine learning",
                "Random variables represent outcomes of random processes",
                "Probability distributions describe likelihood of different outcomes",
                "The normal distribution is a bell shaped continuous distribution",
                "Bayes theorem relates conditional and marginal probabilities",
                "Expected value represents the average outcome of a random variable",
                "Variance measures spread around the expected value",
                "Covariance measures how two variables change together",
                "Maximum likelihood estimation finds parameters that best explain data",
                "Hypothesis testing determines if results are statistically significant"
        ));
        
        // 信息论
        texts.addAll(Arrays.asList(
                "Information theory provides measures for machine learning",
                "Entropy quantifies uncertainty in probability distributions",
                "Cross entropy measures difference between probability distributions",
                "Kullback Leibler divergence measures distance between distributions",
                "Mutual information measures dependence between random variables",
                "Information gain measures reduction in entropy from splitting data",
                "The softmax function converts logits to probability distributions",
                "Cross entropy loss is derived from information theory principles",
                "Bits and nats are units for measuring information",
                "Coding theory connects information theory to data compression"
        ));
        
        return texts;
    }

    /**
     * 生成编程概念相关文本
     */
    private static List<String> generateProgrammingTexts() {
        List<String> texts = new ArrayList<>();
        
        // 编程基础
        texts.addAll(Arrays.asList(
                "Programming is the process of writing instructions for computers",
                "Variables store data values that can be modified during execution",
                "Functions encapsulate reusable blocks of code",
                "Control flow determines the order of statement execution",
                "Loops repeat code blocks until a condition is met",
                "Conditional statements execute code based on boolean conditions",
                "Arrays store multiple values of the same type contiguously",
                "Objects encapsulate data and methods together",
                "Classes define blueprints for creating objects",
                "Inheritance allows classes to inherit properties from parent classes"
        ));
        
        // 数据结构
        texts.addAll(Arrays.asList(
                "Data structures organize and store data efficiently",
                "Linked lists connect nodes through references",
                "Stacks follow last in first out ordering",
                "Queues follow first in first out ordering",
                "Trees organize data in hierarchical structures",
                "Binary trees have at most two children per node",
                "Hash tables provide fast key value lookups",
                "Graphs represent relationships between entities",
                "Heaps maintain ordering for priority queue operations",
                "Tries store strings for efficient prefix matching"
        ));
        
        // 算法
        texts.addAll(Arrays.asList(
                "Algorithms are step by step procedures for solving problems",
                "Sorting algorithms arrange elements in order",
                "Searching algorithms find elements in data structures",
                "Dynamic programming solves problems by breaking them into subproblems",
                "Greedy algorithms make locally optimal choices at each step",
                "Divide and conquer splits problems into smaller subproblems",
                "Recursion solves problems by having functions call themselves",
                "Backtracking explores all possible solutions systematically",
                "Graph algorithms traverse or analyze graph structures",
                "String algorithms process and manipulate text data"
        ));
        
        // 编程范式
        texts.addAll(Arrays.asList(
                "Programming paradigms are approaches to structuring code",
                "Object oriented programming organizes code around objects",
                "Functional programming treats computation as function evaluation",
                "Imperative programming uses statements to change program state",
                "Declarative programming describes what to compute not how",
                "Procedural programming organizes code into procedures",
                "Event driven programming responds to events or messages",
                "Concurrent programming handles multiple computations simultaneously",
                "Generic programming writes code that works with multiple types",
                "Metaprogramming writes code that manipulates code"
        ));
        
        return texts;
    }

    /**
     * 生成AI历史与发展相关文本
     */
    private static List<String> generateAIHistoryTexts() {
        List<String> texts = new ArrayList<>();
        
        // AI早期历史
        texts.addAll(Arrays.asList(
                "Artificial intelligence began as a field in the 1950s",
                "The Turing test was proposed by Alan Turing in 1950",
                "The Dartmouth Conference in 1956 founded AI as a discipline",
                "Early AI focused on symbolic reasoning and rule based systems",
                "Expert systems were popular AI applications in the 1980s",
                "The first AI winter occurred in the 1970s due to funding cuts",
                "Machine learning gained prominence in the 1990s",
                "Deep Blue defeated the world chess champion in 1997",
                "Support vector machines became popular in the late 1990s",
                "Neural networks experienced a revival in the 2000s"
        ));
        
        // 深度学习革命
        texts.addAll(Arrays.asList(
                "Deep learning achieved breakthrough results around 2012",
                "AlexNet won ImageNet competition using GPU training in 2012",
                "Word2Vec introduced powerful word embeddings in 2013",
                "GANs introduced generative adversarial training in 2014",
                "ResNet enabled training of very deep networks in 2015",
                "AlphaGo defeated the world Go champion in 2016",
                "Transformer architecture was introduced in 2017",
                "BERT revolutionized NLP with bidirectional pretraining in 2018",
                "GPT-2 demonstrated impressive text generation in 2019",
                "GPT-3 showed emergent abilities with 175 billion parameters in 2020"
        ));
        
        // 大模型时代
        texts.addAll(Arrays.asList(
                "Large language models have transformed AI capabilities",
                "ChatGPT brought conversational AI to mainstream users in 2022",
                "Instruction tuning improved model following of human instructions",
                "RLHF aligned language models with human preferences",
                "Multimodal models process text images and other modalities",
                "Diffusion models revolutionized image generation",
                "Code generation models assist programmers with writing code",
                "AI assistants help with writing analysis and creativity",
                "Foundation models provide general purpose AI capabilities",
                "AI safety research addresses risks of powerful AI systems"
        ));
        
        // AI应用领域
        texts.addAll(Arrays.asList(
                "AI has transformed many industries and applications",
                "Computer vision enables machines to understand images",
                "Natural language processing helps machines understand text",
                "Speech recognition converts spoken language to text",
                "Recommendation systems personalize user experiences",
                "Autonomous vehicles use AI for navigation and control",
                "Medical AI assists in diagnosis and drug discovery",
                "Financial AI enables algorithmic trading and fraud detection",
                "Robotics combines AI with physical systems",
                "Scientific discovery is accelerated by AI methods"
        ));
        
        return texts;
    }

    /**
     * 生成优化算法相关文本
     */
    private static List<String> generateOptimizationTexts() {
        List<String> texts = new ArrayList<>();
        
        // 梯度下降
        texts.addAll(Arrays.asList(
                "Gradient descent is the core optimization algorithm for neural networks",
                "Stochastic gradient descent uses random subsets of data",
                "Mini batch gradient descent balances efficiency and accuracy",
                "Batch gradient descent computes gradients over all data",
                "Learning rate controls step size in parameter updates",
                "Momentum accelerates convergence by accumulating past gradients",
                "Nesterov momentum looks ahead before computing gradients",
                "Gradient clipping prevents exploding gradients",
                "Learning rate schedules adjust rates during training",
                "Warmup gradually increases learning rate at training start"
        ));
        
        // 自适应优化器
        texts.addAll(Arrays.asList(
                "Adaptive optimizers adjust learning rates per parameter",
                "Adagrad adapts rates based on historical gradient magnitudes",
                "RMSprop uses exponential moving average of squared gradients",
                "Adam combines momentum with adaptive learning rates",
                "AdamW adds weight decay regularization to Adam",
                "AdaFactor reduces memory usage for large models",
                "LAMB enables large batch training for distributed settings",
                "LARS supports layer wise adaptive learning rates",
                "NovoGrad is memory efficient for large batch training",
                "Shampoo uses preconditioning for faster convergence"
        ));
        
        // 二阶优化
        texts.addAll(Arrays.asList(
                "Second order methods use curvature information",
                "Newton method uses the Hessian for faster convergence",
                "Quasi Newton methods approximate the Hessian",
                "BFGS is a popular quasi Newton optimization algorithm",
                "L BFGS is memory efficient for large problems",
                "Natural gradient uses Fisher information matrix",
                "K FAC approximates Fisher for neural networks",
                "Hessian free optimization avoids storing the full Hessian",
                "Trust region methods constrain parameter updates",
                "Line search finds optimal step size along gradient"
        ));
        
        // 约束优化
        texts.addAll(Arrays.asList(
                "Constrained optimization handles restrictions on parameters",
                "Lagrange multipliers incorporate constraints into objectives",
                "Penalty methods add penalties for constraint violations",
                "Barrier methods prevent constraint violations",
                "Projected gradient descent projects onto feasible sets",
                "Proximal methods handle non smooth regularization",
                "ADMM splits problems into alternating subproblems",
                "Dual ascent optimizes the dual problem",
                "Interior point methods traverse feasible region",
                "Augmented Lagrangian combines penalties and multipliers"
        ));
        
        return texts;
    }

    /**
     * 生成计算机视觉相关文本
     */
    private static List<String> generateComputerVisionTexts() {
        List<String> texts = new ArrayList<>();
        
        // 图像处理基础
        texts.addAll(Arrays.asList(
                "Computer vision enables machines to understand images",
                "Images are represented as arrays of pixel values",
                "Convolution extracts features using sliding filters",
                "Pooling reduces spatial dimensions while retaining information",
                "Edge detection identifies boundaries in images",
                "Image filtering removes noise or enhances features",
                "Histogram equalization improves image contrast",
                "Morphological operations process image shapes",
                "Color spaces represent colors in different formats",
                "Image augmentation increases training data diversity"
        ));
        
        // 卷积神经网络
        texts.addAll(Arrays.asList(
                "Convolutional neural networks revolutionized image recognition",
                "Convolutional layers share weights across spatial positions",
                "Receptive fields determine the region each neuron sees",
                "Feature maps are output channels from convolutional layers",
                "Strided convolution reduces spatial resolution",
                "Dilated convolution expands receptive field without parameters",
                "Depthwise separable convolution reduces computation",
                "Batch normalization stabilizes CNN training",
                "Global average pooling reduces spatial dimensions",
                "Skip connections enable very deep CNN architectures"
        ));
        
        // 视觉任务
        texts.addAll(Arrays.asList(
                "Image classification assigns labels to entire images",
                "Object detection finds and classifies objects in images",
                "Semantic segmentation labels each pixel with a class",
                "Instance segmentation separates individual object instances",
                "Pose estimation detects human body keypoints",
                "Face recognition identifies or verifies individuals",
                "Optical flow estimates motion between video frames",
                "Image generation creates new images from noise or text",
                "Image super resolution enhances image quality",
                "Style transfer applies artistic styles to images"
        ));
        
        // 视觉模型
        texts.addAll(Arrays.asList(
                "VGGNet demonstrated the power of deep networks",
                "Inception modules use multiple filter sizes in parallel",
                "ResNet enabled training of hundreds of layers",
                "DenseNet connects layers densely for feature reuse",
                "EfficientNet scales network dimensions optimally",
                "YOLO enables real time object detection",
                "R CNN family introduced region based detection",
                "U Net is popular for biomedical image segmentation",
                "Vision transformer applies transformer to image patches",
                "CLIP learns joint image text representations"
        ));
        
        return texts;
    }

    /**
     * 生成强化学习相关文本
     */
    private static List<String> generateReinforcementLearningTexts() {
        List<String> texts = new ArrayList<>();
        
        // RL基础
        texts.addAll(Arrays.asList(
                "Reinforcement learning trains agents through interaction",
                "Agents take actions in environments to maximize rewards",
                "States represent the current situation of the environment",
                "Actions are choices available to the agent",
                "Rewards provide feedback on action quality",
                "Policies map states to actions",
                "Value functions estimate expected cumulative rewards",
                "The discount factor weights future rewards",
                "Episodes are sequences from start to terminal state",
                "The exploration exploitation tradeoff balances learning and acting"
        ));
        
        // 价值方法
        texts.addAll(Arrays.asList(
                "Value based methods learn value functions",
                "Q learning learns action values without a model",
                "Deep Q networks use neural networks for Q values",
                "Experience replay stores transitions for learning",
                "Target networks stabilize Q learning",
                "Double Q learning reduces overestimation",
                "Dueling networks separate value and advantage",
                "Prioritized replay samples important transitions more",
                "Noisy networks enable exploration through parameter noise",
                "Distributional RL learns value distributions"
        ));
        
        // 策略方法
        texts.addAll(Arrays.asList(
                "Policy based methods directly learn policies",
                "REINFORCE uses Monte Carlo policy gradients",
                "Actor critic methods combine policy and value learning",
                "Advantage functions reduce variance in policy gradients",
                "Trust region policy optimization constrains policy updates",
                "Proximal policy optimization clips policy updates",
                "Deterministic policy gradients handle continuous actions",
                "Soft actor critic maximizes entropy for exploration",
                "Natural policy gradient uses Fisher information",
                "Important sampling reuses past experience"
        ));
        
        // 高级RL
        texts.addAll(Arrays.asList(
                "Model based RL learns environment dynamics",
                "Planning uses models to simulate futures",
                "Monte Carlo tree search plans through simulations",
                "AlphaGo combined deep learning with tree search",
                "Curiosity driven exploration seeks novel states",
                "Inverse reinforcement learning learns from demonstrations",
                "Imitation learning copies expert behavior",
                "Multi agent RL handles multiple interacting agents",
                "Hierarchical RL decomposes tasks into subtasks",
                "Meta RL learns to learn across many tasks"
        ));
        
        return texts;
    }

    /**
     * 生成数据处理相关文本
     */
    private static List<String> generateDataProcessingTexts() {
        List<String> texts = new ArrayList<>();
        
        // 数据收集
        texts.addAll(Arrays.asList(
                "Data collection is the first step in machine learning",
                "Web scraping extracts data from websites",
                "APIs provide structured access to data sources",
                "Surveys collect labeled data from human respondents",
                "Sensors capture real world signals for analysis",
                "Crowdsourcing leverages many people for data labeling",
                "Synthetic data is generated programmatically",
                "Data augmentation expands datasets through transformations",
                "Active learning selects informative samples for labeling",
                "Data labeling assigns ground truth to examples"
        ));
        
        // 数据清洗
        texts.addAll(Arrays.asList(
                "Data cleaning removes errors and inconsistencies",
                "Missing values can be imputed or removed",
                "Outliers may be genuine or errors requiring handling",
                "Duplicate records should be identified and resolved",
                "Data normalization scales features to similar ranges",
                "Standardization centers and scales features",
                "Encoding converts categorical variables to numbers",
                "Text cleaning removes noise from text data",
                "Data validation ensures data quality standards",
                "Data profiling summarizes data characteristics"
        ));
        
        // 特征工程
        texts.addAll(Arrays.asList(
                "Feature engineering creates informative inputs for models",
                "Feature extraction derives features from raw data",
                "Feature selection chooses the most relevant features",
                "Polynomial features capture non linear relationships",
                "Binning converts continuous features to categorical",
                "Interaction features combine multiple features",
                "Time series features capture temporal patterns",
                "Text features represent documents as vectors",
                "Image features extract visual characteristics",
                "Feature importance ranks features by contribution"
        ));
        
        // 数据管道
        texts.addAll(Arrays.asList(
                "Data pipelines automate data processing workflows",
                "ETL processes extract transform and load data",
                "Data validation checks data quality automatically",
                "Data versioning tracks changes to datasets",
                "Feature stores manage and serve features",
                "Data lineage tracks data origins and transformations",
                "Streaming data is processed in real time",
                "Batch processing handles data in large chunks",
                "Data partitioning distributes data for parallel processing",
                "Data caching speeds up repeated data access"
        ));
        
        return texts;
    }

    /**
     * 生成模型评估相关文本
     */
    private static List<String> generateModelEvaluationTexts() {
        List<String> texts = new ArrayList<>();
        
        // 评估指标
        texts.addAll(Arrays.asList(
                "Model evaluation measures how well models perform",
                "Accuracy measures the fraction of correct predictions",
                "Precision measures the fraction of positive predictions that are correct",
                "Recall measures the fraction of actual positives that are found",
                "F1 score balances precision and recall",
                "ROC curves plot true positive rate against false positive rate",
                "AUC summarizes ROC curve performance in one number",
                "Mean squared error measures regression prediction error",
                "Mean absolute error is robust to outliers",
                "R squared measures variance explained by the model"
        ));
        
        // 交叉验证
        texts.addAll(Arrays.asList(
                "Cross validation estimates model performance robustly",
                "K fold cross validation splits data into k folds",
                "Stratified cross validation maintains class distributions",
                "Leave one out cross validation uses each sample as a test",
                "Nested cross validation handles hyperparameter tuning",
                "Time series cross validation respects temporal order",
                "Group cross validation keeps related samples together",
                "Repeated cross validation reduces variance in estimates",
                "Cross validation helps detect overfitting",
                "Cross validation provides confidence intervals"
        ));
        
        // 模型选择
        texts.addAll(Arrays.asList(
                "Model selection chooses between different models",
                "Hyperparameter tuning optimizes model settings",
                "Grid search exhaustively searches parameter combinations",
                "Random search samples parameter combinations randomly",
                "Bayesian optimization uses probabilistic models for search",
                "Early stopping prevents overfitting during training",
                "Model ensembling combines multiple models",
                "Bagging reduces variance through bootstrap sampling",
                "Boosting sequentially corrects previous errors",
                "Stacking uses meta models to combine base models"
        ));
        
        // 模型诊断
        texts.addAll(Arrays.asList(
                "Model diagnosis identifies problems with models",
                "Learning curves show performance versus training size",
                "Validation curves show performance versus hyperparameters",
                "Confusion matrices reveal classification error patterns",
                "Residual analysis checks regression assumptions",
                "Feature importance identifies influential features",
                "Partial dependence plots show feature effects",
                "SHAP values explain individual predictions",
                "Model calibration compares predicted versus actual probabilities",
                "Fairness metrics detect bias in predictions"
        ));
        
        return texts;
    }

    /**
     * 生成分布式训练相关文本
     */
    private static List<String> generateDistributedTrainingTexts() {
        List<String> texts = new ArrayList<>();
        
        // 并行策略
        texts.addAll(Arrays.asList(
                "Distributed training scales training across multiple devices",
                "Data parallelism replicates models across devices",
                "Model parallelism splits models across devices",
                "Pipeline parallelism partitions layers across devices",
                "Tensor parallelism splits tensors across devices",
                "Hybrid parallelism combines multiple strategies",
                "ZeRO optimizer reduces memory for data parallelism",
                "FSDP shards model states across devices",
                "Mesh strategies arrange devices in multidimensional grids",
                "Megatron LM uses tensor parallelism for large models"
        ));
        
        // 通信
        texts.addAll(Arrays.asList(
                "Communication is a key challenge in distributed training",
                "All reduce aggregates gradients across devices",
                "All gather collects tensors from all devices",
                "Reduce scatter distributes aggregation results",
                "Ring all reduce organizes devices in a ring topology",
                "Parameter servers centralize gradient aggregation",
                "Gradient compression reduces communication volume",
                "Overlapping computation and communication improves efficiency",
                "Communication efficient algorithms reduce synchronization",
                "Asynchronous updates allow devices to proceed independently"
        ));
        
        // 系统优化
        texts.addAll(Arrays.asList(
                "System optimizations improve training efficiency",
                "Mixed precision training uses lower precision arithmetic",
                "Gradient checkpointing trades computation for memory",
                "Memory efficient attention reduces memory for transformers",
                "Faster attention implementations speed up transformers",
                "Operator fusion combines operations for efficiency",
                "Just in time compilation optimizes computational graphs",
                "Profiling identifies performance bottlenecks",
                "Batch size scaling improves hardware utilization",
                "Dynamic batching groups similar inputs together"
        ));
        
        // 容错与扩展
        texts.addAll(Arrays.asList(
                "Fault tolerance handles failures in distributed training",
                "Checkpointing saves training state for recovery",
                "Elastic training adapts to available resources",
                "Preemption handling manages interrupted training",
                "Distributed optimizers coordinate across devices",
                "Scalable data loading prevents bottlenecks",
                "Distributed sampling ensures data diversity",
                "Synchronization points coordinate training steps",
                "Consistency models trade freshness for efficiency",
                "Straggler mitigation handles slow devices"
        ));
        
        return texts;
    }

    /**
     * 生成指令-回答对(用于微调)
     */
    private static List<String> generateInstructionQA() {
        List<String> qa = new ArrayList<>();

        // 深度学习基础QA (30条)
        qa.add("Instruction: What is deep learning? Response: Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations from data");
        qa.add("Instruction: Explain backpropagation. Response: Backpropagation is an algorithm that computes gradients of the loss function with respect to network parameters by applying the chain rule backwards through the computational graph");
        qa.add("Instruction: What is overfitting? Response: Overfitting occurs when a model learns the training data too well including noise and fails to generalize to new unseen data");
        qa.add("Instruction: How does dropout work? Response: Dropout randomly sets a fraction of activations to zero during training which prevents co adaptation of neurons and improves generalization");
        qa.add("Instruction: What is transfer learning? Response: Transfer learning reuses knowledge from a pretrained model on one task to improve performance on a related task with limited data");
        qa.add("Instruction: Explain convolutional neural networks. Response: Convolutional neural networks use convolutional layers to extract spatial features from images through learned filters");
        qa.add("Instruction: What are recurrent neural networks? Response: Recurrent neural networks process sequential data by maintaining hidden states that capture information from previous time steps");
        qa.add("Instruction: How does batch normalization work? Response: Batch normalization normalizes layer inputs across mini batches to reduce internal covariate shift and accelerate training");
        qa.add("Instruction: What is gradient descent? Response: Gradient descent is an optimization algorithm that iteratively updates parameters in the direction that reduces the loss function");
        qa.add("Instruction: Explain activation functions. Response: Activation functions introduce non linearity into neural networks enabling them to learn complex patterns beyond linear relationships");

        // NLP相关QA (30条)
        qa.add("Instruction: What is natural language processing? Response: Natural language processing is a field of AI that enables computers to understand analyze and generate human language");
        qa.add("Instruction: Explain word embeddings. Response: Word embeddings are dense vector representations of words that capture semantic relationships in a continuous space");
        qa.add("Instruction: What is tokenization? Response: Tokenization is the process of splitting text into smaller units like words subwords or characters for processing by language models");
        qa.add("Instruction: How do language models work? Response: Language models learn probability distributions over sequences of words and can predict the next word given previous context");
        qa.add("Instruction: What is sentiment analysis? Response: Sentiment analysis determines the emotional tone or opinion expressed in text such as positive negative or neutral");
        qa.add("Instruction: Explain named entity recognition. Response: Named entity recognition identifies and classifies entities like persons organizations and locations in text");
        qa.add("Instruction: What is machine translation? Response: Machine translation automatically converts text from one natural language to another using statistical or neural methods");
        qa.add("Instruction: How does text classification work? Response: Text classification assigns documents to predefined categories based on learned patterns in the text");
        qa.add("Instruction: What is question answering? Response: Question answering systems take natural language questions as input and provide accurate answers from knowledge sources");
        qa.add("Instruction: Explain sequence to sequence models. Response: Sequence to sequence models map input sequences to output sequences and are used for tasks like translation and summarization");

        // Transformer和GPT相关QA (30条)
        qa.add("Instruction: What is the transformer architecture? Response: The transformer is a neural network architecture that uses self attention mechanisms to process sequences in parallel without recurrence");
        qa.add("Instruction: Explain self attention. Response: Self attention computes weighted relationships between all positions in a sequence allowing the model to focus on relevant parts of the input");
        qa.add("Instruction: What is GPT? Response: GPT is a generative pretrained transformer that uses decoder only architecture for autoregressive language modeling and text generation");
        qa.add("Instruction: How does multi head attention work? Response: Multi head attention runs multiple attention mechanisms in parallel each learning different aspects of relationships between tokens");
        qa.add("Instruction: What is positional encoding? Response: Positional encoding adds information about token positions to embeddings since transformers have no built in notion of sequence order");
        qa.add("Instruction: Explain masked language modeling. Response: Masked language modeling trains models to predict masked tokens in a sequence using bidirectional context");
        qa.add("Instruction: What is causal language modeling? Response: Causal language modeling predicts the next token in a sequence using only previous tokens in autoregressive fashion");
        qa.add("Instruction: How does BERT work? Response: BERT uses bidirectional transformers and masked language modeling to learn contextualized representations for language understanding");
        qa.add("Instruction: What is fine tuning? Response: Fine tuning adapts a pretrained model to a specific downstream task by training on task specific data with a smaller learning rate");
        qa.add("Instruction: Explain encoder decoder architecture. Response: Encoder decoder architecture processes input sequences with an encoder and generates output sequences with a decoder connected by attention");

        // TinyAI框架QA (30条)
        qa.add("Instruction: What is TinyAI? Response: TinyAI is a Java based deep learning framework designed for education that implements core concepts like autograd neural networks and transformers");
        qa.add("Instruction: Explain NdArray in TinyAI. Response: NdArray is the fundamental multidimensional array class in TinyAI that stores numerical data and supports operations like matrix multiplication");
        qa.add("Instruction: What is Variable in TinyAI? Response: Variable wraps NdArray and tracks computational graphs enabling automatic differentiation for backpropagation");
        qa.add("Instruction: How does autograd work in TinyAI? Response: The autograd engine builds a computational graph during forward pass and automatically computes gradients during backward pass using the chain rule");
        qa.add("Instruction: What is Module in TinyAI? Response: Module is the base class for neural network layers that defines forward computation and manages learnable parameters");
        qa.add("Instruction: Explain Parameter in TinyAI. Response: Parameter represents learnable weights in neural networks that are updated during training through gradient descent");
        qa.add("Instruction: How does forward pass work? Response: The forward pass computes outputs by propagating inputs through network layers while building the computational graph");
        qa.add("Instruction: What is backward pass? Response: The backward pass computes gradients by traversing the computational graph in reverse and applying the chain rule");
        qa.add("Instruction: Explain SGD optimizer. Response: SGD optimizer updates model parameters by subtracting the gradient scaled by the learning rate");
        qa.add("Instruction: What is Linear layer? Response: Linear layer performs affine transformation by multiplying input with a weight matrix and adding a bias vector");

        // 机器学习QA (30条)
        qa.add("Instruction: What is supervised learning? Response: Supervised learning trains models on labeled data where each input has a corresponding target output");
        qa.add("Instruction: Explain gradient descent. Response: Gradient descent is an optimization algorithm that iteratively updates parameters in the direction that reduces the loss function");
        qa.add("Instruction: What is regularization? Response: Regularization adds constraints to model training to prevent overfitting and improve generalization to new data");
        qa.add("Instruction: How does cross validation work? Response: Cross validation splits data into multiple folds and evaluates model performance on each fold to estimate generalization ability");
        qa.add("Instruction: What is the learning rate? Response: Learning rate controls the step size during gradient descent optimization affecting training speed and convergence");
        qa.add("Instruction: Explain ensemble methods. Response: Ensemble methods combine predictions from multiple models to achieve better performance than any single model");
        qa.add("Instruction: What is feature engineering? Response: Feature engineering transforms raw data into meaningful representations that improve model learning and performance");
        qa.add("Instruction: How does early stopping work? Response: Early stopping monitors validation performance during training and stops when performance stops improving to prevent overfitting");
        qa.add("Instruction: What is the bias variance tradeoff? Response: The bias variance tradeoff balances model simplicity and flexibility to achieve optimal generalization performance");
        qa.add("Instruction: Explain data augmentation. Response: Data augmentation creates variations of training examples through transformations to increase dataset diversity");

        // 神经网络QA (30条)
        qa.add("Instruction: What is a neural network? Response: A neural network is a computing system with interconnected layers of neurons that learn to map inputs to outputs");
        qa.add("Instruction: Explain activation functions. Response: Activation functions introduce non linearity into neural networks enabling them to learn complex patterns beyond linear relationships");
        qa.add("Instruction: What is ReLU? Response: ReLU or rectified linear unit is an activation function that outputs the input if positive and zero otherwise");
        qa.add("Instruction: How does batch normalization work? Response: Batch normalization normalizes layer inputs across mini batches to reduce internal covariate shift and accelerate training");
        qa.add("Instruction: What is the softmax function? Response: Softmax converts a vector of values into a probability distribution where all outputs sum to one");
        qa.add("Instruction: Explain loss functions. Response: Loss functions measure the difference between model predictions and true labels guiding parameter updates during training");
        qa.add("Instruction: What is cross entropy loss? Response: Cross entropy loss measures the difference between predicted and true probability distributions for classification");
        qa.add("Instruction: How does Adam optimizer work? Response: Adam optimizer adapts learning rates for each parameter using estimates of first and second moments of gradients");
        qa.add("Instruction: What are residual connections? Response: Residual connections add skip connections that help train very deep networks by avoiding vanishing gradient problems");
        qa.add("Instruction: Explain weight initialization. Response: Weight initialization sets initial parameter values to break symmetry and enable effective gradient based learning");

        // 应用场景QA (20条)
        qa.add("Instruction: What is image classification? Response: Image classification assigns images to predefined categories based on visual content using convolutional neural networks");
        qa.add("Instruction: Explain object detection. Response: Object detection locates and classifies multiple objects within images by predicting bounding boxes and class labels");
        qa.add("Instruction: What is speech recognition? Response: Speech recognition converts spoken audio into text using acoustic and language models");
        qa.add("Instruction: How does text generation work? Response: Text generation creates coherent text by predicting one token at a time conditioned on previous tokens");
        qa.add("Instruction: What is semantic segmentation? Response: Semantic segmentation assigns a class label to every pixel in an image for fine grained understanding");
        qa.add("Instruction: Explain recommendation systems. Response: Recommendation systems predict user preferences and suggest relevant items based on historical behavior");
        qa.add("Instruction: What is anomaly detection? Response: Anomaly detection identifies unusual patterns or outliers that deviate from normal behavior");
        qa.add("Instruction: How does style transfer work? Response: Style transfer applies artistic style from one image to the content of another using neural networks");
        qa.add("Instruction: What is facial recognition? Response: Facial recognition identifies or verifies individuals by comparing facial features extracted by deep networks");
        qa.add("Instruction: Explain time series forecasting. Response: Time series forecasting predicts future values based on historical temporal patterns using recurrent networks");

        return qa;
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
    private static GPT1Model runPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: GPT-1 预训练 (Pretrain)");
        System.out.println("=".repeat(80));

        // 1. 读取所有数据（预训练+微调）用于构建完整词汇表
        System.out.println("\n📝 加载所有数据以构建词汇表...");
        String pretrainPath = DATA_DIR + "/pretrain.txt";
        String finetuneTrainPath = DATA_DIR + "/finetune_train.txt";
        String finetuneValPath = DATA_DIR + "/finetune_val.txt";

        List<String> pretrainTexts = readFromFile(pretrainPath);
        List<String> finetuneTrainTexts = readFromFile(finetuneTrainPath);
        List<String> finetuneValTexts = readFromFile(finetuneValPath);

        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 微调训练数据: " + finetuneTrainTexts.size() + " 条");
        System.out.println("  ✓ 微调验证数据: " + finetuneValTexts.size() + " 条");

        // 2. 基于所有数据构建完整词汇表
        System.out.println("\n📝 构建完整词汇表...");
        // 收集所有文本
        List<String> allTexts = new ArrayList<>();
        allTexts.addAll(pretrainTexts);
        allTexts.addAll(finetuneTrainTexts);
        allTexts.addAll(finetuneValTexts);

        // 遍历所有文本构建词汇表
        for (String text : allTexts) {
            sharedTokenizer.encode(text);
        }
        int vocabSize = sharedTokenizer.getVocabSize();

        // 冻结词汇表,后续不再添加新词
        sharedTokenizer.freeze();

        System.out.println("  ✓ 完整词汇表大小: " + vocabSize);
        System.out.println("  ✓ 词汇表已冻结,后续不再增加新词");

        // 3. 创建模型（词汇表大小设置为实际大小，无需额外buffer）
        System.out.println("\n📝 创建GPT-1模型...");
        GPT1Config config = GPT1Config.createTinyConfig();
        config.setVocabSize(vocabSize);

        GPT1Model model = new GPT1Model("gpt1-pretrain-v2", config);

        System.out.println("  ✓ 模型配置: Tiny");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getNEmbd());
        System.out.println("  ✓ 层数: " + config.getNLayer());
        System.out.println("  ✓ 注意力头数: " + config.getNHead());
        System.out.println("  ✓ 序列长度: " + config.getNPositions());

        // 4. 准备数据集
        System.out.println("\n📝 准备训练数据集...");
        GPT1Dataset dataset = new GPT1Dataset(
                config.getNPositions(),
                8,
                config.getVocabSize()
        );
        dataset.loadFromTexts(pretrainTexts, sharedTokenizer);

        System.out.println("  ✓ 训练样本: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次大小: 8");
        System.out.println("  ✓ 序列长度: " + config.getNPositions());

        // 5. 配置训练器
        System.out.println("\n📝 配置预训练器...");
        GPT1Pretrain trainer = new GPT1Pretrain(model, dataset);
        trainer.configure(
                30,
                1e-2f,
                5,
                1.0f
        ).setCheckpoint(CHECKPOINT_DIR + "/pretrain", 10);

        System.out.println("  ✓ 最大轮次: 30");
        System.out.println("  ✓ 学习率: 1e-2");
        System.out.println("  ✓ Warmup步数: 5");
        System.out.println("  ✓ 梯度裁剪: 1.0");

        // 6. 开始训练
        System.out.println("\n📝 开始预训练...");
        System.out.println("-".repeat(80));
        trainer.train();
        System.out.println("-".repeat(80));

        System.out.println("\n✅ 预训练完成!");
        System.out.println("\n💡 预训练阶段总结:");
        System.out.println("  - 目标: 学习语言的通用表示和模式");
        System.out.println("  - 任务: 因果语言建模(预测下一个token)");
        System.out.println("  - 数据: 大规模无标注文本");
        System.out.println("  - 结果: 获得了对语言结构的基础理解");

        return model;
    }

    /**
     * 执行微调
     */
    private static GPT1Model runFinetuning(GPT1Model pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: GPT-1 微调 (Finetune/Posttrain)");
        System.out.println("=".repeat(80));

        // 1. 加载微调数据
        System.out.println("\n📝 加载微调数据...");
        String trainPath = DATA_DIR + "/finetune_train.txt";
        String valPath = DATA_DIR + "/finetune_val.txt";

        List<String> trainTexts = readFromFile(trainPath);
        List<String> valTexts = readFromFile(valPath);

        System.out.println("  ✓ 训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 验证集: " + valTexts.size() + " 条");

        // 2. 准备数据集（使用微调专用的数据加载方式）
        // 与预训练不同，微调数据每条样本独立处理，不跨样本拼接
        // 只在Response部分计算loss，Instruction部分通过lossMask屏蔽
        System.out.println("\n📝 准备微调数据集...");
        GPT1Config config = pretrainedModel.getConfig();
        String responseSeparator = "Response:";

        GPT1Dataset trainDataset = new GPT1Dataset(
                config.getNPositions(),
                2,
                config.getVocabSize()
        );
        trainDataset.loadFromInstructionTexts(trainTexts, sharedTokenizer, responseSeparator);

        GPT1Dataset valDataset = new GPT1Dataset(
                config.getNPositions(),
                1,
                config.getVocabSize()
        );
        valDataset.loadFromInstructionTexts(valTexts, sharedTokenizer, responseSeparator);

        System.out.println("  ✓ 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + valDataset.getSampleCount());

        // 3. 配置微调训练器
        System.out.println("\n📝 配置微调训练器...");
        GPT1Finetune finetuner = new GPT1Finetune(
                pretrainedModel,
                trainDataset,
                valDataset
        );

        finetuner.configure(
                10,
                1e-3f,
                3
        ).setCheckpoint(CHECKPOINT_DIR + "/finetune", 3);

        System.out.println("  ✓ 最大轮次: 10");
        System.out.println("  ✓ 学习率: 1e-3 (比预训练小)");
        System.out.println("  ✓ 早停耐心值: 3");

        // 4. 开始微调
        System.out.println("\n📝 开始微调...");
        System.out.println("-".repeat(80));
        finetuner.train();
        System.out.println("-".repeat(80));

        System.out.println("\n✅ 微调完成!");
        System.out.println("\n💡 微调阶段总结:");
        System.out.println("  - 目标: 适应问答任务");
        System.out.println("  - 任务: 指令-回答格式的文本生成");
        System.out.println("  - 数据: 任务特定的指令数据(每条独立处理,不跨样本拼接)");
        System.out.println("  - Loss: 只在Response部分计算loss,Instruction部分被mask屏蔽");
        System.out.println("  - 技巧: 小学习率 + 早停机制 + Loss Mask");
        System.out.println("  - 结果: 模型学会了回答问题的能力");

        return pretrainedModel;
    }

    /**
     * 执行推理测试
     */
    private static void runInference(GPT1Model model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤3: GPT-1 推理与文本生成");
        System.out.println("=".repeat(80));

        // 1. 创建推理器
        System.out.println("\n📝 创建推理器...");
        GPT1Inference inference = new GPT1Inference(model);
        System.out.println("  ✓ 推理器准备完成");

        // 2. 测试用例
        String[] prompts = {
                "Deep learning is",
                "Instruction: What is NLP? Response:",
                "Transformer architecture"
        };

        System.out.println("\n📝 执行文本生成测试...\n");

        for (int i = 0; i < prompts.length; i++) {
            String prompt = prompts[i];
            System.out.println("测试 " + (i + 1) + ": \"" + prompt + "\"");
            System.out.println("-".repeat(80));

            try {
                List<Integer> tokens = sharedTokenizer.encode(prompt);
                int[] promptIds = tokens.stream().mapToInt(Integer::intValue).toArray();

                // Greedy解码
                System.out.println("  策略1 [Greedy]: ");
                int[] greedyResult = inference.generateGreedy(promptIds, 15);
                String greedyText = sharedTokenizer.decode(greedyResult);
                System.out.println("    → " + greedyText);

                // Temperature采样
                System.out.println("  策略2 [Temperature=0.8]: ");
                int[] tempResult = inference.generateWithTemperature(promptIds, 15, 0.8f);
                String tempText = sharedTokenizer.decode(tempResult);
                System.out.println("    → " + tempText);

            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }

            System.out.println();
        }

        System.out.println("✅ 推理测试完成!");
        System.out.println("\n💡 推理阶段总结:");
        System.out.println("  - 输入: 提示词token序列");
        System.out.println("  - 处理: 自回归生成(逐token预测)");
        System.out.println("  - 输出: 生成的完整文本");
        System.out.println("  - 策略: Greedy/Temperature/TopK/TopP/Beam");
    }

    /**
     * 从文件读取文本
     */
    private static List<String> readFromFile(String filePath) throws IOException {
        List<String> lines = new ArrayList<>();
        java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.FileReader(filePath)
        );
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.trim().isEmpty()) {
                lines.add(line);
            }
        }
        reader.close();
        return lines;
    }
}
