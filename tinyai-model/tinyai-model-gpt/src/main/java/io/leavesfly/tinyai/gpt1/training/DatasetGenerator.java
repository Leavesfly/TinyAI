package io.leavesfly.tinyai.gpt1.training;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * GPT-1 训练数据生成器
 * 负责生成预训练和微调数据集
 *
 * @author TinyAI
 * @since 2024
 */
public class DatasetGenerator {

    private static final String DATA_DIR = "./data/gpt1_training";

    /**
     * 准备所有训练数据集
     */
    public void prepareDatasets() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📦 步骤0: 准备训练数据集");
        System.out.println("=".repeat(80));

        File dataDir = new File(DATA_DIR);
        if (!dataDir.exists()) {
            dataDir.mkdirs();
            System.out.println("✓ 创建数据目录: " + DATA_DIR);
        }

        generatePretrainDataset();
        generateFinetuneDataset();

        System.out.println("\n✅ 数据集准备完成!");
    }

    /**
     * 生成预训练数据集
     */
    private void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");

        List<String> pretrainTexts = new ArrayList<>();
        pretrainTexts.addAll(generateDeepLearningTexts());
        pretrainTexts.addAll(generateMachineLearningTexts());
        pretrainTexts.addAll(generateNeuralNetworkTexts());
        pretrainTexts.addAll(generateNLPTexts());
        pretrainTexts.addAll(generateTransformerTexts());
        pretrainTexts.addAll(generateOptimizationTexts());
        pretrainTexts.addAll(generateComputerVisionTexts());
        pretrainTexts.addAll(generateReinforcementLearningTexts());

        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);

        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    /**
     * 生成微调数据集
     */
    private void generateFinetuneDataset() throws IOException {
        System.out.println("\n📝 生成微调数据集...");

        List<String> trainTexts = new ArrayList<>();
        trainTexts.addAll(generateInstructionQA());

        List<String> valTexts = new ArrayList<>();
        for (int i = 0; i < 20 && i < trainTexts.size(); i++) {
            valTexts.add(trainTexts.get(i));
        }

        String trainPath = DATA_DIR + "/finetune_train.txt";
        writeToFile(trainTexts, trainPath);
        System.out.println("  ✓ 微调训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + trainPath);

        String valPath = DATA_DIR + "/finetune_val.txt";
        writeToFile(valTexts, valPath);
        System.out.println("  ✓ 微调验证集: " + valTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + valPath);
    }

    /**
     * 写入数据到文件
     */
    private void writeToFile(List<String> texts, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String text : texts) {
                writer.write(text);
                writer.newLine();
            }
        }
    }

    /**
     * 生成深度学习相关文本
     */
    private List<String> generateDeepLearningTexts() {
        List<String> texts = new ArrayList<>();
        
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
                "Pretrained models enable rapid development of AI applications with limited data"
        ));
        
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
                "Generator networks learn to produce realistic samples that fool the discriminator"
        ));
        
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
                "Distributed data parallel training splits batches across multiple GPUs"
        ));
        
        texts.addAll(Arrays.asList(
                "Cross entropy loss measures the difference between predicted and true probability distributions",
                "Mean squared error loss penalizes large errors more heavily than small ones",
                "Binary cross entropy loss is used for binary classification tasks",
                "Focal loss addresses class imbalance by down weighting easy examples",
                "Contrastive loss learns representations by comparing similar and dissimilar pairs",
                "Triplet loss learns embeddings by anchoring to reference points",
                "Hinge loss is used for training support vector machines and margin based models",
                "Smooth L1 loss is less sensitive to outliers than mean squared error",
                "Dice loss optimizes segmentation by measuring overlap directly",
                "IoU loss measures intersection over union for object detection"
        ));
        
        return texts;
    }

    /**
     * 生成机器学习相关文本
     */
    private List<String> generateMachineLearningTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Machine learning enables computers to learn from data without explicit programming",
                "Supervised learning uses labeled data to train models for prediction tasks",
                "Unsupervised learning discovers patterns in data without labels",
                "Reinforcement learning trains agents through rewards and punishments",
                "Classification assigns discrete labels to input examples",
                "Regression predicts continuous numerical values",
                "Clustering groups similar data points together",
                "Dimensionality reduction simplifies data while preserving important information",
                "Feature engineering creates informative inputs for machine learning models",
                "Model validation ensures models generalize to unseen data"
        ));
        
        texts.addAll(Arrays.asList(
                "Decision trees split data based on feature thresholds",
                "Random forests combine multiple decision trees for better predictions",
                "Gradient boosting sequentially corrects errors of previous models",
                "Support vector machines find optimal decision boundaries",
                "K nearest neighbors classifies based on similarity to training examples",
                "Naive Bayes uses probability theory for classification",
                "Linear regression models relationships with straight lines",
                "Logistic regression predicts probabilities for binary classification",
                "Principal component analysis reduces dimensionality linearly",
                "K means clustering partitions data into k groups"
        ));
        
        texts.addAll(Arrays.asList(
                "Bias variance tradeoff balances model complexity and generalization",
                "Overfitting occurs when models memorize training data",
                "Underfitting occurs when models are too simple to capture patterns",
                "Regularization adds constraints to prevent overfitting",
                "Cross validation estimates model performance robustly",
                "Hyperparameter tuning optimizes model settings",
                "Grid search systematically explores parameter combinations",
                "Random search samples parameter space efficiently",
                "Bayesian optimization uses probabilistic models for tuning",
                "Early stopping prevents overfitting during training"
        ));
        
        texts.addAll(Arrays.asList(
                "Precision measures the fraction of positive predictions that are correct",
                "Recall measures the fraction of actual positives that are found",
                "F1 score balances precision and recall in one metric",
                "Accuracy measures overall prediction correctness",
                "ROC curves plot true positive rate versus false positive rate",
                "AUC summarizes ROC curve performance in one number",
                "Confusion matrices show prediction breakdown by class",
                "Mean squared error measures average squared prediction error",
                "Mean absolute error measures average absolute prediction error",
                "R squared measures variance explained by the model"
        ));
        
        return texts;
    }

    /**
     * 生成神经网络相关文本
     */
    private List<String> generateNeuralNetworkTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Neural networks are computing systems inspired by biological brains",
                "Artificial neurons receive inputs and produce outputs through activation functions",
                "Weights and biases are learnable parameters in neural networks",
                "Feedforward neural networks pass information in one direction",
                "Convolutional neural networks excel at processing grid like data",
                "Recurrent neural networks maintain state for sequential data",
                "Long short term memory networks solve vanishing gradient problems",
                "Gated recurrent units simplify LSTM architecture",
                "Attention mechanisms allow models to focus on relevant inputs",
                "Transformer architecture revolutionized natural language processing"
        ));
        
        texts.addAll(Arrays.asList(
                "Activation functions introduce non linearity into neural networks",
                "ReLU activation is widely used for its simplicity and effectiveness",
                "Sigmoid activation maps inputs to values between zero and one",
                "Tanh activation maps inputs to values between negative one and one",
                "GELU activation combines properties of ReLU and sigmoid",
                "Softmax activation produces probability distributions",
                "Leaky ReLU prevents dying neurons with small negative slopes",
                "Swish activation is a smooth alternative to ReLU",
                "Mish activation combines softplus and tanh for smoothness",
                "Parametric ReLU learns the negative slope during training"
        ));
        
        texts.addAll(Arrays.asList(
                "Weight initialization affects training dynamics significantly",
                "Xavier initialization suits sigmoid and tanh activations",
                "He initialization works well with ReLU activations",
                "Batch normalization stabilizes training by normalizing activations",
                "Layer normalization normalizes across feature dimensions",
                "Dropout randomly zeroes neurons during training",
                "Gradient clipping prevents exploding gradients",
                "Residual connections enable training of very deep networks",
                "Dense connections concatenate features from previous layers",
                "Skip connections allow gradients to flow directly through networks"
        ));
        
        return texts;
    }

    /**
     * 生成NLP相关文本
     */
    private List<String> generateNLPTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Natural language processing enables computers to understand human language",
                "Tokenization splits text into meaningful units called tokens",
                "Word embeddings represent words as dense vectors",
                "Word2Vec learns word embeddings from context windows",
                "GloVe learns word vectors from global co occurrence statistics",
                "FastText extends word embeddings to subword units",
                "ELMo provides contextual word representations",
                "BERT learns bidirectional representations from unlabeled text",
                "GPT generates text using autoregressive language modeling",
                "Transformer models achieve state of the art on NLP benchmarks"
        ));
        
        texts.addAll(Arrays.asList(
                "Sentiment analysis determines the emotional tone of text",
                "Named entity recognition identifies entities like people and places",
                "Part of speech tagging assigns grammatical roles to words",
                "Dependency parsing analyzes grammatical structure",
                "Coreference resolution links mentions to the same entity",
                "Machine translation converts text between languages",
                "Text summarization creates concise versions of documents",
                "Question answering finds answers in text passages",
                "Text generation produces coherent natural language",
                "Dialogue systems enable conversational interactions"
        ));
        
        texts.addAll(Arrays.asList(
                "Language models predict the probability of text sequences",
                "N gram models count word sequences for prediction",
                "Neural language models use neural networks for text",
                "Perplexity measures how well language models predict text",
                "Beam search finds likely sequences during decoding",
                "Sampling methods generate diverse text outputs",
                "Temperature controls randomness in text generation",
                "Top k sampling limits choices to k most likely tokens",
                "Top p sampling uses nucleus of probability mass",
                "Repetition penalty discourages repeated content"
        ));
        
        texts.addAll(Arrays.asList(
                "Text classification assigns categories to documents",
                "Sequence labeling assigns tags to each token",
                "Sequence to sequence models transform input to output sequences",
                "Encoder decoder architectures map source to target sequences",
                "Attention allows decoders to focus on relevant encoder states",
                "Copy mechanisms copy words from input to output",
                "Coverage mechanisms prevent repetition in generation",
                "Multi task learning shares representations across tasks",
                "Transfer learning applies pretrained knowledge to new tasks",
                "Fine tuning adapts pretrained models to specific domains"
        ));
        
        return texts;
    }

    /**
     * 生成Transformer相关文本
     */
    private List<String> generateTransformerTexts() {
        List<String> texts = new ArrayList<>();
        
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
                "Global attention allows special tokens to attend to all positions"
        ));
        
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
                "Zero shot learning generalizes to tasks without examples"
        ));
        
        return texts;
    }

    /**
     * 生成优化算法相关文本
     */
    private List<String> generateOptimizationTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Optimization algorithms minimize loss functions by adjusting model parameters",
                "Gradient descent updates parameters in the direction of steepest descent",
                "Stochastic gradient descent uses mini batches for faster updates",
                "Momentum accelerates gradient descent by accumulating past gradients",
                "Nesterov momentum looks ahead to improve gradient estimates",
                "AdaGrad adapts learning rates for each parameter individually",
                "RMSprop scales learning rates by recent gradient magnitudes",
                "Adam combines momentum and adaptive learning rates",
                "AdamW decouples weight decay from gradient updates",
                "Learning rate schedules adjust learning rates during training"
        ));
        
        texts.addAll(Arrays.asList(
                "Batch gradient descent computes gradients on entire datasets",
                "Mini batch gradient descent balances speed and accuracy",
                "Learning rate controls the step size in parameter updates",
                "Weight decay adds regularization by penalizing large weights",
                "Gradient clipping prevents exploding gradients",
                "Learning rate warmup stabilizes early training",
                "Cosine annealing smoothly decreases learning rates",
                "Step decay reduces learning rates at fixed intervals",
                "Exponential decay continuously reduces learning rates",
                "Polynomial decay smoothly decreases to zero"
        ));
        
        texts.addAll(Arrays.asList(
                "First order optimization uses gradients to update parameters",
                "Second order optimization uses curvature information",
                "Newton method uses second derivatives for faster convergence",
                "Quasi Newton methods approximate second derivatives",
                "L-BFGS is memory efficient for large scale optimization",
                "Natural gradient uses Fisher information for updates",
                "Gradient descent with momentum accelerates convergence",
                "Nesterov accelerated gradient improves on momentum",
                "AdaMax generalizes Adam to infinity norm",
                "NAdam combines Adam with Nesterov momentum"
        ));
        
        return texts;
    }

    /**
     * 生成计算机视觉相关文本
     */
    private List<String> generateComputerVisionTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Computer vision enables machines to understand visual information",
                "Image classification assigns labels to entire images",
                "Object detection locates and classifies objects in images",
                "Semantic segmentation labels each pixel with object classes",
                "Instance segmentation identifies individual object instances",
                "Convolutional layers extract features from images",
                "Pooling layers reduce spatial dimensions",
                "Batch normalization accelerates training of deep networks",
                "Data augmentation expands datasets through transformations",
                "Transfer learning applies pretrained models to new tasks"
        ));
        
        texts.addAll(Arrays.asList(
                "Convolutional neural networks process images hierarchically",
                "Early layers detect edges and simple patterns",
                "Deeper layers recognize complex objects and scenes",
                "Receptive field determines the region a neuron sees",
                "Stride controls spacing between convolution positions",
                "Padding preserves spatial dimensions after convolution",
                "Dilated convolution increases receptive field without parameters",
                "Depthwise separable convolution factorizes convolution operations",
                "Grouped convolution divides channels into groups",
                "Pointwise convolution uses one by one kernels"
        ));
        
        texts.addAll(Arrays.asList(
                "ResNet enables training of very deep networks with skip connections",
                "DenseNet connects each layer to every other layer",
                "MobileNet optimizes for mobile and embedded devices",
                "EfficientNet scales networks systematically",
                "Vision Transformer applies transformers to image patches",
                "ViT achieves competitive results with convolution free architecture",
                "ImageNet is a large scale image classification benchmark",
                "COCO dataset supports object detection and segmentation",
                "Transfer learning from ImageNet improves performance on new tasks",
                "Fine tuning adapts pretrained models to specific domains"
        ));
        
        return texts;
    }

    /**
     * 生成强化学习相关文本
     */
    private List<String> generateReinforcementLearningTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Reinforcement learning trains agents through trial and error",
                "Agents learn by interacting with environments",
                "States represent situations agents encounter",
                "Actions are choices agents make in states",
                "Rewards provide feedback on action quality",
                "Policies map states to actions",
                "Value functions estimate expected future rewards",
                "Q functions estimate action values in states",
                "Episodes are sequences of states actions and rewards",
                "Exploration versus exploitation balances new and known actions"
        ));
        
        texts.addAll(Arrays.asList(
                "Q learning learns action values from experience",
                "Deep Q networks use neural networks for Q functions",
                "Policy gradient methods directly optimize policies",
                "Actor critic methods combine policy and value learning",
                "Advantage functions reduce variance in policy gradients",
                "Trust region policy optimization constrains policy updates",
                "Proximal policy optimization clips policy updates",
                "Deterministic policy gradients handle continuous actions",
                "Soft actor critic maximizes entropy for exploration",
                "Important sampling reuses past experience"
        ));
        
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
     * 生成指令问答数据
     */
    private List<String> generateInstructionQA() {
        List<String> texts = new ArrayList<>();
        
        // 深度学习基础问答
        texts.addAll(Arrays.asList(
                "Question: What is deep learning?\nAnswer: Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations from data.",
                "Question: How does backpropagation work?\nAnswer: Backpropagation computes gradients of the loss with respect to parameters by applying the chain rule backward through the network.",
                "Question: What is the difference between AI, ML, and DL?\nAnswer: AI is the broad concept of machines acting intelligently. ML is a subset of AI that learns from data. DL is a subset of ML using deep neural networks.",
                "Question: Why do we use activation functions?\nAnswer: Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns that linear transformations cannot capture.",
                "Question: What is transfer learning?\nAnswer: Transfer learning reuses a model trained on one task as the starting point for a model on a different but related task, saving training time and data."
        ));
        
        // 神经网络问答
        texts.addAll(Arrays.asList(
                "Question: What is a convolutional neural network?\nAnswer: CNN is a neural network architecture designed for processing grid-like data such as images, using convolution operations to extract spatial features.",
                "Question: What is a recurrent neural network?\nAnswer: RNN is a neural network designed for sequential data, where connections form a directed cycle allowing information to persist across time steps.",
                "Question: What is the vanishing gradient problem?\nAnswer: Vanishing gradients occur when gradients become extremely small during backpropagation, preventing weights in early layers from being updated effectively.",
                "Question: What are LSTM networks?\nAnswer: LSTM (Long Short-Term Memory) networks are RNNs with gating mechanisms that can learn long-term dependencies by controlling information flow.",
                "Question: What is dropout regularization?\nAnswer: Dropout randomly zeros a fraction of neurons during training, preventing overfitting by making the network robust to the loss of any individual neuron."
        ));
        
        // Transformer相关问答
        texts.addAll(Arrays.asList(
                "Question: What is the transformer architecture?\nAnswer: Transformer is a neural network architecture based on self-attention mechanisms, enabling parallel processing of sequences without recurrence.",
                "Question: How does self-attention work?\nAnswer: Self-attention computes attention weights between all positions in a sequence, allowing each position to attend to all other positions.",
                "Question: What is BERT?\nAnswer: BERT (Bidirectional Encoder Representations from Transformers) is a pretrained language model that learns bidirectional representations using masked language modeling.",
                "Question: What is GPT?\nAnswer: GPT (Generative Pre-trained Transformer) is an autoregressive language model trained to predict the next token given previous tokens.",
                "Question: Why are positional encodings needed in transformers?\nAnswer: Transformers process all positions in parallel, so positional encodings inject sequence order information that would otherwise be lost."
        ));
        
        // 优化算法问答
        texts.addAll(Arrays.asList(
                "Question: What is gradient descent?\nAnswer: Gradient descent is an optimization algorithm that iteratively adjusts parameters in the direction that reduces the loss function.",
                "Question: What is the Adam optimizer?\nAnswer: Adam combines momentum and adaptive learning rates, computing individual learning rates for each parameter based on gradient history.",
                "Question: What is learning rate scheduling?\nAnswer: Learning rate scheduling adjusts the learning rate during training, typically starting high and decreasing to fine-tune parameters.",
                "Question: What is batch normalization?\nAnswer: Batch normalization normalizes layer inputs to have zero mean and unit variance, stabilizing and accelerating training.",
                "Question: What is weight decay?\nAnswer: Weight decay adds a regularization term that penalizes large weights, helping prevent overfitting by keeping the model simple."
        ));
        
        // NLP相关问答
        texts.addAll(Arrays.asList(
                "Question: What is word embedding?\nAnswer: Word embedding represents words as dense vectors in continuous space, capturing semantic relationships between words.",
                "Question: What is tokenization?\nAnswer: Tokenization splits text into smaller units called tokens, which can be words, subwords, or characters depending on the strategy.",
                "Question: What is attention mechanism?\nAnswer: Attention allows a model to focus on relevant parts of the input when producing each element of the output, weighting their contributions.",
                "Question: What is perplexity?\nAnswer: Perplexity measures how well a language model predicts text, with lower values indicating better prediction performance.",
                "Question: What is the difference between encoder and decoder?\nAnswer: Encoders process input sequences into representations, while decoders generate output sequences from those representations."
        ));
        
        // 计算机视觉问答
        texts.addAll(Arrays.asList(
                "Question: What is image classification?\nAnswer: Image classification assigns predefined labels to entire images based on their visual content.",
                "Question: What is object detection?\nAnswer: Object detection locates objects in images with bounding boxes and classifies each detected object into categories.",
                "Question: What is semantic segmentation?\nAnswer: Semantic segmentation assigns a class label to every pixel in an image, partitioning it into meaningful regions.",
                "Question: What is data augmentation?\nAnswer: Data augmentation artificially expands training datasets by applying transformations like rotation, scaling, and flipping to existing images.",
                "Question: What is ResNet?\nAnswer: ResNet is a deep neural network architecture that uses skip connections to enable training of very deep networks by avoiding vanishing gradients."
        ));
        
        // 强化学习问答
        texts.addAll(Arrays.asList(
                "Question: What is reinforcement learning?\nAnswer: Reinforcement learning trains agents to make decisions by rewarding desired behaviors and punishing undesired ones through trial and error.",
                "Question: What is the exploration-exploitation tradeoff?\nAnswer: Exploration-exploitation tradeoff is the dilemma between trying new actions to discover rewards and exploiting known rewarding actions.",
                "Question: What is Q-learning?\nAnswer: Q-learning learns the value of actions in states by updating Q-values based on rewards received, without requiring a model of the environment.",
                "Question: What is a policy in RL?\nAnswer: A policy defines the agent's behavior by mapping states to actions, either deterministically or as a probability distribution.",
                "Question: What is reward shaping?\nAnswer: Reward shaping modifies the reward signal to guide learning more efficiently while preserving optimal policies."
        ));
        
        return texts;
    }
}
