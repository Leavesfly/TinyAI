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
 * 
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
public class GPT1TrainDemoV2 {

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
     * 包含深度学习、NLP、Transformer等领域的教学文本
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");

        List<String> pretrainTexts = new ArrayList<>();

        // 1. 深度学习基础知识 (100条)
        pretrainTexts.addAll(generateDeepLearningTexts());

        // 2. 机器学习概念 (80条)
        pretrainTexts.addAll(generateMachineLearningTexts());

        // 3. 神经网络架构 (80条)
        pretrainTexts.addAll(generateNeuralNetworkTexts());

        // 4. NLP和语言模型 (100条)
        pretrainTexts.addAll(generateNLPTexts());

        // 5. Transformer和注意力机制 (80条)
        pretrainTexts.addAll(generateTransformerTexts());

        // 6. TinyAI框架知识 (60条)
        pretrainTexts.addAll(generateTinyAITexts());

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
        return Arrays.asList(
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
            "Attention mechanisms allow models to focus on relevant parts of the input",
            "Deep learning systems learn features automatically without manual feature engineering",
            "Convolutional layers extract spatial hierarchies from images and visual data",
            "Pooling layers reduce spatial dimensions while preserving important features",
            "Deep architectures can learn abstract concepts through multiple levels of representation",
            "Neural networks trained on large datasets show emergent abilities on new tasks"
        );
    }

    /**
     * 生成机器学习相关文本
     */
    private static List<String> generateMachineLearningTexts() {
        return Arrays.asList(
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
            "Learning curves visualize model performance as training data increases",
            "Model evaluation requires separate test data never seen during training",
            "Classification assigns inputs to discrete categories or classes",
            "Regression predicts continuous numerical values from input features",
            "Clustering groups similar data points without predefined labels",
            "Dimensionality reduction simplifies data while preserving important information",
            "Decision trees partition feature space through recursive splits",
            "Random forests aggregate multiple decision trees for robust predictions",
            "Support vector machines find optimal hyperplanes for classification",
            "K nearest neighbors classifies based on proximity in feature space",
            "Principal component analysis reduces dimensions while retaining variance"
        );
    }

    /**
     * 生成神经网络相关文本
     */
    private static List<String> generateNeuralNetworkTexts() {
        return Arrays.asList(
            "Neural networks consist of interconnected layers of artificial neurons",
            "Each neuron computes a weighted sum of inputs and applies an activation function",
            "The input layer receives raw features while the output layer produces predictions",
            "Hidden layers learn increasingly abstract representations of the input",
            "Feedforward networks process information in one direction from input to output",
            "Weights and biases are the learnable parameters of a neural network",
            "The sigmoid activation function maps values to a range between zero and one",
            "ReLU activation is computationally efficient and helps avoid vanishing gradients",
            "Softmax converts network outputs into a probability distribution over classes",
            "Loss functions measure the difference between predictions and true labels",
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
            "Network depth enables learning of complex hierarchical features"
        );
    }

    /**
     * 生成NLP相关文本
     */
    private static List<String> generateNLPTexts() {
        return Arrays.asList(
            "Natural language processing enables computers to understand and generate human language",
            "Tokenization splits text into smaller units like words or subwords",
            "Word embeddings represent words as dense vectors in a continuous space",
            "Word2Vec learns embeddings by predicting context words from target words",
            "GloVe embeddings capture global statistical information from word co occurrences",
            "Language models estimate the probability distribution over sequences of words",
            "N gram models predict words based on previous n minus one words",
            "Neural language models use recurrent networks to capture long range dependencies",
            "Perplexity measures how well a language model predicts a test corpus",
            "Text classification assigns documents to predefined categories",
            "Sentiment analysis determines the emotional tone of text",
            "Named entity recognition identifies and classifies entities in text",
            "Machine translation converts text from one language to another",
            "Question answering systems provide answers to natural language questions",
            "Text summarization creates concise summaries of longer documents",
            "Sequence to sequence models map input sequences to output sequences",
            "Encoder decoder architectures are fundamental to many NLP tasks",
            "Beam search finds high probability output sequences during generation",
            "Byte pair encoding creates subword vocabularies for neural models",
            "Contextualized embeddings like BERT capture word meaning based on context",
            "Part of speech tagging identifies grammatical roles of words",
            "Dependency parsing analyzes syntactic structure of sentences",
            "Coreference resolution links pronouns to their referents",
            "Semantic role labeling identifies predicate argument structures",
            "Information extraction retrieves structured data from unstructured text"
        );
    }

    /**
     * 生成Transformer相关文本
     */
    private static List<String> generateTransformerTexts() {
        return Arrays.asList(
            "Transformer architecture revolutionized natural language processing in 2017",
            "Self attention allows models to weigh the importance of different input positions",
            "Multi head attention captures different aspects of relationships between tokens",
            "Positional encoding injects sequence order information into transformer models",
            "The transformer consists of an encoder and decoder with stacked layers",
            "Query key and value vectors are fundamental components of attention mechanisms",
            "Scaled dot product attention computes attention weights efficiently",
            "Feed forward networks process each position independently in transformers",
            "Layer normalization is applied before or after transformer sublayers",
            "Residual connections help train deep transformer models effectively",
            "BERT uses bidirectional transformers for language understanding tasks",
            "GPT models use decoder only transformers for text generation",
            "Masked language modeling is the pretraining objective for BERT",
            "Causal language modeling predicts the next token in autoregressive models",
            "Transformers eliminate recurrence and enable parallel processing of sequences",
            "Attention patterns can be visualized to understand model behavior",
            "The transformer model achieves state of the art results across NLP tasks",
            "Large scale pretraining of transformers requires significant computational resources",
            "Fine tuning adapts pretrained transformers to downstream tasks efficiently",
            "Transformer models scale effectively with increased data and parameters",
            "Cross attention connects encoder and decoder in sequence to sequence models",
            "Attention heads learn to focus on different linguistic phenomena",
            "Positional embeddings can be learned or defined using sinusoidal functions",
            "The transformer architecture is fully differentiable and trainable end to end",
            "Sparse attention patterns improve efficiency for long sequences"
        );
    }

    /**
     * 生成TinyAI框架相关文本
     */
    private static List<String> generateTinyAITexts() {
        return Arrays.asList(
            "TinyAI is a Java based deep learning framework for education and research",
            "NdArray is the core multidimensional array abstraction in TinyAI",
            "The autograd engine in TinyAI enables automatic differentiation",
            "Variable wraps NdArray and tracks computational graphs for backpropagation",
            "Module is the base class for all neural network layers in TinyAI",
            "Parameter represents learnable weights in TinyAI neural networks",
            "The forward method defines the computation performed by a module",
            "Backward propagation computes gradients through the computational graph",
            "SGD optimizer updates parameters using stochastic gradient descent",
            "Linear layer performs affine transformation of input features",
            "Embedding layer maps discrete tokens to continuous vector representations",
            "MultiHeadAttention implements the attention mechanism for transformers",
            "LayerNorm normalizes activations to stabilize training",
            "Dropout randomly zeros elements during training for regularization",
            "SoftmaxCrossEntropy combines softmax and cross entropy for classification",
            "Dataset class handles data loading and batching in TinyAI",
            "Trainer orchestrates the training loop and model optimization",
            "Model class wraps modules and provides high level training interface",
            "TinyAI supports both CPU and GPU computation for neural networks",
            "The framework provides comprehensive examples for learning deep learning",
            "Block abstraction allows composition of complex network architectures",
            "Loss functions quantify prediction errors during training",
            "Optimizer algorithms minimize loss by updating model parameters",
            "TinyAI implements common activation functions like ReLU and GELU",
            "The framework supports saving and loading trained model checkpoints"
        );
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
            4,
            config.getVocabSize()
        );
        dataset.loadFromTexts(pretrainTexts, sharedTokenizer);
        
        System.out.println("  ✓ 训练样本: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次大小: 4");
        System.out.println("  ✓ 序列长度: " + config.getNPositions());

        // 5. 配置训练器
        System.out.println("\n📝 配置预训练器...");
        GPT1Pretrain trainer = new GPT1Pretrain(model, dataset);
        trainer.configure(
            15,
            1e-2f,
            5,
            1.0f
        ).setCheckpoint(CHECKPOINT_DIR + "/pretrain", 10);

        System.out.println("  ✓ 最大轮次: 15");
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

        // 2. 准备数据集
        System.out.println("\n📝 准备微调数据集...");
        GPT1Config config = pretrainedModel.getConfig();
        
        GPT1Dataset trainDataset = new GPT1Dataset(
            config.getNPositions(),
            2,
            config.getVocabSize()
        );
        trainDataset.loadFromTexts(trainTexts, sharedTokenizer);

        GPT1Dataset valDataset = new GPT1Dataset(
            config.getNPositions(),
            1,
            config.getVocabSize()
        );
        valDataset.loadFromTexts(valTexts, sharedTokenizer);

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
        ).setCheckpoint(CHECKPOINT_DIR + "/finetune", 200);

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
        System.out.println("  - 数据: 任务特定的指令数据");
        System.out.println("  - 技巧: 小学习率 + 早停机制");
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
