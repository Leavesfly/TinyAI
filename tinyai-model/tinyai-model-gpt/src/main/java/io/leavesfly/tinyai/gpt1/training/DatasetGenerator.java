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
                "Deep learning is a powerful subset of machine learning that utilizes artificial neural networks with multiple layers to progressively extract higher-level features from raw input data, enabling remarkable breakthroughs in computer vision, natural language processing, and speech recognition.",
                "Deep learning models can automatically learn hierarchical representations from raw data by building increasingly abstract concepts through successive layers, eliminating the need for manual feature engineering that traditionally required domain expertise.",
                "Deep learning has revolutionized fields like computer vision and natural language processing by achieving superhuman performance on tasks that were previously thought to require human intelligence, from image classification to machine translation.",
                "Convolutional neural networks are the foundation of modern computer vision systems, using learnable filters to automatically detect hierarchical features from edges and textures to complex objects and scenes, powering applications from autonomous vehicles to medical diagnostics.",
                "Recurrent neural networks excel at processing sequential data like text and time series by maintaining hidden states that capture information from previous time steps, with architectures like LSTM and GRU solving the vanishing gradient problem for long-range dependencies.",
                "Deep learning requires large amounts of labeled data for effective training, as the millions of parameters in modern architectures need sufficient examples to learn meaningful patterns without overfitting, leading to techniques like data augmentation and transfer learning.",
                "Transfer learning allows deep learning models to reuse knowledge from related tasks by leveraging pretrained weights, enabling rapid development of specialized applications with limited labeled data by fine-tuning on target domains.",
                "Backpropagation is the fundamental algorithm for training deep neural networks, efficiently computing gradients of the loss function with respect to each weight by propagating error signals backward through the network using the chain rule of calculus.",
                "Gradient descent optimizes neural network parameters by iteratively adjusting weights in the direction that minimizes the loss function, with variants like stochastic gradient descent providing computational efficiency and momentum methods accelerating convergence.",
                "Deep learning frameworks like TensorFlow and PyTorch simplify model development by providing automatic differentiation, GPU acceleration, and high-level APIs that allow researchers and practitioners to quickly prototype and deploy sophisticated neural network architectures.",
                "Activation functions introduce non-linearity into neural network computations, enabling networks to learn complex patterns that linear transformations alone cannot capture, with ReLU being the most widely used due to its simplicity and effectiveness.",
                "Dropout is a regularization technique that prevents overfitting in deep networks by randomly setting a fraction of neurons to zero during training, forcing the network to learn redundant representations and improving generalization to unseen data.",
                "Batch normalization accelerates training and improves model generalization by normalizing layer inputs across the mini-batch dimension, reducing internal covariate shift and allowing higher learning rates while acting as a regularizer.",
                "Deep learning models can achieve superhuman performance on specific tasks when trained on sufficient data, demonstrated by achievements like AlphaGo defeating world champions in Go and models achieving expert-level performance in medical image diagnosis.",
                "Pretrained models enable rapid development of AI applications with limited data by providing learned representations that capture general patterns, allowing practitioners to achieve strong performance through fine-tuning rather than training from scratch."
        ));
        
        texts.addAll(Arrays.asList(
                "Deep learning systems learn features automatically without manual feature engineering by discovering hierarchical representations through successive layers, where early layers detect simple patterns and deeper layers combine them into complex concepts.",
                "Convolutional layers extract spatial hierarchies from images and visual data by applying learned filters across the input, creating feature maps that progressively capture edges, textures, parts, and complete objects at different scales.",
                "Pooling layers reduce spatial dimensions while preserving important features by aggregating information over local regions, providing translation invariance and reducing computational requirements for subsequent layers.",
                "Deep architectures can learn abstract concepts through multiple levels of representation, with each layer transforming the representation from the previous layer into increasingly abstract and composite features that better capture the underlying structure of the data.",
                "Neural networks trained on large datasets show emergent abilities on new tasks that were not explicitly trained, including few-shot learning, chain-of-thought reasoning, and instruction following, demonstrating that scale can unlock unexpected capabilities.",
                "Autoencoders learn compressed representations through encoder-decoder architecture by training to reconstruct their input from a lower-dimensional bottleneck, discovering latent factors that capture the essential structure of the data.",
                "Variational autoencoders add probabilistic elements to learn smooth latent spaces by maximizing a lower bound on the data likelihood, enabling generation of new samples by sampling from the learned latent distribution.",
                "Generative adversarial networks pit two networks against each other for data generation, where a generator learns to produce realistic samples while a discriminator learns to distinguish real from generated data, driving both to improve through competition.",
                "Discriminator networks in GANs learn to distinguish real from generated samples by outputting probability estimates, becoming increasingly sophisticated at detecting artifacts as the generator improves its outputs.",
                "Generator networks learn to produce realistic samples that fool the discriminator by transforming random noise into structured outputs, gradually improving to match the distribution of real training data."
        ));
        
        texts.addAll(Arrays.asList(
                "Data augmentation artificially expands training datasets through transformations like rotation, cropping, flipping, and color jittering, improving model robustness and generalization by exposing the network to variations of the same examples.",
                "Mixup augmentation creates new training samples by linearly interpolating between pairs of examples and their labels, encouraging the model to learn linear behavior between classes and improving calibration and robustness.",
                "Cutout augmentation randomly masks regions of input images during training, forcing the model to rely on multiple features rather than focusing on specific discriminative regions, improving robustness to occlusion.",
                "Label smoothing prevents overconfidence by softening target distributions from hard one-hot vectors to softer distributions, regularizing the model and improving calibration on out-of-distribution examples.",
                "Warmup learning rate schedules gradually increase learning rate at training start to prevent early training instability, allowing the model to settle into a good region of parameter space before applying larger updates.",
                "Cosine annealing schedules decay learning rate following a cosine curve from initial to near-zero values, providing smooth decay that often achieves better final performance than step-based schedules.",
                "One cycle learning rate policy combines warmup and decay in a single cycle, increasing then decreasing the learning rate, enabling faster training with smaller datasets while achieving strong generalization.",
                "Gradient accumulation enables training with larger effective batch sizes by accumulating gradients over multiple forward passes before updating weights, useful when GPU memory limits batch size.",
                "Mixed precision training uses lower precision like float16 for faster computation and reduced memory while maintaining accuracy through techniques like loss scaling and master weights in float32.",
                "Distributed data parallel training splits batches across multiple GPUs, with each device computing gradients on its partition and synchronizing through all-reduce operations, enabling scaling to massive models and datasets."
        ));
        
        texts.addAll(Arrays.asList(
                "Cross-entropy loss measures the difference between predicted and true probability distributions, heavily penalizing confident wrong predictions while encouraging the model to output calibrated probabilities for classification tasks.",
                "Mean squared error loss penalizes large errors more heavily than small ones due to the squaring operation, making it suitable for regression tasks where outliers should have significant influence on the model.",
                "Binary cross-entropy loss is used for binary classification tasks, computing the negative log-likelihood of the correct class probability and working naturally with sigmoid activation for single-output binary prediction.",
                "Focal loss addresses class imbalance by down-weighting easy examples and focusing training on hard negatives, using a modulating factor that reduces loss for well-classified examples to prevent majority classes from dominating.",
                "Contrastive loss learns representations by comparing similar and dissimilar pairs, pulling similar examples closer in embedding space while pushing dissimilar examples apart, used in self-supervised and metric learning.",
                "Triplet loss learns embeddings by anchoring to reference points, ensuring that positive pairs are closer than negative pairs by a margin, used extensively in face recognition and person re-identification.",
                "Hinge loss is used for training support vector machines and margin-based models, penalizing predictions that are not sufficiently confident by requiring a margin between positive and negative examples.",
                "Smooth L1 loss is less sensitive to outliers than mean squared error by using a quadratic term for small errors and linear term for large errors, providing robustness while maintaining differentiability.",
                "Dice loss optimizes segmentation by measuring overlap directly between predicted and ground truth regions, handling class imbalance better than pixel-wise cross-entropy for foreground-background segmentation.",
                "IoU loss measures intersection over union for object detection, directly optimizing the evaluation metric rather than proxy losses, improving bounding box regression by penalizing based on overlap quality."
        ));
        
        return texts;
    }

    /**
     * 生成机器学习相关文本
     */
    private List<String> generateMachineLearningTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Machine learning enables computers to learn from data without explicit programming by automatically discovering patterns and building predictive models, transforming how we approach problems ranging from medical diagnosis to autonomous driving and natural language understanding.",
                "Supervised learning uses labeled data to train models for prediction tasks by learning the mapping between input features and target outputs, requiring human-annotated examples but enabling highly accurate predictions on new, unseen data.",
                "Unsupervised learning discovers patterns in data without labels by identifying inherent structure, clustering similar examples, or reducing dimensionality, valuable when labeled data is scarce or when exploring unknown datasets to reveal hidden insights.",
                "Reinforcement learning trains agents through rewards and punishments by having them interact with an environment, learn optimal policies through trial and error, and balance exploration of new actions with exploitation of known good strategies.",
                "Classification assigns discrete labels to input examples, forming the basis for applications like spam detection, image recognition, sentiment analysis, and medical diagnosis, with algorithms ranging from simple logistic regression to complex neural networks.",
                "Regression predicts continuous numerical values rather than discrete categories, used in applications like house price prediction, demand forecasting, and risk assessment, modeling relationships between input features and continuous target variables.",
                "Clustering groups similar data points together without predefined labels, revealing natural structure in data through algorithms like K-means, hierarchical clustering, and DBSCAN, useful for customer segmentation, anomaly detection, and data exploration.",
                "Dimensionality reduction simplifies data while preserving important information by projecting high-dimensional data into lower-dimensional spaces, using techniques like PCA for linear reduction and t-SNE or UMAP for nonlinear visualization.",
                "Feature engineering creates informative inputs for machine learning models by transforming raw data into features that better represent the underlying problem, often being the most critical factor in model performance and requiring domain expertise.",
                "Model validation ensures models generalize to unseen data by evaluating performance on held-out test sets, using cross-validation for robust estimates, and detecting overfitting before deploying models in production environments."
        ));
        
        texts.addAll(Arrays.asList(
                "Decision trees split data based on feature thresholds by recursively partitioning the feature space, creating interpretable models that can be visualized and understood, though prone to overfitting without proper pruning or ensemble methods.",
                "Random forests combine multiple decision trees for better predictions by training each tree on bootstrap samples with random feature subsets, achieving strong performance through averaging predictions and reducing variance.",
                "Gradient boosting sequentially corrects errors of previous models by training new models on the residuals of existing ensembles, with implementations like XGBoost and LightGBM achieving state-of-the-art results on tabular data competitions.",
                "Support vector machines find optimal decision boundaries by maximizing the margin between classes, using kernel functions to handle nonlinear separation, effective for classification tasks with clear margin of separation in high-dimensional spaces.",
                "K-nearest neighbors classifies based on similarity to training examples by finding the k closest points in feature space, requiring no training phase but needing efficient indexing for fast queries on large datasets.",
                "Naive Bayes uses probability theory for classification by applying Bayes theorem with strong independence assumptions between features, simple yet effective for text classification and other high-dimensional problems.",
                "Linear regression models relationships with straight lines by fitting coefficients that minimize the sum of squared errors between predictions and actual values, providing interpretable models for understanding feature importance and effect sizes.",
                "Logistic regression predicts probabilities for binary classification by applying the logistic function to a linear combination of features, providing calibrated probabilities and interpretable coefficients for understanding feature contributions.",
                "Principal component analysis reduces dimensionality linearly by finding orthogonal directions of maximum variance, enabling data compression, visualization, and noise reduction while preserving as much information as possible.",
                "K-means clustering partitions data into k groups by iteratively assigning points to nearest centroids and updating centroid positions, simple and scalable but requiring specification of the number of clusters and sensitive to initialization."
        ));
        
        texts.addAll(Arrays.asList(
                "Bias-variance tradeoff balances model complexity and generalization, where high bias leads to underfitting simple models while high variance leads to overfitting complex models, requiring careful selection of model complexity for optimal performance.",
                "Overfitting occurs when models memorize training data including noise and outliers rather than learning generalizable patterns, resulting in excellent training performance but poor performance on unseen test data, addressed through regularization and validation.",
                "Underfitting occurs when models are too simple to capture patterns in the data, resulting in poor performance on both training and test sets, addressed by increasing model complexity, adding features, or reducing regularization.",
                "Regularization adds constraints to prevent overfitting by penalizing model complexity through L1 or L2 penalties on weights, dropout for neural networks, or early stopping based on validation performance, improving generalization.",
                "Cross-validation estimates model performance robustly by partitioning data into multiple folds, training on subsets while validating on held-out portions, providing reliable performance estimates that account for data variability.",
                "Hyperparameter tuning optimizes model settings that are not learned during training, including learning rates, regularization strengths, and architecture choices, using methods from grid search to Bayesian optimization for finding optimal configurations.",
                "Grid search systematically explores parameter combinations by evaluating all combinations from predefined sets, guaranteeing finding the best within the search space but scaling poorly with the number of hyperparameters.",
                "Random search samples parameter space efficiently by randomly selecting configurations, often finding good hyperparameters faster than grid search because it explores more diverse values for each parameter.",
                "Bayesian optimization uses probabilistic models for tuning by building a surrogate model of the objective function and selecting promising configurations based on expected improvement, efficient for expensive-to-evaluate functions.",
                "Early stopping prevents overfitting during training by monitoring validation performance and halting when performance stops improving, avoiding unnecessary computation while ensuring the model does not overfit the training data."
        ));
        
        texts.addAll(Arrays.asList(
                "Precision measures the fraction of positive predictions that are correct, calculated as true positives divided by predicted positives, important in scenarios where false positives are costly such as spam detection and medical screening.",
                "Recall measures the fraction of actual positives that are found, calculated as true positives divided by actual positives, critical when missing positive cases is expensive such as disease detection or fraud identification.",
                "F1 score balances precision and recall in one metric by computing their harmonic mean, useful when both false positives and false negatives matter and when dealing with imbalanced class distributions.",
                "Accuracy measures overall prediction correctness as the fraction of correct predictions, simple and intuitive but potentially misleading when classes are imbalanced and the majority class dominates the metric.",
                "ROC curves plot true positive rate versus false positive rate at various classification thresholds, illustrating the trade-off between sensitivity and specificity and enabling selection of optimal thresholds for different application requirements.",
                "AUC summarizes ROC curve performance in one number by computing the area under the curve, providing a threshold-independent measure of classifier quality that is insensitive to class distribution and useful for model comparison.",
                "Confusion matrices show prediction breakdown by class by displaying counts of true positives, true negatives, false positives, and false negatives, enabling detailed analysis of which classes are being confused and the types of errors being made.",
                "Mean squared error measures average squared prediction error, penalizing large errors more heavily than small ones, widely used for regression tasks and serving as the optimization objective for many algorithms.",
                "Mean absolute error measures average absolute prediction error, less sensitive to outliers than MSE and providing a more interpretable metric in the original units of the target variable.",
                "R-squared measures variance explained by the model as the proportion of variance in the dependent variable that is predictable from the independent variables, providing an intuitive measure of model fit quality."
        ));
        
        return texts;
    }

    /**
     * 生成神经网络相关文本
     */
    private List<String> generateNeuralNetworkTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Neural networks are computing systems inspired by biological brains, consisting of interconnected artificial neurons organized in layers that learn to transform input data into useful outputs through adjustable connection weights trained via gradient descent.",
                "Artificial neurons receive inputs and produce outputs through activation functions by computing weighted sums of their inputs plus a bias term, then applying nonlinear activation to produce outputs that serve as inputs to subsequent layers.",
                "Weights and biases are learnable parameters in neural networks that determine how inputs are transformed through each layer, optimized during training to minimize the difference between predicted outputs and target values.",
                "Feedforward neural networks pass information in one direction from input to output through successive layers without cycles, forming the foundation for deep learning architectures that learn hierarchical representations through stacked transformations.",
                "Convolutional neural networks excel at processing grid-like data by using learnable filters that slide across spatial dimensions, sharing weights across positions to achieve translation equivariance and parameter efficiency for image and signal processing.",
                "Recurrent neural networks maintain state for sequential data by feeding outputs back as inputs to the next time step, enabling processing of variable-length sequences while capturing temporal dependencies in the hidden state.",
                "Long short-term memory networks solve vanishing gradient problems through gated architectures that control information flow, with forget, input, and output gates that learn what to remember and forget over long sequences.",
                "Gated recurrent units simplify LSTM architecture by combining the forget and input gates into a single update gate and merging the cell state and hidden state, achieving comparable performance with fewer parameters.",
                "Attention mechanisms allow models to focus on relevant inputs by computing weighted combinations of all positions based on learned compatibility scores, enabling direct modeling of long-range dependencies without recurrence.",
                "Transformer architecture revolutionized natural language processing by replacing recurrence with self-attention, enabling parallel processing of sequences and scaling to unprecedented model sizes that achieved breakthrough results across NLP benchmarks."
        ));
        
        texts.addAll(Arrays.asList(
                "Activation functions introduce non-linearity into neural networks, enabling them to approximate complex functions that would be impossible with linear transformations alone, with different activations suited to different architectures and tasks.",
                "ReLU activation is widely used for its simplicity and effectiveness, computing the maximum of zero and the input, enabling efficient gradient flow for positive values while avoiding vanishing gradients that plagued earlier activation functions.",
                "Sigmoid activation maps inputs to values between zero and one through the logistic function, historically used for binary classification output layers but less common in hidden layers due to vanishing gradient issues with saturated neurons.",
                "Tanh activation maps inputs to values between negative one and one, providing zero-centered outputs that can speed up convergence compared to sigmoid, but still suffering from vanishing gradients for large magnitude inputs.",
                "GELU activation combines properties of ReLU and sigmoid by multiplying the input by the cumulative distribution function of the standard normal distribution, used extensively in transformer models for its smooth nonlinearity and strong empirical performance.",
                "Softmax activation produces probability distributions by exponentiating logits and normalizing to sum to one, used in output layers for multi-class classification to produce interpretable class probabilities.",
                "Leaky ReLU prevents dying neurons with small negative slopes by allowing a small gradient for negative inputs, addressing the issue where standard ReLU neurons can become permanently inactive and stop learning.",
                "Swish activation is a smooth alternative to ReLU computed as input multiplied by sigmoid of input, discovered through neural architecture search and showing improved performance on deep networks while remaining simple to compute.",
                "Mish activation combines softplus and tanh for smoothness by computing input multiplied by tanh of softplus of input, providing smooth nonlinearity that has shown strong empirical results on various architectures.",
                "Parametric ReLU learns the negative slope during training rather than using a fixed value, allowing the network to adapt the activation function shape to the specific task and potentially improving performance over standard ReLU."
        ));
        
        texts.addAll(Arrays.asList(
                "Weight initialization affects training dynamics significantly by setting the starting point for optimization, with poor initialization leading to vanishing or exploding gradients that prevent effective learning regardless of the optimizer used.",
                "Xavier initialization suits sigmoid and tanh activations by setting initial weights proportional to the inverse square root of the fan-in, maintaining activation variance across layers and enabling effective gradient flow in networks with symmetric activations.",
                "He initialization works well with ReLU activations by setting initial weights proportional to the inverse square root of half the fan-in, accounting for the fact that ReLU zeros out half the activations on average and maintaining variance through the network.",
                "Batch normalization stabilizes training by normalizing activations across the mini-batch dimension to zero mean and unit variance, reducing internal covariate shift, allowing higher learning rates, and acting as a regularizer that often improves generalization.",
                "Layer normalization normalizes across feature dimensions for each sample independently, useful for recurrent networks and transformers where batch statistics would vary across time steps or where batch sizes are small.",
                "Dropout randomly zeroes neurons during training with probability p, forcing the network to learn redundant representations and preventing co-adaptation of features, effectively training an ensemble of thinned networks that improves generalization.",
                "Gradient clipping prevents exploding gradients by scaling gradients when their norm exceeds a threshold, stabilizing training in recurrent networks and transformers where gradients can grow exponentially through many layers.",
                "Residual connections enable training of very deep networks by adding the input of a layer to its output, creating identity shortcuts that allow gradients to flow directly through the network without degradation through many layers.",
                "Dense connections concatenate features from previous layers by connecting each layer to every other layer in a dense block, enabling feature reuse and improving gradient flow while requiring fewer parameters than equivalent non-dense networks.",
                "Skip connections allow gradients to flow directly through networks by providing identity paths that bypass one or more layers, enabling training of very deep architectures that would otherwise suffer from vanishing gradients or optimization difficulties."
        ));
        
        return texts;
    }

    /**
     * 生成NLP相关文本
     */
    private List<String> generateNLPTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Natural language processing enables computers to understand human language by applying computational techniques to analyze, interpret, and generate text and speech, powering applications from search engines and virtual assistants to translation services and sentiment analysis.",
                "Tokenization splits text into meaningful units called tokens by segmenting character sequences into words, subwords, or characters, forming the foundational preprocessing step that converts raw text into processable input for neural networks.",
                "Word embeddings represent words as dense vectors in continuous space where semantically similar words are close together, enabling neural networks to process text by mapping discrete vocabulary to learnable continuous representations.",
                "Word2Vec learns word embeddings from context windows by training neural networks to predict context words from target words (skip-gram) or target words from context (CBOW), discovering semantic relationships through distributional patterns.",
                "GloVe learns word vectors from global co-occurrence statistics by factorizing the word co-occurrence matrix, combining the advantages of global matrix factorization and local context window methods for learning word representations.",
                "FastText extends word embeddings to subword units by representing words as bags of character n-grams, enabling handling of out-of-vocabulary words and capturing morphological information that improves embeddings for morphologically rich languages.",
                "ELMo provides contextual word representations by training bidirectional language models on large corpora and extracting hidden states for each word, producing representations that capture syntax, semantics, and context-dependent meaning.",
                "BERT learns bidirectional representations from unlabeled text by masking random tokens and training to predict them based on both left and right context, producing contextual embeddings that achieve state-of-the-art on diverse NLP tasks through fine-tuning.",
                "GPT generates text using autoregressive language modeling by predicting each token given all previous tokens, trained on massive text corpora to learn general language patterns that enable few-shot and zero-shot task performance.",
                "Transformer models achieve state-of-the-art on NLP benchmarks by leveraging self-attention mechanisms that capture long-range dependencies without recurrence, enabling parallel processing and scaling to billions of parameters."
        ));
        
        texts.addAll(Arrays.asList(
                "Sentiment analysis determines the emotional tone of text by classifying text as positive, negative, or neutral, used extensively in social media monitoring, customer feedback analysis, and market research to understand opinions at scale.",
                "Named entity recognition identifies entities like people and places by detecting and classifying mentions of named entities in text into categories such as person, organization, location, and date, crucial for information extraction and knowledge graph construction.",
                "Part-of-speech tagging assigns grammatical roles to words by labeling each word with its grammatical category such as noun, verb, adjective, or adverb, providing syntactic information useful for parsing and downstream NLP tasks.",
                "Dependency parsing analyzes grammatical structure by identifying relationships between words, producing trees that represent which words depend on which others, useful for understanding sentence structure and extracting semantic relationships.",
                "Coreference resolution links mentions to the same entity by identifying which noun phrases refer to the same real-world entity, essential for understanding pronoun references and building coherent representations of documents.",
                "Machine translation converts text between languages by learning mappings from source to target languages using parallel corpora, with neural machine translation achieving human-level quality on many language pairs through encoder-decoder architectures.",
                "Text summarization creates concise versions of documents by extracting key sentences or generating new summaries that preserve important information, helping users quickly understand long documents and enabling efficient information consumption.",
                "Question answering finds answers in text passages by identifying relevant spans in documents that answer natural language questions, used in search engines, virtual assistants, and knowledge management systems.",
                "Text generation produces coherent natural language by predicting sequences of words that follow naturally from given prompts, powering applications from creative writing assistance to code generation and conversational agents.",
                "Dialogue systems enable conversational interactions by maintaining context across multiple turns, understanding user intents, and generating appropriate responses, used in virtual assistants, customer service bots, and interactive applications."
        ));
        
        texts.addAll(Arrays.asList(
                "Language models predict the probability of text sequences by learning distributions over sequences of tokens, enabling text generation, completion, and evaluation through computing likelihoods of sequences.",
                "N-gram models count word sequences for prediction by estimating probabilities based on counts of n-word sequences in training data, simple but effective for many applications despite inability to capture long-range dependencies.",
                "Neural language models use neural networks for text by learning continuous representations and complex patterns in language, enabling generalization beyond training sequences and achieving superior perplexity on held-out data.",
                "Perplexity measures how well language models predict text as the exponential of average negative log-likelihood, providing a standardized metric for comparing language models with lower values indicating better prediction.",
                "Beam search finds likely sequences during decoding by maintaining the k most probable partial sequences at each step, potentially finding better outputs than greedy decoding at the cost of increased computation.",
                "Sampling methods generate diverse text outputs by randomly selecting from probability distributions rather than always choosing the most likely token, enabling creative and varied generation while controlling quality through temperature.",
                "Temperature controls randomness in text generation by scaling logits before softmax, with higher temperatures producing more random and diverse outputs while lower temperatures make generation more deterministic and focused.",
                "Top-k sampling limits choices to k most likely tokens by zeroing out probabilities of tokens outside the top k before sampling, preventing selection of extremely unlikely tokens while maintaining some randomness.",
                "Top-p sampling uses nucleus of probability mass by selecting from the smallest set of tokens whose cumulative probability exceeds p, adapting the candidate pool based on distribution shape for more natural generation.",
                "Repetition penalty discourages repeated content by reducing the log-probability of tokens that have recently appeared, preventing loops and repetition that commonly occur in autoregressive generation from language models."
        ));
        
        texts.addAll(Arrays.asList(
                "Text classification assigns categories to documents by training models to predict labels from text content, used for spam detection, topic labeling, intent recognition, and sentiment analysis across diverse applications.",
                "Sequence labeling assigns tags to each token by predicting a label for every position in the sequence, used for named entity recognition, part-of-speech tagging, and chunking to provide token-level annotations.",
                "Sequence-to-sequence models transform input to output sequences by encoding the source sequence into a representation and decoding it into the target sequence, enabling translation, summarization, and question answering.",
                "Encoder-decoder architectures map source to target sequences by using separate networks for encoding input and generating output, with attention mechanisms allowing the decoder to focus on relevant parts of the encoded input.",
                "Attention allows decoders to focus on relevant encoder states by computing weighted combinations of encoder hidden states based on learned compatibility scores, enabling direct access to source information during generation.",
                "Copy mechanisms copy words from input to output by learning when to generate from vocabulary versus copy from source, useful for handling rare words and proper nouns in summarization and question answering.",
                "Coverage mechanisms prevent repetition in generation by tracking which source positions have been attended to and penalizing attention to already-covered positions, addressing the problem of repetitive outputs in sequence generation.",
                "Multi-task learning shares representations across tasks by training a single model on multiple related tasks simultaneously, enabling transfer between tasks and improving efficiency through shared parameters.",
                "Transfer learning applies pretrained knowledge to new tasks by initializing models with weights from pretrained models and fine-tuning on target tasks, dramatically reducing data requirements and training time.",
                "Fine-tuning adapts pretrained models to specific domains by continuing training on domain-specific data, adjusting learned representations to the nuances of the target domain while preserving general language knowledge."
        ));
        
        return texts;
    }

    /**
     * 生成Transformer相关文本
     */
    private List<String> generateTransformerTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Transformer architecture revolutionized natural language processing in 2017 by introducing self-attention mechanisms that replaced recurrent and convolutional layers, enabling parallel processing of sequences and scaling to models with hundreds of billions of parameters.",
                "Self-attention allows models to weigh the importance of different input positions by computing attention weights between all pairs of positions, enabling direct modeling of long-range dependencies without the sequential bottleneck of recurrent networks.",
                "Multi-head attention captures different aspects of relationships between tokens by running multiple attention operations in parallel with different learned projections, allowing the model to jointly attend to information from different representation subspaces.",
                "Positional encoding injects sequence order information into transformer models by adding position-dependent signals to input embeddings, using either sinusoidal functions of different frequencies or learned positional embeddings to distinguish token positions.",
                "The transformer consists of an encoder and decoder with stacked layers, where each layer contains multi-head self-attention and position-wise feed-forward networks with residual connections and layer normalization, enabling training of very deep architectures.",
                "Query, key, and value vectors are fundamental components of attention mechanisms, where queries are matched against keys to compute attention weights that determine how values are combined, enabling flexible information retrieval from the input.",
                "Scaled dot-product attention computes attention weights efficiently by taking the dot product of queries and keys, scaling by the square root of dimension, applying softmax for normalization, and using the result to weight the values.",
                "Feed-forward networks process each position independently in transformers by applying two linear transformations with a non-linear activation between them, providing the model with additional expressive power beyond the attention mechanism.",
                "Layer normalization is applied before or after transformer sublayers to normalize activations across feature dimensions, stabilizing training and enabling faster convergence by reducing the dependence of gradients on other layers.",
                "Residual connections help train deep transformer models effectively by adding the input of each sublayer to its output, creating identity shortcuts that allow gradients to flow directly through the network during backpropagation."
        ));
        
        texts.addAll(Arrays.asList(
                "BERT uses bidirectional transformers for language understanding tasks by masking random tokens and training to predict them from both left and right context, producing contextualized embeddings that capture rich semantic information.",
                "GPT models use decoder-only transformers for text generation by predicting each token given all previous tokens in an autoregressive manner, trained on large corpora to learn general language patterns that enable few-shot task performance.",
                "Masked language modeling is the pre-training objective for BERT where random tokens are replaced with a mask token and the model learns to predict the original tokens, enabling bidirectional representation learning without the need for labels.",
                "Causal language modeling predicts the next token in autoregressive models by computing the probability of each token given all previous tokens, used in GPT-style models for pre-training on large text corpora for generation tasks.",
                "Transformers eliminate recurrence and enable parallel processing of sequences by computing attention between all positions simultaneously, dramatically improving training efficiency compared to sequential RNN processing.",
                "Attention patterns can be visualized to understand model behavior by displaying the attention weights between query and key positions, revealing which tokens the model focuses on when making predictions and providing interpretability insights.",
                "The transformer model achieves state-of-the-art results across NLP tasks including machine translation, text classification, question answering, and named entity recognition, demonstrating the power of self-attention for learning transferable representations.",
                "Large-scale pre-training of transformers requires significant computational resources with models like GPT-3 using thousands of GPU-years, but enables strong few-shot performance and transfer learning to diverse downstream tasks.",
                "Fine-tuning adapts pretrained transformers to downstream tasks efficiently by continuing training on task-specific labeled data, typically requiring only a few epochs and small learning rates to achieve strong performance on specialized tasks.",
                "Transformer models scale effectively with increased data and parameters, with empirical scaling laws predicting performance based on compute budget, data size, and parameter count, guiding decisions about model and training configuration."
        ));
        
        texts.addAll(Arrays.asList(
                "Cross-attention connects encoder and decoder in sequence-to-sequence models by allowing the decoder to attend to encoder outputs, enabling the transfer of relevant source information during target sequence generation.",
                "Attention heads learn to focus on different linguistic phenomena such as syntactic dependencies, semantic relationships, and coreference, with different heads specializing in different aspects of language structure.",
                "Positional embeddings can be learned or defined using sinusoidal functions, with learned embeddings offering flexibility while sinusoidal encodings provide generalization to longer sequences and require no parameters.",
                "The transformer architecture is fully differentiable and trainable end-to-end, allowing gradients to flow from the loss function through all components including attention, feed-forward networks, and embedding layers.",
                "Sparse attention patterns improve efficiency for long sequences by limiting attention to subsets of positions rather than all pairs, reducing quadratic complexity to linear or near-linear while maintaining model quality.",
                "Flash attention reduces memory access for faster attention computation by reformulating the attention algorithm to minimize memory reads and writes between GPU high-bandwidth memory and on-chip SRAM, enabling training with longer sequences.",
                "Multi-query attention shares keys and values across attention heads while keeping separate queries, reducing memory bandwidth requirements and improving inference speed with minimal impact on model quality.",
                "Grouped-query attention balances efficiency and quality by sharing keys and values within groups of heads, offering a middle ground between multi-head and multi-query attention for better trade-offs.",
                "Sliding window attention limits attention to local neighborhoods around each position, enabling efficient processing of long documents while capturing local context that is often most relevant for language tasks.",
                "Global attention allows special tokens to attend to all positions and be attended by all positions, enabling the model to maintain global context while using efficient local attention for most tokens."
        ));
        
        texts.addAll(Arrays.asList(
                "Scaling laws predict model performance based on compute and data, showing power-law relationships between performance and scale, guiding decisions about optimal allocation of training compute across model size and data quantity.",
                "Emergent abilities appear in large language models unexpectedly, including arithmetic, reasoning, and task following that emerge only beyond certain scale thresholds, demonstrating qualitative changes from quantitative scaling.",
                "In-context learning allows models to learn from examples in prompts by recognizing patterns from few-shot demonstrations without gradient updates, leveraging the model's pretrained knowledge for rapid adaptation to new tasks.",
                "Chain-of-thought prompting improves reasoning step by step by encouraging models to generate intermediate reasoning steps before final answers, dramatically improving performance on arithmetic and logical reasoning tasks.",
                "Instruction tuning trains models to follow human instructions by fine-tuning on datasets of instructions and responses, improving the model's ability to understand and execute natural language commands across diverse tasks.",
                "RLHF uses human feedback to align models with human preferences by training reward models on human comparisons and optimizing the language model using reinforcement learning, producing more helpful and harmless outputs.",
                "Constitutional AI provides principles for model behavior by defining rules that the model should follow, enabling self-improvement through self-critique and revision based on constitutional principles without human labeling.",
                "Prompt engineering designs effective inputs for language models by carefully crafting instructions, examples, and formatting to maximize performance on specific tasks, becoming a crucial skill for deploying language models.",
                "Few-shot learning adapts to new tasks with minimal examples by conditioning on a few input-output pairs provided in the context, enabling rapid task adaptation without fine-tuning or gradient updates.",
                "Zero-shot learning generalizes to tasks without examples by following natural language instructions, demonstrating the impressive generalization capabilities of large language models trained on diverse text corpora."
        ));
        
        return texts;
    }

    /**
     * 生成优化算法相关文本
     */
    private List<String> generateOptimizationTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Optimization algorithms minimize loss functions by adjusting model parameters through iterative updates based on computed gradients, forming the computational foundation of training neural networks and machine learning models.",
                "Gradient descent updates parameters in the direction of steepest descent by computing the gradient of the loss with respect to parameters, taking steps proportional to the learning rate to minimize the objective function.",
                "Stochastic gradient descent uses mini-batches for faster updates by computing gradients on random subsets of training data rather than the full dataset, providing noisy but unbiased gradient estimates that enable faster iterations.",
                "Momentum accelerates gradient descent by accumulating past gradients in a velocity vector that smooths optimization trajectories, enabling faster convergence through ravines and past local minima in the loss landscape.",
                "Nesterov momentum looks ahead to improve gradient estimates by computing gradients at a predicted future position, providing more accurate updates that achieve better convergence rates than standard momentum.",
                "AdaGrad adapts learning rates for each parameter individually by dividing the learning rate by the square root of accumulated squared gradients, giving larger updates for infrequent parameters and smaller updates for frequent ones.",
                "RMSprop scales learning rates by recent gradient magnitudes by maintaining an exponentially weighted moving average of squared gradients, addressing AdaGrad's aggressive learning rate decay for non-convex optimization.",
                "Adam combines momentum and adaptive learning rates by maintaining estimates of first and second moments of gradients, providing bias-corrected updates that work well across a wide range of architectures and hyperparameters.",
                "AdamW decouples weight decay from gradient updates by applying L2 regularization directly rather than through the gradient, improving generalization compared to Adam with L2 regularization especially for large models.",
                "Learning rate schedules adjust learning rates during training to balance exploration and fine-tuning, with warmup for stability, constant phases for learning, and decay for convergence to final solutions."
        ));
        
        texts.addAll(Arrays.asList(
                "Batch gradient descent computes gradients on entire datasets for accurate but slow updates, requiring full passes through data before each parameter update but providing stable convergence for convex objectives.",
                "Mini-batch gradient descent balances speed and accuracy by using random subsets of data, enabling frequent updates with reasonable computational cost while providing stochastic noise that can help escape local minima.",
                "Learning rate controls the step size in parameter updates, with too large values causing divergence and too small values leading to slow convergence, making it the most important hyperparameter for optimization.",
                "Weight decay adds regularization by penalizing large weights through L2 penalties added to the loss or directly to updates, preventing overfitting and improving generalization to unseen data.",
                "Gradient clipping prevents exploding gradients by scaling gradient vectors when their norm exceeds a threshold, stabilizing training in recurrent networks and transformers where gradients can grow exponentially.",
                "Learning rate warmup stabilizes early training by starting with small learning rates and gradually increasing to the target value, preventing instability from large updates on randomly initialized parameters.",
                "Cosine annealing smoothly decreases learning rates following a cosine curve from initial to near-zero values, providing gradual refinement that often achieves better final performance than step-based schedules.",
                "Step decay reduces learning rates at fixed intervals by multiplying by a factor at specified epochs, providing sudden drops that enable exploration followed by fine-tuning in discrete phases.",
                "Exponential decay continuously reduces learning rates by multiplying by a factor each epoch or step, providing smooth monotonic decrease that gradually transitions from exploration to exploitation.",
                "Polynomial decay smoothly decreases learning rates to zero following a polynomial function, providing controlled decay that can be adjusted through the polynomial power for different optimization dynamics."
        ));
        
        texts.addAll(Arrays.asList(
                "First-order optimization uses gradients to update parameters by following the negative gradient direction, computationally efficient and widely applicable but potentially slow near optima due to lack of curvature information.",
                "Second-order optimization uses curvature information from the Hessian or approximations to determine update directions, potentially achieving faster convergence but with much higher computational cost per iteration.",
                "Newton method uses second derivatives for faster convergence by scaling gradients with the inverse Hessian, achieving quadratic convergence near optima but impractical for high-dimensional problems due to Hessian computation.",
                "Quasi-Newton methods approximate second derivatives by building estimates of the Hessian from gradient observations, achieving superlinear convergence without the computational cost of exact Hessian computation.",
                "L-BFGS is memory efficient for large-scale optimization by storing only a limited history of gradient differences to approximate the Hessian, enabling second-order methods for problems with millions of parameters.",
                "Natural gradient uses Fisher information for updates by preconditioning gradients with the inverse Fisher matrix, following the steepest descent direction in the space of probability distributions rather than parameter space.",
                "Gradient descent with momentum accelerates convergence by accumulating velocity in consistent gradient directions, dampening oscillations and enabling faster progress through ravines in the loss landscape.",
                "Nesterov accelerated gradient improves on momentum by computing the gradient at the extrapolated position, providing a look-ahead that achieves optimal convergence rate for smooth convex functions.",
                "AdaMax generalizes Adam to infinity norm by using the maximum of past gradients rather than their sum, sometimes providing more stable updates for problems with sparse gradients or noisy updates.",
                "NAdam combines Adam with Nesterov momentum by incorporating the momentum look-ahead into the Adam update rule, providing the benefits of both adaptive learning rates and momentum-based acceleration."
        ));
        
        return texts;
    }


    /**
     * 生成强化学习相关文本
     */
    private List<String> generateReinforcementLearningTexts() {
        List<String> texts = new ArrayList<>();
        
        texts.addAll(Arrays.asList(
                "Reinforcement learning trains agents through trial and error by having them interact with an environment, receive rewards or penalties based on their actions, and learn policies that maximize cumulative reward over time.",
                "Agents learn by interacting with environments through a cycle of observing states, selecting actions, receiving rewards, and updating their behavior based on experience, gradually improving their decision-making strategies.",
                "States represent situations agents encounter by encoding the current configuration of the environment, providing the information needed to select appropriate actions based on the agent's learned policy.",
                "Actions are choices agents make in states that affect the environment, ranging from discrete selections among finite options to continuous controls in robotics and autonomous systems.",
                "Rewards provide feedback on action quality by signaling the immediate desirability of state-action pairs, guiding agents toward behaviors that maximize long-term cumulative reward rather than short-term gains.",
                "Policies map states to actions by defining the agent's behavior, implemented as deterministic functions that select specific actions or stochastic distributions that assign probabilities to actions in each state.",
                "Value functions estimate expected future rewards by computing the discounted sum of rewards expected from a state under a given policy, enabling evaluation and comparison of different states and policies.",
                "Q-functions estimate action values in states by computing the expected return from taking a specific action in a state and following the policy thereafter, enabling comparison of action choices.",
                "Episodes are sequences of states, actions, and rewards from initial state to terminal state, forming complete interaction traces that provide training data for learning algorithms in episodic tasks.",
                "Exploration versus exploitation balances new and known actions by trading off between gathering information about unfamiliar states and actions versus exploiting current knowledge to maximize immediate rewards."
        ));
        
        texts.addAll(Arrays.asList(
                "Q-learning learns action values from experience by updating Q-value estimates based on observed rewards and estimated future values, converging to optimal Q-values under appropriate conditions.",
                "Deep Q-networks use neural networks for Q-functions by training networks to approximate Q-values for all state-action pairs, enabling generalization to unseen states in high-dimensional environments.",
                "Policy gradient methods directly optimize policies by computing gradients of expected return with respect to policy parameters, enabling learning of stochastic policies and handling continuous action spaces.",
                "Actor-critic methods combine policy and value learning by using a critic to estimate value functions that reduce variance in policy gradient estimates, achieving more stable and sample-efficient learning.",
                "Advantage functions reduce variance in policy gradients by measuring how much better an action is compared to the average action in a state, providing lower-variance estimates for policy improvement.",
                "Trust region policy optimization constrains policy updates by limiting the KL divergence between old and new policies, preventing destructive updates that could destabilize learning.",
                "Proximal policy optimization clips policy updates to prevent large changes, achieving similar benefits to trust region methods with simpler implementation and better sample efficiency.",
                "Deterministic policy gradients handle continuous actions by computing gradients through deterministic policies that map states directly to actions, enabling efficient learning in high-dimensional action spaces.",
                "Soft actor-critic maximizes entropy for exploration by adding an entropy bonus to the reward, encouraging policies to remain stochastic and explore more thoroughly while maximizing returns.",
                "Importance sampling reuses past experience by weighting samples according to the ratio of target and behavior policy probabilities, enabling off-policy learning and efficient use of historical data."
        ));
        
        texts.addAll(Arrays.asList(
                "Model-based RL learns environment dynamics by training models that predict next states and rewards given current states and actions, enabling planning and more sample-efficient learning through simulated experience.",
                "Planning uses models to simulate futures by generating predicted trajectories through learned environment models, enabling look-ahead search and decision-making without real environment interaction.",
                "Monte Carlo tree search plans through simulations by building search trees through random rollouts, balancing exploration and exploitation to find promising actions in games and planning problems.",
                "AlphaGo combined deep learning with tree search by using neural networks for policy and value evaluation within Monte Carlo tree search, achieving superhuman performance in the game of Go.",
                "Curiosity-driven exploration seeks novel states by using prediction error as intrinsic reward, encouraging agents to explore poorly understood regions of the state space to improve their environment models.",
                "Inverse reinforcement learning learns from demonstrations by inferring the reward function that would make expert behavior optimal, enabling learning of complex behaviors without manual reward specification.",
                "Imitation learning copies expert behavior by training policies to match demonstrated actions, providing a way to initialize policies or learn behaviors when reward specification is difficult.",
                "Multi-agent RL handles multiple interacting agents by extending single-agent algorithms to settings where multiple learners simultaneously adapt their policies, requiring consideration of strategic interactions.",
                "Hierarchical RL decomposes tasks into subtasks by learning policies at multiple temporal scales, with high-level policies selecting subgoals and low-level policies executing primitive actions to achieve them.",
                "Meta-RL learns to learn across many tasks by training agents that can quickly adapt to new tasks from few examples, learning general learning strategies that transfer across task distributions."
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
