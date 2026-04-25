package io.leavesfly.tinyai.minimind.training.demo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.util.*;

import static io.leavesfly.tinyai.minimind.training.demo.DemoConfig.*;

/**
 * MiniMind 训练演示 - 数据生成器
 * 
 * 负责生成各阶段训练数据：
 * - 预训练数据（通用语言知识）
 * - SFT数据（指令-回答对）
 * - DPO数据（偏好对）
 * - RL数据（带奖励样本）
 * 
 * @author TinyAI Team
 */
public class DemoDataGenerator {

    /**
     * 准备所有训练数据集
     * 如果数据文件已存在，则跳过生成
     */
    public static void prepareDatasets() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📦 步骤0: 准备训练数据集");
        System.out.println("=".repeat(80));

        File dataDir = new File(DATA_DIR);
        if (!dataDir.exists()) {
            dataDir.mkdirs();
            System.out.println("✓ 创建数据目录: " + DATA_DIR);
        }

        // 检查是否已有数据文件（从 minimind3 提取的中文数据）
        boolean hasPretrainData = new File(DATA_DIR + "/pretrain.jsonl").exists();
        boolean hasSFTData = new File(DATA_DIR + "/sft_train.jsonl").exists();
        boolean hasDPOData = new File(DATA_DIR + "/dpo_train.jsonl").exists();
        boolean hasRLData = new File(DATA_DIR + "/rl_train.jsonl").exists();
        boolean hasAgentData = new File(DATA_DIR + "/agent_rl.jsonl").exists();

        if (hasPretrainData && hasSFTData && hasDPOData && hasRLData && hasAgentData) {
            System.out.println("  ✓ 数据文件已存在，跳过生成");
            System.out.println("  ✓ pretrain.jsonl, sft_train.jsonl, dpo_train.jsonl, rl_train.jsonl, agent_rl.jsonl");
            System.out.println("\n✅ 数据集准备完成!");
            return;
        }

        // 如果缺少数据文件，生成默认演示数据
        if (!hasPretrainData) generatePretrainDataset();
        if (!hasSFTData) generateSFTDataset();
        if (!hasDPOData) generateDPODataset();
        if (!hasRLData) generateRLDataset();
        if (!hasAgentData) generateAgentDataset();

        System.out.println("\n✅ 数据集准备完成!");
    }

    /**
     * 生成预训练数据集 - 通用语言知识
     */
    public static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");

        List<String> texts = new ArrayList<>();

        // 1. 深度学习基础知识 (30条)
        texts.addAll(Arrays.asList(
            "Deep learning is a powerful subset of machine learning that utilizes artificial neural networks with multiple layers to progressively extract higher-level features from raw input data, enabling remarkable breakthroughs in computer vision, natural language processing, and speech recognition.",
            "Neural networks are computational systems inspired by biological brains, consisting of interconnected layers of artificial neurons that process and transform information through weighted connections, with each layer learning to recognize increasingly abstract patterns in the data.",
            "Backpropagation is the fundamental algorithm used to train neural networks by computing gradients of the loss function with respect to each weight in the network, propagating error signals backward from the output layer through hidden layers to update parameters via gradient descent.",
            "Gradient descent is an iterative optimization algorithm that minimizes the loss function by adjusting model parameters in the direction of steepest descent, with variants like stochastic gradient descent, mini-batch gradient descent, and momentum-based methods offering different trade-offs between speed and stability.",
            "Activation functions are essential components of neural networks that introduce non-linearity into the model, enabling it to learn complex patterns and decision boundaries, with popular choices including ReLU for hidden layers, sigmoid and tanh for specific architectures, and softmax for output classification layers.",
            "Convolutional neural networks are specialized deep learning architectures designed for processing grid-like data such as images, using convolutional layers to automatically learn hierarchical features, pooling layers to reduce spatial dimensions, and fully connected layers for final classification or regression tasks.",
            "Recurrent neural networks are a class of neural networks specifically designed to process sequential data by maintaining hidden states that capture information from previous time steps, with variants like LSTM and GRU addressing the vanishing gradient problem for long-range dependencies in sequences.",
            "The Transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to process entire sequences in parallel while capturing long-range dependencies, replacing sequential processing with efficient parallel computation that enables training on massive datasets.",
            "Attention mechanisms enable neural networks to dynamically focus on relevant parts of the input when producing each element of the output, allowing models to learn complex alignments between sequences and significantly improving performance in tasks like machine translation, text summarization, and question answering.",
            "The pre-training and fine-tuning paradigm has become a cornerstone of modern deep learning, where large models are first trained on vast amounts of unlabeled data to learn general representations, then adapted to specific downstream tasks with smaller labeled datasets, dramatically reducing the need for task-specific training data.",
            "Overfitting is a common challenge in deep learning where models learn to memorize training data including noise and outliers rather than generalizing underlying patterns, resulting in excellent performance on training data but poor performance on unseen test data, requiring careful regularization strategies to prevent.",
            "Regularization techniques are essential methods for preventing overfitting in neural networks by adding constraints or noise to the training process, including L1 and L2 weight penalties that encourage simpler models, dropout that randomly disables neurons, and data augmentation that expands the effective training set.",
            "Dropout is a powerful regularization technique that randomly sets a fraction of neurons to zero during each training forward pass, forcing the network to learn redundant representations and preventing co-adaptation of features, effectively training an ensemble of thinned networks that improves generalization.",
            "Batch normalization is a technique that normalizes the activations of each layer across the mini-batch dimension during training, reducing internal covariate shift, allowing higher learning rates, and acting as a regularizer, leading to faster convergence and more stable training of deep neural networks.",
            "The learning rate is a critical hyperparameter that controls the step size during gradient descent optimization, with too high values causing divergence and too low values leading to slow convergence, requiring careful tuning through techniques like learning rate schedules, warmup strategies, and adaptive methods.",
            "The Adam optimizer combines the benefits of momentum-based gradient descent with adaptive learning rates for each parameter, using estimates of first and second moments of gradients to adjust update sizes, making it one of the most popular optimization algorithms for training deep neural networks.",
            "Loss functions quantify the discrepancy between model predictions and ground truth labels, providing the optimization objective that guides parameter updates through gradient descent, with different tasks requiring appropriate loss functions such as cross-entropy for classification and mean squared error for regression.",
            "Cross-entropy loss is the predominant loss function for classification tasks, measuring the difference between predicted probability distributions and true class labels, penalizing confident wrong predictions heavily while rewarding correct predictions with high probability, working naturally with softmax output layers.",
            "Mean squared error is a standard loss function for regression problems that computes the average squared difference between predicted and target values, penalizing larger errors more heavily than smaller ones, providing a smooth and differentiable objective for gradient-based optimization.",
            "Early stopping is a practical regularization technique that monitors validation loss during training and halts training when validation performance stops improving, preventing overfitting by not allowing the model to continue training on the training set beyond the point of best generalization.",
            "Data augmentation artificially expands the training dataset by applying label-preserving transformations to existing samples, such as rotation, flipping, and cropping for images, or synonym replacement and back-translation for text, improving model robustness and reducing overfitting.",
            "Transfer learning leverages knowledge learned from source tasks with abundant data to improve performance on target tasks with limited data, by reusing pre-trained model weights either as fixed feature extractors or as initialization for fine-tuning, dramatically accelerating training on new tasks.",
            "Embedding layers transform discrete categorical inputs like words or items into dense continuous vectors in a lower-dimensional space, where semantically similar entities are mapped to nearby points, enabling neural networks to process categorical data effectively and capture latent relationships.",
            "Positional encoding addresses the lack of position awareness in self-attention mechanisms by adding position-dependent signals to input embeddings, using either sinusoidal functions of different frequencies or learned positional embeddings to help models understand the order of elements in sequences.",
            "Multi-head attention extends the basic attention mechanism by running multiple attention operations in parallel with different learned projections, allowing the model to jointly attend to information from different representation subspaces at different positions, capturing diverse types of relationships simultaneously.",
            "Feedforward networks in transformers consist of two linear transformations with a non-linear activation in between, applied independently to each position, providing the model with additional expressive power to transform attention outputs and learn complex functions of the representations.",
            "Layer normalization normalizes activations across the feature dimension for each sample independently, stabilizing the hidden state dynamics in recurrent networks and enabling faster training of transformers by reducing the dependence of gradients on other layers' parameters.",
            "Residual connections, also known as skip connections, add the input of a layer to its output, creating identity shortcuts that allow gradients to flow directly through the network, enabling training of very deep architectures that would otherwise suffer from vanishing or exploding gradients.",
            "The softmax function converts a vector of real-valued logits into a probability distribution where all values are non-negative and sum to one, making it ideal for multi-class classification output layers and attention weight computation in sequence-to-sequence models.",
            "Tokenization is the fundamental preprocessing step that breaks raw text into meaningful units called tokens, with approaches ranging from simple whitespace splitting to sophisticated subword methods like Byte-Pair Encoding and WordPiece that balance vocabulary size with coverage of rare words and morphological variations."
        ));

        // 2. 自然语言处理知识 (30条)
        texts.addAll(Arrays.asList(
            "Language models are neural networks trained to predict the probability distribution over sequences of words or tokens, forming the foundation for many natural language processing tasks including text generation, machine translation, and question answering by learning statistical patterns from large corpora.",
            "Autoregressive models generate text by predicting each subsequent token conditioned on all previously generated tokens, using techniques like teacher forcing during training and sampling strategies during inference to produce coherent and contextually appropriate sequences.",
            "BERT introduced the paradigm of bidirectional pre-training by masking random tokens in the input and training the model to predict them based on both left and right context, enabling rich contextual representations that can be fine-tuned for various downstream tasks.",
            "GPT models employ unidirectional causal language modeling where each token can only attend to previous tokens in the sequence, making them particularly well-suited for text generation tasks while being pretrained on massive text corpora to learn general language patterns.",
            "Fine-tuning adapts pretrained language models to specific downstream tasks by continuing training on task-specific labeled data, allowing the model to specialize its general knowledge while requiring significantly less data than training from scratch.",
            "Text classification is a fundamental NLP task that assigns predefined categories to documents or sentences, with applications ranging from sentiment analysis and topic classification to spam detection and intent recognition, often achieved by adding a classification head to pretrained models.",
            "Named entity recognition identifies and classifies named entities mentioned in unstructured text into predefined categories such as person names, organizations, locations, and dates, serving as a crucial component in information extraction and knowledge graph construction pipelines.",
            "Sentiment analysis determines the subjective emotional tone expressed in text, classifying opinions as positive, negative, or neutral, with applications in social media monitoring, customer feedback analysis, and market research to understand public opinion at scale.",
            "Machine translation automatically converts text from one natural language to another while preserving meaning and fluency, with modern systems using encoder-decoder transformer architectures trained on parallel corpora to achieve human-level translation quality on many language pairs.",
            "Question answering systems extract precise answers from given context passages or knowledge bases in response to natural language queries, with extractive QA identifying relevant spans and generative QA producing abstractive answers using knowledge from pretrained models.",
            "Text summarization condenses lengthy documents into shorter versions that retain key information and main ideas, with extractive approaches selecting important sentences and abstractive approaches generating novel summaries using seq2seq models with attention mechanisms.",
            "Text generation produces fluent and coherent natural language text for various applications including creative writing, dialogue systems, and code generation, with modern language models achieving impressive quality through autoregressive decoding and sophisticated sampling strategies.",
            "Perplexity is a standard evaluation metric for language models that measures how well the model predicts a held-out test set, computed as the exponential of the average negative log-likelihood, with lower values indicating better predictive performance.",
            "BLEU score evaluates machine translation quality by comparing n-gram overlap between candidate translations and reference translations, with modifications for brevity penalty and multiple reference handling, serving as the standard automatic metric for translation systems.",
            "Word embeddings like Word2Vec and GloVe learn dense vector representations of words that capture semantic relationships, placing similar words close in the embedding space and enabling arithmetic operations that reveal analogies between word relationships.",
            "Byte-pair encoding is a subword tokenization algorithm that iteratively merges the most frequent pairs of characters or character sequences, building a vocabulary that balances between character-level and word-level tokenization to handle rare and out-of-vocabulary words.",
            "Subword tokenization methods like BPE, WordPiece, and Unigram tokenize text into meaningful subword units that are larger than characters but smaller than words, enabling models to handle morphologically rich languages and unknown words while keeping vocabulary sizes manageable.",
            "Masked language modeling is the pre-training objective used by BERT where random tokens in the input are replaced with a special mask token, and the model learns to predict the original tokens based on surrounding context, enabling bidirectional representations.",
            "Causal language modeling predicts each token given only the preceding tokens, used in decoder-only models like GPT for pre-training, with the model learning to generate coherent text through next-token prediction on large corpora.",
            "Few-shot learning enables language models to perform new tasks by providing just a handful of examples in the prompt context, leveraging the model's pretrained knowledge and in-context learning abilities without updating model parameters.",
            "Zero-shot learning allows models to perform tasks they were not explicitly trained for by providing natural language instructions, demonstrating the impressive generalization capabilities of large language models pretrained on diverse text corpora.",
            "Prompt engineering is the practice of designing and optimizing input prompts to elicit desired behaviors from language models, involving careful phrasing, example selection, and formatting to maximize performance on specific tasks without model retraining.",
            "In-context learning refers to the ability of large language models to learn from examples provided in the input prompt, adapting their behavior to new tasks through pattern recognition without gradient updates or fine-tuning.",
            "Instruction tuning trains language models on datasets of instructions paired with desired responses, teaching models to follow user commands and improving their ability to understand and execute natural language instructions across diverse tasks.",
            "Reinforcement learning from human feedback trains language models to align with human preferences by collecting comparison data from human annotators, training a reward model on preferences, and optimizing the language model using PPO or similar algorithms.",
            "Temperature is a hyperparameter that controls the randomness of text generation by scaling the logits before softmax, with higher values producing more diverse and creative outputs and lower values making generations more deterministic and focused.",
            "Top-k sampling restricts the selection of next tokens to the k most probable candidates, providing a balance between quality and diversity in text generation by preventing the model from selecting extremely low probability tokens.",
            "Top-p sampling, also called nucleus sampling, dynamically selects from the smallest set of tokens whose cumulative probability exceeds threshold p, adapting the candidate pool based on the probability distribution's shape for more natural generation.",
            "Beam search explores multiple generation paths simultaneously by keeping the k most probable partial sequences at each step, potentially finding higher quality outputs than greedy decoding at the cost of increased computation and risk of generic outputs.",
            "Greedy decoding always selects the single most probable next token at each generation step, producing deterministic outputs that may be repetitive or generic, often serving as a baseline for comparison with more sophisticated decoding strategies."
        ));

        // 3. 机器学习概念 (30条)
        texts.addAll(Arrays.asList(
            "Supervised learning is the most common machine learning paradigm where models learn from labeled training data consisting of input-output pairs, learning to map inputs to outputs by minimizing prediction error on training examples while generalizing to unseen data.",
            "Unsupervised learning discovers hidden patterns and structures in data without explicit labels, using techniques like clustering, dimensionality reduction, and density estimation to extract meaningful representations from unlabeled datasets.",
            "Reinforcement learning trains agents to make sequential decisions by interacting with an environment, receiving rewards or penalties based on actions, and learning optimal policies that maximize cumulative reward through trial and error exploration.",
            "Classification is a supervised learning task that assigns input instances to discrete categories or classes, used in applications like spam detection, image recognition, and medical diagnosis, with algorithms ranging from logistic regression to deep neural networks.",
            "Regression predicts continuous numerical values rather than discrete categories, modeling the relationship between input features and a continuous target variable through techniques like linear regression, polynomial regression, and neural network regression.",
            "Clustering algorithms partition data into groups of similar instances without prior knowledge of group labels, with methods like K-means, hierarchical clustering, and DBSCAN each offering different approaches to defining and discovering cluster structure.",
            "Dimensionality reduction transforms high-dimensional data into lower-dimensional representations while preserving important information, using techniques like PCA for linear projection and t-SNE or UMAP for nonlinear embedding that reveal latent structure.",
            "Feature engineering creates informative input variables from raw data through domain knowledge and transformation techniques, often being the most critical factor in model performance and requiring expertise in both the domain and machine learning.",
            "Cross-validation provides robust estimates of model generalization by partitioning data into multiple folds, training on subsets while validating on held-out portions, with k-fold cross-validation being the most common approach for reliable performance assessment.",
            "Train-test split is the basic data partitioning strategy that divides the dataset into separate portions for training and evaluation, typically using 70-80 percent for training and the remainder for testing to assess how well the model generalizes.",
            "Validation sets are held out from training data specifically for hyperparameter tuning and model selection, providing an unbiased estimate of model performance during development while the test set remains untouched for final evaluation.",
            "Precision measures the proportion of positive predictions that are actually correct, calculated as true positives divided by the sum of true positives and false positives, important in scenarios where false positives are costly.",
            "Recall, also known as sensitivity, measures the proportion of actual positives that are correctly identified, calculated as true positives divided by the sum of true positives and false negatives, crucial when missing positive cases is expensive.",
            "F1 score provides a balanced measure of classification performance by computing the harmonic mean of precision and recall, useful when both false positives and false negatives are important and the class distribution is imbalanced.",
            "Accuracy measures the overall proportion of correct predictions, calculated as the number of correct predictions divided by total predictions, but can be misleading for imbalanced datasets where one class dominates.",
            "Confusion matrices provide detailed visualization of classification performance by showing the counts of true positives, true negatives, false positives, and false negatives, enabling analysis of which classes are being confused with others.",
            "ROC curves plot the true positive rate against the false positive rate at various classification thresholds, illustrating the trade-off between sensitivity and specificity and enabling threshold selection for different application requirements.",
            "AUC, or area under the ROC curve, provides a single scalar measure of classifier performance that is insensitive to class distribution and classification threshold, with values ranging from 0.5 for random guessing to 1.0 for perfect classification.",
            "The bias-variance tradeoff describes the fundamental tension between model complexity and generalization, where high bias leads to underfitting simple models while high variance leads to overfitting complex models, requiring careful balance.",
            "Ensemble methods combine predictions from multiple models to improve performance over individual models, leveraging the wisdom of crowds effect where diverse models make different errors that cancel out when aggregated.",
            "Bagging, or bootstrap aggregating, reduces variance by training multiple models on different bootstrap samples of the training data and averaging their predictions, with random forests being the most famous bagging-based algorithm.",
            "Boosting sequentially trains weak learners, with each new model focusing on correcting errors made by previous models, combining them through weighted voting to create a strong ensemble, with AdaBoost and gradient boosting as prominent examples.",
            "Random forests are ensemble methods that build multiple decision trees on bootstrap samples with random feature subsets at each split, combining their predictions through averaging or voting to achieve robust and accurate classifications.",
            "Gradient boosting builds decision trees sequentially where each tree corrects the residual errors of previous trees, with implementations like XGBoost, LightGBM, and CatBoost achieving state-of-the-art results on tabular data competitions.",
            "Neural architecture search automates the design of neural network architectures using optimization techniques like evolutionary algorithms, reinforcement learning, or gradient-based methods, potentially discovering novel architectures that outperform human-designed ones.",
            "Hyperparameter tuning optimizes model configuration parameters that are not learned during training, using methods like grid search, random search, and Bayesian optimization to find settings that maximize validation performance.",
            "Grid search exhaustively evaluates all combinations of hyperparameter values from predefined sets, guaranteeing finding the best combination within the search space but scaling poorly with the number of hyperparameters.",
            "Random search samples hyperparameter combinations randomly from specified distributions, often finding good configurations faster than grid search because it explores the hyperparameter space more efficiently.",
            "Bayesian optimization uses probabilistic models to guide the hyperparameter search, building a surrogate model of the objective function and selecting promising configurations to evaluate based on acquisition functions.",
            "Meta-learning, or learning to learn, develops algorithms that can quickly adapt to new tasks with minimal data by learning from the experience of learning many related tasks, enabling few-shot and one-shot learning capabilities."
        ));

        // 4. AI伦理与应用 (30条)
        texts.addAll(Arrays.asList(
            "Artificial intelligence is transforming virtually every industry from healthcare and finance to transportation and entertainment, automating routine tasks, augmenting human capabilities, and creating new possibilities that were previously unimaginable.",
            "AI ethics encompasses the principles and practices that ensure artificial intelligence systems are developed and deployed in ways that are fair, transparent, accountable, and beneficial to humanity, addressing concerns about bias, privacy, and societal impact.",
            "Fairness in AI requires careful attention to ensure models do not discriminate against protected groups based on race, gender, age, or other characteristics, involving both technical approaches like fairness constraints and organizational practices like diverse teams.",
            "Bias in training data reflects historical inequities and societal prejudices that can be amplified by machine learning models, requiring careful data collection, preprocessing, and ongoing monitoring to identify and mitigate discriminatory outcomes.",
            "Transparency in AI systems enables stakeholders to understand how models make decisions, what factors influence predictions, and what limitations exist, building trust and enabling meaningful human oversight of automated decision-making processes.",
            "Explainable AI develops methods to make complex model predictions interpretable to humans, including techniques like feature attribution, rule extraction, and counterfactual explanations that help users understand why a model made a particular prediction.",
            "Privacy protection in AI systems addresses concerns about personal data collection, storage, and use, employing techniques like differential privacy, federated learning, and secure multi-party computation to enable learning while preserving individual privacy.",
            "Data security ensures that training data and model parameters are protected from unauthorized access, tampering, and theft, implementing encryption, access controls, and secure computation environments to safeguard sensitive information.",
            "AI safety research focuses on ensuring that AI systems behave as intended, avoiding unintended consequences and catastrophic failures, addressing challenges like reward hacking, distributional shift, and safe exploration in reinforcement learning.",
            "Robustness in AI systems means they perform reliably under adversarial conditions, distribution shift, and unexpected inputs, requiring techniques like adversarial training, uncertainty estimation, and out-of-distribution detection.",
            "Adversarial examples are carefully crafted inputs that cause machine learning models to make incorrect predictions, highlighting vulnerabilities in neural networks and motivating research into adversarial defense and verification methods.",
            "Model interpretability techniques reveal which factors influence model predictions, including permutation feature importance, SHAP values, and attention visualization, helping practitioners debug models and build user trust.",
            "Feature importance quantifies the contribution of each input variable to model predictions, enabling feature selection, model debugging, and domain insights that help practitioners understand what patterns the model has learned.",
            "Attention visualization shows which parts of the input the model focuses on when making predictions, particularly useful in transformers and attention-based models for understanding how information flows through the network.",
            "Counterfactual explanations describe what minimal changes to the input would result in a different prediction, helping users understand decision boundaries and providing actionable insights for how to achieve desired outcomes.",
            "AI applications in healthcare include diagnostic imaging analysis, drug discovery, personalized treatment recommendations, and clinical workflow optimization, potentially improving patient outcomes while raising important ethical considerations.",
            "Computer vision enables autonomous vehicles to perceive their environment through cameras and sensors, detecting pedestrians, vehicles, and road conditions while making real-time decisions that ensure safe and efficient navigation.",
            "Natural language processing powers virtual assistants like Siri, Alexa, and ChatGPT that can understand and respond to user queries, translate languages, summarize documents, and engage in conversations on diverse topics.",
            "Recommendation systems personalize user experiences on platforms like Netflix, Amazon, and Spotify by predicting preferences and suggesting relevant content, balancing accuracy with diversity, novelty, and fairness considerations.",
            "Fraud detection systems use machine learning to identify suspicious transactions and activities in real-time, protecting financial institutions and their customers from various types of fraud while minimizing false positives that inconvenience legitimate users.",
            "Predictive maintenance uses sensor data and machine learning to anticipate equipment failures before they occur, enabling proactive maintenance that reduces downtime and costs in manufacturing, aviation, and energy sectors.",
            "Drug discovery leverages AI to accelerate the identification and optimization of potential drug candidates, predicting molecular properties, designing novel compounds, and reducing the time and cost of bringing new medicines to market.",
            "Climate modeling employs machine learning to improve predictions of weather patterns, climate change impacts, and extreme events, helping policymakers and communities prepare for and mitigate environmental challenges.",
            "Robotics combines AI perception and decision-making with physical actuators to create machines that can interact with the physical world, from industrial automation to service robots and autonomous systems.",
            "Speech recognition converts spoken language into text, enabling voice-based interfaces, transcription services, and accessibility tools that bridge the gap between human communication and digital systems.",
            "Image generation models create realistic synthetic images from textual descriptions or other inputs, enabling applications in art, design, entertainment, and data augmentation while raising concerns about deepfakes and misinformation.",
            "Style transfer applies the artistic style of one image to the content of another, creating novel artworks that combine the subject matter of one image with the visual aesthetics of another using deep neural networks.",
            "Anomaly detection identifies unusual patterns that deviate from expected behavior, used in fraud detection, system monitoring, quality control, and security applications where identifying outliers is critical.",
            "Time series forecasting predicts future values based on historical sequences, used in financial markets, energy demand, inventory management, and weather prediction where understanding temporal patterns is essential.",
            "Knowledge graphs organize information as networks of entities and relationships, enabling semantic search, question answering, and reasoning over structured knowledge that complements the capabilities of language models."
        ));

        // 5. 编程与软件开发 (30条)
        texts.addAll(Arrays.asList(
            "Programming languages serve as the fundamental interface between human thought and machine execution, providing syntax and semantics that allow developers to express algorithms, manage data, and build complex software systems that power modern technology infrastructure.",
            "Python has become the dominant language for machine learning and data science due to its clean syntax, extensive library ecosystem including NumPy, Pandas, and PyTorch, and strong community support that accelerates development and experimentation.",
            "Java provides a robust platform for enterprise software development with its strong type system, garbage collection, cross-platform compatibility through the JVM, and mature ecosystem of frameworks like Spring that enable scalable backend systems.",
            "JavaScript is essential for modern web development, running in browsers to create interactive user interfaces and on servers through Node.js for backend services, with frameworks like React, Vue, and Express enabling full-stack JavaScript applications.",
            "Data structures are fundamental building blocks that organize and store data efficiently, including arrays for sequential access, hash tables for fast lookups, trees for hierarchical data, and graphs for relationship modeling.",
            "Algorithms provide systematic procedures for solving computational problems, from basic sorting and searching to advanced optimization and machine learning, with analysis focusing on time complexity, space complexity, and correctness.",
            "Version control systems like Git track changes to source code over time, enabling collaboration among developers, maintaining complete history of modifications, supporting branching for parallel development, and facilitating code review processes.",
            "Git has become the standard version control system in software development, providing distributed repositories, efficient branching and merging, and integration with platforms like GitHub and GitLab for collaborative development workflows.",
            "Code review is a quality assurance practice where developers examine each other's code before integration, catching bugs, ensuring adherence to coding standards, sharing knowledge across the team, and improving overall code maintainability.",
            "Unit testing verifies that individual components or functions work correctly in isolation, using frameworks like JUnit for Java and pytest for Python, providing fast feedback and enabling confident refactoring through automated regression detection.",
            "Integration testing validates that different components work correctly together when combined, testing interfaces between modules, database interactions, and external API integrations to ensure the system functions as a cohesive whole.",
            "Continuous integration automatically builds and tests code changes as they are committed, catching integration issues early, providing rapid feedback to developers, and ensuring the main branch remains in a deployable state at all times.",
            "Software design patterns provide reusable solutions to commonly occurring problems in software architecture, including creational patterns for object creation, structural patterns for composition, and behavioral patterns for communication between objects.",
            "Object-oriented programming organizes software around objects that encapsulate data and behavior, using concepts like classes, inheritance, and polymorphism to create modular, reusable, and maintainable code structures.",
            "Functional programming treats computation as evaluation of mathematical functions, emphasizing immutability, pure functions without side effects, and higher-order functions, enabling easier reasoning about code and natural parallelization.",
            "Debugging is the systematic process of identifying, analyzing, and fixing defects in software, using tools like debuggers, logging, and assertions, with skills including reproducing bugs, isolating causes, and verifying fixes.",
            "Profiling measures the runtime performance of software, identifying bottlenecks in CPU usage, memory allocation, and I/O operations, enabling data-driven optimization decisions that improve application responsiveness and resource efficiency.",
            "Optimization improves software performance through techniques like algorithm selection, caching, lazy evaluation, and vectorization, balancing speed improvements against code complexity and maintainability considerations.",
            "Memory management ensures efficient allocation and deallocation of memory resources, with languages like Java and Python using garbage collection while C and C++ require manual memory management, preventing leaks and fragmentation.",
            "Exception handling provides structured mechanisms for dealing with runtime errors, using try-catch blocks to gracefully recover from unexpected conditions, ensuring robust error propagation and preventing crashes in production systems.",
            "API design defines the interfaces through which software components communicate, balancing ease of use with flexibility, establishing contracts between providers and consumers, and evolving over time while maintaining backward compatibility.",
            "Documentation communicates the purpose, usage, and implementation details of software, including API references, tutorials, and inline comments, enabling other developers to understand, use, and maintain the code effectively.",
            "Code refactoring improves the internal structure of existing code without changing its external behavior, addressing technical debt, improving readability, and making the codebase more maintainable and extensible over time.",
            "Modularity decomposes complex systems into independent, interchangeable modules with well-defined interfaces, enabling parallel development, easier testing, and the ability to replace or upgrade components independently.",
            "Abstraction hides implementation complexity behind simpler interfaces, allowing developers to work at higher levels of abstraction and focus on what code does rather than how it accomplishes it, reducing cognitive load and enabling reuse.",
            "Encapsulation bundles data with the methods that operate on that data, restricting direct access to internal state and providing controlled interfaces, protecting invariants and enabling implementation changes without affecting clients.",
            "Inheritance enables code reuse by allowing classes to inherit properties and methods from parent classes, establishing is-a relationships and supporting hierarchical organization of types in object-oriented systems.",
            "Polymorphism allows objects of different types to be treated uniformly through shared interfaces, enabling flexible and extensible code that can work with new types without modification, supporting the open-closed principle.",
            "Dependency injection is a design pattern that removes hard-coded dependencies by injecting them from outside, improving testability through easy mocking, enabling loose coupling between components, and facilitating configuration.",
            "Clean code principles emphasize writing readable, maintainable, and expressive code through meaningful naming, small focused functions, minimal comments, consistent formatting, and avoiding code duplication, creating software that is easy to understand and modify."
        ));

        String filePath = DATA_DIR + "/pretrain.jsonl";
        List<String> jsonLines = new ArrayList<>();
        for (String text : texts) {
            jsonLines.add(new JSONObject().put("text", text).toString());
        }
        writeJsonlFile(jsonLines, filePath);
        System.out.println("  ✓ 预训练数据: " + texts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    /**
     * 生成监督微调数据集 - 指令-回答对
     */
    public static void generateSFTDataset() throws IOException {
        System.out.println("\n📝 生成监督微调数据集...");

        List<String> trainTexts = new ArrayList<>();

        trainTexts.addAll(Arrays.asList(
            "Question: What is deep learning? Answer: Deep learning is a subset of machine learning using neural networks with multiple layers",
            "Question: Explain backpropagation Answer: Backpropagation is an algorithm that computes gradients to update neural network weights",
            "Question: What is overfitting? Answer: Overfitting occurs when a model memorizes training data instead of learning general patterns",
            "Question: Define gradient descent Answer: Gradient descent is an optimization algorithm that minimizes loss by updating parameters",
            "Question: What are transformers? Answer: Transformers are neural network architectures using self-attention mechanisms",
            "Question: Explain attention mechanism Answer: Attention allows models to focus on relevant parts of input when processing information",
            "Question: What is fine-tuning? Answer: Fine-tuning adapts pre-trained models to specific tasks with additional training",
            "Question: Define reinforcement learning Answer: Reinforcement learning trains agents through rewards and penalties for actions",
            "Question: What is tokenization? Answer: Tokenization splits text into smaller units like words or subwords for processing",
            "Question: Explain embedding layers Answer: Embedding layers convert discrete tokens into continuous vector representations",
            "Instruction: Write a Python function to add two numbers Answer: def add(a, b): return a + b",
            "Instruction: Create a loop to print numbers 1 to 5 Answer: for i in range(1, 6): print(i)",
            "Instruction: Define a class for a person Answer: class Person: def __init__(self, name): self.name = name",
            "Instruction: Implement binary search Answer: Binary search finds elements in sorted arrays by dividing search space",
            "Instruction: Explain list comprehension Answer: List comprehension creates lists using concise syntax: [x*2 for x in range(10)]",
            "Task: Summarize this concept: Neural networks Answer: Networks of artificial neurons that learn patterns from data",
            "Task: Classify this as positive or negative: I love this product Answer: Positive sentiment",
            "Task: Translate to simple terms: Convolutional neural network Answer: Network specialized for processing grid-like data such as images",
            "Task: Generate a creative name for an AI assistant Answer: MindBot - your intelligent companion",
            "Task: Suggest improvements for code readability Answer: Use meaningful variable names and add comments",
            "Question: How does BERT work? Answer: BERT uses bidirectional transformers to understand context from both directions",
            "Question: What is GPT? Answer: GPT is a generative pre-trained transformer for autoregressive language modeling",
            "Question: Explain cross-entropy loss Answer: Cross-entropy measures difference between predicted and true probability distributions",
            "Question: What is batch normalization? Answer: Batch normalization normalizes layer inputs to stabilize and speed up training",
            "Question: Define learning rate Answer: Learning rate controls step size in gradient descent optimization",
            "Instruction: Sort a list in Python Answer: sorted_list = sorted(my_list) or my_list.sort()",
            "Instruction: Handle exceptions in Python Answer: try: risky_code() except Exception as e: handle_error(e)",
            "Instruction: Read a file in Python Answer: with open('file.txt', 'r') as f: content = f.read()",
            "Instruction: Create a dictionary Answer: my_dict = {'key1': 'value1', 'key2': 'value2'}",
            "Instruction: Use list slicing Answer: first_three = my_list[:3], last_two = my_list[-2:]",
            "Task: Explain AI ethics Answer: AI ethics ensures responsible development considering fairness bias and transparency",
            "Task: Compare supervised and unsupervised learning Answer: Supervised uses labels unsupervised finds patterns without labels",
            "Task: Recommend a machine learning algorithm Answer: For classification try random forest or neural networks",
            "Task: Debug this error: IndexError Answer: Check array bounds and ensure index is within valid range",
            "Task: Optimize slow code Answer: Profile to find bottlenecks use efficient algorithms and data structures",
            "Question: What is transfer learning? Answer: Transfer learning reuses pre-trained models for new related tasks",
            "Question: Explain dropout regularization Answer: Dropout randomly disables neurons during training to prevent overfitting",
            "Question: What is a loss function? Answer: Loss function quantifies difference between model predictions and true values",
            "Question: Define activation functions Answer: Activation functions introduce non-linearity enabling networks to learn complex patterns",
            "Question: What is early stopping? Answer: Early stopping halts training when validation performance stops improving",
            "Instruction: Import libraries in Python Answer: import numpy as np, import pandas as pd",
            "Instruction: Create a virtual environment Answer: python -m venv myenv, source myenv/bin/activate",
            "Instruction: Install packages Answer: pip install package_name",
            "Instruction: Format strings in Python Answer: f'Hello {name}' or 'Hello {}'.format(name)",
            "Instruction: Use lambda functions Answer: square = lambda x: x**2",
            "Task: Improve model accuracy Answer: Try feature engineering data augmentation or ensemble methods",
            "Task: Reduce training time Answer: Use smaller batches GPU acceleration or model pruning",
            "Task: Prevent data leakage Answer: Split data before preprocessing keep test set completely separate",
            "Task: Handle imbalanced data Answer: Use oversampling undersampling or class weights",
            "Task: Validate model performance Answer: Use cross-validation and multiple metrics",
            "Question: What is ensemble learning? Answer: Ensemble learning combines multiple models to improve predictions",
            "Question: Explain feature engineering Answer: Feature engineering creates informative variables from raw data",
            "Question: What is regularization? Answer: Regularization adds penalties to prevent overfitting and improve generalization",
            "Question: Define precision and recall Answer: Precision is accuracy of positive predictions recall is coverage of actual positives",
            "Question: What is the bias-variance tradeoff? Answer: Balancing model complexity to minimize both underfitting and overfitting",
            "Instruction: Use NumPy arrays Answer: import numpy as np, arr = np.array([1, 2, 3])",
            "Instruction: Plot data with Matplotlib Answer: import matplotlib.pyplot as plt, plt.plot(x, y), plt.show()",
            "Instruction: Create pandas DataFrame Answer: import pandas as pd, df = pd.DataFrame(data)",
            "Instruction: Apply function to DataFrame Answer: df['new_col'] = df['col'].apply(lambda x: x*2)",
            "Instruction: Split train-test data Answer: from sklearn.model_selection import train_test_split"
        ));

        // 验证集
        List<String> valTexts = new ArrayList<>();
        for (int i = 0; i < 10 && i < trainTexts.size(); i++) {
            valTexts.add(trainTexts.get(i));
        }

        String trainPath = DATA_DIR + "/sft_train.jsonl";
        List<String> trainJsonLines = new ArrayList<>();
        for (String text : trainTexts) {
            String[] parts = text.split(" Answer: ", 2);
            if (parts.length == 2) {
                trainJsonLines.add(new JSONObject()
                        .put("instruction", parts[0])
                        .put("input", "")
                        .put("output", parts[1]).toString());
            }
        }
        writeJsonlFile(trainJsonLines, trainPath);
        System.out.println("  ✓ SFT训练集: " + trainTexts.size() + " 条");

        String valPath = DATA_DIR + "/sft_val.jsonl";
        List<String> valJsonLines = new ArrayList<>();
        for (String text : valTexts) {
            String[] parts = text.split(" Answer: ", 2);
            if (parts.length == 2) {
                valJsonLines.add(new JSONObject()
                        .put("instruction", parts[0])
                        .put("input", "")
                        .put("output", parts[1]).toString());
            }
        }
        writeJsonlFile(valJsonLines, valPath);
        System.out.println("  ✓ SFT验证集: " + valTexts.size() + " 条");
    }

    /**
     * 生成DPO偏好数据集 - 偏好对 (prompt, chosen, rejected)
     */
    public static void generateDPODataset() throws IOException {
        System.out.println("\n📝 生成DPO偏好数据集...");

        List<String> dpoTexts = new ArrayList<>();

        // 格式: prompt|||chosen|||rejected
        dpoTexts.addAll(Arrays.asList(
            "Question: What is deep learning?|||Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data.|||Deep learning uses neural networks.",
            "Question: Explain backpropagation|||Backpropagation is the algorithm that computes gradients by applying the chain rule backwards through the network, enabling efficient weight updates.|||It updates weights.",
            "Question: What is overfitting?|||Overfitting occurs when a model learns training data too well including noise, resulting in poor generalization. Solutions include regularization and dropout.|||Model memorizes data.",
            "Question: Define gradient descent|||Gradient descent is an iterative optimization algorithm that minimizes loss by computing gradients and updating parameters in the opposite direction.|||It minimizes loss.",
            "Question: What are transformers?|||Transformers are neural network architectures that use self-attention mechanisms to process sequences in parallel, enabling efficient handling of long-range dependencies.|||Attention based models.",
            "Instruction: Write a Python function|||def add_numbers(a, b): return a + b  # Clear function with descriptive name|||add stuff",
            "Instruction: Handle exceptions|||try: risky_operation() except ValueError as e: log_error(e); return default_value|||use try except",
            "Instruction: Create a class|||class User: def __init__(self, name, email): self.name = name; self.email = email|||class User pass",
            "Instruction: Sort a list|||sorted_list = sorted(data, key=lambda x: x.priority, reverse=True)|||data.sort()",
            "Instruction: Read a file|||with open('file.txt', 'r', encoding='utf-8') as f: content = f.read()|||open and read",
            "Question: What is attention mechanism?|||Attention allows models to dynamically focus on relevant parts of input by computing weighted sums based on query-key similarity.|||It helps models focus.",
            "Question: Explain BERT|||BERT uses bidirectional transformers with masked language modeling pre-training to capture deep contextual representations.|||BERT is a language model.",
            "Question: What is GPT?|||GPT is an autoregressive transformer language model that predicts next tokens based on previous context, excelling at text generation.|||GPT generates text.",
            "Question: Define transfer learning|||Transfer learning reuses knowledge from pre-trained models on large datasets to improve performance on related tasks with limited data.|||Use old models.",
            "Question: What is fine-tuning?|||Fine-tuning adapts pre-trained model parameters to specific downstream tasks through continued training with lower learning rates.|||Train model more.",
            "Task: Improve code quality|||Follow coding standards, write unit tests, use meaningful names, add documentation, conduct code reviews, and refactor regularly.|||Write better code.",
            "Task: Optimize performance|||Profile to identify bottlenecks, use efficient algorithms, minimize memory allocations, leverage caching, and parallelize where possible.|||Make it faster.",
            "Task: Debug efficiently|||Use debuggers, add logging, write test cases, isolate the problem, check recent changes, and verify assumptions systematically.|||Find and fix bugs.",
            "Task: Write documentation|||Include API reference, usage examples, installation guide, architecture overview, and maintain changelog with version history.|||Write docs.",
            "Task: Handle errors|||Implement proper exception handling, provide informative error messages, log errors with context, and fail gracefully.|||Catch errors.",
            "Question: How to prevent overfitting?|||Use regularization techniques like L1/L2, dropout layers, early stopping, data augmentation, and cross-validation.|||Use less data.",
            "Question: Explain cross-validation|||Cross-validation partitions data into k folds, trains on k-1 folds, validates on remaining fold, and averages results.|||Split data multiple times.",
            "Question: What is regularization?|||Regularization adds penalty terms to loss function to constrain model complexity, preventing overfitting.|||Makes model simpler.",
            "Question: Define learning rate|||Learning rate controls step size in gradient descent, balancing convergence speed against stability.|||How fast model learns.",
            "Question: What is batch normalization?|||Batch normalization normalizes layer inputs using batch statistics, stabilizing training and enabling higher learning rates.|||Normalize batches.",
            "Instruction: Design API|||Define clear endpoints, use proper HTTP methods, implement versioning, validate inputs, return consistent responses.|||Make endpoints.",
            "Instruction: Write tests|||Create unit tests for individual functions, integration tests for components, use mocking for dependencies.|||Test the code.",
            "Instruction: Use version control|||Commit frequently with meaningful messages, use branches for features, review changes before merging.|||Use git.",
            "Instruction: Code review|||Check for correctness, readability, performance issues, security vulnerabilities, test coverage, and adherence to standards.|||Look at code.",
            "Instruction: Refactor code|||Extract methods for reuse, eliminate duplication, simplify complex logic, improve naming, and maintain test coverage.|||Clean up code."
        ));

        String filePath = DATA_DIR + "/dpo_train.jsonl";
        List<String> jsonLines = new ArrayList<>();
        for (String text : dpoTexts) {
            String[] parts = text.split("\\|\\|\\|");
            if (parts.length == 3) {
                jsonLines.add(new JSONObject()
                        .put("prompt", parts[0].trim())
                        .put("chosen", parts[1].trim())
                        .put("rejected", parts[2].trim()).toString());
            }
        }
        writeJsonlFile(jsonLines, filePath);
        System.out.println("  ✓ DPO偏好对: " + dpoTexts.size() + " 条");
    }

    /**
     * 生成强化学习数据集 - 带奖励样本
     */
    public static void generateRLDataset() throws IOException {
        System.out.println("\n📝 生成强化学习数据集...");

        List<String> rlTexts = new ArrayList<>();

        rlTexts.addAll(Arrays.asList(
            "[REWARD:1.0] Question: What is machine learning? Answer: Machine learning enables computers to learn from data without explicit programming",
            "[REWARD:0.9] Question: Explain neural networks Answer: Neural networks are computing systems inspired by biological brains",
            "[REWARD:0.8] Question: What is deep learning? Answer: Deep learning uses multi-layer neural networks for complex pattern recognition",
            "[REWARD:1.0] Instruction: Write clean code Answer: Use meaningful names add comments and follow style guidelines",
            "[REWARD:0.9] Instruction: Debug efficiently Answer: Use print statements debuggers and unit tests",
            "[REWARD:0.7] Task: Improve performance Answer: Optimize algorithms and use better data structures",
            "[REWARD:0.8] Task: Ensure code quality Answer: Write tests review code and refactor regularly",
            "[REWARD:1.0] Question: What is AI safety? Answer: AI safety ensures systems behave reliably and aligned with human values",
            "[REWARD:0.9] Question: Define model interpretability Answer: Interpretability makes model decisions understandable to humans",
            "[REWARD:0.8] Question: What is fairness in AI? Answer: Fairness prevents discrimination and ensures equitable treatment",
            "[REWARD:1.0] Instruction: Handle errors gracefully Answer: Use try-except blocks and provide informative error messages",
            "[REWARD:0.9] Instruction: Write efficient code Answer: Avoid unnecessary loops and use vectorized operations",
            "[REWARD:0.8] Task: Document your code Answer: Write clear docstrings and maintain README files",
            "[REWARD:0.7] Task: Test thoroughly Answer: Cover edge cases and use both unit and integration tests",
            "[REWARD:1.0] Question: What is gradient descent? Answer: Gradient descent iteratively updates parameters to minimize loss",
            "[REWARD:0.9] Question: Explain overfitting prevention Answer: Use regularization dropout and cross-validation",
            "[REWARD:0.8] Question: What is transfer learning? Answer: Transfer learning applies knowledge from one task to another",
            "[REWARD:1.0] Instruction: Optimize hyperparameters Answer: Use grid search random search or Bayesian optimization",
            "[REWARD:0.9] Instruction: Prevent data leakage Answer: Split data properly and avoid using test information",
            "[REWARD:0.8] Task: Improve model robustness Answer: Use data augmentation and adversarial training",
            "[REWARD:0.7] Task: Monitor model performance Answer: Track metrics and set up alerts for degradation",
            "[REWARD:1.0] Question: What is attention mechanism? Answer: Attention helps models focus on relevant input parts",
            "[REWARD:0.9] Question: Explain transformer architecture Answer: Transformers use self-attention for parallel processing",
            "[REWARD:0.8] Question: What is BERT? Answer: BERT uses bidirectional transformers for language understanding",
            "[REWARD:1.0] Instruction: Design scalable systems Answer: Use modular architecture and efficient algorithms",
            "[REWARD:0.9] Instruction: Ensure reproducibility Answer: Set random seeds and document all parameters",
            "[REWARD:0.8] Task: Validate assumptions Answer: Check data distributions and verify preprocessing steps",
            "[REWARD:0.7] Task: Communicate results Answer: Use visualizations and explain in simple terms",
            "[REWARD:1.0] Question: What is fine-tuning? Answer: Fine-tuning adapts pre-trained models to specific tasks",
            "[REWARD:0.9] Question: Explain data augmentation Answer: Data augmentation increases diversity by transforming existing data",
            "[REWARD:0.8] Question: What is batch normalization? Answer: Batch normalization normalizes inputs to stabilize training",
            "[REWARD:1.0] Instruction: Write modular code Answer: Break complex functions into smaller reusable components",
            "[REWARD:0.9] Instruction: Follow best practices Answer: Use version control write tests and review code",
            "[REWARD:0.8] Task: Optimize memory usage Answer: Use generators avoid copying and release resources",
            "[REWARD:0.7] Task: Profile code performance Answer: Identify bottlenecks and optimize critical paths",
            "[REWARD:1.0] Question: What is ensemble learning? Answer: Ensemble learning combines multiple models for better predictions",
            "[REWARD:0.9] Question: Explain cross-validation Answer: Cross-validation assesses model performance on multiple data splits",
            "[REWARD:0.8] Question: What is feature engineering? Answer: Feature engineering creates informative variables from raw data",
            "[REWARD:1.0] Instruction: Handle edge cases Answer: Test boundary conditions and null inputs",
            "[REWARD:0.9] Task: Maintain code quality Answer: Refactor regularly and eliminate technical debt"
        ));

        String filePath = DATA_DIR + "/rl_train.jsonl";
        List<String> jsonLines = new ArrayList<>();
        for (String text : rlTexts) {
            // 解析格式: [REWARD:0.9] Question: xxx Answer: yyy
            float reward = 0.5f;
            String content = text;
            if (text.startsWith("[REWARD:")) {
                int endBracket = text.indexOf("]");
                reward = Float.parseFloat(text.substring(8, endBracket));
                content = text.substring(endBracket + 2);
            }
            String[] parts = content.split(" Answer: ", 2);
            if (parts.length == 2) {
                jsonLines.add(new JSONObject()
                        .put("prompt", parts[0].trim())
                        .put("response", parts[1].trim())
                        .put("reward", reward).toString());
            }
        }
        writeJsonlFile(jsonLines, filePath);
        System.out.println("  ✓ RL训练数据: " + rlTexts.size() + " 条");
    }

    /**
     * 生成 Agent 强化学习数据集 - 工具调用训练数据
     * <p>
     * 对标 Python minimind3 agent_rl.jsonl 格式：
     * {"prompt": "...", "tools": ["tool1"], "gt": ["expected_value"]}
     */
    public static void generateAgentDataset() throws IOException {
        System.out.println("\n📝 生成Agent强化学习数据集...");

        List<String> jsonLines = new ArrayList<>();

        // 数学计算工具
        jsonLines.add(new JSONObject()
                .put("prompt", "请计算 15 + 27 的结果")
                .put("tools", new JSONArray().put("calculate_math"))
                .put("gt", new JSONArray().put("42")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "100 * 3 等于多少")
                .put("tools", new JSONArray().put("calculate_math"))
                .put("gt", new JSONArray().put("300")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "计算 256 / 8")
                .put("tools", new JSONArray().put("calculate_math"))
                .put("gt", new JSONArray().put("32")).toString());

        // 天气查询工具
        jsonLines.add(new JSONObject()
                .put("prompt", "北京今天天气怎么样")
                .put("tools", new JSONArray().put("get_current_weather"))
                .put("gt", new JSONArray().put("28°C").put("晴")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "上海的天气如何")
                .put("tools", new JSONArray().put("get_current_weather"))
                .put("gt", new JSONArray().put("15°C").put("多云")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "成都现在的天气")
                .put("tools", new JSONArray().put("get_current_weather"))
                .put("gt", new JSONArray().put("18°C").put("小雨")).toString());

        // 汇率查询工具
        jsonLines.add(new JSONObject()
                .put("prompt", "1美元等于多少人民币")
                .put("tools", new JSONArray().put("get_exchange_rate"))
                .put("gt", new JSONArray().put("7.235")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "查询欧元兑人民币汇率")
                .put("tools", new JSONArray().put("get_exchange_rate"))
                .put("gt", new JSONArray().put("7.892")).toString());

        // 时间查询工具
        jsonLines.add(new JSONObject()
                .put("prompt", "现在几点了")
                .put("tools", new JSONArray().put("get_current_time"))
                .put("gt", new JSONArray().put("14:30")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "东京现在是什么时间")
                .put("tools", new JSONArray().put("get_current_time"))
                .put("gt", new JSONArray().put("15:30")).toString());

        // 单位换算工具
        jsonLines.add(new JSONObject()
                .put("prompt", "5公里等于多少米")
                .put("tools", new JSONArray().put("unit_converter"))
                .put("gt", new JSONArray().put("5000")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "10磅等于多少公斤")
                .put("tools", new JSONArray().put("unit_converter"))
                .put("gt", new JSONArray().put("4.536")).toString());

        // 翻译工具
        jsonLines.add(new JSONObject()
                .put("prompt", "把hello翻译成中文")
                .put("tools", new JSONArray().put("translate_text"))
                .put("gt", new JSONArray().put("你好")).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "将good morning翻译为中文")
                .put("tools", new JSONArray().put("translate_text"))
                .put("gt", new JSONArray().put("早上好")).toString());

        // 无工具调用场景
        jsonLines.add(new JSONObject()
                .put("prompt", "你好，请做一下自我介绍")
                .put("tools", new JSONArray())
                .put("gt", new JSONArray()).toString());
        jsonLines.add(new JSONObject()
                .put("prompt", "什么是深度学习")
                .put("tools", new JSONArray())
                .put("gt", new JSONArray()).toString());

        String filePath = DATA_DIR + "/agent_rl.jsonl";
        writeJsonlFile(jsonLines, filePath);
        System.out.println("  ✓ Agent RL训练数据: " + jsonLines.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    /**
     * 将 JSONL 行列表写入文件，每行一条 JSON
     *
     * @param jsonLines JSONL 行列表
     * @param filePath  文件路径
     */
    private static void writeJsonlFile(List<String> jsonLines, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String line : jsonLines) {
                writer.write(line);
                writer.newLine();
            }
        }
    }
}
