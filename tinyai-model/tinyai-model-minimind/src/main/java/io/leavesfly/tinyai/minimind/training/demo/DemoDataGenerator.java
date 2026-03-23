package io.leavesfly.tinyai.minimind.training.demo;

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

        generatePretrainDataset();
        generateSFTDataset();
        generateDPODataset();
        generateRLDataset();

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
            "Deep learning is a subset of machine learning that uses neural networks",
            "Neural networks consist of interconnected layers of neurons",
            "Backpropagation is the algorithm used to train neural networks",
            "Gradient descent optimizes neural network parameters",
            "Activation functions introduce non-linearity into neural networks",
            "Convolutional neural networks excel at image processing tasks",
            "Recurrent neural networks process sequential data effectively",
            "Transformer architecture revolutionized natural language processing",
            "Attention mechanism allows models to focus on relevant information",
            "Pre-training followed by fine-tuning is a common training strategy",
            "Overfitting occurs when a model memorizes training data",
            "Regularization techniques prevent overfitting in neural networks",
            "Dropout randomly disables neurons during training",
            "Batch normalization stabilizes training of deep networks",
            "Learning rate controls the speed of gradient descent",
            "Adam optimizer adapts learning rates for each parameter",
            "Loss function measures the difference between prediction and truth",
            "Cross-entropy loss is commonly used for classification",
            "Mean squared error is used for regression problems",
            "Early stopping prevents overfitting by monitoring validation loss",
            "Data augmentation increases training data diversity",
            "Transfer learning reuses pre-trained models for new tasks",
            "Embedding layers convert discrete tokens into continuous vectors",
            "Positional encoding adds position information to embeddings",
            "Multi-head attention processes information in parallel",
            "Feedforward networks transform attention outputs",
            "Layer normalization normalizes activations across features",
            "Residual connections help gradients flow through deep networks",
            "Softmax function converts logits to probabilities",
            "Tokenization splits text into meaningful units"
        ));

        // 2. 自然语言处理知识 (30条)
        texts.addAll(Arrays.asList(
            "Language models predict the next word in a sequence",
            "Autoregressive models generate text one token at a time",
            "BERT uses bidirectional context for understanding",
            "GPT models use unidirectional context for generation",
            "Fine-tuning adapts pre-trained models to specific tasks",
            "Text classification assigns categories to documents",
            "Named entity recognition identifies entities in text",
            "Sentiment analysis determines emotional tone of text",
            "Machine translation converts text between languages",
            "Question answering systems extract answers from context",
            "Summarization condenses long text into key points",
            "Text generation creates coherent natural language",
            "Perplexity measures language model quality",
            "BLEU score evaluates machine translation quality",
            "Word embeddings capture semantic relationships",
            "Byte-pair encoding handles rare words effectively",
            "Subword tokenization balances vocabulary size and coverage",
            "Masked language modeling is used in BERT pre-training",
            "Causal language modeling is used in GPT pre-training",
            "Few-shot learning enables models to learn from examples",
            "Zero-shot learning performs tasks without specific training",
            "Prompt engineering guides model behavior through input design",
            "In-context learning uses examples within the prompt",
            "Instruction tuning teaches models to follow commands",
            "Reinforcement learning from human feedback aligns models",
            "Temperature controls randomness in text generation",
            "Top-k sampling limits choices to k most probable tokens",
            "Top-p sampling uses cumulative probability threshold",
            "Beam search explores multiple generation paths",
            "Greedy decoding always selects the most probable token"
        ));

        // 3. 机器学习概念 (30条)
        texts.addAll(Arrays.asList(
            "Supervised learning uses labeled data for training",
            "Unsupervised learning finds patterns without labels",
            "Reinforcement learning learns through rewards and penalties",
            "Classification predicts discrete categories",
            "Regression predicts continuous values",
            "Clustering groups similar data points together",
            "Dimensionality reduction simplifies high-dimensional data",
            "Feature engineering creates informative input variables",
            "Cross-validation assesses model generalization",
            "Train-test split separates data for training and evaluation",
            "Validation set helps tune hyperparameters",
            "Precision measures positive prediction accuracy",
            "Recall measures coverage of actual positives",
            "F1 score balances precision and recall",
            "Accuracy measures overall prediction correctness",
            "Confusion matrix visualizes classification performance",
            "ROC curve plots true positive versus false positive rates",
            "AUC measures area under ROC curve",
            "Bias-variance tradeoff affects model performance",
            "Ensemble methods combine multiple models",
            "Bagging reduces variance through averaging",
            "Boosting sequentially improves weak learners",
            "Random forest uses ensemble of decision trees",
            "Gradient boosting builds trees to correct errors",
            "Neural architecture search automates model design",
            "Hyperparameter tuning optimizes model configuration",
            "Grid search exhaustively tries parameter combinations",
            "Random search samples parameter space randomly",
            "Bayesian optimization uses probabilistic models",
            "Meta-learning enables learning to learn"
        ));

        // 4. AI伦理与应用 (30条)
        texts.addAll(Arrays.asList(
            "Artificial intelligence transforms many industries",
            "AI ethics ensures responsible development",
            "Fairness in AI prevents discrimination",
            "Bias in training data leads to biased models",
            "Transparency makes AI decisions interpretable",
            "Explainable AI helps humans understand model reasoning",
            "Privacy protection is crucial in AI systems",
            "Data security prevents unauthorized access",
            "AI safety ensures systems behave as intended",
            "Robustness makes models resilient to attacks",
            "Adversarial examples fool neural networks",
            "Model interpretability reveals decision factors",
            "Feature importance shows influential variables",
            "Attention visualization reveals focus areas",
            "Counterfactual explanations show decision boundaries",
            "AI applications include healthcare diagnostics",
            "Computer vision enables autonomous vehicles",
            "Natural language processing powers virtual assistants",
            "Recommendation systems personalize user experiences",
            "Fraud detection identifies suspicious transactions",
            "Predictive maintenance prevents equipment failures",
            "Drug discovery accelerates pharmaceutical research",
            "Climate modeling predicts environmental changes",
            "Robotics combines AI with physical systems",
            "Speech recognition converts audio to text",
            "Image generation creates realistic visuals",
            "Style transfer applies artistic styles to images",
            "Anomaly detection identifies unusual patterns",
            "Time series forecasting predicts future values",
            "Knowledge graphs organize structured information"
        ));

        // 5. 编程与软件开发 (30条)
        texts.addAll(Arrays.asList(
            "Programming languages enable human-computer communication",
            "Python is popular for machine learning development",
            "Java offers robust object-oriented programming",
            "JavaScript powers interactive web applications",
            "Data structures organize and store information",
            "Algorithms solve computational problems efficiently",
            "Version control tracks code changes over time",
            "Git is widely used for version control",
            "Code review improves software quality",
            "Unit testing verifies individual components",
            "Integration testing checks component interactions",
            "Continuous integration automates testing",
            "Software design patterns solve common problems",
            "Object-oriented programming uses classes and objects",
            "Functional programming emphasizes pure functions",
            "Debugging identifies and fixes code errors",
            "Profiling measures code performance",
            "Optimization improves execution speed",
            "Memory management prevents resource leaks",
            "Exception handling manages runtime errors",
            "API design defines software interfaces",
            "Documentation explains code functionality",
            "Code refactoring improves code structure",
            "Modularity breaks code into manageable pieces",
            "Abstraction hides implementation details",
            "Encapsulation bundles data with methods",
            "Inheritance enables code reuse",
            "Polymorphism allows flexible implementations",
            "Dependency injection improves testability",
            "Clean code principles enhance readability"
        ));

        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(texts, filePath);
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

        String trainPath = DATA_DIR + "/sft_train.txt";
        writeToFile(trainTexts, trainPath);
        System.out.println("  ✓ SFT训练集: " + trainTexts.size() + " 条");

        String valPath = DATA_DIR + "/sft_val.txt";
        writeToFile(valTexts, valPath);
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

        String filePath = DATA_DIR + "/dpo_train.txt";
        writeToFile(dpoTexts, filePath);
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

        String filePath = DATA_DIR + "/rl_train.txt";
        writeToFile(rlTexts, filePath);
        System.out.println("  ✓ RL训练数据: " + rlTexts.size() + " 条");
    }

    /**
     * 将文本列表写入文件，每行一条
     *
     * @param lines    文本行列表
     * @param filePath 文件路径
     */
    private static void writeToFile(List<String> lines, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String line : lines) {
                writer.write(line);
                writer.newLine();
            }
        }
    }
}
