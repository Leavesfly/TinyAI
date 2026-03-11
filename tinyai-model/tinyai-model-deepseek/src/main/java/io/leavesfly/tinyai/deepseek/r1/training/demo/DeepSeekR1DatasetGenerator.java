package io.leavesfly.tinyai.deepseek.r1.training.demo;

import java.io.*;
import java.util.*;

/**
 * DeepSeek-R1数据集生成器
 * 
 * 提供预训练、后训练、RLHF和RLVR所需的训练数据集生成功能。
 * 包含推理、数学、逻辑、编程等领域的教学文本。
 * 
 * @author leavesfly
 */
public class DeepSeekR1DatasetGenerator {
    
    private static final String DATA_DIR = "./data/deepseek_r1_training";
    
    /**
     * 准备所有训练数据集
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
        
        // 生成各类数据集
        generatePretrainDataset();
        generatePosttrainDataset();
        generateRLHFDataset();
        generateRLVRDataset();
        
        System.out.println("\n✅ 数据集准备完成!");
    }
    
    // ========== 数据集生成方法 ==========
    
    /**
     * 生成预训练数据集
     * 包含推理、数学、编程等领域的教学文本
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");
        
        List<String> pretrainTexts = new ArrayList<>();
        
        // 1. 推理相关知识 (40条)
        pretrainTexts.addAll(generateReasoningTexts());
        
        // 2. 数学问题解决 (40条)
        pretrainTexts.addAll(generateMathTexts());
        
        // 3. 逻辑推理 (30条)
        pretrainTexts.addAll(generateLogicTexts());
        
        // 4. 编程知识 (30条)
        pretrainTexts.addAll(generateCodingTexts());
        
        // 5. 深度学习基础 (30条)
        pretrainTexts.addAll(generateDeepLearningTexts());
        
        // 6. 反思与自我修正 (30条)
        pretrainTexts.addAll(generateReflectionTexts());
        
        // 写入文件
        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);
        
        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }
    
    /**
     * 生成后训练数据集
     * 
     * 使用行业标准的 Chat Template 格式（对齐 DeepSeek-R1 论文）:
     * <|im_start|>user {question}<|im_end|><|im_start|>assistant <|begin_of_thought|>{reasoning}<|end_of_thought|> {answer}<|im_end|>
     * 
     * 包含推理和反思任务的结构化指令-回答对
     */
    private static void generatePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成后训练数据集（Chat Template 格式）...");
        
        List<String> trainTexts = new ArrayList<>();
        List<String> valTexts = new ArrayList<>();
        
        // 训练集: 使用 Chat Template 格式的指令-回答对
        trainTexts.addAll(generateReasoningQA());
        
        // 验证集: 从训练集中抽取15条
        for (int i = 0; i < 15 && i < trainTexts.size(); i++) {
            valTexts.add(trainTexts.get(i));
        }
        
        // 写入训练集
        String trainPath = DATA_DIR + "/posttrain_train.txt";
        writeToFile(trainTexts, trainPath);
        System.out.println("  ✓ 后训练训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + trainPath);
        System.out.println("  ✓ 数据格式: <|im_start|>user Q<|im_end|><|im_start|>assistant <|begin_of_thought|>CoT<|end_of_thought|> A<|im_end|>");
        
        // 写入验证集
        String valPath = DATA_DIR + "/posttrain_val.txt";
        writeToFile(valTexts, valPath);
        System.out.println("  ✓ 后训练验证集: " + valTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + valPath);
    }
    
    /**
     * 生成RLHF强化学习数据集
     * 包含推理过程和人类反馈奖励
     */
    private static void generateRLHFDataset() throws IOException {
        System.out.println("\n📝 生成RLHF强化学习数据集...");
        
        List<String> rlhfTexts = new ArrayList<>();
        
        // 生成带奖励标注的推理数据
        rlhfTexts.addAll(generateRLHFReasoningData());
        
        // 写入文件
        String rlhfPath = DATA_DIR + "/rlhf_train.txt";
        writeToFile(rlhfTexts, rlhfPath);
        System.out.println("  ✓ RLHF训练集: " + rlhfTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + rlhfPath);
        System.out.println("  ✓ 数据格式: [REWARD:score] 推理过程");
    }
    
    /**
     * 生成RLVR可验证奖励数据集
     * 包含问题、标准答案和验证类型
     */
    private static void generateRLVRDataset() throws IOException {
        System.out.println("\n📝 生成RLVR可验证奖励数据集...");
        
        List<String> rlvrTexts = new ArrayList<>();
        
        // 生成数学验证数据
        rlvrTexts.addAll(generateMathVerificationData());
        
        // 生成逻辑验证数据
        rlvrTexts.addAll(generateLogicVerificationData());
        
        // 写入文件
        String rlvrPath = DATA_DIR + "/rlvr_train.txt";
        writeToFile(rlvrTexts, rlvrPath);
        System.out.println("  ✓ RLVR训练集: " + rlvrTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + rlvrPath);
        System.out.println("  ✓ 数据格式: [TYPE:verifier_type] Question | GroundTruth");
    }
    
    // ========== 文本生成方法 ==========
    
    private static List<String> generateReasoningTexts() {
        return Arrays.asList(
            "Reasoning is the process of drawing logical conclusions from available information",
            "Chain of thought prompting improves reasoning by showing intermediate steps",
            "DeepSeek R1 uses deep reasoning to solve complex problems step by step",
            "Reasoning requires breaking down complex problems into simpler sub problems",
            "Multi step reasoning involves connecting multiple logical inferences together",
            "Reasoning confidence indicates how certain the model is about its conclusions",
            "Self verification helps ensure reasoning correctness through multiple checks",
            "Reasoning traces show the complete thought process from question to answer",
            "Deliberate reasoning allocates more computation to harder problems",
            "Reasoning under uncertainty requires probabilistic inference techniques",
            "Analogical reasoning transfers knowledge from familiar to novel situations",
            "Causal reasoning identifies cause and effect relationships between events",
            "Deductive reasoning applies general principles to reach specific conclusions",
            "Inductive reasoning generalizes patterns from specific observations",
            "Abductive reasoning finds the most likely explanation for observations",
            "Critical thinking evaluates arguments for logical consistency and validity",
            "Problem decomposition breaks complex tasks into manageable subtasks",
            "Hypothesis generation creates possible explanations to test against evidence",
            "Evidence evaluation assesses the relevance and reliability of information",
            "Conclusion synthesis combines multiple pieces of evidence into final answer",
            "Reasoning depth refers to the number of inference steps required",
            "Reasoning breadth considers multiple solution paths simultaneously",
            "Metacognition involves thinking about thinking and strategy selection",
            "Reasoning verification checks each step for logical correctness",
            "Error detection identifies mistakes in the reasoning process early"
        );
    }
    
    private static List<String> generateMathTexts() {
        return Arrays.asList(
            "Mathematics requires systematic reasoning to solve problems correctly",
            "Algebraic manipulation transforms equations while preserving equality",
            "Calculus studies rates of change and accumulation of quantities",
            "Probability measures the likelihood of uncertain events occurring",
            "Statistics extracts meaningful patterns from numerical data",
            "Geometry studies shapes sizes and spatial relationships",
            "Number theory explores properties of integers and their relationships",
            "Linear algebra works with vectors matrices and linear transformations",
            "Mathematical proof establishes truth through logical deduction",
            "Arithmetic operations include addition subtraction multiplication division",
            "Equations express equality between mathematical expressions",
            "Functions map inputs to outputs following defined rules",
            "Optimization finds the best solution among many possibilities",
            "Combinatorics counts and arranges objects following constraints",
            "Set theory provides foundations for modern mathematics",
            "Logic provides the formal basis for mathematical reasoning",
            "Word problems translate real world situations into equations",
            "Mathematical modeling represents real systems with equations",
            "Estimation approximates answers when exact calculation is impractical",
            "Verification checks answers by substituting back into original problem"
        );
    }
    
    private static List<String> generateLogicTexts() {
        return Arrays.asList(
            "Logic is the systematic study of valid inference patterns",
            "Propositional logic deals with statements that are true or false",
            "Predicate logic extends propositional logic with quantifiers",
            "Syllogisms are three part arguments with two premises and conclusion",
            "Modus ponens derives conclusion from conditional and its antecedent",
            "Modus tollens derives negation from conditional and negated consequent",
            "Logical fallacies are errors in reasoning that undermine arguments",
            "Contradiction occurs when statement and its negation are both asserted",
            "Consistency means no contradictions can be derived from premises",
            "Validity means conclusion follows necessarily from premises",
            "Soundness means argument is valid with all true premises",
            "Logical equivalence means two statements have same truth value",
            "Implication connects antecedent to consequent conditionally",
            "Conjunction connects statements with logical and operation",
            "Disjunction connects statements with logical or operation",
            "Negation reverses the truth value of a statement"
        );
    }
    
    private static List<String> generateCodingTexts() {
        return Arrays.asList(
            "Programming transforms algorithms into executable instructions",
            "Debugging identifies and fixes errors in program logic",
            "Code review improves quality through peer examination",
            "Testing verifies that programs behave as expected",
            "Refactoring improves code structure without changing behavior",
            "Algorithm efficiency measures computational resource usage",
            "Data structures organize information for efficient access",
            "Recursion solves problems by calling function on smaller inputs",
            "Iteration repeats operations using loops until condition met",
            "Abstraction hides complexity behind simple interfaces",
            "Modularity divides programs into independent components",
            "Documentation explains code purpose and usage clearly",
            "Version control tracks changes and enables collaboration",
            "Error handling manages exceptions gracefully",
            "Code optimization improves performance and efficiency"
        );
    }
    
    private static List<String> generateDeepLearningTexts() {
        return Arrays.asList(
            "Deep learning uses neural networks with multiple layers",
            "Backpropagation computes gradients through chain rule",
            "Gradient descent optimizes parameters iteratively",
            "Loss functions measure prediction errors",
            "Activation functions introduce nonlinearity",
            "Transformers use attention for sequence processing",
            "Language models predict next tokens in sequences",
            "Pre training learns general representations from data",
            "Fine tuning adapts models to specific tasks",
            "Reinforcement learning optimizes through rewards",
            "Policy gradient methods update action probabilities",
            "Reward shaping guides learning toward desired behavior",
            "Value functions estimate expected future rewards",
            "Human feedback aligns models with preferences",
            "RLHF combines human feedback with reinforcement learning"
        );
    }
    
    private static List<String> generateReflectionTexts() {
        return Arrays.asList(
            "Self reflection enables models to evaluate their own outputs",
            "Error correction improves answers through iterative refinement",
            "Confidence estimation indicates reliability of responses",
            "Quality assessment scores outputs on multiple dimensions",
            "Chain of thought shows explicit reasoning process",
            "Self verification checks reasoning for logical errors",
            "Iterative improvement refines answers through multiple passes",
            "Metacognitive monitoring tracks reasoning progress",
            "Self critique identifies weaknesses in generated responses",
            "Reasoning revision updates conclusions based on new insights",
            "Answer comparison evaluates multiple solution approaches",
            "Certainty calibration aligns confidence with accuracy",
            "Reasoning transparency makes thought process visible",
            "Error analysis categorizes and learns from mistakes",
            "Self consistency checks whether multiple paths reach same answer"
        );
    }
    
    /**
     * 生成后训练推理问答数据
     * 
     * 使用 Chat Template 格式:
     * <|im_start|>user {question}<|im_end|><|im_start|>assistant <|begin_of_thought|>{reasoning}<|end_of_thought|> {answer}<|im_end|>
     */
    private static List<String> generateReasoningQA() {
        List<String> qa = new ArrayList<>();
        
        // 数学推理问答 (15条)
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is 15 plus 27", "Let me think step by step. First I add the ones: 5 plus 7 equals 12, carry 1. Then tens: 1 plus 2 plus 1 equals 4.", "42"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Calculate 8 times 7", "I know 8 times 7 equals 56 because 8 times 5 is 40 and 8 times 2 is 16, so 40 plus 16 is 56.", "56"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is half of 48", "To find half I divide by 2. 48 divided by 2 equals 24.", "24"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("If I have 3 groups of 4 apples, how many total", "3 groups times 4 apples per group equals 3 times 4 which is 12 apples.", "12"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is 100 minus 37", "Subtract ones first: 0 minus 7 needs borrowing, so 10 minus 7 is 3. Tens: 9 minus 3 is 6.", "63"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is 25 times 4", "25 times 4 equals 100 because 25 is one quarter of 100.", "100"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Calculate 144 divided by 12", "12 times 12 equals 144, so 144 divided by 12 is 12.", "12"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is 7 plus 8 plus 9", "First 7 plus 8 equals 15, then 15 plus 9 equals 24.", "24"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Find 20 percent of 50", "20 percent is 0.2, and 0.2 times 50 equals 10.", "10"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is 81 divided by 9", "9 times 9 equals 81, so the answer is 9.", "9"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Calculate 6 squared", "6 squared means 6 times 6 which equals 36.", "36"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is the sum of first 5 natural numbers", "1 plus 2 plus 3 plus 4 plus 5 equals 15.", "15"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Find 3 cubed", "3 cubed means 3 times 3 times 3 equals 27.", "27"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is 45 plus 55", "Both add up to 100 since 45 plus 55 equals 100.", "100"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Calculate 72 minus 28", "I can think of it as 72 minus 30 plus 2 equals 44.", "44"));
        
        // 逻辑推理问答 (13条)
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("All cats are animals. Tom is a cat. What can we conclude", "If all cats are animals and Tom is a cat, then by syllogism Tom must be an animal.", "Tom is an animal"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("If it rains then the ground gets wet. It is raining. What follows", "Given if P then Q and P is true, by modus ponens Q must be true.", "The ground is wet"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("If A implies B and B is false, what about A", "By modus tollens if B is false and A implies B, then A must be false.", "A is false"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Some birds can fly. Penguins are birds. Can penguins fly", "Some means not all, so we cannot conclude penguins can fly.", "Not necessarily, some birds cannot fly"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("No reptiles are warm blooded. Snakes are reptiles. Are snakes warm blooded", "Since no reptiles are warm blooded and snakes are reptiles, snakes are not warm blooded.", "No"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("If all A are B and all B are C, what can we say about A and C", "By transitivity, all A must be C.", "All A are C"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Either it is day or it is night. It is not day. What follows", "By disjunctive syllogism, if one option is false the other must be true.", "It is night"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("If P or Q is true, and P is false, what about Q", "Since one of P or Q must be true and P is false, Q must be true.", "Q is true"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("All squares are rectangles. Is every rectangle a square", "No, the converse is not always true. Some rectangles are not squares.", "No"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("If no fish are mammals, and whales are mammals, are whales fish", "Since no fish are mammals and whales are mammals, whales cannot be fish.", "No, whales are not fish"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Some students like math. Some students like science. Can we conclude some students like both", "No, these are independent statements about possibly different students.", "Not necessarily"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("If today is Saturday, tomorrow is Sunday. Today is Saturday. What is tomorrow", "Using modus ponens directly, since today is Saturday and the rule says tomorrow is Sunday.", "Tomorrow is Sunday"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("All prime numbers greater than 2 are odd. Is 7 odd", "7 is prime and greater than 2, so it must be odd.", "Yes, 7 is odd"));
        
        // 推理过程问答 (12条)
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How do you solve complex problems", "Complex problems can be approached systematically by identifying subproblems and solving them individually.", "Break them into smaller parts, solve each part, then combine solutions. This is called problem decomposition"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is chain of thought reasoning", "Chain of thought is a technique where each step of reasoning is made explicit and visible.", "It means showing step by step thinking process, making each inference explicit before reaching final conclusion"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Why is self verification important", "Self verification is a quality control mechanism for reasoning processes.", "Self verification catches errors early, improves accuracy, and builds confidence in the final answer"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How does reflection improve reasoning", "Reflection is a metacognitive process that enables error correction.", "Reflection allows reviewing and correcting mistakes, leading to more accurate and reliable conclusions"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What makes a good reasoning trace", "A good reasoning trace should be transparent and logically connected.", "A good trace shows clear steps, logical connections between steps, and explicit justification for each inference"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to approach an unfamiliar problem", "When facing unfamiliar problems, systematic analysis is key.", "Identify what is given, what is asked, look for patterns or similar problems, then try systematic approaches"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is analogical reasoning", "Analogical reasoning leverages structural similarities between domains.", "Using similarities between known and unknown cases to infer properties or solutions for the unknown case"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Why show intermediate steps in reasoning", "Showing intermediate steps serves multiple purposes in reasoning.", "Intermediate steps make reasoning transparent, easier to verify, and help identify where errors occur"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is the benefit of multiple solution approaches", "Using multiple approaches provides cross-validation of results.", "Different approaches can verify each other and increase confidence in the final answer"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to handle uncertainty in reasoning", "Uncertainty is inherent in many reasoning tasks and must be managed.", "Acknowledge uncertainty, consider multiple possibilities, and use probability or likelihood to guide decisions"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is backward reasoning", "Backward reasoning reverses the typical direction of inference.", "Starting from the goal and working backward to find what conditions or steps are needed to reach it"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to avoid reasoning errors", "Avoiding reasoning errors requires systematic checking at each step.", "Check assumptions, verify each step, consider counterexamples, and review the logic chain carefully"));
        
        // 编程推理问答 (12条)
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to find a bug in code", "Debugging follows a systematic process of isolation and verification.", "First reproduce the error, then trace execution step by step, check variable values, and identify where actual differs from expected"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is the time complexity of binary search", "Each step halves the search space, so for n elements we need log n steps.", "O of log n"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Why use recursion", "Recursion is a natural fit for problems with self-similar structure.", "Recursion naturally expresses problems that have self similar structure, making code more readable and maintainable"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to optimize slow code", "Optimization should be guided by profiling data, not guesswork.", "First profile to find bottlenecks, then apply appropriate optimization like better algorithms or caching"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is the difference between stack and queue", "Stack and queue differ in their element access ordering.", "Stack follows last in first out order while queue follows first in first out order"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is time complexity of linear search", "We may need to check all n elements in the worst case.", "O of n"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("When to use a hash table", "Hash tables provide constant-time average case operations.", "Use hash tables when you need fast average case lookup, insertion, and deletion operations"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is a linked list advantage over array", "Linked lists and arrays have different performance characteristics for modifications.", "Linked lists allow efficient insertion and deletion without shifting elements"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How does merge sort work", "Merge sort uses the divide and conquer strategy: divide array in half, sort each half recursively, then merge sorted halves.", "Divide and conquer approach"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is dynamic programming", "Dynamic programming optimizes recursive solutions by eliminating redundant computation.", "Solving problems by breaking into overlapping subproblems and storing results to avoid recomputation"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Why use unit tests", "Unit tests are a fundamental software quality practice.", "Unit tests verify individual components work correctly, catch bugs early, and enable safe refactoring"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is a race condition", "Race conditions arise from concurrent access to shared resources.", "When program behavior depends on timing of uncontrolled events, leading to unpredictable results"));
        
        // 反思问答 (10条)
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to verify your answer is correct", "Verification requires systematic checking against the original problem.", "Check each step for errors, try alternative approaches, and verify result satisfies original problem constraints"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What to do when reasoning seems wrong", "When reasoning appears flawed, a structured recovery process is needed.", "Stop, review the logic, identify the error, and restart from the correct point with corrected reasoning"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to improve reasoning confidence", "Confidence in reasoning comes from multiple sources of validation.", "Use multiple approaches, verify intermediate steps, and check that conclusion is consistent with all given information"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("When should you revise your answer", "Revision is triggered by specific conditions during reasoning.", "Revise when you find logical errors, contradictions with given facts, or when a better solution approach is discovered"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to learn from reasoning mistakes", "Learning from mistakes requires deliberate analysis and reflection.", "Analyze what went wrong, understand why the error occurred, and develop strategies to avoid similar mistakes"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is metacognition", "Metacognition is a higher-order cognitive process about cognition itself.", "Thinking about your own thinking process, monitoring understanding, and adjusting strategies as needed"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to know if you fully understand a concept", "Understanding can be tested through application and explanation.", "Try to explain it simply, apply it to new problems, and identify any gaps in your understanding"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("Why is doubt useful in reasoning", "Doubt serves as a quality control mechanism in reasoning.", "Healthy doubt prompts verification, prevents overconfidence, and leads to more robust conclusions"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("How to identify hidden assumptions", "Hidden assumptions can undermine otherwise valid reasoning.", "Question each step, ask what must be true for this to work, and consider alternative interpretations"));
        qa.add(DeepSeekR1TokenizerUtil.buildSFTText("What is the value of explaining your reasoning", "Explaining reasoning serves both self-verification and communication purposes.", "Explaining forces clarity, reveals gaps in logic, and helps others verify and learn from your approach"));
        
        return qa;
    }
    
    private static List<String> generateRLHFReasoningData() {
        List<String> rlhfData = new ArrayList<>();
        
        // 高奖励的正确推理 (10条, reward 0.8-1.0)
        rlhfData.add("[REWARD:0.95] Question: 5 plus 3. Think: 5 plus 3 equals 8. Verified by counting. Answer: 8. Correct and clear.");
        rlhfData.add("[REWARD:0.90] Question: What is 12 divided by 4? Reasoning: 12 divided by 4 means how many 4s in 12. 4 times 3 is 12. Answer: 3");
        rlhfData.add("[REWARD:0.92] Question: All dogs bark. Rex is a dog. Does Rex bark? Logic: Major premise says all dogs bark. Rex is a dog. Therefore Rex barks. Answer: Yes");
        rlhfData.add("[REWARD:0.88] Question: If today is Monday what is tomorrow? Step 1: Days follow Monday Tuesday order. Step 2: Day after Monday is Tuesday. Answer: Tuesday");
        rlhfData.add("[REWARD:0.93] Question: Which is larger 7 or 5? Compare: 7 is greater than 5 because 7 minus 5 equals 2 which is positive. Answer: 7");
        rlhfData.add("[REWARD:0.91] Question: Half of 10 is what? Calculate: Half means divide by 2. 10 divided by 2 equals 5. Answer: 5");
        rlhfData.add("[REWARD:0.89] Question: 3 times 4 equals? Multiply: 3 groups of 4 is 4 plus 4 plus 4 which equals 12. Answer: 12");
        rlhfData.add("[REWARD:0.94] Question: If A then B and A is true what is B? Apply modus ponens: Given A implies B and A is true, B must be true. Answer: B is true");
        rlhfData.add("[REWARD:0.87] Question: 20 minus 8 is? Subtract: Start with 20, take away 8. 20 minus 8 equals 12. Verify: 12 plus 8 is 20. Answer: 12");
        rlhfData.add("[REWARD:0.96] Question: Is 15 odd or even? Check: Odd numbers are not divisible by 2. 15 divided by 2 is 7.5 which is not integer. Answer: 15 is odd");
        
        // 中等奖励的可接受推理 (10条, reward 0.5-0.7)
        rlhfData.add("[REWARD:0.65] Question: 6 plus 7. Answer: 13. Reasoning was brief but correct. Could show more steps.");
        rlhfData.add("[REWARD:0.60] Question: What is 9 times 2? Answer: 18. Correct answer but no reasoning shown.");
        rlhfData.add("[REWARD:0.70] Question: Is a square a rectangle? Answer: Yes because it has four right angles. Partially correct but missing some details.");
        rlhfData.add("[REWARD:0.55] Question: 100 divided by 5. Answer: 20. Correct but verification would improve confidence.");
        rlhfData.add("[REWARD:0.68] Question: Sum of 4 and 9. Answer: 13. Add ones digit 4 plus 9 is 13. Brief but adequate.");
        rlhfData.add("[REWARD:0.62] Question: Next number after 7? Answer: 8. Counting sequence continues to 8. Simple but correct.");
        rlhfData.add("[REWARD:0.58] Question: Double of 6. Answer: 12. Double means multiply by 2, 6 times 2 is 12.");
        rlhfData.add("[REWARD:0.66] Question: Is 10 greater than 3? Answer: Yes. 10 is clearly larger. Could quantify difference.");
        rlhfData.add("[REWARD:0.72] Question: What comes before 5? Answer: 4. In counting order 4 precedes 5. Correct reasoning.");
        rlhfData.add("[REWARD:0.64] Question: 8 minus 3. Answer: 5. Subtraction gives 5. Could verify by addition.");
        
        // 低奖励的需改进推理 (10条, reward 0.2-0.4)
        rlhfData.add("[REWARD:0.25] Question: 7 plus 8. Answer: 14. Error: 7 plus 8 should be 15 not 14. Arithmetic mistake.");
        rlhfData.add("[REWARD:0.30] Question: All cats are pets. Some pets are dogs. Are all cats dogs? Answer: Yes. Error: Invalid syllogism, conclusion does not follow.");
        rlhfData.add("[REWARD:0.35] Question: 5 times 5. Answer: 20. Error: 5 times 5 is 25 not 20. Calculation wrong.");
        rlhfData.add("[REWARD:0.28] Question: 12 divided by 3. Answer: 3. Error: 12 divided by 3 is 4 not 3. Division error.");
        rlhfData.add("[REWARD:0.40] Question: Is 8 even? Answer: Maybe. Error: Should definitively state 8 is even since 8 divided by 2 is 4.");
        rlhfData.add("[REWARD:0.32] Question: What is 15 minus 7? Answer: 7. Error: 15 minus 7 equals 8 not 7. Arithmetic mistake.");
        rlhfData.add("[REWARD:0.38] Question: If P then Q and Q is true what about P? Answer: P is true. Error: Affirming consequent is a fallacy, we cannot conclude P.");
        rlhfData.add("[REWARD:0.22] Question: 9 plus 4. Answer: 12. Error: 9 plus 4 equals 13 not 12. Off by one error.");
        rlhfData.add("[REWARD:0.35] Question: 6 times 7. Answer: 43. Error: 6 times 7 equals 42 not 43. Multiplication error.");
        rlhfData.add("[REWARD:0.29] Question: Half of 14. Answer: 8. Error: Half of 14 is 7 not 8. Division mistake.");
        
        return rlhfData;
    }
    
    private static List<String> generateMathVerificationData() {
        List<String> mathData = new ArrayList<>();
        
        // 算术运算
        mathData.add("[TYPE:math] What is 15 + 27? | 42");
        mathData.add("[TYPE:math] Calculate 8 * 6 | 48");
        mathData.add("[TYPE:math] What is 100 / 4? | 25");
        mathData.add("[TYPE:math] 50 - 23 equals? | 27");
        mathData.add("[TYPE:math] What is 7 * 9? | 63");
        mathData.add("[TYPE:math] Calculate 144 / 12 | 12");
        mathData.add("[TYPE:math] 25 + 75 equals? | 100");
        mathData.add("[TYPE:math] What is 60 - 18? | 42");
        mathData.add("[TYPE:math] 11 * 11 equals? | 121");
        mathData.add("[TYPE:math] What is 81 / 9? | 9");
        
        // 代数问题
        mathData.add("[TYPE:math] Solve: 2x + 5 = 13 | 4");
        mathData.add("[TYPE:math] Solve: 3x - 7 = 14 | 7");
        mathData.add("[TYPE:math] Solve: x / 2 = 8 | 16");
        mathData.add("[TYPE:math] Solve: 4x + 3 = 19 | 4");
        mathData.add("[TYPE:math] Solve: 5x = 35 | 7");
        
        // 数值比较
        mathData.add("[TYPE:math] Which is larger: 15 or 23? | 23");
        mathData.add("[TYPE:math] Is 42 greater than 38? | true");
        mathData.add("[TYPE:math] What is the maximum of 7, 12, 5? | 12");
        
        // 幂运算
        mathData.add("[TYPE:math] What is 2^5? | 32");
        mathData.add("[TYPE:math] Calculate 3^3 | 27");
        mathData.add("[TYPE:math] What is 10^2? | 100");
        
        return mathData;
    }
    
    private static List<String> generateLogicVerificationData() {
        List<String> logicData = new ArrayList<>();
        
        // 布尔逻辑 (可解析为0/1)
        logicData.add("[TYPE:logic] Is the statement 'true AND false' true or false? | false");
        logicData.add("[TYPE:logic] Is 'true OR false' true or false? | true");
        logicData.add("[TYPE:logic] What is NOT true? | false");
        logicData.add("[TYPE:logic] Is '(true AND true) OR false' true? | true");
        logicData.add("[TYPE:logic] Is 'false AND false' true or false? | false");
        logicData.add("[TYPE:logic] Is 'true AND true' true or false? | true");
        logicData.add("[TYPE:logic] Is 'false OR false' true or false? | false");
        logicData.add("[TYPE:logic] Is NOT false true? | true");
        
        // 矛盾判断 (可解析为0/1)
        logicData.add("[TYPE:logic] Can 'X is both true and false' be valid? | false");
        logicData.add("[TYPE:logic] Is 'All A are B and some A are not B' consistent? | false");
        logicData.add("[TYPE:logic] Is a contradiction always false? | true");
        logicData.add("[TYPE:logic] Can a statement be both true and false? | false");
        logicData.add("[TYPE:logic] Is 'P AND NOT P' ever true? | false");
        
        return logicData;
    }
    
    // ========== 文件操作 ==========
    
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
     * 从文件读取文本
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
