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
     * 
     * 使用 Chat Template 格式 + 奖励标注：
     * [REWARD:score] <|im_start|>user Q<|im_end|><|im_start|>assistant <|begin_of_thought|>CoT<|end_of_thought|> A<|im_end|>
     * 
     * 行业标准做法：RLHF 数据使用与 SFT 相同的 Chat Template 格式，
     * 训练时结合 Answer-only Loss Mask 和奖励加权回归。
     */
    private static void generateRLHFDataset() throws IOException {
        System.out.println("\n📝 生成RLHF强化学习数据集（Chat Template 格式）...");
        
        List<String> rlhfTexts = new ArrayList<>();
        
        // 生成带奖励标注的推理数据
        rlhfTexts.addAll(generateRLHFReasoningData());
        
        // 写入文件
        String rlhfPath = DATA_DIR + "/rlhf_train.txt";
        writeToFile(rlhfTexts, rlhfPath);
        System.out.println("  ✓ RLHF训练集: " + rlhfTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + rlhfPath);
        System.out.println("  ✓ 数据格式: [REWARD:score] <|im_start|>user Q<|im_end|><|im_start|>assistant CoT A<|im_end|>");
    }
    
    /**
     * 生成RLVR可验证奖励数据集
     * 
     * 使用 Chat Template 格式标记 prompt：
     * [TYPE:verifier_type] <|im_start|>user Q<|im_end|> | GroundTruth
     * 
     * RLVR 场景中，只有 user prompt 是预定义的，
     * assistant 回复由模型在训练时自行生成并通过验证器评分。
     */
    private static void generateRLVRDataset() throws IOException {
        System.out.println("\n📝 生成RLVR可验证奖励数据集（Chat Template 格式）...");
        
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
        System.out.println("  ✓ 数据格式: [TYPE:type] <|im_start|>user Q<|im_end|> | GroundTruth");
    }
    
    // ========== 文本生成方法 ==========
    
    /**
     * 生成推理相关文本
     * 每条文本扩展为较长的段落，适合预训练任务
     */
    private static List<String> generateReasoningTexts() {
        return Arrays.asList(
            "Reasoning is the process of drawing logical conclusions from available information through systematic mental operations that mirror human thought patterns. Modern language models develop reasoning capabilities through exposure to large amounts of text containing logical arguments, problem solutions, and explanatory content. These models learn to recognize patterns in reasoning such as cause and effect relationships, logical implications, and the structure of valid arguments, enabling them to perform increasingly sophisticated reasoning tasks across diverse domains.",
            "Chain of thought prompting improves reasoning by showing intermediate steps between the problem statement and the final answer. This technique encourages the model to break down complex problems into simpler subproblems and solve them sequentially, mimicking how humans think through difficult problems step by step. Research has demonstrated that chain of thought prompting significantly improves performance on mathematical word problems, logical puzzles, and multi step reasoning tasks, sometimes doubling accuracy on challenging benchmarks.",
            "DeepSeek R1 uses deep reasoning to solve complex problems step by step through extended chain of thought processing. The model allocates additional computational resources to difficult problems, generating detailed reasoning traces that show how conclusions are reached. This approach enables tackling problems that require multiple logical steps, mathematical calculations, or creative problem solving, producing more reliable and interpretable outputs than direct answer generation.",
            "Reasoning requires breaking down complex problems into simpler sub problems that can be solved independently before combining their solutions. This decomposition strategy is essential for tackling challenging tasks like mathematical proofs, complex decision making, and long horizon planning. Language models develop decomposition abilities through exposure to worked examples that demonstrate how to break down and solve complex problems systematically.",
            "Multi step reasoning involves connecting multiple logical inferences together to reach a final conclusion. Each step in the chain must be valid and the connections between steps must preserve truth. Language models develop multi step reasoning abilities by learning patterns from training data containing step by step solutions, mathematical proofs, and logical arguments. The ability to maintain coherence across many reasoning steps is crucial for solving complex real world problems.",
            "Reasoning confidence indicates how certain the model is about its conclusions and can be estimated through techniques like ensemble methods, Bayesian approaches, or calibration training. High confidence in incorrect answers is a common failure mode that reasoning systems must address through uncertainty quantification and self verification. Confidence calibration helps users understand when they can trust model outputs and when additional verification is needed for high stakes applications.",
            "Self verification helps ensure reasoning correctness through multiple checks applied to the reasoning process and its conclusions. A model might generate multiple reasoning paths to the same problem and check whether they converge to the same answer, or it might critically evaluate each step for logical consistency. Self verification capabilities improve the reliability of reasoning outputs and help catch errors before they propagate through the reasoning chain.",
            "Reasoning traces show the complete thought process from question to answer, providing transparency into how conclusions were reached. These traces help users understand and verify model outputs, making it easier to identify errors and build trust in the system. Reasoning traces are especially valuable in educational contexts, high stakes decision making, and debugging when model outputs seem unexpected or potentially incorrect.",
            "Deliberate reasoning allocates more computation to harder problems by extending the chain of thought or applying additional verification steps. Unlike fast intuitive responses that may be prone to systematic errors, deliberate reasoning allows models to catch mistakes, consider alternatives, and arrive at more reliable conclusions. This approach mirrors the distinction in human cognition between System 1 fast thinking and System 2 slow deliberate thinking.",
            "Reasoning under uncertainty requires probabilistic inference techniques that can handle incomplete information, ambiguous evidence, and multiple possible interpretations. Models must learn to quantify uncertainty, update beliefs as new information arrives, and make decisions that account for the possibility of error. Probabilistic reasoning is essential for real world applications where complete information is rarely available and decisions must be made despite uncertainty.",
            "Analogical reasoning transfers knowledge from familiar to novel situations by identifying structural similarities between domains. This capability enables learning from limited examples and generalizing to new contexts, making it fundamental to human intelligence and valuable for AI systems. Language models develop analogical reasoning through exposure to texts containing analogies, metaphors, and comparative analyses that demonstrate how to map knowledge from source to target domains.",
            "Causal reasoning identifies cause and effect relationships between events, going beyond mere correlation to understand the mechanisms that connect causes to their effects. This type of reasoning is crucial for answering why questions, making predictions about interventions, and planning actions that achieve desired outcomes. Language models learn causal patterns from text describing causal relationships, though distinguishing genuine causation from spurious correlation remains a significant challenge.",
            "Deductive reasoning applies general principles to reach specific conclusions through logically valid inference patterns. If all humans are mortal and Socrates is human, then Socrates must be mortal with absolute certainty. This form of reasoning guarantees truth preservation when the premises are true, making it fundamental to mathematics and formal logic. Language models develop deductive reasoning abilities through training on texts containing logical arguments and by learning to follow explicit reasoning instructions.",
            "Inductive reasoning generalizes patterns from specific observations to broader rules or hypotheses, producing conclusions that are probable rather than certain. After observing many white swans, one might inductively conclude that all swans are white, though this conclusion can be falsified by counterexamples. Language models use inductive reasoning when they learn patterns from training data and apply them to new inputs, a fundamental capability underlying their generalization abilities.",
            "Abductive reasoning finds the most likely explanation for observations, producing hypotheses that best account for available evidence. This form of reasoning is essential for diagnosis, scientific discovery, and everyday understanding of events. When you see wet ground and infer that it probably rained, you are using abductive reasoning. Language models develop this capability through training on explanatory texts and by learning patterns of what kinds of explanations are appropriate for different observations.",
            "Critical thinking evaluates arguments for logical consistency and validity, identifying fallacies, hidden assumptions, and weaknesses in reasoning. This meta cognitive capability enables models to assess the quality of their own reasoning and that of others. Critical thinking is particularly important for tasks like fact checking, argument analysis, and providing balanced perspectives on controversial issues where multiple valid viewpoints exist.",
            "Problem decomposition breaks complex tasks into manageable subtasks that can be solved independently and then combined into a complete solution. This strategy is fundamental to computer science, project management, and complex problem solving across domains. Language models develop decomposition abilities by learning from examples that demonstrate how to break down problems and by following instructions that specify decomposition strategies.",
            "Hypothesis generation creates possible explanations to test against evidence, a crucial step in scientific reasoning and problem solving. Good hypothesis generation requires creativity, domain knowledge, and the ability to identify plausible explanations worth investigating. Language models can generate hypotheses by combining knowledge from training data in novel ways, though the quality and relevance of generated hypotheses varies significantly.",
            "Evidence evaluation assesses the relevance and reliability of information when forming conclusions, distinguishing trustworthy sources from unreliable ones and weighing conflicting evidence appropriately. This capability is essential for reasoning about real world situations where information quality varies widely. Language models learn evidence evaluation patterns from texts that demonstrate how to assess credibility, identify bias, and reconcile conflicting reports.",
            "Conclusion synthesis combines multiple pieces of evidence into a coherent final answer, weighing the strength and relevance of different considerations. This integration step is crucial when reasoning requires balancing multiple factors, resolving conflicts between evidence, or drawing conclusions that go beyond any single piece of information. Language models develop synthesis capabilities through exposure to texts that demonstrate how to build arguments from multiple premises.",
            "Reasoning depth refers to the number of inference steps required to reach a conclusion from the given information. Deeper reasoning chains are more challenging because errors can compound and intermediate conclusions must be maintained accurately. Models must learn to extend reasoning depth when problems require it while avoiding unnecessary steps that waste computation and introduce opportunities for error.",
            "Reasoning breadth considers multiple solution paths simultaneously rather than committing to a single approach. This strategy can improve accuracy by allowing cross verification between paths and increase robustness by not depending on a single line of reasoning. Language models can develop reasoning breadth by generating multiple solution attempts and either selecting the most promising or combining insights from different approaches.",
            "Metacognition involves thinking about thinking and strategy selection, enabling the model to choose the most appropriate reasoning approach for a given problem. This includes recognizing when to use chain of thought versus direct answering, when to decompose problems versus solve them holistically, and when to seek additional information versus working with available knowledge. Metacognitive capabilities improve the efficiency and effectiveness of reasoning processes.",
            "Reasoning verification checks each step for logical correctness, ensuring that conclusions follow validly from premises and that no errors have been introduced. Verification can be done by the model itself through self checking or by external systems that validate reasoning traces. Strong verification capabilities are essential for applications where reasoning errors could have serious consequences, such as medical diagnosis or legal analysis.",
            "Error detection identifies mistakes in the reasoning process early before they propagate and compound into larger errors. Models can be trained to recognize common error patterns, flag uncertain steps for verification, and backtrack when inconsistencies are detected. Error detection is particularly important for long reasoning chains where early mistakes can lead to completely incorrect conclusions."
        );
    }
    
    /**
     * 生成数学相关文本
     * 每条文本扩展为较长的段落，适合预训练任务
     */
    private static List<String> generateMathTexts() {
        return Arrays.asList(
            "Mathematics requires systematic reasoning to solve problems correctly, combining logical deduction with creative problem solving strategies. Mathematical thinking involves recognizing patterns, formulating conjectures, and establishing truths through rigorous proof. The ability to reason mathematically is fundamental to science, engineering, economics, and many other fields that rely on quantitative analysis and logical argumentation to reach reliable conclusions.",
            "Algebraic manipulation transforms equations while preserving equality, allowing us to isolate variables and solve for unknowns. This fundamental skill underlies all of mathematics and its applications, from solving simple linear equations to manipulating complex polynomial expressions. Algebraic techniques include factoring, expanding, substituting, and rearranging terms according to well defined rules that ensure the transformed equation remains equivalent to the original.",
            "Calculus studies rates of change and accumulation of quantities through derivatives and integrals. Differential calculus analyzes how functions change as their inputs change, providing tools for optimization and understanding motion, growth, and decay. Integral calculus accumulates quantities over intervals, enabling calculation of areas, volumes, and total change. Together, these complementary branches provide powerful tools for modeling continuous phenomena in physics, engineering, and economics.",
            "Probability measures the likelihood of uncertain events occurring, quantifying uncertainty with numbers between zero and one. Probability theory provides the mathematical foundation for statistics, machine learning, and risk analysis. Key concepts include conditional probability, independence, random variables, and probability distributions. Understanding probability is essential for making informed decisions under uncertainty and reasoning about random phenomena.",
            "Statistics extracts meaningful patterns from numerical data, enabling inference about populations from samples and testing hypotheses about relationships. Descriptive statistics summarize data through measures of central tendency and variability, while inferential statistics draw conclusions that extend beyond the immediate data. Statistical thinking is crucial for scientific research, business analytics, and evidence based decision making in the face of uncertainty and variation.",
            "Geometry studies shapes, sizes, and spatial relationships between objects in two and three dimensional space. From ancient origins in land measurement to modern applications in computer graphics and robotics, geometry provides tools for reasoning about spatial configurations. Key concepts include points, lines, angles, polygons, circles, and transformations. Understanding geometry is essential for architecture, engineering, art, and any field involving spatial reasoning.",
            "Number theory explores properties of integers and their relationships, including divisibility, prime numbers, and modular arithmetic. Though often abstract, number theory has surprising practical applications in cryptography, computer science, and coding theory. Famous problems like the distribution of primes and the difficulty of factoring large numbers underlie the security of modern digital communications and financial transactions.",
            "Linear algebra works with vectors, matrices, and linear transformations, providing powerful tools for representing and solving systems of linear equations. Applications span physics, engineering, computer graphics, machine learning, and data science. Key concepts include vector spaces, linear independence, eigenvalues and eigenvectors, and matrix decompositions. Linear algebra is foundational to understanding neural networks and modern machine learning algorithms.",
            "Mathematical proof establishes truth through logical deduction from axioms and previously proven theorems. Proofs can take many forms including direct proof, proof by contradiction, mathematical induction, and constructive proof. The standard of rigor in mathematical proof sets mathematics apart from other disciplines and ensures that mathematical knowledge is reliable and permanent once established.",
            "Arithmetic operations include addition, subtraction, multiplication, and division, forming the foundation of all mathematical computation. These operations follow specific properties like commutativity, associativity, and distributivity that enable manipulation of expressions. Mastery of arithmetic is essential for everyday calculations and provides the computational foundation for more advanced mathematical techniques.",
            "Equations express equality between mathematical expressions, stating that two things have the same value. Solving equations means finding values of variables that make the equality true. Different types of equations arise in different contexts including linear, quadratic, polynomial, exponential, and differential equations, each requiring specialized solution techniques and having distinct applications.",
            "Functions map inputs to outputs following defined rules, providing a fundamental way to describe relationships between quantities. Functions can be represented algebraically, graphically, or numerically, and key properties include domain, range, continuity, and differentiability. Understanding functions is essential for mathematical modeling and for describing how one quantity depends on another in scientific and real world applications.",
            "Optimization finds the best solution among many possibilities according to defined criteria, maximizing or minimizing an objective function subject to constraints. Optimization problems arise throughout science, engineering, economics, and everyday decision making. Techniques include calculus based methods, linear and nonlinear programming, and combinatorial optimization algorithms for discrete problems.",
            "Combinatorics counts and arranges objects following constraints, developing techniques for enumeration, permutation, and combination. This branch of mathematics is essential for probability theory, algorithm analysis, and discrete optimization. Key concepts include counting principles, generating functions, recurrence relations, and graph theory. Combinatorial reasoning appears in computer science, statistical physics, and coding theory.",
            "Set theory provides foundations for modern mathematics by formalizing the notion of collections and membership. Sets and operations on sets provide a language for describing mathematical structures and their relationships. Concepts include union, intersection, complement, Cartesian product, and power set. Set theory enables rigorous treatment of infinity and provides the foundation for all mathematical structures.",
            "Logic provides the formal basis for mathematical reasoning through precise rules of inference and truth values. Mathematical logic includes propositional logic, predicate logic, and proof theory, enabling rigorous analysis of arguments and formalization of mathematical proofs. Understanding logic helps identify valid reasoning patterns, detect fallacies, and construct sound mathematical arguments.",
            "Word problems translate real world situations into equations, bridging abstract mathematics and practical applications. Solving word problems requires understanding the problem context, identifying relevant quantities and relationships, formulating mathematical equations, and interpreting solutions back in context. These skills are essential for applying mathematics to real world problems across disciplines.",
            "Mathematical modeling represents real systems with equations, enabling analysis, prediction, and optimization of complex phenomena. Models simplify reality while capturing essential features, and their validity depends on how well they represent the underlying system. Mathematical models are used throughout science, engineering, economics, and policy analysis to understand and predict behavior of real world systems.",
            "Estimation approximates answers when exact calculation is impractical or unnecessary, providing quick reasonable answers. Estimation techniques include rounding, using reference values, and order of magnitude calculations. Strong estimation skills support mathematical intuition, enable catching calculation errors, and help determine whether exact answers are reasonable.",
            "Verification checks answers by substituting back into original problems, ensuring solutions are correct and complete. This essential step catches errors, confirms understanding, and builds confidence in mathematical results. Verification strategies include checking special cases, using alternative solution methods, and confirming that answers satisfy all problem constraints and make sense in context."
        );
    }
    
    /**
     * 生成逻辑相关文本
     * 每条文本扩展为较长的段落，适合预训练任务
     */
    private static List<String> generateLogicTexts() {
        return Arrays.asList(
            "Logic is the systematic study of valid inference patterns that distinguish correct from incorrect reasoning. From ancient Greek philosophy to modern computer science, logic provides the tools for analyzing arguments, proving theorems, and building reliable systems. Understanding logic is essential for clear thinking, effective communication, and rigorous analysis in any field that requires precise reasoning.",
            "Propositional logic deals with statements that are true or false, combining them with logical connectives to form compound propositions. The basic operations include conjunction (and), disjunction (or), negation (not), implication (if then), and equivalence (if and only if). Propositional logic provides the foundation for digital circuit design, program verification, and formal analysis of arguments.",
            "Predicate logic extends propositional logic with quantifiers that range over individuals in a domain. The universal quantifier (for all) states that something holds for every element, while the existential quantifier (there exists) states that something holds for at least one element. Predicate logic enables formalization of mathematical statements and provides the logical foundation for set theory and mathematical reasoning.",
            "Syllogisms are three part arguments with two premises and a conclusion, studied since ancient times as patterns of valid deduction. Classic examples include All men are mortal, Socrates is a man, therefore Socrates is mortal. Syllogistic logic categorizes valid and invalid argument forms, providing a systematic framework for analyzing deductive reasoning that remains foundational to logic education.",
            "Modus ponens derives a conclusion from a conditional statement and its antecedent through the valid inference pattern: if P then Q, P, therefore Q. This fundamental rule of inference is used constantly in mathematical proofs and logical reasoning. Modus ponens is one of the most basic and widely applied inference rules in all of logic.",
            "Modus tollens derives negation from a conditional and the negated consequent: if P then Q, not Q, therefore not P. This valid inference pattern allows reasoning backward from the falsity of a conclusion to the falsity of a hypothesis. Modus tollens is essential for proof by contrapositive and for reasoning about what cannot be true given observed outcomes.",
            "Logical fallacies are errors in reasoning that undermine arguments by making the conclusion fail to follow from the premises. Common fallacies include affirming the consequent, denying the antecedent, ad hominem attacks, and begging the question. Recognizing fallacies is crucial for critical thinking, debate, and avoiding manipulation by unsound arguments that may seem persuasive despite being logically invalid.",
            "Contradiction occurs when a statement and its negation are both asserted or implied, making the logical system inconsistent. In classical logic, contradictions are fatal because anything can be derived from a contradiction through the principle of explosion. Detecting and resolving contradictions is essential for maintaining consistency in mathematical theories, databases, and knowledge representation systems.",
            "Consistency means no contradictions can be derived from a set of premises or within a logical system. A consistent set of statements contains no internal conflicts and can all be true simultaneously. Consistency is a fundamental requirement for any useful logical or mathematical system, as inconsistent systems allow deriving any statement and thus provide no useful information.",
            "Validity means the conclusion follows necessarily from the premises in an argument, regardless of whether the premises are actually true. A valid argument form guarantees that if the premises are true, the conclusion must also be true. Distinguishing validity from soundness is crucial: a valid argument with false premises may have a false conclusion, while a sound argument is both valid and has true premises.",
            "Soundness means an argument is valid with all true premises, guaranteeing a true conclusion. Soundness is the gold standard for deductive arguments, combining logical validity with factual correctness. When evaluating arguments, one must check both whether the reasoning is valid and whether the premises are true to determine whether the argument is sound.",
            "Logical equivalence means two statements have the same truth value in all possible situations, making them interchangeable in logical contexts. Equivalent statements may look very different but express the same logical content. Recognizing equivalences enables simplification of logical expressions and finding alternative formulations that may be more useful for particular purposes.",
            "Implication connects antecedent to consequent conditionally, expressing that if the antecedent is true then the consequent must also be true. The truth table for implication is often counterintuitive because an implication with false antecedent is considered true. Understanding implication is crucial for reasoning about conditional statements, causation, and logical relationships between propositions.",
            "Conjunction connects statements with the logical and operation, requiring both conjuncts to be true for the conjunction to be true. Conjunction is associative and commutative, allowing reordering and grouping of conjuncts. In natural language, conjunction often carries additional meaning about temporal or causal relationships that pure logical conjunction does not capture.",
            "Disjunction connects statements with the logical or operation, requiring at least one disjunct to be true for the disjunction to be true. The inclusive interpretation of disjunction allows both disjuncts to be true, while the exclusive interpretation requires exactly one. In logic and mathematics, disjunction is typically inclusive, matching the truth table used in formal systems.",
            "Negation reverses the truth value of a statement, making true statements false and false statements true. Double negation returns a statement to its original truth value in classical logic, though this fails in some non classical logics. Negation is fundamental to expressing denial, contradiction, and the opposite of a claim, enabling rich logical expression."
        );
    }
    
    /**
     * 生成编程相关文本
     * 每条文本扩展为较长的段落，适合预训练任务
     */
    private static List<String> generateCodingTexts() {
        return Arrays.asList(
            "Programming transforms algorithms into executable instructions that computers can process, bridging the gap between human thinking and machine execution. This translation requires understanding both the problem domain and the constraints of the target platform. Good programmers develop skills in decomposition, pattern recognition, abstraction, and algorithm design that enable them to express complex logic in code that is both correct and maintainable.",
            "Debugging identifies and fixes errors in program logic through systematic investigation of program behavior. Effective debugging involves reproducing the error, understanding expected versus actual behavior, isolating the problematic code, and verifying that fixes resolve the issue without introducing new problems. Debugging tools like debuggers, loggers, and profilers help developers understand program execution and locate error sources efficiently.",
            "Code review improves quality through peer examination, catching bugs, design issues, and maintainability problems before code reaches production. Reviews spread knowledge across teams, enforce coding standards, and provide learning opportunities for both reviewers and authors. Effective code reviews focus on clarity, correctness, performance, security, and alignment with architectural goals while maintaining constructive and respectful communication.",
            "Testing verifies that programs behave as expected across a range of inputs and conditions, providing confidence that code works correctly. Testing approaches include unit testing for individual components, integration testing for interactions between components, and end to end testing for complete system behavior. Comprehensive test suites serve as executable documentation and catch regressions when code is modified.",
            "Refactoring improves code structure without changing behavior, making code more maintainable, readable, and efficient while preserving functionality. Common refactorings include extracting methods, renaming variables, moving code between classes, and simplifying conditional logic. Refactoring is most effective when done incrementally alongside comprehensive tests that verify behavior preservation at each step.",
            "Algorithm efficiency measures computational resource usage in terms of time complexity and space complexity, typically expressed using big O notation. Understanding efficiency enables developers to choose appropriate algorithms for their use cases and predict how code will scale with larger inputs. Common complexity classes include constant time, logarithmic, linear, quadratic, and exponential, each with different implications for performance.",
            "Data structures organize information for efficient access and manipulation, providing the foundation for algorithms and software systems. Common structures include arrays, linked lists, stacks, queues, trees, hash tables, and graphs, each offering different tradeoffs between access time, insertion cost, and memory usage. Choosing appropriate data structures is crucial for achieving good performance in software applications.",
            "Recursion solves problems by calling functions on smaller inputs, breaking complex problems into simpler subproblems with the same structure. Recursive solutions often mirror the natural structure of problems, producing elegant and intuitive code. Key concepts include base cases that terminate recursion, recursive cases that reduce problem size, and understanding the call stack that tracks pending recursive calls.",
            "Iteration repeats operations using loops until a condition is met, providing the primary mechanism for processing collections and repeating actions in most programming languages. Loop constructs include for loops for known iteration counts, while loops for condition based iteration, and do while loops that execute at least once. Understanding iteration is fundamental to processing data and implementing algorithms.",
            "Abstraction hides complexity behind simple interfaces, allowing users to work with high level concepts without understanding implementation details. Well designed abstractions reduce cognitive load, enable code reuse, and allow implementation changes without affecting users. Finding the right level of abstraction is a key skill in software design, balancing simplicity with the flexibility needed for different use cases.",
            "Modularity divides programs into independent components with well defined interfaces, improving maintainability, testability, and reusability. Modules encapsulate related functionality and expose only necessary interfaces to other parts of the system. Good modular design follows principles like single responsibility, separation of concerns, and high cohesion within modules with low coupling between them.",
            "Documentation explains code purpose and usage clearly, serving as a guide for developers who need to understand, use, or modify the codebase. Effective documentation includes high level architecture descriptions, API documentation for public interfaces, and inline comments for complex implementation details. Maintaining documentation alongside code changes prevents it from becoming outdated and misleading.",
            "Version control tracks changes and enables collaboration on code by maintaining a complete history of modifications. Git has become the dominant version control system, supporting distributed development workflows, branching for feature development, and pull requests for code review. Version control is essential for coordinating team development, maintaining multiple versions, and recovering from mistakes.",
            "Error handling manages exceptional conditions gracefully, preventing crashes and providing useful feedback when problems occur. Robust error handling includes validating inputs, checking for error conditions, and providing informative error messages. Different languages offer different mechanisms including exceptions, error codes, and result types, each with tradeoffs in explicitness and ease of use.",
            "Code optimization improves performance and resource efficiency, making programs run faster or use less memory. Optimization techniques include choosing efficient algorithms and data structures, reducing unnecessary computation, and leveraging hardware capabilities. Premature optimization should be avoided in favor of writing clear code first, then optimizing based on profiling data that identifies actual bottlenecks."
        );
    }
    
    /**
     * 生成深度学习相关文本
     * 每条文本扩展为较长的段落，适合预训练任务
     */
    private static List<String> generateDeepLearningTexts() {
        return Arrays.asList(
            "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. Each layer transforms input into increasingly abstract features, with early layers capturing simple patterns and deeper layers capturing complex concepts. This approach has achieved remarkable success across computer vision, natural language processing, speech recognition, and many other domains that require learning from complex, high dimensional data.",
            "Backpropagation computes gradients through the chain rule of calculus, propagating error signals backward through the network to update parameters efficiently. Starting from the loss at the output, gradients are computed layer by layer by multiplying local gradients with upstream gradients. This elegant algorithm makes it practical to train networks with millions or billions of parameters and is fundamental to understanding how neural networks learn from data.",
            "Gradient descent optimizes parameters iteratively by moving in the direction that reduces the loss function, with the learning rate controlling the step size. Modern optimizers extend basic gradient descent with momentum to accelerate convergence, adaptive learning rates for each parameter, and techniques to escape local minima. Choosing appropriate optimization strategies is crucial for training deep networks effectively.",
            "Loss functions measure prediction errors during training, providing the signal that drives learning in neural networks. Common loss functions include mean squared error for regression tasks, cross entropy for classification, and specialized losses for tasks like object detection and sequence labeling. The choice of loss function encodes assumptions about what constitutes good predictions and significantly impacts learned behavior.",
            "Activation functions introduce nonlinearity into neural networks, enabling them to approximate complex functions that linear transformations alone cannot represent. Popular activation functions include ReLU which outputs the input if positive and zero otherwise, sigmoid and tanh for bounded outputs, and newer variants like GELU and Swish that provide smoother gradients. The choice of activation function affects training dynamics and model performance.",
            "Transformers use attention mechanisms for sequence processing, enabling parallel processing of all positions rather than sequential processing like recurrent networks. The self attention mechanism allows each position to attend to all other positions directly, capturing long range dependencies efficiently. Transformers have become the dominant architecture for natural language processing and are increasingly applied to other domains including computer vision.",
            "Language models predict next tokens in sequences, learning to assign probability distributions over possible continuations. Training on large text corpora teaches models about grammar, semantics, world knowledge, and reasoning patterns. Modern language models can generate coherent text, answer questions, translate between languages, and perform many other tasks through their learned representations of language.",
            "Pre training learns general representations from large scale data, providing a foundation that can be adapted to many downstream tasks. During pre training, models learn to predict masked tokens or next tokens, forcing them to develop useful representations of language and knowledge. Pre trained models provide strong initializations that dramatically reduce the data and computation needed for task specific training.",
            "Fine tuning adapts pre trained models to specific tasks by continuing training on task specific data with a smaller learning rate. The pre trained weights encode general knowledge that transfers well to related tasks, so only modest adaptation is typically needed. This transfer learning paradigm has made it practical to apply large models to many tasks without training from scratch on limited task specific data.",
            "Reinforcement learning optimizes agent behavior through rewards received from the environment, learning to take actions that maximize cumulative reward over time. Key concepts include states, actions, rewards, policies, and value functions. Reinforcement learning is applicable to control problems, game playing, robotics, and increasingly to training language models through human feedback.",
            "Policy gradient methods update action probabilities directly based on received rewards, learning stochastic policies that map states to action distributions. Unlike value based methods that must derive policies from value estimates, policy gradients can optimize policies directly and handle continuous action spaces. Modern policy gradient algorithms include REINFORCE, actor critic methods, and proximal policy optimization.",
            "Reward shaping guides learning toward desired behavior by providing intermediate rewards that help agents learn more efficiently than sparse final rewards alone. Well designed reward shaping can dramatically accelerate learning while poorly designed rewards can lead to unintended behaviors where agents exploit reward loopholes. Careful reward engineering is crucial for successful reinforcement learning applications.",
            "Value functions estimate expected future rewards from states or state action pairs, providing a foundation for many reinforcement learning algorithms. The optimal value function captures the best possible performance from each state and enables deriving optimal policies. Value based methods learn value estimates and derive policies from them, while actor critic methods combine value learning with policy optimization.",
            "Human feedback aligns models with human preferences by incorporating human judgments into the training process. Rather than specifying exact desired behaviors, humans can compare model outputs and indicate preferences, which are then used to train reward models or directly update policies. This approach enables aligning models with complex human values that are difficult to specify formally.",
            "RLHF combines human feedback with reinforcement learning to train models that are helpful, harmless, and honest according to human preferences. The typical pipeline involves collecting human preference data, training a reward model to predict preferences, and then optimizing the policy to maximize predicted reward. RLHF has become essential for training language models that interact safely and helpfully with users."
        );
    }
    
    /**
     * 生成反思与自我修正相关文本
     * 每条文本扩展为较长的段落，适合预训练任务
     */
    private static List<String> generateReflectionTexts() {
        return Arrays.asList(
            "Self reflection enables models to evaluate their own outputs critically, identifying errors and opportunities for improvement before presenting final answers. This meta cognitive capability allows models to catch mistakes, consider alternatives, and refine initial responses through internal review processes. Self reflection is crucial for producing reliable outputs and is a key capability of advanced reasoning systems like DeepSeek R1.",
            "Error correction improves answers through iterative refinement, allowing models to identify and fix problems in their initial attempts. This process can involve spotting factual errors, logical inconsistencies, or missing information, and then making targeted corrections. Error correction transforms initial draft responses into polished final answers through systematic review and improvement cycles.",
            "Confidence estimation indicates reliability of responses, helping users understand when they can trust model outputs and when additional verification is needed. Well calibrated confidence scores reflect actual accuracy, with high confidence indicating high probability of correctness and low confidence flagging uncertainty. Confidence estimation is essential for responsible deployment of AI systems in high stakes applications.",
            "Quality assessment scores outputs on multiple dimensions including accuracy, completeness, relevance, clarity, and adherence to constraints. Multi dimensional evaluation provides richer feedback than binary correct or incorrect judgments, enabling targeted improvements. Quality assessment helps models understand what makes responses good and guides the refinement process toward higher quality outputs.",
            "Chain of thought shows explicit reasoning process, making each step of inference visible and verifiable. This transparency helps users follow the logic leading to conclusions, identify where reasoning might go wrong, and build trust in model outputs. Chain of thought prompting has been shown to significantly improve performance on complex reasoning tasks by encouraging systematic step by step problem solving.",
            "Self verification checks reasoning for logical errors by applying verification procedures to each step in the reasoning chain. A model might check that intermediate conclusions follow validly from premises, that mathematical calculations are correct, or that logical implications are sound. Self verification catches errors early before they propagate through the reasoning process and lead to incorrect final answers.",
            "Iterative improvement refines answers through multiple passes, with each pass identifying weaknesses and making targeted improvements. This approach mirrors how humans revise drafts into polished final work, progressively improving quality with each iteration. Iterative improvement can involve revising for accuracy, clarity, completeness, and other quality dimensions across multiple refinement cycles.",
            "Metacognitive monitoring tracks reasoning progress, enabling models to recognize when they are on the right track, when they need to try a different approach, or when they have reached a satisfactory answer. This awareness of the reasoning process itself helps models allocate effort efficiently and avoid wasting computation on unproductive solution paths while knowing when additional effort is warranted.",
            "Self critique identifies weaknesses in generated responses by applying critical analysis to one own outputs. This capability enables models to spot logical flaws, missing information, unclear explanations, or other problems that might reduce response quality. Self critique provides the foundation for improvement by highlighting specific aspects that need attention in the revision process.",
            "Reasoning revision updates conclusions based on new insights, allowing models to change their minds when evidence warrants. This flexibility is essential for accurate reasoning because initial conclusions may be wrong or incomplete. Effective reasoning revision requires recognizing when updates are needed, determining what changes to make, and maintaining consistency throughout the reasoning chain.",
            "Answer comparison evaluates multiple solution approaches by generating alternative answers and analyzing their differences. This technique can identify errors by finding inconsistencies between approaches or increase confidence when different methods converge to the same answer. Answer comparison provides a robust way to verify correctness and explore the solution space more thoroughly.",
            "Certainty calibration aligns confidence with accuracy so that expressed confidence reliably indicates actual probability of correctness. Well calibrated models know what they know and know what they do not know, expressing appropriate uncertainty rather than being overconfident in wrong answers or underconfident in correct ones. Calibration is essential for appropriate reliance on model outputs.",
            "Reasoning transparency makes thought process visible by exposing intermediate steps, assumptions, and decision points in the reasoning chain. Transparent reasoning enables users to verify logic, understand how conclusions were reached, and identify potential errors. This transparency builds trust and enables effective human AI collaboration in problem solving.",
            "Error analysis categorizes and learns from mistakes by identifying patterns in errors and understanding their causes. Systematic error analysis enables models to recognize situations where they are prone to specific error types and take corrective action. Learning from mistakes is essential for improving performance over time and avoiding repeated errors on similar problems.",
            "Self consistency checks whether multiple paths reach the same answer, providing a powerful method for verifying correctness without external ground truth. When different reasoning approaches converge to the same conclusion, confidence in that conclusion increases. Divergent results signal potential errors that require investigation, making self consistency a valuable tool for improving reliability."
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
    
    /**
     * 生成 RLHF 推理数据（Chat Template 格式 + 奖励标注）
     * 
     * 格式: [REWARD:score] <|im_start|>user Q<|im_end|><|im_start|>assistant <|begin_of_thought|>CoT<|end_of_thought|> A<|im_end|>
     * 
     * 包含三个奖励等级：
     * - 高奖励 (0.8-1.0): 推理正确、步骤清晰
     * - 中等奖励 (0.5-0.7): 答案正确但推理不够详细
     * - 低奖励 (0.2-0.4): 推理有错误或答案错误
     */
    private static List<String> generateRLHFReasoningData() {
        List<String> rlhfData = new ArrayList<>();
        
        // 高奖励的正确推理 (10条, reward 0.8-1.0)
        rlhfData.add("[REWARD:0.95] " + DeepSeekR1TokenizerUtil.buildSFTText("5 plus 3", "5 plus 3 equals 8. Verified by counting.", "8"));
        rlhfData.add("[REWARD:0.90] " + DeepSeekR1TokenizerUtil.buildSFTText("What is 12 divided by 4", "12 divided by 4 means how many 4s in 12. 4 times 3 is 12.", "3"));
        rlhfData.add("[REWARD:0.92] " + DeepSeekR1TokenizerUtil.buildSFTText("All dogs bark. Rex is a dog. Does Rex bark", "Major premise says all dogs bark. Rex is a dog. Therefore Rex barks by syllogism.", "Yes"));
        rlhfData.add("[REWARD:0.88] " + DeepSeekR1TokenizerUtil.buildSFTText("If today is Monday what is tomorrow", "Days follow Monday Tuesday order. Day after Monday is Tuesday.", "Tuesday"));
        rlhfData.add("[REWARD:0.93] " + DeepSeekR1TokenizerUtil.buildSFTText("Which is larger 7 or 5", "7 is greater than 5 because 7 minus 5 equals 2 which is positive.", "7"));
        rlhfData.add("[REWARD:0.91] " + DeepSeekR1TokenizerUtil.buildSFTText("Half of 10 is what", "Half means divide by 2. 10 divided by 2 equals 5.", "5"));
        rlhfData.add("[REWARD:0.89] " + DeepSeekR1TokenizerUtil.buildSFTText("3 times 4 equals what", "3 groups of 4 is 4 plus 4 plus 4 which equals 12.", "12"));
        rlhfData.add("[REWARD:0.94] " + DeepSeekR1TokenizerUtil.buildSFTText("If A then B and A is true what is B", "Apply modus ponens: Given A implies B and A is true, B must be true.", "B is true"));
        rlhfData.add("[REWARD:0.87] " + DeepSeekR1TokenizerUtil.buildSFTText("20 minus 8 is what", "Start with 20, take away 8. 20 minus 8 equals 12. Verify: 12 plus 8 is 20.", "12"));
        rlhfData.add("[REWARD:0.96] " + DeepSeekR1TokenizerUtil.buildSFTText("Is 15 odd or even", "Odd numbers are not divisible by 2. 15 divided by 2 is 7.5 which is not integer.", "15 is odd"));
        
        // 中等奖励的可接受推理 (10条, reward 0.5-0.7)
        rlhfData.add("[REWARD:0.65] " + DeepSeekR1TokenizerUtil.buildSFTText("6 plus 7", "Brief calculation.", "13"));
        rlhfData.add("[REWARD:0.60] " + DeepSeekR1TokenizerUtil.buildSFTText("What is 9 times 2", "Direct multiplication.", "18"));
        rlhfData.add("[REWARD:0.70] " + DeepSeekR1TokenizerUtil.buildSFTText("Is a square a rectangle", "It has four right angles.", "Yes"));
        rlhfData.add("[REWARD:0.55] " + DeepSeekR1TokenizerUtil.buildSFTText("100 divided by 5", "Simple division.", "20"));
        rlhfData.add("[REWARD:0.68] " + DeepSeekR1TokenizerUtil.buildSFTText("Sum of 4 and 9", "Add ones digit 4 plus 9 is 13.", "13"));
        rlhfData.add("[REWARD:0.62] " + DeepSeekR1TokenizerUtil.buildSFTText("Next number after 7", "Counting sequence.", "8"));
        rlhfData.add("[REWARD:0.58] " + DeepSeekR1TokenizerUtil.buildSFTText("Double of 6", "Double means multiply by 2.", "12"));
        rlhfData.add("[REWARD:0.66] " + DeepSeekR1TokenizerUtil.buildSFTText("Is 10 greater than 3", "10 is clearly larger.", "Yes"));
        rlhfData.add("[REWARD:0.72] " + DeepSeekR1TokenizerUtil.buildSFTText("What comes before 5", "In counting order 4 precedes 5.", "4"));
        rlhfData.add("[REWARD:0.64] " + DeepSeekR1TokenizerUtil.buildSFTText("8 minus 3", "Subtraction gives 5.", "5"));
        
        // 低奖励的需改进推理 (10条, reward 0.2-0.4)
        rlhfData.add("[REWARD:0.25] " + DeepSeekR1TokenizerUtil.buildSFTText("7 plus 8", "I think it is 14.", "14"));
        rlhfData.add("[REWARD:0.30] " + DeepSeekR1TokenizerUtil.buildSFTText("All cats are pets. Some pets are dogs. Are all cats dogs", "Since cats are pets and some pets are dogs, cats must be dogs.", "Yes"));
        rlhfData.add("[REWARD:0.35] " + DeepSeekR1TokenizerUtil.buildSFTText("5 times 5", "5 times 5 should be around 20.", "20"));
        rlhfData.add("[REWARD:0.28] " + DeepSeekR1TokenizerUtil.buildSFTText("12 divided by 3", "12 divided by 3 is 3.", "3"));
        rlhfData.add("[REWARD:0.40] " + DeepSeekR1TokenizerUtil.buildSFTText("Is 8 even", "Not sure about this.", "Maybe"));
        rlhfData.add("[REWARD:0.32] " + DeepSeekR1TokenizerUtil.buildSFTText("What is 15 minus 7", "15 minus 7 is 7.", "7"));
        rlhfData.add("[REWARD:0.38] " + DeepSeekR1TokenizerUtil.buildSFTText("If P then Q and Q is true what about P", "Since Q is true and P implies Q, P must be true.", "P is true"));
        rlhfData.add("[REWARD:0.22] " + DeepSeekR1TokenizerUtil.buildSFTText("9 plus 4", "9 plus 4 is 12.", "12"));
        rlhfData.add("[REWARD:0.35] " + DeepSeekR1TokenizerUtil.buildSFTText("6 times 7", "6 times 7 is 43.", "43"));
        rlhfData.add("[REWARD:0.29] " + DeepSeekR1TokenizerUtil.buildSFTText("Half of 14", "Half of 14 is 8.", "8"));
        
        return rlhfData;
    }
    
    /**
     * 生成数学验证数据（Chat Template 格式）
     * 
     * 格式: [TYPE:math] <|im_start|>user Q<|im_end|> | GroundTruth
     * RLVR 场景中 prompt 使用 Chat Template 的 user 部分，
     * assistant 回复由模型在训练时自行生成。
     */
    private static List<String> generateMathVerificationData() {
        List<String> mathData = new ArrayList<>();
        String imStart = DeepSeekR1TokenizerUtil.IM_START_TOKEN;
        String imEnd = DeepSeekR1TokenizerUtil.IM_END_TOKEN;
        String userRole = DeepSeekR1TokenizerUtil.ROLE_USER;
        
        // 算术运算
        mathData.add("[TYPE:math] " + imStart + userRole + " What is 15 + 27?" + imEnd + " | 42");
        mathData.add("[TYPE:math] " + imStart + userRole + " Calculate 8 * 6" + imEnd + " | 48");
        mathData.add("[TYPE:math] " + imStart + userRole + " What is 100 / 4?" + imEnd + " | 25");
        mathData.add("[TYPE:math] " + imStart + userRole + " 50 - 23 equals?" + imEnd + " | 27");
        mathData.add("[TYPE:math] " + imStart + userRole + " What is 7 * 9?" + imEnd + " | 63");
        mathData.add("[TYPE:math] " + imStart + userRole + " Calculate 144 / 12" + imEnd + " | 12");
        mathData.add("[TYPE:math] " + imStart + userRole + " 25 + 75 equals?" + imEnd + " | 100");
        mathData.add("[TYPE:math] " + imStart + userRole + " What is 60 - 18?" + imEnd + " | 42");
        mathData.add("[TYPE:math] " + imStart + userRole + " 11 * 11 equals?" + imEnd + " | 121");
        mathData.add("[TYPE:math] " + imStart + userRole + " What is 81 / 9?" + imEnd + " | 9");
        
        // 代数问题
        mathData.add("[TYPE:math] " + imStart + userRole + " Solve: 2x + 5 = 13" + imEnd + " | 4");
        mathData.add("[TYPE:math] " + imStart + userRole + " Solve: 3x - 7 = 14" + imEnd + " | 7");
        mathData.add("[TYPE:math] " + imStart + userRole + " Solve: x / 2 = 8" + imEnd + " | 16");
        mathData.add("[TYPE:math] " + imStart + userRole + " Solve: 4x + 3 = 19" + imEnd + " | 4");
        mathData.add("[TYPE:math] " + imStart + userRole + " Solve: 5x = 35" + imEnd + " | 7");
        
        // 数值比较
        mathData.add("[TYPE:math] " + imStart + userRole + " Which is larger: 15 or 23?" + imEnd + " | 23");
        mathData.add("[TYPE:math] " + imStart + userRole + " Is 42 greater than 38?" + imEnd + " | true");
        mathData.add("[TYPE:math] " + imStart + userRole + " What is the maximum of 7, 12, 5?" + imEnd + " | 12");
        
        // 幂运算
        mathData.add("[TYPE:math] " + imStart + userRole + " What is 2^5?" + imEnd + " | 32");
        mathData.add("[TYPE:math] " + imStart + userRole + " Calculate 3^3" + imEnd + " | 27");
        mathData.add("[TYPE:math] " + imStart + userRole + " What is 10^2?" + imEnd + " | 100");
        
        return mathData;
    }
    
    /**
     * 生成逻辑验证数据（Chat Template 格式）
     */
    private static List<String> generateLogicVerificationData() {
        List<String> logicData = new ArrayList<>();
        String imStart = DeepSeekR1TokenizerUtil.IM_START_TOKEN;
        String imEnd = DeepSeekR1TokenizerUtil.IM_END_TOKEN;
        String userRole = DeepSeekR1TokenizerUtil.ROLE_USER;
        
        // 布尔逻辑
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is the statement true AND false true or false?" + imEnd + " | false");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is true OR false true or false?" + imEnd + " | true");
        logicData.add("[TYPE:logic] " + imStart + userRole + " What is NOT true?" + imEnd + " | false");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is true AND true OR false true?" + imEnd + " | true");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is false AND false true or false?" + imEnd + " | false");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is true AND true true or false?" + imEnd + " | true");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is false OR false true or false?" + imEnd + " | false");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is NOT false true?" + imEnd + " | true");
        
        // 矛盾判断
        logicData.add("[TYPE:logic] " + imStart + userRole + " Can X is both true and false be valid?" + imEnd + " | false");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is All A are B and some A are not B consistent?" + imEnd + " | false");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is a contradiction always false?" + imEnd + " | true");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Can a statement be both true and false?" + imEnd + " | false");
        logicData.add("[TYPE:logic] " + imStart + userRole + " Is P AND NOT P ever true?" + imEnd + " | false");
        
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
