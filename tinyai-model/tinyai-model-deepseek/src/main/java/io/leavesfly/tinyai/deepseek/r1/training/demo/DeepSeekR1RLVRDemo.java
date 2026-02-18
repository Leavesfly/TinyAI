package io.leavesfly.tinyai.deepseek.r1.training.demo;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1RLVRDataset;
import io.leavesfly.tinyai.deepseek.r1.training.verifier.CodeVerifier;
import io.leavesfly.tinyai.deepseek.r1.training.verifier.LogicVerifier;
import io.leavesfly.tinyai.deepseek.r1.training.verifier.MathVerifier;
import io.leavesfly.tinyai.deepseek.r1.training.verifier.VerificationResult;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.deepseek.r1.training.DeepSeekR1RLVRTrainer;

/**
 * DeepSeek-R1 RLVR训练演示
 * 
 * 演示RLVR (Reinforcement Learning from Verifiable Rewards) 训练流程
 * 
 * 对比RLHF和RLVR:
 * 
 * RLHF演示 (DeepSeekR1RLHFTrainer):
 * - 使用人类标注的奖励分数
 * - 奖励是连续值(0-1)
 * - 需要人工标注数据
 * - 适合开放性任务
 * 
 * RLVR演示 (本类):
 * - 使用可验证的规则/测试
 * - 奖励是二值(0或1)
 * - 自动验证，无需人工
 * - 适合可验证任务(数学、代码、逻辑)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1RLVRDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-R1 RLVR训练演示");
        System.out.println("Reinforcement Learning from Verifiable Rewards");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 示例1: 验证器演示
        example1_VerifierDemo();
        
        // 示例2: 数据集准备
        example2_DatasetPreparation();
        
        // 示例3: RLVR训练
        example3_RLVRTraining();
        
        // 示例4: RLVR vs RLHF对比
        example4_RLVRvsRLHF();
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("所有演示完成!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 示例1: 验证器演示
     */
    private static void example1_VerifierDemo() {
        System.out.println("=".repeat(80));
        System.out.println("示例1: 验证器演示");
        System.out.println("=".repeat(80));
        
        // 1. 数学验证器演示
        System.out.println("\n【数学验证器】");
        MathVerifier mathVerifier = new MathVerifier();
        
        String mathOutput1 = "Let me solve this: 15 + 27 = 42. The answer is 42.";
        VerificationResult mathResult1 = mathVerifier.verify(mathOutput1, "42");
        System.out.println("问题: 15 + 27 = ?");
        System.out.println("模型输出: " + mathOutput1);
        System.out.println(mathResult1);
        
        String mathOutput2 = "The answer is 40.";
        VerificationResult mathResult2 = mathVerifier.verify(mathOutput2, "42");
        System.out.println("\n错误示例:");
        System.out.println("模型输出: " + mathOutput2);
        System.out.println(mathResult2);
        
        // 2. 代码验证器演示
        System.out.println("\n【代码验证器】");
        CodeVerifier codeVerifier = new CodeVerifier();
        
        String codeOutput1 = "Here's the solution:\n```java\npublic int add(int a, int b) { return a + b; }\n```";
        String expectedCode = "public int add(int a, int b) { return a + b; }";
        VerificationResult codeResult = codeVerifier.verify(codeOutput1, expectedCode);
        System.out.println("问题: 实现加法函数");
        System.out.println("模型输出: " + codeOutput1);
        System.out.println(codeResult);
        
        // 3. 逻辑验证器演示
        System.out.println("\n【逻辑验证器】");
        LogicVerifier logicVerifier = new LogicVerifier();
        
        String logicOutput = "Given that all humans are mortal, and Socrates is human, therefore Socrates is mortal.";
        VerificationResult logicResult = logicVerifier.verify(logicOutput, "mortal");
        System.out.println("问题: 三段论推理");
        System.out.println("模型输出: " + logicOutput);
        System.out.println(logicResult);
        
        System.out.println("\n✓ 验证器演示完成!");
    }
    
    /**
     * 示例2: 数据集准备
     */
    private static void example2_DatasetPreparation() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例2: RLVR数据集准备");
        System.out.println("=".repeat(80));
        
        // 创建RLVR数据集
        DeepSeekR1RLVRDataset dataset = new DeepSeekR1RLVRDataset(
            2,      // batchSize
            64,     // maxSeqLen
            5000    // vocabSize
        );
        
        // 添加数学问题样本
        System.out.println("\n添加数学问题样本...");
        dataset.addSample("What is 25 + 17?", "42", "math");
        dataset.addSample("Calculate 8 * 6", "48", "math");
        dataset.addSample("Solve: 2x + 5 = 13", "4", "math");
        System.out.println("✓ 已添加 3 个数学样本");
        
        // 添加代码问题样本
        System.out.println("\n添加代码问题样本...");
        dataset.addSample(
            "Write a function to add two numbers",
            "public int add(int a, int b) { return a + b; }",
            "code"
        );
        System.out.println("✓ 已添加 1 个代码样本");
        
        // 添加逻辑推理样本
        System.out.println("\n添加逻辑推理样本...");
        dataset.addSample(
            "If all A are B, and C is A, what can we conclude?",
            "C is B",
            "logic"
        );
        System.out.println("✓ 已添加 1 个逻辑样本");
        
        System.out.println("\n数据集统计:");
        System.out.println("  总样本数: " + dataset.getSampleCount());
        System.out.println("  批次大小: " + dataset.getBatchSize());
        
        // 测试批次迭代
        System.out.println("\n测试批次迭代...");
        dataset.prepare(false);
        int batchCount = 0;
        while (dataset.hasNext()) {
            DeepSeekR1RLVRDataset.Batch batch = dataset.nextBatch();
            batchCount++;
            System.out.printf("  批次 %d: %d 个样本\n", batchCount, batch.getBatchSize());
        }
        
        System.out.println("\n✓ 数据集准备完成!");
    }
    
    /**
     * 示例3: RLVR训练
     */
    private static void example3_RLVRTraining() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例3: RLVR训练流程");
        System.out.println("=".repeat(80));
        
        // 1. 创建微型模型
        System.out.println("\n创建DeepSeek-R1微型模型...");
        DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("DeepSeek-R1-RLVR");
        DeepSeekR1Config config = model.getConfig();
        System.out.println("✓ 模型已创建");
        System.out.println("  参数量: " + config.estimateParameterCount());
        System.out.println("  嵌入维度: " + config.getNEmbd());
        System.out.println("  专家数量: " + config.getNumExperts());
        System.out.println("  Top-K选择: " + config.getTopK());
        
        // 2. 准备RLVR数据集
        System.out.println("\n准备RLVR训练数据...");
        DeepSeekR1RLVRDataset dataset = new DeepSeekR1RLVRDataset(
            2,
            config.getNPositions(),
            config.getVocabSize()
        );
        
        // 添加多种类型的可验证样本
        addMathSamples(dataset);
        addCodeSamples(dataset);
        addLogicSamples(dataset);
        
        System.out.println("✓ 数据集已准备");
        System.out.println("  训练样本: " + dataset.getSampleCount());
        
        // 3. 创建RLVR训练器
        System.out.println("\n创建RLVR训练器...");
        DeepSeekR1RLVRTrainer trainer = new DeepSeekR1RLVRTrainer(model, dataset);
        
        // 配置训练参数
        trainer.configure(
            50,      // maxEpochs (增加训练轮次以充分学习)
            0.05f,   // learningRate (降低学习率提高稳定性)
            0.7f,    // correctnessWeight (正确性最重要)
            0.2f,    // reasoningQualityWeight
            0.1f     // verificationWeight
        );
        
        System.out.println("✓ 训练器已配置");
        System.out.println("  奖励权重: 70% 正确性 + 20% 推理质量 + 10% 验证完整性");
        
        // 4. 开始训练
        System.out.println("\n开始RLVR训练...");
        trainer.train();
        
        // 5. 查看训练统计
        System.out.println("\n训练统计:");
        var stats = trainer.getTrainingStats();
        System.out.println("  总步数: " + stats.get("total_steps"));
        System.out.printf("  平均正确率: %.4f\n", stats.get("avg_correctness"));
        System.out.printf("  平均奖励: %.4f\n", stats.get("avg_reward"));
        System.out.printf("  平均质量: %.4f\n", stats.get("avg_quality"));
        
        System.out.println("\n✓ RLVR训练完成!");
    }
    
    /**
     * 示例4: RLVR vs RLHF对比
     */
    private static void example4_RLVRvsRLHF() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例4: RLVR vs RLHF 对比");
        System.out.println("=".repeat(80));
        
        System.out.println("\n【训练方式对比】");
        System.out.println();
        System.out.printf("%-20s | %-30s | %-30s\n", "维度", "RLHF", "RLVR");
        System.out.println("-".repeat(85));
        
        printComparison("奖励来源", "人类主观反馈", "可验证的客观标准");
        printComparison("奖励类型", "连续值 (0-1)", "二值 (0 或 1)");
        printComparison("验证方式", "奖励模型近似", "规则/测试用例验证");
        printComparison("数据准备", "需人工标注", "自动验证");
        printComparison("训练速度", "慢 (受标注速度限制)", "快 (可大规模并行)");
        printComparison("适用场景", "开放性任务", "可验证任务");
        printComparison("抗奖励欺骗", "弱", "强");
        printComparison("可扩展性", "受限", "高");
        
        System.out.println();
        System.out.println("【使用建议】");
        System.out.println();
        System.out.println("使用RLHF当:");
        System.out.println("  ✓ 任务是开放性的 (如创意写作、对话)");
        System.out.println("  ✓ 质量难以量化 (如风格、语气)");
        System.out.println("  ✓ 需要人类主观判断");
        System.out.println();
        System.out.println("使用RLVR当:");
        System.out.println("  ✓ 任务有明确正确答案 (如数学、代码)");
        System.out.println("  ✓ 可以编写验证规则");
        System.out.println("  ✓ 需要大规模训练");
        System.out.println("  ✓ 追求客观正确性");
        System.out.println();
        System.out.println("【DeepSeek-R1的选择】");
        System.out.println();
        System.out.println("DeepSeek-R1模型特别适合RLVR，因为:");
        System.out.println("  1. 内置推理模块 - 可生成完整推理过程");
        System.out.println("  2. 反思机制 - 可自我验证推理正确性");
        System.out.println("  3. 质量评分 - 可辅助验证");
        System.out.println("  4. 多步推理 - 便于逐步验证");
        
        System.out.println("\n✓ 对比演示完成!");
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 添加数学样本
     */
    private static void addMathSamples(DeepSeekR1RLVRDataset dataset) {
        dataset.addSample("What is 15 + 27?", "42", "math");
        dataset.addSample("Calculate 8 * 6", "48", "math");
        dataset.addSample("What is 100 / 4?", "25", "math");
        dataset.addSample("Solve: 3x + 7 = 22", "5", "math");
        dataset.addSample("What is 2^5?", "32", "math");
    }
    
    /**
     * 添加代码样本
     */
    private static void addCodeSamples(DeepSeekR1RLVRDataset dataset) {
        dataset.addSample(
            "Write a function to add two numbers",
            "public int add(int a, int b) { return a + b; }",
            "code"
        );
        dataset.addSample(
            "Write a function to check if number is even",
            "public boolean isEven(int n) { return n % 2 == 0; }",
            "code"
        );
    }
    
    /**
     * 添加逻辑样本
     */
    private static void addLogicSamples(DeepSeekR1RLVRDataset dataset) {
        dataset.addSample(
            "If all humans are mortal, and Socrates is human, what can we conclude?",
            "Socrates is mortal",
            "logic"
        );
        dataset.addSample(
            "If A > B and B > C, what is the relationship between A and C?",
            "A > C",
            "logic"
        );
    }
    
    /**
     * 打印对比行
     */
    private static void printComparison(String dimension, String rlhf, String rlvr) {
        System.out.printf("%-20s | %-30s | %-30s\n", dimension, rlhf, rlvr);
    }
}
