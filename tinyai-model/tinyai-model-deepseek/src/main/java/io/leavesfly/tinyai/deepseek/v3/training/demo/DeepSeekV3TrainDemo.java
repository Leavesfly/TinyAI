package io.leavesfly.tinyai.deepseek.v3.training.demo;

import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.deepseek.v3.training.*;
import io.leavesfly.tinyai.deepseek.base.TaskType;

import java.io.*;
import java.util.*;

/**
 * DeepSeek-V3 完整训练演示
 * <p>
 * 提供完整的 DeepSeek-V3 训练流程编排：
 * 1. 数据准备阶段 - 委托 {@link DeepSeekV3DatasetGenerator} 生成训练数据集
 * 2. 预训练阶段 - 因果语言建模（MoE 负载均衡）
 * 3. 通用后训练阶段 - 任务感知微调（ChatML + Answer-Only Loss）
 * 4. 代码专项后训练阶段 - 强化 MoE 专家代码特化
 * 5. RLHF 训练阶段 - 人类反馈强化学习（奖励加权回归）
 * 6. 推理测试阶段 - 多种生成策略演示
 * <p>
 * 架构说明：
 * - 数据生成逻辑拆分到 {@link DeepSeekV3DatasetGenerator}
 * - 分词工具提取到 {@link DeepSeekV3TokenizerUtil}
 * - 本类专注训练流程编排和数据集组装
 * <p>
 * 数据格式对比：
 * - 预训练：纯文本 → 全序列 CLM loss
 * - 后训练：ChatML 对话模板 → Answer-Only loss（只对 assistant 回答计算损失）
 * - RLHF：ChatML 模板 + 奖励标注 → Reward-weighted loss（奖励加权回归）
 *
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3TrainDemo {

    private static final DeepSeekV3TokenizerUtil sharedTokenizer = new DeepSeekV3TokenizerUtil();

    private static final String DATA_DIR = DeepSeekV3DatasetGenerator.DATA_DIR;
    private static final String CHECKPOINT_DIR = "./checkpoints/deepseek_v3";

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 完整训练与推理演示 V2");
        System.out.println("适用于教学和学习的小型数据集训练方案");
        System.out.println("=".repeat(80));

        try {
            // 步骤0: 准备数据集文件
            DeepSeekV3DatasetGenerator.prepareAllDatasets();

            // 步骤1: 预训练
            DeepSeekV3Model pretrainedModel = runPretraining();

            // 步骤2: 通用后训练（任务感知微调）
            DeepSeekV3Model finetunedModel = runPosttraining(pretrainedModel);

            // 步骤2B (可选): 代码生成专项后训练，强化 MoE 专家对代码任务的特化能力
            DeepSeekV3Model codeSpecializedModel = runCodePosttraining(finetunedModel);

            // 步骤3: RLHF 强化学习训练（对齐人类偏好）
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

    // ========== 步骤1: 预训练 ==========

    /**
     * 执行预训练
     */
    private static DeepSeekV3Model runPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: DeepSeek-V3 预训练 (Pretrain)");
        System.out.println("=".repeat(80));

        // 1. 读取全部数据以构建完整词汇表
        System.out.println("\n📝 加载所有数据以构建词汇表...");
        List<String> pretrainTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/pretrain.txt");
        List<String> posttrainTrainTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/posttrain_train.txt");
        List<String> posttrainValTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/posttrain_val.txt");

        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练训练数据: " + posttrainTrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练验证数据: " + posttrainValTexts.size() + " 条");

        // 2. 基于所有数据构建并冻结词汇表
        System.out.println("\n📝 构建完整词汇表...");
        List<String> allTexts = new ArrayList<>();
        allTexts.addAll(pretrainTexts);
        allTexts.addAll(posttrainTrainTexts);
        allTexts.addAll(posttrainValTexts);
        for (String text : allTexts) {
            sharedTokenizer.encode(DeepSeekV3TokenizerUtil.removeTaskLabel(text));
        }
        int vocabSize = sharedTokenizer.getVocabSize();
        sharedTokenizer.freeze();
        System.out.println("  ✓ 完整词汇表大小: " + vocabSize);
        System.out.println("  ✓ 词汇表已冻结，后续不再增加新词");

        // 3. 创建 DeepSeek-V3 模型
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
                pretrainTexts, config.getNPositions(), 4, config.getVocabSize(), false
        );
        System.out.println("  ✓ 训练样本: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次大小: 4, 序列长度: " + config.getNPositions());

        // 5. 配置并运行预训练器
        System.out.println("\n📝 配置预训练器...");
        DeepSeekV3Pretrainer trainer = new DeepSeekV3Pretrainer(model, dataset);
        trainer.configure(30, 2e-3f, 10, 1.0f)
                .setCheckpoint(CHECKPOINT_DIR + "/pretrain", 500);

        System.out.println("  ✓ 最大轮次: 30  ✓ 学习率: 2e-3  ✓ Warmup步数: 10");
        System.out.println("  ✓ MoE负载均衡权重: " + config.getLoadBalanceLossWeight());

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

    // ========== 步骤2: 后训练/微调 ==========

    /**
     * 执行通用后训练（任务感知微调，ChatML + Answer-Only Loss）
     */
    private static DeepSeekV3Model runPosttraining(DeepSeekV3Model pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: DeepSeek-V3 后训练/微调 (Posttrain)");
        System.out.println("=".repeat(80));

        List<String> trainTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/posttrain_train.txt");
        List<String> valTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/posttrain_val.txt");
        System.out.println("\n📝 加载后训练数据... 训练集: " + trainTexts.size() + " 条, 验证集: " + valTexts.size() + " 条");

        DeepSeekV3Config config = pretrainedModel.getConfig();

        DeepSeekV3Dataset trainDataset = createDatasetFromTexts(
                trainTexts, config.getNPositions(), 4, config.getVocabSize(), true
        );
        DeepSeekV3Dataset valDataset = createDatasetFromTexts(
                valTexts, config.getNPositions(), 1, config.getVocabSize(), true
        );

        System.out.println("  ✓ 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + valDataset.getSampleCount());
        System.out.println("  ✓ 数据格式: ChatML 对话模板 (<|im_start|>user/assistant<|im_end|>)");
        System.out.println("  ✓ Loss策略: Answer-Only (仅对 assistant 回答部分计算 loss)");

        DeepSeekV3SFTrainer posttrain = new DeepSeekV3SFTrainer(pretrainedModel, trainDataset, valDataset);
        posttrain.configure(10, 3e-4f, 3);

        System.out.println("\n📝 开始后训练...");
        System.out.println("-".repeat(80));
        posttrain.train();
        System.out.println("-".repeat(80));

        System.out.println("\n✅ 后训练完成!");
        System.out.println("\n💡 后训练阶段总结:");
        System.out.println("  - 目标: 适应任务特定的推理和生成");
        System.out.println("  - 数据: ChatML 模板 + 任务标签（行业主流 SFT 格式）");
        System.out.println("  - Loss: Answer-Only Loss Masking（仅对回答部分计算损失）");

        return pretrainedModel;
    }

    // ========== 步骤2B: 代码专项后训练 ==========

    /**
     * 执行代码生成专项后训练（纯 CODING 任务，强化 MoE 专家代码特化）
     */
    private static DeepSeekV3Model runCodePosttraining(DeepSeekV3Model finetunedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("💻 步骤2B: DeepSeek-V3 代码生成专项后训练");
        System.out.println("=".repeat(80));
        System.out.println("💡 目标：强化MoE专家对代码任务的特化能力");

        List<String> codeTrainTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/code_posttrain_train.txt");
        List<String> codeValTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/code_posttrain_val.txt");
        System.out.println("\n📝 加载代码专项数据... 训练集: " + codeTrainTexts.size() + " 条, 验证集: " + codeValTexts.size() + " 条");

        DeepSeekV3Config config = finetunedModel.getConfig();

        DeepSeekV3Dataset codeTrainDataset = createDatasetFromTexts(
                codeTrainTexts, config.getNPositions(), 4, config.getVocabSize(), true
        );
        DeepSeekV3Dataset codeValDataset = createDatasetFromTexts(
                codeValTexts, config.getNPositions(), 1, config.getVocabSize(), true
        );

        System.out.println("  ✓ 训练样本: " + codeTrainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + codeValDataset.getSampleCount());
        System.out.println("  ✓ 任务类型: 纯CODING (Python/Java/JS/C++)");
        System.out.println("  ✓ 数据格式: ChatML 对话模板 + Answer-Only Loss");

        DeepSeekV3SFTrainer codePosttrain = new DeepSeekV3SFTrainer(finetunedModel, codeTrainDataset, codeValDataset);
        codePosttrain.configure(30, 2e-4f, 3);

        System.out.println("\n📝 开始代码生成专项后训练...");
        System.out.println("-".repeat(80));
        codePosttrain.train();
        System.out.println("-".repeat(80));

        System.out.println("\n✅ 代码专项后训练完成!");
        System.out.println("\n💡 代码专项后训练总结:");
        System.out.println("  - 目标: 强化MoE专家对代码任务的特化");
        System.out.println("  - 特色: 持续激活同一批专家 -> 专家特化能力增强");
        System.out.println("  - 预期效果: Expert 2,5成为代码专家，CODING任务激活概率大幅提升");

        return finetunedModel;
    }

    // ========== 步骤3: RLHF 训练 ==========

    /**
     * 执行 RLHF 训练（人类反馈强化学习，奖励加权回归）
     */
    private static DeepSeekV3Model runRLHFTraining(DeepSeekV3Model model) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤3: DeepSeek-V3 RLHF训练 - 人类反馈强化学习");
        System.out.println("=".repeat(80));
        System.out.println("💡 算法：Reward-weighted Regression（奖励加权回归）");

        List<String> rlhfTexts = DeepSeekV3DatasetGenerator.readFromFile(DATA_DIR + "/rlhf_train.txt");
        System.out.println("\n📝 加载RLHF训练数据... 样本: " + rlhfTexts.size() + " 条");

        DeepSeekV3Config config = model.getConfig();
        DeepSeekV3Dataset rlhfDataset = createRLHFDatasetFromTexts(
                rlhfTexts, config.getNPositions(), 4, config.getVocabSize()
        );

        System.out.println("  ✓ RLHF训练样本: " + rlhfDataset.getSampleCount());
        System.out.println("  ✓ 数据格式: ChatML 模板 + 奖励标注（高/中/低三档）");

        DeepSeekV3RLHFTrainer rlhfTrainer = new DeepSeekV3RLHFTrainer(model, rlhfDataset);
        rlhfTrainer.configure(20, 5e-4f, 1.0f, 0.5f);

        System.out.println("\n📝 开始RLHF强化学习训练...");
        System.out.println("-".repeat(80));
        rlhfTrainer.train();
        System.out.println("-".repeat(80));

        System.out.println("\n✅ RLHF训练完成!");
        System.out.println("\n💡 RLHF阶段总结:");
        System.out.println("  - 目标: 通过人类反馈对齐模型行为");
        System.out.println("  - 效果: 高奖励样本被强化，低奖励样本被弱化");
        System.out.println("  - 结果: 模型更符合人类偏好，回答更礼貌、更安全");

        return model;
    }

    // ========== 步骤4: 推理测试 ==========

    /**
     * 执行推理测试（展示多种生成策略）
     */
    private static void runInference(DeepSeekV3Model model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤4: DeepSeek-V3 推理与文本生成");
        System.out.println("=".repeat(80));

        DeepSeekV3Inference inference = new DeepSeekV3Inference(model);
        inference.setSeed(42);
        System.out.println("  ✓ 推理器准备完成");

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

                System.out.println("  策略1 [Greedy贪婪]: ");
                var greedyResult = inference.generateGreedy(promptIds, model.getConfig().getNPositions());
                System.out.println("    → " + sharedTokenizer.decode(greedyResult.tokens));

                System.out.println("  策略2 [Temperature=0.8]: ");
                var tempResult = inference.generateWithTemperature(promptIds, model.getConfig().getNPositions(), 0.8f);
                System.out.println("    → " + sharedTokenizer.decode(tempResult.tokens));

                System.out.println("  策略3 [Top-K=50]: ");
                var topKResult = inference.generateTopK(promptIds, model.getConfig().getNPositions(), 50);
                System.out.println("    → " + sharedTokenizer.decode(topKResult.tokens));

            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }
            System.out.println();
        }

        System.out.println("✅ 推理测试完成!");
        System.out.println("\n💡 推理策略对比: Greedy（确定性）/ Temperature（随机性）/ Top-K（截断采样）");
    }

    // ========== 数据集组装工具方法 ==========

    /**
     * 从文本列表创建数据集
     * <p>
     * useTaskLabels=true 时解析 ChatML 格式并生成 answer-only loss mask；
     * useTaskLabels=false 时（预训练模式）所有 token 位置都参与 loss 计算。
     */
    private static DeepSeekV3Dataset createDatasetFromTexts(
            List<String> texts, int maxSeqLength, int batchSize, int vocabSize, boolean useTaskLabels) {

        List<int[]> sequences = new ArrayList<>();
        List<float[]> lossMasks = new ArrayList<>();

        for (String text : texts) {
            String cleanText = useTaskLabels ? DeepSeekV3TokenizerUtil.removeTaskLabel(text) : text;

            if (useTaskLabels && cleanText.contains("<|im_start|>assistant")) {
                int assistantStart = cleanText.indexOf("<|im_start|>assistant");
                if (assistantStart < 0) {
                    // 防御性处理：找不到 assistant 标记时按普通文本处理
                    sequences.add(padSequence(sharedTokenizer.encode(cleanText), maxSeqLength));
                    continue;
                }
                String userPart = cleanText.substring(0, assistantStart);
                String assistantPart = cleanText.substring(assistantStart);

                List<Integer> userTokens = sharedTokenizer.encode(userPart);
                List<Integer> assistantTokens = sharedTokenizer.encode(assistantPart);

                List<Integer> allTokens = new ArrayList<>(userTokens);
                allTokens.addAll(assistantTokens);

                // loss mask：user 部分为 0.0，assistant 部分为 1.0
                float[] mask = new float[maxSeqLength];
                int totalLen = Math.min(allTokens.size(), maxSeqLength);
                for (int j = userTokens.size(); j < totalLen; j++) {
                    mask[j] = 1.0f;
                }

                sequences.add(padSequence(allTokens, maxSeqLength));
                lossMasks.add(mask);
            } else {
                sequences.add(padSequence(sharedTokenizer.encode(cleanText), maxSeqLength));
            }
        }

        if (!lossMasks.isEmpty()) {
            return new DeepSeekV3Dataset(sequences, new ArrayList<>(), lossMasks,
                    maxSeqLength, batchSize, true, true);
        }
        return new DeepSeekV3Dataset(sequences, maxSeqLength, batchSize, true);
    }

    /**
     * 从 RLHF 文本列表创建数据集（解析奖励标注 + 构建 loss mask）
     */
    private static DeepSeekV3Dataset createRLHFDatasetFromTexts(
            List<String> texts, int maxSeqLength, int batchSize, int vocabSize) {

        List<int[]> sequences = new ArrayList<>();
        List<float[]> lossMasks = new ArrayList<>();
        List<Float> rewardList = new ArrayList<>();

        for (String text : texts) {
            float reward = 0.5f;
            String content = text;

            // 提取奖励标注 [REWARD:0.x]
            int rewardStart = text.indexOf("[REWARD:");
            if (rewardStart >= 0) {
                int rewardEnd = text.indexOf("]", rewardStart);
                if (rewardEnd > rewardStart) {
                    try {
                        reward = Float.parseFloat(text.substring(rewardStart + 8, rewardEnd));
                    } catch (NumberFormatException e) {
                        reward = 0.5f;
                    }
                    content = text.substring(rewardEnd + 1).trim();
                }
            }

            if (content.contains("<|im_start|>assistant")) {
                int assistantStart = content.indexOf("<|im_start|>assistant");
                String userPart = content.substring(0, assistantStart);
                String assistantPart = content.substring(assistantStart);

                List<Integer> userTokens = sharedTokenizer.encode(userPart);
                List<Integer> assistantTokens = sharedTokenizer.encode(assistantPart);

                List<Integer> allTokens = new ArrayList<>(userTokens);
                allTokens.addAll(assistantTokens);

                float[] mask = new float[maxSeqLength];
                int totalLen = Math.min(allTokens.size(), maxSeqLength);
                for (int j = userTokens.size(); j < totalLen; j++) {
                    mask[j] = 1.0f;
                }

                sequences.add(padSequence(allTokens, maxSeqLength));
                lossMasks.add(mask);
                rewardList.add(reward);
            }
        }

        List<TaskType> taskTypeList = new ArrayList<>();
        for (int i = 0; i < sequences.size(); i++) {
            taskTypeList.add(TaskType.GENERAL);
        }

        return new DeepSeekV3Dataset(sequences, taskTypeList, lossMasks, rewardList, maxSeqLength, batchSize, true);
    }

    /**
     * 将 token ID 列表截断或填充到指定长度（PAD 填充）
     */
    private static int[] padSequence(List<Integer> tokens, int maxSeqLength) {
        int[] paddedSeq = new int[maxSeqLength];
        Arrays.fill(paddedSeq, DeepSeekV3TokenizerUtil.PAD_TOKEN_ID);
        int copyLen = Math.min(tokens.size(), maxSeqLength);
        for (int i = 0; i < copyLen; i++) {
            paddedSeq[i] = tokens.get(i);
        }
        return paddedSeq;
    }

    // ========== 内部辅助类 ==========

    /**
     * 推理测试用例
     */
    private static class TestCase {
        final String prompt;

        TestCase(String prompt) {
            this.prompt = prompt;
        }
    }
}
