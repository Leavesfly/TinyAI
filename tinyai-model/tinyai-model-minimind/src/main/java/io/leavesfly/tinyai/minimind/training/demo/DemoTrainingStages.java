package io.leavesfly.tinyai.minimind.training.demo;

import io.leavesfly.tinyai.minimind.cli.ModelLoader;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.minimind.training.PretrainTrainer;
import io.leavesfly.tinyai.minimind.training.SFTTrainer;
import io.leavesfly.tinyai.minimind.training.dataset.DPODataset;
import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.grpo.GRPOConfig;
import io.leavesfly.tinyai.minimind.training.rlaif.grpo.GRPOTrainer;
import io.leavesfly.tinyai.minimind.training.rlaif.ppo.ValueNetwork;
import io.leavesfly.tinyai.minimind.training.dataset.PretrainDataset;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.minimind.training.dpo.DPOConfig;
import io.leavesfly.tinyai.minimind.training.dpo.DPOTrainer;
import io.leavesfly.tinyai.minimind.training.lora.LoRAConfig;
import io.leavesfly.tinyai.minimind.training.lora.LoRATrainer;
import org.json.JSONObject;

import java.io.*;
import java.util.*;

import static io.leavesfly.tinyai.minimind.training.demo.DemoConfig.*;

/**
 * MiniMind 训练演示 - 训练阶段执行器
 * <p>
 * 包含各训练阶段的执行逻辑：
 * - 步骤1: 无监督预训练
 * - 步骤2: 监督微调 (SFT)
 * - 步骤3: DPO训练（人类偏好对齐）
 * - 步骤4: 强化学习训练
 * - 步骤5: LoRA微调（特定任务适配）
 * - 步骤6: 推理测试
 *
 * @author TinyAI Team
 */
public class DemoTrainingStages {

    // ========== 步骤1: 无监督预训练 ==========

    /**
     * 执行无监督预训练 - 使用标准 PretrainTrainer
     */
    public static MiniMindModel runUnsupervisedPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: MiniMind 无监督预训练 (Unsupervised Pretraining)");
        System.out.println("=".repeat(80));

        // 1. 创建分词器
        System.out.println("\n📝 创建分词器...");
        int maxSeqLen = 32;
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createSimpleTokenizer(maxSeqLen);
        setSharedTokenizer(tokenizer);
        System.out.println("  ✓ 分词器类型: 动态词汇表 (Simple-GPT1风格)");
        System.out.println("  ✓ 最大序列长度: " + maxSeqLen);

        // 2. 加载数据（动态构建词汇表）
        System.out.println("\n📝 准备预训练数据集...");
        String pretrainPath = DATA_DIR + "/pretrain.jsonl";
        List<String> pretrainJsonLines = readJsonlFile(pretrainPath);
        List<String> pretrainTexts = new ArrayList<>();
        for (String jsonLine : pretrainJsonLines) {
            pretrainTexts.add(new JSONObject(jsonLine).getString("text"));
        }

        int batchSize = 4;
        PretrainDataset dataset = new PretrainDataset(tokenizer, maxSeqLen, batchSize);
        dataset.loadFromTexts(pretrainTexts);
        dataset.prepare(true);

        // 冻结词汇表（类似GPT1 SimpleTokenizer）
        tokenizer.freeze();
        System.out.println("  ✓ 词汇表大小: " + tokenizer.getVocabSize() + " (已冻结)");
        System.out.println("  ✓ 预训练样本数: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次数量: " + dataset.getBatchCount());

        // 3. 创建MiniMind模型（使用实际词汇表大小）
        System.out.println("\n📝 创建MiniMind模型...");
        MiniMindConfig config = createMicroConfig(tokenizer.getVocabSize());
        MiniMindModel model = new MiniMindModel("tiny-minimind", config);

        System.out.println("  ✓ 模型配置: Micro (教学专用)");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getHiddenSize());
        System.out.println("  ✓ 层数: " + config.getNumLayers());
        System.out.println("  ✓ 注意力头数: " + config.getNumHeads());

        // 打印网络架构
        System.out.println("\n📐 模型网络架构:");
        model.getMiniMindBlock().printArchitecture();

        // 4. 训练
        System.out.println("\n📝 开始无监督预训练...");
        System.out.println("  - 训练目标: 因果语言建模 (下一个词预测)");
        System.out.println("  - 学习率: 1e-2");
        System.out.println("  - 训练轮次: 30 epochs");
        System.out.println("-".repeat(80));

        PretrainTrainer trainer = new PretrainTrainer(model, dataset);
        trainer.configure(30, 1e-2f, 0, 1.0f);
        trainer.setCheckpoint(CHECKPOINT_DIR + "/pretrain", 100);
        trainer.setLogInterval(10);
        trainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ 无监督预训练完成!");
        printPretrainSummary();

        return model;
    }

    // ========== 步骤2: 监督微调 ==========

    /**
     * 执行监督微调（SFT）
     */
    public static MiniMindModel runSupervisedFinetuning(MiniMindModel pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: MiniMind 监督微调 (Supervised Fine-tuning)");
        System.out.println("=".repeat(80));

        // 1. 加载数据（标准 Alpaca JSONL 格式）
        System.out.println("\n📝 加载监督微调数据...");
        String trainPath = DATA_DIR + "/sft_train.jsonl";

        // 2. 准备数据集（直接使用 SFTDataset 的 JSONL 加载能力）
        System.out.println("\n📝 准备监督微调数据集...");
        MiniMindConfig config = pretrainedModel.getConfig();
        int batchSize = 2;

        SFTDataset dataset = new SFTDataset(getSharedTokenizer(), config.getMaxSeqLen(), batchSize);
        dataset.loadFromJsonl(trainPath);
        dataset.prepare(true);
        System.out.println("  ✓ 训练样本数: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次数量: " + dataset.getBatchCount());

        // 3. 训练
        System.out.println("\n📝 开始监督微调训练...");
        System.out.println("  - 训练目标: 指令跟随和对话生成");
        System.out.println("  - 学习率: 1e-3");
        System.out.println("  - 训练轮次: 10 epochs");
        System.out.println("-".repeat(80));

        SFTTrainer trainer = new SFTTrainer(pretrainedModel, dataset);
        trainer.configure(10, 1e-3f, 1.0f);
        trainer.setCheckpointDir(CHECKPOINT_DIR + "/sft");
        trainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ 监督微调完成!");
        printSFTSummary();

        return pretrainedModel;
    }


    // ========== 步骤3: DPO训练 ==========

    /**
     * 执行DPO训练 - 直接偏好优化（人类偏好对齐）
     */
    public static MiniMindModel runDPOTraining(MiniMindModel loraModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤3: MiniMind DPO训练 (Direct Preference Optimization)");
        System.out.println("=".repeat(80));
        System.out.println("💡 DPO核心思想: 无需奖励模型，直接从偏好对优化策略");

        // 1. 配置DPO
        System.out.println("\n📝 配置DPO参数...");
        DPOConfig dpoConfig = DPOConfig.createDefault();
        dpoConfig.setBeta(0.2f);
        dpoConfig.setLabelSmoothing(0.0f);
        dpoConfig.setUseLengthNormalization(false);
        dpoConfig.setResponseOnlyLoss(true);

        System.out.println("  ✓ Beta (β): " + dpoConfig.getBeta());
        System.out.println("  ✓ Response损失: " + dpoConfig.isResponseOnlyLoss());

        // 2. 加载数据（标准 DPO JSONL 格式）
        System.out.println("\n📝 准备DPO偏好数据集...");
        String dpoPath = DATA_DIR + "/dpo_train.jsonl";
        List<String> dpoJsonLines = readJsonlFile(dpoPath);

        MiniMindConfig config = loraModel.getConfig();
        int batchSize = 1;
        DPODataset dpoDataset = new DPODataset(getSharedTokenizer(), config.getMaxSeqLen(), batchSize);

        for (String jsonLine : dpoJsonLines) {
            JSONObject json = new JSONObject(jsonLine);
            dpoDataset.addSample(json.getString("prompt"), json.getString("chosen"), json.getString("rejected"));
        }
        dpoDataset.prepare(true);
        System.out.println("  ✓ 偏好对数量: " + dpoDataset.getSampleCount());

        // 3. 训练
        System.out.println("\n📝 开始DPO训练...");
        System.out.println("  - 学习率: 5e-4");
        System.out.println("  - 训练轮次: 20 epochs");
        System.out.println("-".repeat(80));

        DPOTrainer dpoTrainer = new DPOTrainer(loraModel, dpoDataset, dpoConfig);
        dpoTrainer.configure(20, 5e-4f, 1.0f);
        dpoTrainer.setCheckpoint(CHECKPOINT_DIR + "/dpo", 50);
        dpoTrainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ DPO训练完成!");
        printDPOSummary();

        return loraModel;
    }

    // ========== 步骤4: 强化学习训练 ==========

    /**
     * 执行强化学习训练（GRPO）
     */
    public static MiniMindModel runReinforcementLearningTraining(MiniMindModel model) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤4: MiniMind 强化学习训练 (GRPO - Group Relative Policy Optimization)");
        System.out.println("=".repeat(80));
        System.out.println("💡 使用GRPO算法进行组相对策略优化");

        // 1. 加载数据（标准 RL JSONL 格式）
        System.out.println("\n📝 加载强化学习训练数据...");
        String rlPath = DATA_DIR + "/rl_train.jsonl";
        List<String> rlJsonLines = readJsonlFile(rlPath);
        System.out.println("  ✓ RL训练数据: " + rlJsonLines.size() + " 条");

        // 2. 准备RLAIF数据集
        System.out.println("\n📝 准备RLAIF数据集...");
        MiniMindConfig config = model.getConfig();
        int batchSize = 2;
        int numCandidates = 4;  // 每个prompt生成4个候选回答

        RLAIFDataset dataset = new RLAIFDataset(getSharedTokenizer(), config.getMaxSeqLen(), batchSize);

        for (String jsonLine : rlJsonLines) {
            JSONObject json = new JSONObject(jsonLine);
            String prompt = json.getString("prompt");
            String response = json.getString("response");
            float reward = (float) json.getDouble("reward");

            // 基于原始response生成多个候选，模拟不同质量的回答
            List<String> candidates = new ArrayList<>();
            float[] rewards = new float[numCandidates];
            for (int i = 0; i < numCandidates; i++) {
                candidates.add(response);
                rewards[i] = reward * (0.8f + i * 0.1f);
            }

            dataset.addSample(prompt, candidates, rewards);
        }

        dataset.prepare(true);
        System.out.println("  ✓ RLAIF样本数: " + dataset.getSampleCount());
        System.out.println("  ✓ 每组候选数: " + numCandidates);
        System.out.println("  ✓ 批次数量: " + dataset.getBatchCount());

        // 3. 创建GRPO配置
        System.out.println("\n📝 配置GRPO参数...");
        GRPOConfig grpoConfig = new GRPOConfig();
        grpoConfig.setNumCandidates(numCandidates);
        grpoConfig.setGroupSize(2);
        grpoConfig.setActorLearningRate(1e-4f);
        grpoConfig.setClipEpsilon(0.2f);
        grpoConfig.setGrpoEpochs(3);
        grpoConfig.setNormalizeAdvantage(true);
        grpoConfig.setUseGroupContrast(true);

        System.out.println("  ✓ 候选数量: " + grpoConfig.getNumCandidates());
        System.out.println("  ✓ 组大小: " + grpoConfig.getGroupSize());
        System.out.println("  ✓ Actor学习率: " + grpoConfig.getActorLearningRate());
        System.out.println("  ✓ Clip范围: " + grpoConfig.getClipEpsilon());
        System.out.println("  ✓ GRPO轮数: " + grpoConfig.getGrpoEpochs());

        // 4. 创建Critic网络(可选,这里简化为null)
        System.out.println("\n📝 创建训练器...");
        ValueNetwork critic = null;  // 简化版本不使用critic

        // 5. 创建GRPO训练器
        GRPOTrainer trainer = new GRPOTrainer(model, critic, dataset, grpoConfig);
        trainer.configure(3, 10);  // 3 epochs, 每10步打印一次
        trainer.setCheckpointDir(CHECKPOINT_DIR + "/grpo");

        // 6. 训练
        System.out.println("\n📝 开始GRPO训练...");
        System.out.println("  - 算法: Group Relative Policy Optimization");
        System.out.println("  - 训练轮次: 3 epochs");
        System.out.println("-".repeat(80));

        trainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ GRPO强化学习训练完成!");
        printRLSummary();

        return model;
    }

    // ========== LoRA微调前推理对比 ==========

    /**
     * 在 LoRA 微调前执行推理测试，用于与微调后效果对比
     * 直接使用内存中的模型进行推理，无需从文件加载
     *
     * @param model     当前模型（LoRA微调前）
     * @param stageLabel 阶段标签，用于输出区分
     */
    public static void runInferenceForComparison(MiniMindModel model, String stageLabel) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🔍 " + stageLabel);
        System.out.println("=".repeat(80));

        MiniMindTokenizer tokenizer = getSharedTokenizer();
        model.setTraining(false);

        List<String> testPrompts = Arrays.asList(
                "Machine learning is",
                "Neural networks are",
                "Deep learning",
                "AI technology"
        );

        System.out.println("\n📝 推理输出 (" + stageLabel + "):");
        System.out.println("-".repeat(80));

        for (String prompt : testPrompts) {
            System.out.println("\n📌 Prompt: \"" + prompt + "\"");
            try {
                List<Integer> promptTokens = tokenizer.encode(prompt);
                int[] promptIds = promptTokens.stream().mapToInt(Integer::intValue).toArray();
                int promptLen = promptIds.length;

                // 贪婪解码
                int[] greedyResult = model.generate(promptIds, 30, 0.0f, 0, 0.0f, 1.5f);
                String greedyGenerated = extractGenerated(tokenizer, greedyResult, promptLen);
                if (greedyGenerated.equals(" [无新生成]") || greedyGenerated.equals(" [空]")) {
                    int[] fallbackResult = model.generate(promptIds, 30, 0.5f, 0, 0.0f, 1.5f);
                    greedyGenerated = extractGenerated(tokenizer, fallbackResult, promptLen);
                    System.out.println("  [Greedy→T=0.5] → " + prompt + greedyGenerated);
                } else {
                    System.out.println("  [Greedy]      → " + prompt + greedyGenerated);
                }

                // Temperature 采样
                int[] tempResult = model.generate(promptIds, 30, 0.8f, 0, 0.0f);
                String tempGenerated = extractGenerated(tokenizer, tempResult, promptLen);
                System.out.println("  [Temp=0.8]    → " + prompt + tempGenerated);

            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }
        }

        System.out.println("\n" + "-".repeat(80));

        // 恢复为训练模式，以便后续训练阶段使用
        model.setTraining(true);
    }

    // ========== 步骤5: LoRA微调 ==========

    /**
     * 执行LoRA微调 - 参数高效微调（特定任务适配）
     */
    public static MiniMindModel runLoRAFinetuning(MiniMindModel sftModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🔧 步骤5: MiniMind LoRA微调 (Low-Rank Adaptation)");
        System.out.println("=".repeat(80));
        System.out.println("💡 LoRA核心思想: 冻结原始参数，只训练低秩分解矩阵");

        // 1. 配置LoRA
        System.out.println("\n📝 配置LoRA参数...");
        LoRAConfig loraConfig = new LoRAConfig();
        loraConfig.setRank(8);
        loraConfig.setAlpha(16.0f);
        loraConfig.setDropout(0.1f);
        loraConfig.setTargetModules(new String[]{"queryProj", "valueProj"});
        loraConfig.setFreezeOriginal(true);

        System.out.println("  ✓ LoRA秩 (r): " + loraConfig.getRank());
        System.out.println("  ✓ 缩放因子 (α): " + loraConfig.getAlpha());
        System.out.println("  ✓ 缩放系数 (α/r): " + loraConfig.getScaling());
        System.out.println("  ✓ 目标模块: " + String.join(", ", loraConfig.getTargetModules()));

        // 2. 注入 LoRA 层
        System.out.println("\n📝 注入 LoRA 层到模型...");
        int injectedCount = sftModel.applyLoRA(loraConfig);
        if (injectedCount > 0) {
            sftModel.printLoRAStats();
        }

        // 3. 准备数据（标准 Alpaca JSONL 格式）
        System.out.println("\n📝 准备LoRA微调数据...");
        String trainPath = DATA_DIR + "/sft_train.jsonl";

        MiniMindConfig config = sftModel.getConfig();
        int batchSize = 2;
        SFTDataset dataset = new SFTDataset(getSharedTokenizer(), config.getMaxSeqLen(), batchSize);
        dataset.loadFromJsonl(trainPath);
        dataset.prepare(true);
        System.out.println("  ✓ 训练样本数: " + dataset.getSampleCount());

        // 3. 训练
        System.out.println("\n📝 开始LoRA微调...");
        System.out.println("  - 学习率: 5e-4");
        System.out.println("  - 训练轮次: 20 epochs");
        System.out.println("-".repeat(80));

        LoRATrainer loraTrainer = new LoRATrainer(sftModel, dataset, loraConfig);
        loraTrainer.configure(20, 5e-4f, 1.0f);
        loraTrainer.setCheckpointDir(CHECKPOINT_DIR + "/lora");
        loraTrainer.printTrainableParams();
        loraTrainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ LoRA微调完成!");
        printLoRASummary();

        return sftModel;
    }

    // ========== 步骤6: 推理测试 ==========

    /**
     * 从模型文件加载模型后执行推理测试
     *
     * @param modelName 模型名称，用于从 CHECKPOINT_DIR/model/{modelName}/ 加载
     */
    public static void runInference(String modelName) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤6: MiniMind 推理测试（从模型文件加载）");
        System.out.println("=".repeat(80));

        // 从模型文件加载模型和分词器
        MiniMindModel model;
        MiniMindTokenizer tokenizer;
        try {
            System.out.println("\n📂 从模型文件加载模型: " + modelName);
            ModelLoader.ModelLoadResult loadResult = ModelLoader.loadModel("", modelName);
            model = loadResult.getModel();
            tokenizer = loadResult.getTokenizer();
        } catch (Exception e) {
            System.err.println("❌ 模型加载失败: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        model.setTraining(false);

        // 使用训练数据中出现过的短语作为 prompt，这样模型更容易续写
        List<String> testPrompts = Arrays.asList("Machine learning is", "Neural networks are", "Deep learning", "AI technology");

        System.out.println("\n📝 测试不同生成策略...");
        System.out.println("-".repeat(80));

        for (String prompt : testPrompts) {
            System.out.println("\n📌 Prompt: \"" + prompt + "\"");

            try {
                List<Integer> promptTokens = tokenizer.encode(prompt);
                int[] promptIds = promptTokens.stream().mapToInt(Integer::intValue).toArray();
                int promptLen = promptIds.length;

                // 1. 贪婪解码
                int[] greedyResult = model.generate(promptIds, 30, 0.0f, 0, 0.0f, 1.5f);
                String greedyGenerated = extractGenerated(tokenizer, greedyResult, promptLen);
                if (greedyGenerated.equals(" [无新生成]") || greedyGenerated.equals(" [空]")) {
                    // 贪婪解码生成EOS，尝试低温度采样
                    int[] fallbackResult = model.generate(promptIds, 30, 0.5f, 0, 0.0f, 1.5f);
                    greedyGenerated = extractGenerated(tokenizer, fallbackResult, promptLen);
                    System.out.println("  [Greedy→T=0.5] → " + prompt + greedyGenerated);
                } else {
                    System.out.println("  [Greedy]      → " + prompt + greedyGenerated);
                }

                // 2. Temperature 采样 (增加多样性)
                int[] tempResult = model.generate(promptIds, 30, 0.8f, 0, 0.0f);
                String tempGenerated = extractGenerated(tokenizer, tempResult, promptLen);
                System.out.println("  [Temp=0.8]    → " + prompt + tempGenerated);

                // 3. Top-K 采样
                int[] topkResult = model.generate(promptIds, 30, 1.0f, 10, 0.0f);
                String topkGenerated = extractGenerated(tokenizer, topkResult, promptLen);
                System.out.println("  [Top-K=10]    → " + prompt + topkGenerated);

                // 4. Top-P 采样 (Nucleus)
                int[] toppResult = model.generate(promptIds, 30, 1.0f, 0, 0.9f);
                String toppGenerated = extractGenerated(tokenizer, toppResult, promptLen);
                System.out.println("  [Top-P=0.9]   → " + prompt + toppGenerated);

            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }
        }

        System.out.println("\n" + "-".repeat(80));
        System.out.println("\n✅ 推理测试完成!");
        printInferenceSummary();
        printInferenceNotes();
    }

    /**
     * 从生成结果中提取新生成的部分
     */
    private static String extractGenerated(MiniMindTokenizer tokenizer, int[] result, int promptLen) {
        if (result.length <= promptLen) {
            return " [无新生成]";
        }
        // 提取 prompt 之后的 token
        int[] generatedIds = Arrays.copyOfRange(result, promptLen, result.length);
        List<Integer> generatedList = new ArrayList<>();
        for (int id : generatedIds) {
            generatedList.add(id);
        }
        String generated = tokenizer.decode(generatedList);
        if (generated.isEmpty()) {
            return " [空]";
        }
        // 确保生成内容与 prompt 之间有空格分隔
        if (!generated.startsWith(" ") && !generated.startsWith("\n")) {
            generated = " " + generated;
        }
        return generated;
    }

    /**
     * 打印推理注意事项
     */
    private static void printInferenceNotes() {
        System.out.println("\n💡 推理效果说明:");
        System.out.println("  ⚠ 当前为超小规模教学演示 (150条预训练+60条SFT数据)");
        System.out.println("  ⚠ 模型仅有0.4M参数，词汇表约600个token");
        System.out.println("  ⚠ 生成质量受限于数据量，主要用于理解训练流程");
        System.out.println("  ✓ 如需更好效果，请增加训练数据和训练轮次");
    }

    // ========== 阶段总结输出 ==========

    private static void printPretrainSummary() {
        System.out.println("\n💡 预训练阶段总结:");
        System.out.println("  - 目标: 学习语言的通用表示和语法");
        System.out.println("  - 任务: 因果语言建模（预测下一个词）");
        System.out.println("  - 数据: 大规模无标注文本");
        System.out.println("  - 技巧: 较高学习率 + 多轮训练");
    }

    private static void printSFTSummary() {
        System.out.println("\n💡 SFT阶段总结:");
        System.out.println("  - 目标: 学习遵循指令和生成高质量回答");
        System.out.println("  - 任务: 指令微调（问答对）");
        System.out.println("  - 数据: 带标签的指令-回答数据");
        System.out.println("  - 技巧: 小学习率 + 早停防止过拟合");
    }

    private static void printLoRASummary() {
        System.out.println("\n💡 LoRA阶段总结:");
        System.out.println("  - 目标: 在冻结原始参数的情况下进行低成本微调");
        System.out.println("  - 方法: 低秩分解 W = W0 + BA (只训练B和A矩阵)");
        System.out.println("  - 优势: 可训练参数量显著减少 (通常<1%)");
        System.out.println("  - 应用: 资源受限场景的模型定制");
    }

    private static void printDPOSummary() {
        System.out.println("\n💡 DPO阶段总结:");
        System.out.println("  - 目标: 使模型输出符合人类偏好");
        System.out.println("  - 方法: 直接从(prompt, chosen, rejected)三元组学习");
        System.out.println("  - 优势: 无需单独训练奖励模型，简化流程");
        System.out.println("  - 损失: L = -logσ(β(r_chosen - r_rejected))");
    }

    private static void printRLSummary() {
        System.out.println("\n💡 RL阶段总结:");
        System.out.println("  - 目标: 通过奖励信号对齐模型行为");
        System.out.println("  - 方法: 奖励加权的交叉熵损失");
        System.out.println("  - 效果: 高奖励样本获得更大梯度贡献");
        System.out.println("  - 技巧: 小学习率 + 奖励引导");
    }

    private static void printInferenceSummary() {
        System.out.println("\n💡 推理阶段总结:");
        System.out.println("  - 输入: 提示词文本");
        System.out.println("  - 处理: 自回归生成");
        System.out.println("  - 输出: 生成的完整文本");
        System.out.println("  - 策略: Greedy/Temperature/Top-K/Top-P");
    }
}
