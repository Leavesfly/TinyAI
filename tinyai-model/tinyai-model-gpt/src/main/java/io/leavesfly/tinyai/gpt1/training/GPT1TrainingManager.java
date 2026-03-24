package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * GPT-1 训练管理器
 * 负责预训练、微调和推理流程
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT1TrainingManager {

    private static final String DATA_DIR = "./data/gpt1_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/gpt1";
    private final GPT1Dataset.SimpleTokenizer tokenizer;

    public GPT1TrainingManager() {
        this.tokenizer = new GPT1Dataset.SimpleTokenizer();
    }

    public GPT1Dataset.SimpleTokenizer getTokenizer() {
        return tokenizer;
    }

    /**
     * 执行预训练
     */
    public GPT1Model runPretraining() throws IOException {
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
        List<String> allTexts = new ArrayList<>();
        allTexts.addAll(pretrainTexts);
        allTexts.addAll(finetuneTrainTexts);
        allTexts.addAll(finetuneValTexts);

        for (String text : allTexts) {
            tokenizer.encode(text);
        }
        int vocabSize = tokenizer.getVocabSize();
        tokenizer.freeze();

        System.out.println("  ✓ 完整词汇表大小: " + vocabSize);
        System.out.println("  ✓ 词汇表已冻结,后续不再增加新词");

        // 3. 创建模型
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
        dataset.loadFromTexts(pretrainTexts, tokenizer);

        System.out.println("  ✓ 训练样本: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次大小: 8");
        System.out.println("  ✓ 序列长度: " + config.getNPositions());

        // 5. 配置训练器
        System.out.println("\n📝 配置预训练器...");
        GPT1Pretrain trainer = new GPT1Pretrain(model, dataset);
        trainer.configure(
                10,
                1e-2f,
                5,
                1.0f
        ).setCheckpoint(CHECKPOINT_DIR + "/pretrain", 10);

        System.out.println("  ✓ 最大轮次: 30");
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
    public GPT1Model runFinetuning(GPT1Model pretrainedModel) throws IOException {
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

        // 2. 准备数据集（使用微调专用的数据加载方式）
        System.out.println("\n📝 准备微调数据集...");
        GPT1Config config = pretrainedModel.getConfig();
        String responseSeparator = "Response:";

        GPT1Dataset trainDataset = new GPT1Dataset(
                config.getNPositions(),
                4,
                config.getVocabSize()
        );
        trainDataset.loadFromInstructionTexts(trainTexts, tokenizer, responseSeparator);

        GPT1Dataset valDataset = new GPT1Dataset(
                config.getNPositions(),
                4,
                config.getVocabSize()
        );
        valDataset.loadFromInstructionTexts(valTexts, tokenizer, responseSeparator);

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
                5,
                1e-3f,
                3
        ).setCheckpoint(CHECKPOINT_DIR + "/finetune", 3);

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
        System.out.println("  - 数据: 任务特定的指令数据(每条独立处理,不跨样本拼接)");
        System.out.println("  - Loss: 只在Response部分计算loss,Instruction部分被mask屏蔽");
        System.out.println("  - 技巧: 小学习率 + 早停机制 + Loss Mask");
        System.out.println("  - 结果: 模型学会了回答问题的能力");

        return pretrainedModel;
    }

    /**
     * 执行推理测试
     */
    public void runInference(GPT1Model model) {
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
                List<Integer> tokens = tokenizer.encode(prompt);
                int[] promptIds = tokens.stream().mapToInt(Integer::intValue).toArray();

                // Greedy解码
                System.out.println("  策略1 [Greedy]: ");
                int[] greedyResult = inference.generateGreedy(promptIds, 15);
                String greedyText = tokenizer.decode(greedyResult);
                System.out.println("    → " + greedyText);

                // Temperature采样
                System.out.println("  策略2 [Temperature=0.8]: ");
                int[] tempResult = inference.generateWithTemperature(promptIds, 15, 0.8f);
                String tempText = tokenizer.decode(tempResult);
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
    private List<String> readFromFile(String filePath) throws IOException {
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
