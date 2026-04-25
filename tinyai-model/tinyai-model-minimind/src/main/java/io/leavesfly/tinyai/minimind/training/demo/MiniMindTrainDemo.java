package io.leavesfly.tinyai.minimind.training.demo;

import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.cli.ModelLoader;

import static io.leavesfly.tinyai.minimind.training.demo.DemoConfig.*;
import static io.leavesfly.tinyai.minimind.training.demo.DemoDataGenerator.*;
import static io.leavesfly.tinyai.minimind.training.demo.DemoTrainingStages.*;

/**
 * MiniMind 完整训练演示 - 主入口
 * <p>
 * 提供完整的 LLM 训练流程演示：
 * 1. 数据准备 - 生成各阶段训练数据
 * 2. 预训练 - 无监督语言建模
 * 3. SFT微调 - 监督指令微调（全参数）
 * 4. 知识蒸馏 - 教师模型指导学生模型
 * 5. DPO训练 - 直接偏好优化（人类偏好对齐）
 * 6. RL训练 - 强化学习优化
 * 7. Agent RL训练 - 工具调用强化学习
 * 8. LoRA微调 - 参数高效微调（特定任务适配）
 * 9. 推理测试 - 文本生成
 * <p>
 * 数据集特点：超小规模、适合教学、覆盖完整流程
 * <p>
 * 代码结构：
 * - {@link DemoConfig} - 配置与工具类
 * - {@link DemoDataGenerator} - 数据生成器
 * - {@link DemoTrainingStages} - 训练阶段执行器
 *
 * @author TinyAI Team
 * @version 2.0
 */
public class MiniMindTrainDemo {

    public static void main(String[] args) {
        printBanner();

        try {
            // 步骤0: 准备数据集
            prepareDatasets();

            // 步骤1: 无监督预训练
            MiniMindModel pretrainedModel = runUnsupervisedPretraining();

            // 步骤2: 监督微调（SFT）
            MiniMindModel sftModel = runSupervisedFinetuning(pretrainedModel);

            // 步骤3: 知识蒸馏训练（教师模型指导学生模型）
            MiniMindModel distilledModel = runDistillationTraining(sftModel);

            // 步骤4: DPO训练（人类偏好对齐）
            MiniMindModel dpoModel = runDPOTraining(distilledModel);

            // 步骤5: 强化学习训练
            MiniMindModel rlModel = runReinforcementLearningTraining(dpoModel);

            // 步骤6: Agent RL训练（工具调用强化学习）
            MiniMindModel agentModel = runAgentTraining(rlModel);

            // LoRA微调前推理 - 记录基线效果
            runInferenceForComparison(agentModel, "LoRA微调前推理（基线效果）");

            // 步骤7: LoRA微调（特定任务适配）
            MiniMindModel loraModel = runLoRAFinetuning(agentModel);

            // LoRA微调后推理 - 与基线对比
            runInferenceForComparison(loraModel, "LoRA微调后推理（微调效果）");

            // 保存最终模型到 CHECKPOINT_DIR/model/ 目录
            String modelName = loraModel.getName();
            System.out.println("\n💾 保存最终模型...");
            // 合并 LoRA 权重到原始权重，确保保存/加载后推理结果一致
            loraModel.mergeLoRA();
            ModelLoader.saveModel(loraModel, getSharedTokenizer(), modelName);

            // 步骤8: 从模型文件加载并推理测试
            runInference(modelName);

            printSuccess();

        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void printBanner() {
        System.out.println("=".repeat(80));
        System.out.println("MiniMind 完整训练与推理演示");
        System.out.println("适用于教学和学习的超小规模数据集训练方案");
        System.out.println("=".repeat(80));
        System.out.println();
        System.out.println("训练流程:");
        System.out.println("  [0] 数据准备 → [1] 预训练 → [2] SFT → [3] 蒸馏 → [4] DPO → [5] RL → [6] Agent → [7] LoRA → [8] 推理");
        System.out.println();
    }

    private static void printSuccess() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("✅ 完整训练流程演示成功!");
        System.out.println("=".repeat(80));
    }
}
