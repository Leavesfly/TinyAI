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
 * 4. DPO训练 - 直接偏好优化（人类偏好对齐）
 * 5. RL训练 - 强化学习优化
 * 6. LoRA微调 - 参数高效微调（特定任务适配）
 * 7. 推理测试 - 文本生成
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

            // 步骤3: DPO训练（人类偏好对齐）
            MiniMindModel dpoModel = runDPOTraining(sftModel);

            // 步骤4: 强化学习训练
            MiniMindModel rlModel = runReinforcementLearningTraining(dpoModel);

            // 步骤5: LoRA微调（特定任务适配）
            MiniMindModel loraModel = runLoRAFinetuning(rlModel);

            // 保存最终模型到 CHECKPOINT_DIR/model/ 目录
            String modelName = loraModel.getName();
            System.out.println("\n💾 保存最终模型...");
            ModelLoader.saveModel(loraModel, getSharedTokenizer(), modelName);

            // 步骤6: 从模型文件加载并推理测试
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
        System.out.println("  [0] 数据准备 → [1] 预训练 → [2] SFT → [3] DPO → [4] RL → [5] LoRA → [6] 推理");
        System.out.println();
    }

    private static void printSuccess() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("✅ 完整训练流程演示成功!");
        System.out.println("=".repeat(80));
    }
}
