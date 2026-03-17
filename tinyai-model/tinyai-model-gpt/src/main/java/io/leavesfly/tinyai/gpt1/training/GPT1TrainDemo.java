package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.gpt1.GPT1Model;

/**
 * GPT-1完整训练演示 V2版本
 * <p>
 * 重构说明:
 * 1. 数据生成逻辑提取到 DatasetGenerator 类
 * 2. 训练流程管理提取到 GPT1TrainingManager 类
 * 3. 本类仅作为入口，协调各组件工作
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT1TrainDemo {

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("GPT-1 完整训练与推理演示 V2");
        System.out.println("适用于教学和学习的小型数据集训练方案");
        System.out.println("=".repeat(80));

        try {
            // 步骤0: 准备数据集文件
            DatasetGenerator datasetGenerator = new DatasetGenerator();
            datasetGenerator.prepareDatasets();

            // 步骤1-3: 训练流程
            GPT1TrainingManager trainingManager = new GPT1TrainingManager();
            
            // 步骤1: 预训练
            GPT1Model pretrainedModel = trainingManager.runPretraining();

            // 步骤2: 微调
            GPT1Model finetunedModel = trainingManager.runFinetuning(pretrainedModel);

            // 步骤3: 推理测试
            trainingManager.runInference(finetunedModel);

            System.out.println("\n" + "=".repeat(80));
            System.out.println("✅ 完整训练流程演示成功!");
            System.out.println("=".repeat(80));

        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
