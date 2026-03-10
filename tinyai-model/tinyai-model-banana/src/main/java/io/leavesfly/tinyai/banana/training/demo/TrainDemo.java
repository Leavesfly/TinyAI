package io.leavesfly.tinyai.banana.training.demo;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.banana.training.PretrainTrainer;
import io.leavesfly.tinyai.banana.training.FinetuneTrainer;
import io.leavesfly.tinyai.banana.training.dataset.BananaDataset;

/**
 * Banana训练和推理完整演示
 * 
 * 展示完整的训练流程:
 * 1. 预训练 (Pretrain) - 多模态对比学习
 * 2. 微调 (Finetune) - 任务特定优化
 * 3. 推理 (Inference) - 文本生成图像
 * 
 * @author TinyAI
 * @since 2024
 */
public class TrainDemo {
    
    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Banana多模态模型训练演示");
        System.out.println("=".repeat(80));
        
        try {
            // 选择演示场景
            String mode = args.length > 0 ? args[0] : "all";
            
            switch (mode) {
                case "pretrain":
                    runPretrainDemo();
                    break;
                case "finetune":
                    runFinetuneDemo();
                    break;
                case "all":
                default:
                    runFullTrainingPipeline();
                    break;
            }
            
        } catch (Exception e) {
            System.err.println("训练失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 运行预训练演示
     */
    private static void runPretrainDemo() {
        System.out.println("\n【演示一：多模态预训练】");
        System.out.println("-".repeat(80));
        
        // 1. 创建Nano模型(演示用,低内存)
        System.out.println("\n1. 创建Banana Nano模型...");
        BananaModel model = BananaModel.create("banana_nano_pretrain", "nano");
        System.out.println(model.getConfigSummary());
        
        // 2. 准备合成训练数据
        System.out.println("\n2. 准备训练数据...");
        BananaConfig config = model.getConfig();
        BananaDataset dataset = new BananaDataset(
            32,                    // maxTextLen
            config.getImageSize(), // imageSize
            4                      // batchSize
        );
        dataset.loadSyntheticData(100);  // 100个合成样本
        
        // 3. 创建预训练器
        System.out.println("\n3. 配置预训练器...");
        PretrainTrainer trainer = new PretrainTrainer(model, dataset);
        trainer.configure(
            2,        // maxEpochs (教学用,只训练2轮)
            1e-3f,    // learningRate
            0,        // warmupSteps (教学用,不预热)
            1.0f      // maxGradNorm
        );
        trainer.setCheckpoint("./checkpoints/banana_pretrain", 50);
        trainer.setLogInterval(10);
        
        // 4. 开始预训练
        System.out.println("\n4. 开始预训练...");
        System.out.println("  - 训练目标: 多模态特征对齐");
        System.out.println("  - 学习率: 1e-3");
        System.out.println("  - 训练轮次: 2 epochs");
        System.out.println("-".repeat(80));
        
        trainer.train();
        
        System.out.println("-".repeat(80));
        System.out.println("\n✅ 预训练完成!");
        printPretrainSummary(trainer);
    }
    
    /**
     * 运行微调演示
     */
    private static void runFinetuneDemo() {
        System.out.println("\n【演示二：任务特定微调】");
        System.out.println("-".repeat(80));
        
        // 1. 加载预训练模型(这里为演示创建新模型)
        System.out.println("\n1. 加载预训练模型...");
        BananaModel model = BananaModel.create("banana_nano_finetune", "nano");
        System.out.println(model.getConfigSummary());
        
        // 2. 准备微调数据集
        System.out.println("\n2. 准备微调数据集...");
        BananaConfig config = model.getConfig();
        
        BananaDataset trainDataset = new BananaDataset(32, config.getImageSize(), 4);
        trainDataset.loadSyntheticData(80);  // 80个训练样本
        
        BananaDataset valDataset = new BananaDataset(32, config.getImageSize(), 4);
        valDataset.loadSyntheticData(20);    // 20个验证样本
        
        // 3. 创建微调器
        System.out.println("\n3. 配置微调器...");
        FinetuneTrainer trainer = new FinetuneTrainer(model, trainDataset, valDataset);
        trainer.configure(
            3,        // maxEpochs
            1e-4f,    // learningRate (比预训练小10倍)
            2         // patience (早停耐心值)
        );
        trainer.setCheckpoint("./checkpoints/banana_finetune");
        
        // 4. 开始微调
        System.out.println("\n4. 开始微调...");
        System.out.println("  - 微调任务: 文本到图像生成");
        System.out.println("  - 学习率: 1e-4 (较小以避免灾难性遗忘)");
        System.out.println("  - 早停策略: patience=2");
        System.out.println("-".repeat(80));
        
        trainer.train();
        
        System.out.println("-".repeat(80));
        System.out.println("\n✅ 微调完成!");
        printFinetuneSummary(trainer);
    }
    
    /**
     * 运行完整训练流程
     */
    private static void runFullTrainingPipeline() {
        System.out.println("\n【完整训练流程：预训练 → 微调】");
        System.out.println("=".repeat(80));
        
        // 阶段一: 预训练
        System.out.println("\n【阶段一：多模态预训练】");
        BananaModel model = BananaModel.create("banana_nano", "nano");
        BananaConfig config = model.getConfig();
        
        System.out.println("创建模型: " + model.getName());
        System.out.println("配置规模: Banana " + config.getClass().getSimpleName());
        
        // 预训练数据
        BananaDataset pretrainDataset = new BananaDataset(32, config.getImageSize(), 4);
        pretrainDataset.loadSyntheticData(100);
        
        // 预训练器
        PretrainTrainer pretrainer = new PretrainTrainer(model, pretrainDataset);
        pretrainer.configure(2, 1e-3f, 0, 1.0f);
        pretrainer.setCheckpoint("./checkpoints/banana_full_pretrain", 50);
        pretrainer.setLogInterval(10);
        
        System.out.println("\n开始预训练...");
        pretrainer.train();
        System.out.println("✅ 预训练完成");
        
        // 阶段二: 微调
        System.out.println("\n【阶段二：任务微调】");
        
        BananaDataset trainDataset = new BananaDataset(32, config.getImageSize(), 4);
        trainDataset.loadSyntheticData(80);
        
        BananaDataset valDataset = new BananaDataset(32, config.getImageSize(), 4);
        valDataset.loadSyntheticData(20);
        
        // 微调器
        FinetuneTrainer finetuner = new FinetuneTrainer(model, trainDataset, valDataset);
        finetuner.configure(3, 1e-4f, 2);
        finetuner.setCheckpoint("./checkpoints/banana_full_finetune");
        
        System.out.println("\n开始微调...");
        finetuner.train();
        System.out.println("✅ 微调完成");
        
        // 总结
        System.out.println("\n" + "=".repeat(80));
        System.out.println("【训练流程总结】");
        System.out.println("=".repeat(80));
        System.out.println("1. 预训练轮次: 2 epochs");
        System.out.println("2. 微调轮次: " + finetuner.getValLossHistory().size() + " epochs");
        System.out.println("3. 最佳验证损失: " + String.format("%.4f", finetuner.getBestValLoss()));
        System.out.println("4. 模型已保存到: ./checkpoints/");
        System.out.println("\n✅ 完整训练流程成功!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 打印预训练总结
     */
    private static void printPretrainSummary(PretrainTrainer trainer) {
        var lossHistory = trainer.getLossHistory();
        if (lossHistory.isEmpty()) {
            return;
        }
        
        System.out.println("\n【预训练总结】");
        System.out.println("- 训练步数: " + lossHistory.size());
        System.out.println("- 初始损失: " + String.format("%.4f", lossHistory.get(0)));
        System.out.println("- 最终损失: " + String.format("%.4f", lossHistory.get(lossHistory.size() - 1)));
        
        // 计算平均损失
        double avgLoss = lossHistory.stream()
            .mapToDouble(Float::doubleValue)
            .average()
            .orElse(0.0);
        System.out.println("- 平均损失: " + String.format("%.4f", avgLoss));
        
        System.out.println("\n提示: 预训练模型已保存,可用于下游任务微调");
    }
    
    /**
     * 打印微调总结
     */
    private static void printFinetuneSummary(FinetuneTrainer trainer) {
        var trainLoss = trainer.getTrainLossHistory();
        var valLoss = trainer.getValLossHistory();
        
        System.out.println("\n【微调总结】");
        System.out.println("- 训练步数: " + trainLoss.size());
        System.out.println("- 验证轮数: " + valLoss.size());
        System.out.println("- 最佳验证损失: " + String.format("%.4f", trainer.getBestValLoss()));
        
        if (!trainLoss.isEmpty()) {
            System.out.println("- 初始训练损失: " + String.format("%.4f", trainLoss.get(0)));
            System.out.println("- 最终训练损失: " + String.format("%.4f", trainLoss.get(trainLoss.size() - 1)));
        }
        
        System.out.println("\n提示: 最佳模型已保存到 checkpoints/banana_finetune/best_model.model");
    }
}
