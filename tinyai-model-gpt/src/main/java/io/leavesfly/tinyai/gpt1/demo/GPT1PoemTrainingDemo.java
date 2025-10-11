package io.leavesfly.tinyai.gpt1.demo;

import io.leavesfly.tinyai.gpt1.trainer.ChinesePoemDataSet;
import io.leavesfly.tinyai.gpt1.trainer.GPT1PoemTrainer;

/**
 * GPT-1古诗词训练演示
 * 
 * 这是一个完整的古诗词语言模型训练演示程序
 * 展示了如何使用TinyAI框架训练一个中文古诗词生成模型
 * 
 * 主要功能：
 * 1. 加载和预处理古诗词数据
 * 2. 构建和配置GPT-1模型
 * 3. 执行完整的训练循环
 * 4. 生成古诗词样本
 * 5. 评估模型性能
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT1PoemTrainingDemo {
    
    public static void main(String[] args) {
        System.out.println("=== GPT-1古诗词训练演示 ===\n");
        
        try {
            // 1. 配置训练参数
            System.out.println("1. 配置训练参数...");
            GPT1PoemTrainer.TrainingConfig config = createTrainingConfig();
            printTrainingConfig(config);
            
            // 2. 创建训练器
            System.out.println("\n2. 创建训练器...");
            GPT1PoemTrainer trainer = new GPT1PoemTrainer(config);
            
            // 3. 数据文件路径（相对于项目根目录）
            String dataPath = ChinesePoemDataSet.dataPath;
            
            // 4. 初始化训练组件
            System.out.println("\n3. 初始化训练组件...");
            trainer.initialize(dataPath);
            
            // 5. 开始训练
            System.out.println("\n4. 开始训练...");
            trainer.train();
            
            // 6. 训练后测试
            System.out.println("\n=== 训练完成，开始测试 ===");
            performPostTrainingTests(trainer);
            
        } catch (Exception e) {
            System.err.println("训练过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== 演示程序结束 ===");
    }
    
    /**
     * 创建训练配置
     * 
     * @return 训练配置对象
     */
    private static GPT1PoemTrainer.TrainingConfig createTrainingConfig() {
        GPT1PoemTrainer.TrainingConfig config = new GPT1PoemTrainer.TrainingConfig();
        
        // 模型规模（适合演示的小模型）
        config.maxSequenceLength = 32;        // 最大序列长度
        config.hiddenSize = 128;              // 隐藏层维度
        config.numLayers = 4;                 // Transformer层数
        config.numAttentionHeads = 4;         // 注意力头数
        
        // 训练参数
        config.epochs = 20;                   // 训练轮数
        config.batchSize = 4;                 // 批量大小
        config.learningRate = 0.001;          // 学习率
        config.validationRatio = 0.2;         // 验证集比例
        
        // 生成参数
        config.generateLength = 15;           // 生成文本长度
        config.temperature = 0.8;             // 生成温度
        
        // 监控参数
        config.enableMonitoring = true;       // 启用训练监控
        config.printInterval = 5;             // 打印间隔
        config.saveInterval = 10;             // 保存间隔
        config.modelSavePath = "gpt1_poem_model"; // 模型保存路径
        
        return config;
    }
    
    /**
     * 打印训练配置信息
     * 
     * @param config 训练配置
     */
    private static void printTrainingConfig(GPT1PoemTrainer.TrainingConfig config) {
        System.out.println("训练配置详情:");
        System.out.println("  模型配置:");
        System.out.printf("    最大序列长度: %d\n", config.maxSequenceLength);
        System.out.printf("    隐藏层维度: %d\n", config.hiddenSize);
        System.out.printf("    Transformer层数: %d\n", config.numLayers);
        System.out.printf("    注意力头数: %d\n", config.numAttentionHeads);
        
        System.out.println("  训练配置:");
        System.out.printf("    训练轮数: %d\n", config.epochs);
        System.out.printf("    批量大小: %d\n", config.batchSize);
        System.out.printf("    学习率: %.4f\n", config.learningRate);
        System.out.printf("    验证集比例: %.2f\n", config.validationRatio);
        
        System.out.println("  生成配置:");
        System.out.printf("    生成长度: %d\n", config.generateLength);
        System.out.printf("    生成温度: %.2f\n", config.temperature);
    }
    
    /**
     * 执行训练后测试
     * 
     * @param trainer 训练器
     */
    private static void performPostTrainingTests(GPT1PoemTrainer trainer) {
        System.out.println("1. 文本生成测试:");
        testTextGeneration(trainer);
        
        System.out.println("\n2. 不同温度参数测试:");
        testDifferentTemperatures(trainer);
        
        System.out.println("\n3. 不同长度生成测试:");
        testDifferentLengths(trainer);
        
        System.out.println("\n4. 模型困惑度评估:");
        testPerplexity(trainer);
    }
    
    /**
     * 测试文本生成
     * 
     * @param trainer 训练器
     */
    private static void testTextGeneration(GPT1PoemTrainer trainer) {
        String[] testPrompts = {
            "春",      // 春天主题
            "月",      // 月亮主题  
            "山",      // 山水主题
            "水",      // 水景主题
            "花",      // 花卉主题
            "风",      // 风景主题
            "云",      // 云彩主题
            "雪"       // 雪景主题
        };
        
        System.out.println("使用不同提示词生成古诗:");
        for (String prompt : testPrompts) {
            String generated = trainer.generateText(prompt, 12, 0.8);
            System.out.printf("  '%s' -> '%s'\n", prompt, generated);
        }
    }
    
    /**
     * 测试不同温度参数
     * 
     * @param trainer 训练器
     */
    private static void testDifferentTemperatures(GPT1PoemTrainer trainer) {
        String prompt = "春江";
        double[] temperatures = {0.3, 0.6, 0.8, 1.0, 1.2};
        
        System.out.printf("使用提示词'%s'，不同温度参数生成:\n", prompt);
        for (double temp : temperatures) {
            String generated = trainer.generateText(prompt, 15, temp);
            System.out.printf("  温度%.1f: '%s'\n", temp, generated);
        }
    }
    
    /**
     * 测试不同长度生成
     * 
     * @param trainer 训练器
     */
    private static void testDifferentLengths(GPT1PoemTrainer trainer) {
        String prompt = "明月";
        int[] lengths = {8, 12, 16, 20};
        
        System.out.printf("使用提示词'%s'，不同长度生成:\n", prompt);
        for (int length : lengths) {
            String generated = trainer.generateText(prompt, length, 0.8);
            System.out.printf("  长度%d: '%s'\n", length, generated);
        }
    }
    
    /**
     * 测试模型困惑度
     * 
     * @param trainer 训练器
     */
    private static void testPerplexity(GPT1PoemTrainer trainer) {
        try {
            double perplexity = trainer.evaluatePerplexity();
            System.out.printf("模型困惑度: %.2f\n", perplexity);
            
            // 困惑度解释
            if (perplexity < 10) {
                System.out.println("困惑度很低，模型性能优秀");
            } else if (perplexity < 50) {
                System.out.println("困惑度适中，模型性能良好");
            } else if (perplexity < 100) {
                System.out.println("困惑度较高，模型还有改进空间");
            } else {
                System.out.println("困惑度很高，建议增加训练轮数或调整模型参数");
            }
        } catch (Exception e) {
            System.out.println("困惑度计算失败: " + e.getMessage());
        }
    }
    
    /**
     * 演示训练过程的详细配置选项
     */
    public static void demonstrateAdvancedConfig() {
        System.out.println("=== 高级配置演示 ===");
        
        // 创建更大规模的模型配置
        GPT1PoemTrainer.TrainingConfig advancedConfig = new GPT1PoemTrainer.TrainingConfig();
        
        // 大模型配置
        advancedConfig.maxSequenceLength = 64;
        advancedConfig.hiddenSize = 256;
        advancedConfig.numLayers = 8;
        advancedConfig.numAttentionHeads = 8;
        
        // 更长的训练
        advancedConfig.epochs = 100;
        advancedConfig.batchSize = 8;
        advancedConfig.learningRate = 0.0005;
        
        System.out.println("高级配置: " + advancedConfig);
        System.out.println("注意: 此配置需要更多计算资源和时间");
    }
    
    /**
     * 演示模型性能分析
     */
    public static void demonstratePerformanceAnalysis() {
        System.out.println("=== 性能分析演示 ===");
        
        System.out.println("模型性能指标:");
        System.out.println("1. 训练损失 - 衡量模型在训练数据上的拟合程度");
        System.out.println("2. 验证损失 - 衡量模型的泛化能力");
        System.out.println("3. 困惑度 - 衡量语言模型的预测质量");
        System.out.println("4. 生成质量 - 主观评估生成文本的流畅性和合理性");
        
        System.out.println("\n优化建议:");
        System.out.println("- 如果训练损失高: 增加模型容量或训练轮数");
        System.out.println("- 如果验证损失高: 可能过拟合，需要正则化");
        System.out.println("- 如果生成质量差: 调整温度参数或改进数据质量");
    }
}