package io.leavesfly.tinyai.gpt1.demo;

import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import io.leavesfly.tinyai.gpt1.trainer.ChinesePoemDataSet;
import io.leavesfly.tinyai.nlp.ChineseTokenizer;

import java.io.IOException;
import java.util.List;

/**
 * 简化的GPT-1古诗词训练测试
 * 
 * 这个类用于测试古诗词数据处理和分词功能
 * 不依赖复杂的训练框架，便于调试和验证
 * 
 * @author 山泽
 * @version 1.0
 */
public class SimpleDataSetTest {
    
    public static void main(String[] args) {
        System.out.println("=== GPT-1古诗词训练测试 ===\n");
        
        try {
            // 1. 测试分词器
            testTokenizer();
            
            // 2. 测试数据集加载
            testDatasetLoading();
            
            // 3. 测试模型创建
            testModelCreation();
            
        } catch (Exception e) {
            System.err.println("测试过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== 测试完成 ===");
    }
    
    /**
     * 测试中文分词器
     */
    private static void testTokenizer() throws IOException {
        System.out.println("1. 测试中文分词器");
        
        // 创建分词器
        ChineseTokenizer tokenizer = new ChineseTokenizer();
        
        // 构建词汇表 - 修复文件路径，使用相对于项目根目录的路径

        tokenizer.buildVocabFromFile(ChinesePoemDataSet.dataPath);
        
        // 显示词汇表信息
        tokenizer.printVocabStats();
        
        // 测试编码和解码
        String testText = "春江花月夜";
        List<Integer> encoded = tokenizer.encode(testText);
        String decoded = tokenizer.decode(encoded);
        
        System.out.println("\n编码测试:");
        System.out.printf("原文: %s\n", testText);
        System.out.printf("编码: %s\n", encoded);
        System.out.printf("解码: %s\n", decoded);
        
        System.out.println("✓ 分词器测试通过\n");
    }
    
    /**
     * 测试数据集加载
     */
    private static void testDatasetLoading() throws Exception {
        System.out.println("2. 测试数据集加载");
        
        // 创建分词器和数据集
        ChineseTokenizer tokenizer = new ChineseTokenizer();

        tokenizer.buildVocabFromFile(ChinesePoemDataSet.dataPath);
        
        ChinesePoemDataSet dataset = new ChinesePoemDataSet(tokenizer, 32, 4);
        dataset.loadFromFile(ChinesePoemDataSet.dataPath);
        
        // 显示数据集信息
        dataset.printDatasetStats();
        
        // 测试批次生成
        dataset.prepare();
        int batchCount = 0;
        while (dataset.hasNextBatch() && batchCount < 3) {
            ChinesePoemDataSet.BatchData batch = dataset.getNextBatch();
            System.out.printf("批次 %d: 输入形状 %s, 目标形状 %s\n", 
                batchCount + 1,
                batch.getInput().getValue().getShape(),
                batch.getTarget().getValue().getShape());
            batchCount++;
        }
        
        System.out.println("✓ 数据集加载测试通过\n");
    }
    
    /**
     * 测试模型创建
     */
    private static void testModelCreation() {
        System.out.println("3. 测试模型创建");
        
        try {
            // 创建小型模型配置
            GPT1Config config = new GPT1Config(1000, 32, 128, 4, 4);
            config.validate();
            
            System.out.println("模型配置: " + config);
            
            // 创建模型
            GPT1Model model = new GPT1Model("测试模型", config);
            model.printModelInfo();
            
            System.out.println("✓ 模型创建测试通过");
            
        } catch (Exception e) {
            System.out.println("✗ 模型创建测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 演示完整的数据处理流程
     */
    public static void demonstrateDataProcessing() throws Exception {
        System.out.println("=== 数据处理流程演示 ===");
        
        // 1. 初始化组件
        ChineseTokenizer tokenizer = new ChineseTokenizer();

        
        // 2. 构建词汇表
        System.out.println("构建词汇表...");
        tokenizer.buildVocabFromFile(ChinesePoemDataSet.dataPath);
        
        // 3. 加载和编码数据
        System.out.println("加载和编码数据...");
        List<List<Integer>> sequences = tokenizer.loadAndEncodeData(ChinesePoemDataSet.dataPath);
        System.out.printf("加载了 %d 个序列\n", sequences.size());
        
        // 4. 创建训练对
        System.out.println("创建训练对...");
        List<ChineseTokenizer.TrainingPair> pairs = tokenizer.createTrainingPairs(sequences, 32);
        System.out.printf("创建了 %d 个训练对\n", pairs.size());
        
        // 5. 显示样本
        System.out.println("\n前3个训练样本:");
        for (int i = 0; i < Math.min(3, pairs.size()); i++) {
            ChineseTokenizer.TrainingPair pair = pairs.get(i);
            String input = tokenizer.decode(pair.getInput());
            String target = tokenizer.decode(pair.getTarget());
            System.out.printf("样本 %d:\n", i + 1);
            System.out.printf("  输入: %s\n", input);
            System.out.printf("  目标: %s\n", target);
        }
        
        System.out.println("\n数据处理流程演示完成");
    }
    
    /**
     * 性能测试
     */
    public static void performanceTest() throws Exception {
        System.out.println("=== 性能测试 ===");
        
        long startTime = System.currentTimeMillis();
        
        // 分词器性能测试
        ChineseTokenizer tokenizer = new ChineseTokenizer();
        tokenizer.buildVocabFromFile(ChinesePoemDataSet.dataPath);
        
        long vocabTime = System.currentTimeMillis();
        System.out.printf("词汇表构建耗时: %d ms\n", vocabTime - startTime);
        
        // 数据加载性能测试
        ChinesePoemDataSet dataset = new ChinesePoemDataSet(tokenizer, 32, 4);
        dataset.loadFromFile(ChinesePoemDataSet.dataPath);
        
        long dataTime = System.currentTimeMillis();
        System.out.printf("数据集加载耗时: %d ms\n", dataTime - vocabTime);
        
        // 批次生成性能测试
        dataset.prepare();
        int batchCount = 0;
        while (dataset.hasNextBatch()) {
            dataset.getNextBatch();
            batchCount++;
        }
        
        long batchTime = System.currentTimeMillis();
        System.out.printf("批次生成耗时: %d ms (共 %d 批次)\n", batchTime - dataTime, batchCount);
        
        System.out.printf("总耗时: %d ms\n", batchTime - startTime);
    }
}