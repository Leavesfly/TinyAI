package io.leavesfly.tinyai.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt2.GPT2Model;
import io.leavesfly.tinyai.gpt2.GPT2Config;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nlp.moe.MoELayer;
import io.leavesfly.tinyai.nlp.moe.MoETransformerBlock;


import java.util.List;
import java.util.Random;

/**
 * MoE-GPT模型演示类
 * <p>
 * 演示MoE-GPT模型的基本功能，包括：
 * 1. 模型创建和配置
 * 2. 前向传播
 * 3. 负载均衡分析
 * 4. 性能对比
 * 5. 专家使用情况统计
 *
 * @author leavesfly
 * @version 1.0
 */
public class MoEGPTModelDemo {

    public static void main(String[] args) {
        System.out.println("=== MoE-GPT 模型演示 ===");
        System.out.println();
        
        // 1. 创建不同规模的MoE-GPT模型
        System.out.println("1. 创建不同规模的MoE-GPT模型:");
        
        // 小型MoE模型 (适合测试)
        MoEGPTModel smallMoEModel = new MoEGPTModel(
            "moe_gpt_small", 1000, 128, 6, 8, 4, 256, 2, 64, 0.1, true, 0.1
        );
        System.out.println("小型MoE模型配置:");
        System.out.println(smallMoEModel.getModelConfig());
        System.out.println();
        
        // 先只测试小型模型后续功能
        /*
        try {
            // 中型MoE模型
            MoEGPTModel mediumMoEModel = new MoEGPTModel(
                "moe_gpt_medium", 5000, 240, 8, 12, 8, 2, 128
            );
            System.out.println("中型MoE模型配置:");
            System.out.println(mediumMoEModel.getModelConfig());
            System.out.println();
        } catch (Exception e) {
            System.out.println("中型MoE模型创建失败: " + e.getMessage());
        }
        */
        
        // 2. 准备输入数据
        System.out.println("2. 准备输入数据:");
        int batchSize = 2;
        int seqLen = 16;
        
        // 创建模拟的token序列
        NdArray inputTokens = createSampleTokens(batchSize, seqLen, smallMoEModel.getVocabSize());
        System.out.printf("输入形状: %s (batch_size=%d, seq_len=%d)%n", 
                         inputTokens.getShape(), batchSize, seqLen);
        
        // 打印输入token ID示例
        System.out.println("输入token ID示例 (第一个样本的前8个token):");
        for (int i = 0; i < Math.min(8, seqLen); i++) {
            System.out.printf("  位置%d: token_id=%d%n", i, (int)inputTokens.get(0, i));
        }
        System.out.println();
        
        // 3. 执行前向传播
        System.out.println("3. 执行MoE-GPT前向传播:");
        long startTime = System.currentTimeMillis();
        
        Variable input = new Variable(inputTokens);
        Variable output = smallMoEModel.layerForward(input);
        
        long endTime = System.currentTimeMillis();
        
        System.out.printf("前向传播完成，耗时: %d ms%n", endTime - startTime);
        
        // 4. 分析输出结果
        System.out.println();
        System.out.println("4. 分析输出结果:");
        NdArray outputData = output.getValue();
        Shape outputShape = outputData.getShape();
        
        System.out.printf("输出形状: %s (batch_size=%d, seq_len=%d, vocab_size=%d)%n",
                         outputShape, outputShape.getDimension(0), 
                         outputShape.getDimension(1), outputShape.getDimension(2));
        
        // 分析最后一个位置的输出概率分布
        System.out.println();
        System.out.println("最后一个位置的输出分析 (第一个样本):");
        int lastPos = seqLen - 1;
        float[] lastTokenLogits = new float[Math.min(10, smallMoEModel.getVocabSize())];
        for (int i = 0; i < lastTokenLogits.length; i++) {
            lastTokenLogits[i] = outputData.get(0, lastPos, i);
        }
        
        System.out.println("前10个token的logits:");
        for (int i = 0; i < lastTokenLogits.length; i++) {
            System.out.printf("  token_%d: %.4f%n", i, lastTokenLogits[i]);
        }
        
        // 5. MoE负载均衡分析
        System.out.println();
        System.out.println("5. MoE负载均衡分析:");
        
        // 多次前向传播以收集统计信息
        System.out.println("正在收集负载均衡统计信息...");
        for (int i = 0; i < 10; i++) {
            NdArray randomInput = createSampleTokens(2, 16, smallMoEModel.getVocabSize());
            smallMoEModel.layerForward(new Variable(randomInput));
        }
        
        // 打印负载均衡报告
        System.out.println(smallMoEModel.getLoadBalancingReport());
        
        // 6. 验证模型完整性
        System.out.println();
        System.out.println("6. 模型组件验证:");
        
        System.out.println("Token嵌入层: " + (smallMoEModel.getTokenEmbedding() != null ? "已初始化" : "未初始化"));
        System.out.println("MoE Transformer块数量: " + smallMoEModel.getMoeTransformerBlocks().size());
        System.out.println("最终层归一化: " + (smallMoEModel.getFinalLayerNorm() != null ? "已初始化" : "未初始化"));
        System.out.println("输出头: " + (smallMoEModel.getOutputHead() != null ? "已初始化" : "未初始化"));
        
        // 验证每个MoE块的专家数量
        for (int i = 0; i < smallMoEModel.getNumLayers(); i++) {
            MoETransformerBlock block = smallMoEModel.getMoeTransformerBlock(i);
            System.out.printf("  第%d层专家数量: %d, Top-K: %d%n", 
                            i, block.getNumExperts(), block.getTopK());
        }
        
        // 7. 参数统计对比
        System.out.println();
        System.out.println("7. 参数统计对比:");
        
        // 创建对应的传统GPT-2模型进行对比
        GPT2Config traditionalGPTConfig = new GPT2Config(
            smallMoEModel.getVocabSize(),                // vocabSize
            smallMoEModel.getMaxSeqLength(),             // nPositions
            smallMoEModel.getDModel(),                   // nEmbd
            smallMoEModel.getNumLayers(),                // nLayer
            smallMoEModel.getNumHeads(),                 // nHead
            smallMoEModel.getDModel() * 4,               // nInner (传统FFN隐藏维度)
            "gelu",                                      // activationFunction
            smallMoEModel.getDropoutRate(),              // residPdrop
            smallMoEModel.getDropoutRate(),              // embdPdrop
            smallMoEModel.getDropoutRate(),              // attnPdrop
            1e-5,                                        // layerNormEpsilon
            0.02                                         // initializerRange
        );
        GPT2Model traditionalGPT = new GPT2Model("traditional_gpt2", traditionalGPTConfig);
        
        long moeParams = smallMoEModel.getTotalParameterCount();
        long traditionalParams = traditionalGPT.getGPT2Block().getParameterCount();
        
        System.out.printf("MoE-GPT参数数量: %,d%n", moeParams);
        System.out.printf("传统GPT-2参数数量: %,d%n", traditionalParams);
        System.out.printf("参数增加比例: %.2fx%n", smallMoEModel.getParameterIncreaseRatio());
        
        // 8. 不同序列长度测试
        System.out.println();
        System.out.println("8. 不同序列长度测试:");
        int[] testLengths = {1, 4, 8, 16, 32};
        
        for (int len : testLengths) {
            if (len <= smallMoEModel.getMaxSeqLength()) {
                try {
                    NdArray testInput = createSampleTokens(1, len, smallMoEModel.getVocabSize());
                    Variable testVar = new Variable(testInput);
                    Variable testOutput = smallMoEModel.layerForward(testVar);
                    System.out.printf("  序列长度 %2d: 成功 (输出形状: %s)%n", 
                                     len, testOutput.getValue().getShape());
                } catch (Exception e) {
                    System.out.printf("  序列长度 %2d: 失败 (%s)%n", len, e.getMessage());
                }
            }
        }
        
        // 9. 专家激活模式分析
        System.out.println();
        System.out.println("9. 专家激活模式分析:");
        
        // 重置统计信息并进行专门的激活分析
        smallMoEModel.resetAllMoEStats();
        
        // 使用不同的输入模式
        System.out.println("测试不同输入模式的专家激活:");
        
        // 模式1: 随机输入
        NdArray randomInput = createSampleTokens(4, 8, smallMoEModel.getVocabSize());
        smallMoEModel.layerForward(new Variable(randomInput));
        System.out.println("随机输入后的专家使用情况:");
        printExpertUsageSummary(smallMoEModel.getAllMoEStats());
        
        // 模式2: 重复模式输入
        NdArray patternInput = createPatternTokens(4, 8, smallMoEModel.getVocabSize());
        smallMoEModel.layerForward(new Variable(patternInput));
        System.out.println("模式输入后的专家使用情况:");
        printExpertUsageSummary(smallMoEModel.getAllMoEStats());
        
        System.out.println();
        System.out.println("=== MoE-GPT 模型演示完成 ===");
    }
    
    /**
     * 创建样本token序列
     */
    private static NdArray createSampleTokens(int batchSize, int seqLen, int vocabSize) {
        NdArray tokens = NdArray.of(Shape.of(batchSize, seqLen));
        Random random = new Random(42);  // 固定种子以确保可重现性
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = random.nextInt(vocabSize);
                tokens.set(tokenId, b, s);
            }
        }
        
        return tokens;
    }
    
    /**
     * 创建有模式的token序列
     */
    private static NdArray createPatternTokens(int batchSize, int seqLen, int vocabSize) {
        NdArray tokens = NdArray.of(Shape.of(batchSize, seqLen));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 创建简单的重复模式
                int tokenId = (s % 4) + (b * 10);  // 确保在词汇表范围内
                tokens.set(tokenId % vocabSize, b, s);
            }
        }
        
        return tokens;
    }
    
    /**
     * 打印专家使用情况摘要
     */
    private static void printExpertUsageSummary(List<MoELayer.LoadBalancingStats> statsList) {
        for (int layerIdx = 0; layerIdx < statsList.size(); layerIdx++) {
            MoELayer.LoadBalancingStats stats = statsList.get(layerIdx);
            System.out.printf("  第%d层: 总tokens=%d, 负载均衡=%.4f%n", 
                            layerIdx, stats.totalTokens, stats.loadImbalance);
        }
    }
}