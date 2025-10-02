package io.leavesfly.tinyai.gpt2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * GPT2Model使用示例
 * 
 * @author leavesfly
 */
public class GPT2ModelDemo {
    
    public static void main(String[] args) {
        System.out.println("=== GPT-2 模型演示 ===");
        System.out.println();
        
        // 1. 创建不同规模的GPT-2模型
        System.out.println("1. 创建不同规模的GPT-2模型:");
        
        // 小型模型 (适合测试)
        GPT2Model smallModel = new GPT2Model("gpt2_small", 1000, 128, 6, 8, 512, 64, 0.1);
        System.out.println("小型模型配置:");
        System.out.println(smallModel.getModelConfig());
        System.out.println();
        
        // 2. 准备输入数据
        System.out.println("2. 准备输入数据:");
        int batchSize = 2;
        int seqLen = 16;
        
        // 创建模拟的token序列
        NdArray inputTokens = createSampleTokens(batchSize, seqLen, smallModel.getVocabSize());
        System.out.printf("输入形状: %s (batch_size=%d, seq_len=%d)%n", 
                         inputTokens.getShape(), batchSize, seqLen);
        
        // 打印输入token ID示例
        System.out.println("输入token ID示例 (第一个样本的前8个token):");
        for (int i = 0; i < Math.min(8, seqLen); i++) {
            System.out.printf("  位置%d: token_id=%d%n", i, (int)inputTokens.get(0, i));
        }
        System.out.println();
        
        // 3. 执行前向传播
        System.out.println("3. 执行前向传播:");
        long startTime = System.currentTimeMillis();
        
        Variable input = new Variable(inputTokens);
        Variable output = smallModel.layerForward(input);
        
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
        float[] lastTokenLogits = new float[Math.min(10, smallModel.getVocabSize())];
        for (int i = 0; i < lastTokenLogits.length; i++) {
            lastTokenLogits[i] = outputData.get(0, lastPos, i);
        }
        
        System.out.println("前10个token的logits:");
        for (int i = 0; i < lastTokenLogits.length; i++) {
            System.out.printf("  token_%d: %.4f%n", i, lastTokenLogits[i]);
        }
        
        // 5. 验证模型完整性
        System.out.println();
        System.out.println("5. 模型组件验证:");
        
        System.out.println("Token嵌入层: " + (smallModel.getTokenEmbedding() != null ? "已初始化" : "未初始化"));
        System.out.println("Transformer块数量: " + smallModel.getTransformerBlocks().size());
        System.out.println("最终层归一化: " + (smallModel.getFinalLayerNorm() != null ? "已初始化" : "未初始化"));
        System.out.println("输出头: " + (smallModel.getOutputHead() != null ? "已初始化" : "未初始化"));
        
        // 6. 参数统计
        System.out.println();
        System.out.println("6. 参数统计:");
        java.util.Map<String, io.leavesfly.tinyai.nnet.Parameter> allParams = smallModel.getAllParams();
        System.out.println("总参数数量: " + allParams.size());
        
        long totalElements = 0;
        for (io.leavesfly.tinyai.nnet.Parameter param : allParams.values()) {
            totalElements += param.getValue().getShape().size();
        }
        System.out.printf("估计参数总数: %,d%n", totalElements);
        
        // 7. 不同序列长度测试
        System.out.println();
        System.out.println("7. 不同序列长度测试:");
        int[] testLengths = {1, 4, 8, 16, 32};
        
        for (int len : testLengths) {
            if (len <= smallModel.getMaxSeqLength()) {
                try {
                    NdArray testInput = createSampleTokens(1, len, smallModel.getVocabSize());
                    Variable testVar = new Variable(testInput);
                    Variable testOutput = smallModel.layerForward(testVar);
                    System.out.printf("  序列长度 %2d: 成功 (输出形状: %s)%n", 
                                     len, testOutput.getValue().getShape());
                } catch (Exception e) {
                    System.out.printf("  序列长度 %2d: 失败 (%s)%n", len, e.getMessage());
                }
            }
        }
        
        System.out.println();
        System.out.println("=== GPT-2 模型演示完成 ===");
    }
    
    /**
     * 创建示例token序列
     */
    private static NdArray createSampleTokens(int batchSize, int seqLen, int vocabSize) {
        NdArray tokens = NdArray.zeros(Shape.of(batchSize, seqLen));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 创建有一定模式的token序列，避免纯随机
                int tokenId = (b * 1000 + s * 7 + 1) % Math.min(vocabSize, 100); // 限制在较小范围内
                tokens.set(tokenId, b, s);
            }
        }
        
        return tokens;
    }
}