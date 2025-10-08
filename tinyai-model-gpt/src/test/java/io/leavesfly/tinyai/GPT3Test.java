package io.leavesfly.tinyai;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt3.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * GPT-3模型测试类
 * 
 * 验证GPT-3实现的正确性，包括：
 * 1. 模型构建测试
 * 2. 前向传播测试
 * 3. 配置验证测试
 * 4. 组件功能测试
 * 5. 边界条件测试
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3Test {
    
    /**
     * 运行所有测试
     */
    public static void main(String[] args) {
        System.out.println("🧪 GPT-3模型测试套件");
        System.out.println("====================\n");
        
        int passed = 0;
        int total = 0;
        
        // 执行各项测试
        total++; passed += testModelCreation() ? 1 : 0;
        total++; passed += testConfigValidation() ? 1 : 0;
        total++; passed += testForwardPass() ? 1 : 0;
        total++; passed += testTextGeneration() ? 1 : 0;
        total++; passed += testRotaryEmbedding() ? 1 : 0;
        total++; passed += testSparseAttention() ? 1 : 0;
        total++; passed += testTransformerBlock() ? 1 : 0;
        total++; passed += testEdgeCases() ? 1 : 0;
        
        // 输出测试结果
        System.out.println("\n" + "=".repeat(50));
        System.out.println("📊 测试结果汇总");
        System.out.println("=".repeat(50));
        System.out.printf("通过: %d/%d (%.1f%%)\n", passed, total, (100.0 * passed / total));
        
        if (passed == total) {
            System.out.println("✅ 所有测试通过！GPT-3实现验证成功。");
        } else {
            System.out.println("❌ 部分测试失败，请检查实现。");
        }
    }
    
    /**
     * 测试模型创建
     */
    private static boolean testModelCreation() {
        System.out.println("1️⃣ 测试模型创建");
        try {
            // 测试不同规模模型的创建
            GPT3Model smallModel = GPT3Model.createSmallModel("test-small");
            GPT3Model mediumModel = GPT3Model.createMediumModel("test-medium");
            GPT3Model largeModel = GPT3Model.createLargeModel("test-large");
            
            // 验证模型属性
            assert smallModel.getConfig().getNLayer() == 12;
            assert mediumModel.getConfig().getNLayer() == 24;
            assert largeModel.getConfig().getNLayer() == 24;
            
            assert smallModel.getConfig().getNEmbd() == 768;
            assert mediumModel.getConfig().getNEmbd() == 1024;
            assert largeModel.getConfig().getNEmbd() == 2048;
            
            System.out.println("   ✓ 不同规模模型创建成功");
            
            // 测试自定义配置
            GPT3Config customConfig = new GPT3Config(1000, 512, 256, 6, 8);
            GPT3Model customModel = new GPT3Model("test-custom", customConfig);
            
            assert customModel.getConfig().getVocabSize() == 1000;
            assert customModel.getConfig().getNPositions() == 512;
            assert customModel.getConfig().getNEmbd() == 256;
            
            System.out.println("   ✓ 自定义配置模型创建成功");
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ 模型创建测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试配置验证
     */
    private static boolean testConfigValidation() {
        System.out.println("2️⃣ 测试配置验证");
        try {
            // 测试有效配置
            GPT3Config validConfig = new GPT3Config(1000, 512, 768, 12, 12);
            validConfig.validate();  // 应该通过
            System.out.println("   ✓ 有效配置验证通过");
            
            // 测试无效配置：嵌入维度不能被头数整除
            try {
                GPT3Config invalidConfig = new GPT3Config(1000, 512, 770, 12, 13);
                invalidConfig.validate();
                System.out.println("   ❌ 无效配置应该抛出异常");
                return false;
            } catch (IllegalArgumentException e) {
                System.out.println("   ✓ 无效配置正确抛出异常: " + e.getMessage());
            }
            
            // 测试参数估算
            long estimatedParams = validConfig.estimateParameterCount();
            assert estimatedParams > 0;
            System.out.println("   ✓ 参数数量估算: " + String.format("%,d", estimatedParams));
            
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ 配置验证测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试前向传播
     */
    private static boolean testForwardPass() {
        System.out.println("3️⃣ 测试前向传播");
        try {
            GPT3Model model = GPT3Model.createSmallModel("test-forward");
            GPT3Config config = model.getConfig();
            
            // 测试不同输入大小
            int[][] testCases = {
                {1, 10},   // 单样本，短序列
                {2, 32},   // 小批次，中等序列
                {1, 128}   // 单样本，长序列
            };
            
            for (int[] testCase : testCases) {
                int batchSize = testCase[0];
                int seqLen = testCase[1];
                
                // 创建输入
                NdArray input = NdArray.of(Shape.of(batchSize, seqLen));
                for (int b = 0; b < batchSize; b++) {
                    for (int s = 0; s < seqLen; s++) {
                        input.set((s + b) % config.getVocabSize(), b, s);
                    }
                }
                
                // 前向传播
                Variable output = model.forward(new Variable(input));
                NdArray outputData = output.getValue();
                
                // 验证输出形状
                Shape expectedShape = Shape.of(batchSize, seqLen, config.getVocabSize());
                assert outputData.getShape().equals(expectedShape) : 
                    "输出形状不匹配: " + outputData.getShape() + " vs " + expectedShape;
                
                System.out.printf("   ✓ 输入%dx%d -> 输出%s\n", 
                                batchSize, seqLen, outputData.getShape());
            }
            
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ 前向传播测试失败: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * 测试文本生成
     */
    private static boolean testTextGeneration() {
        System.out.println("4️⃣ 测试文本生成");
        try {
            GPT3Model model = GPT3Model.createSmallModel("test-generation");
            
            // 测试单步预测
            NdArray singleInput = NdArray.of(Shape.of(1, 5));
            for (int i = 0; i < 5; i++) {
                singleInput.set(i + 1, 0, i);
            }
            
            int nextToken = model.predictNextToken(singleInput);
            assert nextToken >= 0 && nextToken < model.getConfig().getVocabSize();
            System.out.println("   ✓ 单步预测成功，预测token: " + nextToken);
            
            // 测试序列生成
            NdArray startTokens = NdArray.of(Shape.of(1, 3));
            for (int i = 0; i < 3; i++) {
                startTokens.set(i + 10, 0, i);
            }
            
            NdArray generated = model.generateSequence(startTokens, 7);
            int finalLength = generated.getShape().getDimension(1);
            assert finalLength >= 3;  // 至少包含原始序列
            System.out.printf("   ✓ 序列生成成功，从长度%d生成到长度%d\n", 3, finalLength);
            
            // 测试Few-shot生成
            NdArray context = NdArray.of(Shape.of(1, 10));
            for (int i = 0; i < 10; i++) {
                context.set(i + 20, 0, i);
            }
            
            NdArray fewShotResult = model.fewShotGenerate(context, 5);
            assert fewShotResult.getShape().getDimension(1) >= 10;
            System.out.println("   ✓ Few-shot生成成功");
            
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ 文本生成测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试旋转位置编码
     */
    private static boolean testRotaryEmbedding() {
        System.out.println("5️⃣ 测试旋转位置编码");
        try {
            int rotaryDim = 64;
            int maxSeqLen = 128;
            GPT3RotaryEmbedding rope = new GPT3RotaryEmbedding("test-rope", rotaryDim, maxSeqLen);
            
            // 测试编码生成
            int testSeqLen = 32;
            NdArray[] cosAndSin = rope.generateRotaryEmbedding(testSeqLen);
            NdArray cos = cosAndSin[0];
            NdArray sin = cosAndSin[1];
            
            assert cos.getShape().equals(Shape.of(testSeqLen, rotaryDim));
            assert sin.getShape().equals(Shape.of(testSeqLen, rotaryDim));
            System.out.println("   ✓ 旋转编码生成成功");
            
            // 测试旋转变换
            int batchSize = 2;
            int numHeads = 8;
            int headDim = 64;
            
            NdArray query = NdArray.of(Shape.of(batchSize, testSeqLen, numHeads, headDim));
            NdArray key = NdArray.of(Shape.of(batchSize, testSeqLen, numHeads, headDim));
            
            // 填充测试数据
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < testSeqLen; s++) {
                    for (int h = 0; h < numHeads; h++) {
                        for (int d = 0; d < headDim; d++) {
                            query.set((float)(Math.random() - 0.5), b, s, h, d);
                            key.set((float)(Math.random() - 0.5), b, s, h, d);
                        }
                    }
                }
            }
            
            Variable[] rotated = rope.applyRotaryPositionEmbedding(
                new Variable(query), new Variable(key), testSeqLen
            );
            
            assert rotated.length == 2;
            assert rotated[0].getValue().getShape().equals(query.getShape());
            assert rotated[1].getValue().getShape().equals(key.getShape());
            System.out.println("   ✓ 旋转变换应用成功");
            
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ 旋转位置编码测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试稀疏注意力
     */
    private static boolean testSparseAttention() {
        System.out.println("6️⃣ 测试稀疏注意力");
        try {
            // 创建启用稀疏注意力的配置
            GPT3Config config = GPT3Config.createLargeConfig();
            config.setSparseAttention(true);
            
            GPT3SparseAttention sparseAttn = new GPT3SparseAttention(
                "test-sparse", config.getNEmbd(), config.getNHead(), 0, config
            );
            
            // 创建测试输入
            int batchSize = 1;
            int seqLen = 64;
            NdArray input = NdArray.of(Shape.of(batchSize, seqLen, config.getNEmbd()));
            
            // 填充随机数据
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int d = 0; d < config.getNEmbd(); d++) {
                        input.set((float)(Math.random() - 0.5), b, s, d);
                    }
                }
            }
            
            // 测试稀疏注意力前向传播
            Variable output = sparseAttn.layerForward(new Variable(input));
            assert output.getValue().getShape().equals(input.getShape());
            System.out.println("   ✓ 稀疏注意力前向传播成功");
            
            // 验证稀疏注意力配置
            assert sparseAttn.isSparseMode();
            assert sparseAttn.getLocalWindowSize() > 0;
            assert sparseAttn.getGlobalStride() > 0;
            System.out.println("   ✓ 稀疏注意力配置验证成功");
            
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ 稀疏注意力测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试Transformer块
     */
    private static boolean testTransformerBlock() {
        System.out.println("7️⃣ 测试Transformer块");
        try {
            GPT3Config config = GPT3Config.createSmallConfig();
            GPT3TransformerBlock block = new GPT3TransformerBlock("test-block", config, 0);
            
            // 测试块的组件
            assert block.getLayerNorm1() != null;
            assert block.getAttention() != null;
            assert block.getLayerNorm2() != null;
            assert block.getFeedForward() != null;
            System.out.println("   ✓ Transformer块组件初始化成功");
            
            // 测试前向传播
            int batchSize = 2;
            int seqLen = 16;
            NdArray input = NdArray.of(Shape.of(batchSize, seqLen, config.getNEmbd()));
            
            // 填充测试数据
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int d = 0; d < config.getNEmbd(); d++) {
                        input.set((float)(Math.random() - 0.5), b, s, d);
                    }
                }
            }
            
            Variable output = block.layerForward(new Variable(input));
            assert output.getValue().getShape().equals(input.getShape());
            System.out.println("   ✓ Transformer块前向传播成功");
            
            // 测试并行与串行模式
            config.setParallelAttention(false);
            GPT3TransformerBlock sequentialBlock = new GPT3TransformerBlock("test-seq", config, 1);
            Variable seqOutput = sequentialBlock.layerForward(new Variable(input));
            assert seqOutput.getValue().getShape().equals(input.getShape());
            System.out.println("   ✓ 串行模式Transformer块测试成功");
            
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ Transformer块测试失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 测试边界条件
     */
    private static boolean testEdgeCases() {
        System.out.println("8️⃣ 测试边界条件");
        try {
            GPT3Model model = GPT3Model.createSmallModel("test-edge");
            
            // 测试最小输入
            NdArray minInput = NdArray.of(Shape.of(1, 1));
            minInput.set(0, 0, 0);
            
            Variable minOutput = model.forward(new Variable(minInput));
            assert minOutput.getValue().getShape().getDimension(0) == 1;
            assert minOutput.getValue().getShape().getDimension(1) == 1;
            System.out.println("   ✓ 最小输入测试通过");
            
            // 测试输入验证
            try {
                NdArray invalidInput = NdArray.of(Shape.of(1, model.getConfig().getNPositions() + 1));
                model.validateInput(invalidInput);
                System.out.println("   ❌ 应该检测到序列过长");
                return false;
            } catch (IllegalArgumentException e) {
                System.out.println("   ✓ 序列长度验证成功");
            }
            
            // 测试非法配置
            try {
                new GPT3Config(-1, 512, 768, 12, 12);
                System.out.println("   ❌ 应该检测到非法词汇表大小");
                return false;
            } catch (IllegalArgumentException e) {
                System.out.println("   ✓ 非法配置检测成功");
            }
            
            // 测试空模型信息
            String summary = model.getConfigSummary();
            assert summary != null && !summary.isEmpty();
            System.out.println("   ✓ 模型信息生成成功");
            
            return true;
            
        } catch (Exception e) {
            System.out.println("   ❌ 边界条件测试失败: " + e.getMessage());
            return false;
        }
    }
}