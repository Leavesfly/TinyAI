package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * DeepSeek-V3 模型测试类
 * 
 * 提供简单的测试方法验证DeepSeek-V3模型的核心功能
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Test {
    
    public static void main(String[] args) {
        System.out.println("=== DeepSeek-V3 模型测试开始 ===");
        
        try {
            DeepSeekV3Test tester = new DeepSeekV3Test();
            
            // 运行所有测试
            tester.testModelCreation();
            tester.testForwardPass();
            tester.testMLAAttention();
            tester.testMoELayer();
            tester.testLoadBalancing();
            tester.testKVCache();
            tester.testParameterCounting();
            tester.testFactory();
            tester.testConfig();
            tester.testMemoryOptimization();
            
            System.out.println("\n=== 所有测试通过！ ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public void testModelCreation() {
        System.out.println("1. 测试模型创建...");
        
        DeepSeekV3Config config = DeepSeekV3Config.createTinyConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("test_model", config);
        
        // 验证模型组件
        if (model == null) throw new RuntimeException("模型创建失败");
        if (model.getConfig() == null) throw new RuntimeException("配置为空");
        if (model.getTokenEmbedding() == null) throw new RuntimeException("Token嵌入层为空");
        if (model.getTransformerBlocks() == null) throw new RuntimeException("Transformer块列表为空");
        if (model.getFinalLayerNorm() == null) throw new RuntimeException("最终层归一化为空");
        if (model.getOutputHead() == null) throw new RuntimeException("输出头为空");
        
        // 验证层数
        if (model.getTransformerBlocks().size() != config.getNumLayers()) {
            throw new RuntimeException("Transformer块数量不匹配");
        }
        
        System.out.println("  ✓ 模型创建测试通过");
    }
    
    public void testForwardPass() {
        System.out.println("2. 测试前向传播...");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("forward_test");
        DeepSeekV3Config config = model.getConfig();
        
        int batchSize = 2;
        int seqLen = 8;
        NdArray inputTokens = createRandomTokens(batchSize, seqLen, config.getVocabSize());
        
        Variable input = new Variable(inputTokens);
        Variable output = model.layerForward(input);
        
        if (output == null) throw new RuntimeException("前向传播输出为空");
        
        Shape outputShape = output.getValue().getShape();
        if (outputShape.getDimNum() != 3) throw new RuntimeException("输出维度数量错误");
        if (outputShape.getDimension(0) != batchSize) throw new RuntimeException("批大小不匹配");
        if (outputShape.getDimension(1) != seqLen) throw new RuntimeException("序列长度不匹配");
        if (outputShape.getDimension(2) != config.getVocabSize()) throw new RuntimeException("词汇表大小不匹配");
        
        System.out.println("  ✓ 前向传播测试通过");
    }
    
    public void testMLAAttention() {
        System.out.println("3. 测试MLA注意力...");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("mla_test");
        DeepSeekV3Config config = model.getConfig();
        
        DeepSeekV3TransformerBlock firstBlock = model.getTransformerBlock(0);
        MultiHeadLatentAttention mlaAttention = firstBlock.getMlaAttention();
        
        if (mlaAttention == null) throw new RuntimeException("MLA注意力层为空");
        
        // 验证配置
        if (mlaAttention.getDModel() != config.getDModel()) throw new RuntimeException("MLA模型维度不匹配");
        if (mlaAttention.getNumHeads() != config.getNumHeads()) throw new RuntimeException("MLA注意力头数不匹配");
        if (mlaAttention.getDMLA() != config.getDMLA()) throw new RuntimeException("MLA潜在维度不匹配");
        
        // 验证压缩比
        double compressionRatio = mlaAttention.getCacheCompressionRatio();
        if (compressionRatio <= 0 || compressionRatio >= 1) throw new RuntimeException("缓存压缩比异常");
        
        // 验证内存节省
        long memorySavings = mlaAttention.getMemorySavingsBytes(1024);
        if (memorySavings <= 0) throw new RuntimeException("内存节省应该大于0");
        
        System.out.println("  ✓ MLA注意力测试通过");
    }
    
    public void testMoELayer() {
        System.out.println("4. 测试MoE层...");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("moe_test");
        DeepSeekV3Config config = model.getConfig();
        
        DeepSeekV3TransformerBlock firstBlock = model.getTransformerBlock(0);
        DeepSeekMoELayer moeLayer = firstBlock.getMoeLayer();
        
        if (moeLayer == null) throw new RuntimeException("MoE层为空");
        
        // 验证配置
        if (moeLayer.getDModel() != config.getDModel()) throw new RuntimeException("MoE模型维度不匹配");
        if (moeLayer.getNumExperts() != config.getNumExperts()) throw new RuntimeException("专家数量不匹配");
        if (moeLayer.getTopK() != config.getTopK()) throw new RuntimeException("Top-K值不匹配");
        
        // 执行前向传播生成统计数据
        NdArray inputTokens = createRandomTokens(2, 8, config.getVocabSize());
        Variable input = new Variable(inputTokens);
        model.layerForward(input);
        
        // 验证统计
        long totalTokens = moeLayer.getTotalTokens();
        if (totalTokens <= 0) throw new RuntimeException("处理的token数应该大于0");
        
        long[] expertUsage = moeLayer.getExpertUsageCount();
        if (expertUsage == null) throw new RuntimeException("专家使用统计为空");
        if (expertUsage.length != config.getNumExperts()) throw new RuntimeException("专家使用统计长度不匹配");
        
        System.out.println("  ✓ MoE层测试通过");
    }
    
    public void testLoadBalancing() {
        System.out.println("5. 测试负载均衡...");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("balance_test");
        
        // 执行多次前向传播
        for (int i = 0; i < 3; i++) {
            NdArray inputTokens = createRandomTokens(2, 8, model.getConfig().getVocabSize());
            Variable input = new Variable(inputTokens);
            model.layerForward(input);
        }
        
        // 验证负载均衡损失
        double loadBalanceLoss = model.computeTotalLoadBalancingLoss();
        if (loadBalanceLoss < 0) throw new RuntimeException("负载均衡损失应该非负");
        
        // 测试统计重置
        model.resetAllMoEStats();
        DeepSeekV3TransformerBlock firstBlock = model.getTransformerBlock(0);
        if (firstBlock.getTotalTokens() != 0) throw new RuntimeException("统计重置后token数应该为0");
        
        System.out.println("  ✓ 负载均衡测试通过");
    }
    
    public void testKVCache() {
        System.out.println("6. 测试KV缓存...");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("cache_test");
        DeepSeekV3TransformerBlock firstBlock = model.getTransformerBlock(0);
        MultiHeadLatentAttention mlaAttention = firstBlock.getMlaAttention();
        
        // 测试缓存启用/禁用
        mlaAttention.enableKVCache();
        if (!mlaAttention.isUseKVCache()) throw new RuntimeException("KV缓存应该被启用");
        
        mlaAttention.disableKVCache();
        if (mlaAttention.isUseKVCache()) throw new RuntimeException("KV缓存应该被禁用");
        
        // 测试全局缓存控制
        model.enableKVCache();
        model.disableKVCache();
        model.clearAllKVCache();
        
        System.out.println("  ✓ KV缓存测试通过");
    }
    
    public void testParameterCounting() {
        System.out.println("7. 测试参数计数...");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("param_test");
        
        long totalParams = model.getTotalParameterCount();
        long activeParams = model.getActiveParameterCount();
        
        if (totalParams <= 0) throw new RuntimeException("总参数数应该大于0");
        if (activeParams <= 0) throw new RuntimeException("激活参数数应该大于0");
        if (activeParams > totalParams) throw new RuntimeException("激活参数数不应该超过总参数数");
        
        double efficiency = (double) activeParams / totalParams;
        if (efficiency <= 0 || efficiency > 1) throw new RuntimeException("参数效率应该在0到1之间");
        
        System.out.printf("    总参数: %,d, 激活参数: %,d, 效率: %.2f%%\n", 
                         totalParams, activeParams, efficiency * 100);
        
        System.out.println("  ✓ 参数计数测试通过");
    }
    
    public void testFactory() {
        System.out.println("8. 测试工厂方法...");
        
        DeepSeekV3Model tinyModel = DeepSeekV3Factory.createTinyModel("factory_tiny");
        DeepSeekV3Model smallModel = DeepSeekV3Factory.createSmallModel("factory_small");
        
        if (tinyModel == null) throw new RuntimeException("工厂创建的tiny模型为空");
        if (smallModel == null) throw new RuntimeException("工厂创建的small模型为空");
        
        // 验证模型
        if (!DeepSeekV3Factory.validateModel(tinyModel)) throw new RuntimeException("Tiny模型验证失败");
        if (!DeepSeekV3Factory.validateModel(smallModel)) throw new RuntimeException("Small模型验证失败");
        
        System.out.println("  ✓ 工厂方法测试通过");
    }
    
    public void testConfig() {
        System.out.println("9. 测试配置类...");
        
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        // 测试配置验证
        try {
            config.validate();
        } catch (Exception e) {
            throw new RuntimeException("默认配置验证失败: " + e.getMessage());
        }
        
        // 测试参数计算
        long totalParams = config.getTotalParameterCount();
        long activeParams = config.getActiveParameterCount();
        
        if (totalParams <= 0) throw new RuntimeException("配置计算的总参数数应该大于0");
        if (activeParams <= 0) throw new RuntimeException("配置计算的激活参数数应该大于0");
        
        // 测试配置摘要
        String summary = config.getSummary();
        if (summary == null || summary.isEmpty()) throw new RuntimeException("配置摘要为空");
        
        System.out.println("  ✓ 配置测试通过");
    }
    
    public void testMemoryOptimization() {
        System.out.println("10. 测试内存优化...");
        
        DeepSeekV3Config baseConfig = DeepSeekV3Config.createTinyConfig();
        DeepSeekV3Model optimizedModel = DeepSeekV3Factory.createInferenceOptimizedModel(
            "optimized_test", baseConfig);
        
        if (optimizedModel == null) throw new RuntimeException("优化模型创建失败");
        if (optimizedModel.isTraining()) throw new RuntimeException("优化模型应该处于推理模式");
        
        // 测试内存节省计算
        long memorySavings = optimizedModel.getTotalMemorySavings(1024);
        if (memorySavings < 0) throw new RuntimeException("内存节省计算异常");
        
        System.out.println("  ✓ 内存优化测试通过");
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 创建随机token输入
     */
    private NdArray createRandomTokens(int batchSize, int seqLen, int vocabSize) {
        NdArray tokens = NdArray.zeros(Shape.of(batchSize, seqLen));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int randomToken = (int) (Math.random() * vocabSize);
                tokens.set(randomToken, b, s);
            }
        }
        
        return tokens;
    }
}