package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Qwen3模型测试类
 * 
 * 该类提供了Qwen3模型的全面测试，包括：
 * 1. 基本功能测试
 * 2. 模型架构验证
 * 3. 前向传播测试
 * 4. 文本生成测试
 * 5. 性能基准测试
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Test {

    /**
     * 主测试方法
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== Qwen3模型测试开始 ===\n");
        
        try {
            // 1. 基本功能测试
            testBasicFunctionality();
            
            // 2. 模型架构测试
            testModelArchitecture();
            
            // 3. 前向传播测试
            testForwardPass();
            
            // 4. 文本生成测试
            testTextGeneration();
            
            // 5. 组件单元测试
            testComponents();
            
            System.out.println("\n=== 所有测试通过！ ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试基本功能
     */
    private static void testBasicFunctionality() {
        System.out.println("1. 基本功能测试");
        
        // 创建小型模型
        Qwen3Model model = Qwen3Model.createTinyModel("test_qwen3");
        System.out.println("✓ 模型创建成功");
        
        // 验证配置
        Qwen3Config config = model.getConfig();
        assert config.getVocabSize() == 1000 : "词汇表大小错误";
        assert config.getHiddenSize() == 256 : "隐藏层维度错误";
        assert config.getNumHiddenLayers() == 4 : "层数错误";
        System.out.println("✓ 配置验证成功");
        
        // 打印模型信息
        model.printModelInfo();
        
        System.out.println("✓ 基本功能测试通过\n");
    }
    
    /**
     * 测试模型架构
     */
    private static void testModelArchitecture() {
        System.out.println("2. 模型架构测试");
        
        Qwen3Model model = Qwen3Model.createTinyModel("arch_test");
        Qwen3Block block = model.getQwen3Block();
        
        // 验证组件
        assert block.getEmbedTokens() != null : "词嵌入层未初始化";
        assert block.getDecoderLayers() != null : "解码器层未初始化";
        assert block.getDecoderLayers().length == 4 : "解码器层数量错误";
        assert block.getNorm() != null : "最终归一化层未初始化";
        System.out.println("✓ 架构组件验证成功");
        
        // 验证每层的子组件
        Qwen3DecoderLayer layer0 = block.getLayer(0);
        assert layer0.getSelfAttn() != null : "注意力层未初始化";
        assert layer0.getMlp() != null : "MLP层未初始化";
        assert layer0.getInputLayerNorm() != null : "输入归一化层未初始化";
        assert layer0.getPostAttentionLayerNorm() != null : "注意力后归一化层未初始化";
        System.out.println("✓ 子组件验证成功");
        
        System.out.println("✓ 模型架构测试通过\n");
    }
    
    /**
     * 测试前向传播
     */
    private static void testForwardPass() {
        System.out.println("3. 前向传播测试");
        
        Qwen3Model model = Qwen3Model.createTinyModel("forward_test");
        
        // 测试不同输入格式
        testSingleSequence(model);
        testBatchSequences(model);
        
        System.out.println("✓ 前向传播测试通过\n");
    }
    
    /**
     * 测试单序列输入
     */
    private static void testSingleSequence(Qwen3Model model) {
        // 1D输入测试
        NdArray singleInput = NdArray.of(new float[]{1, 2, 3, 4, 5});
        Variable singleOutput = model.forwardWithLogits(new Variable(singleInput));
        
        Shape outputShape = singleOutput.getValue().getShape();
        assert outputShape.getDimNum() == 3 : "单序列输出维度错误";
        assert outputShape.getDimension(0) == 1 : "batch维度错误";
        assert outputShape.getDimension(1) == 5 : "序列长度错误";
        assert outputShape.getDimension(2) == 1000 : "词汇表维度错误";
        
        System.out.println("✓ 单序列前向传播成功");
    }
    
    /**
     * 测试批次序列输入
     */
    private static void testBatchSequences(Qwen3Model model) {
        // 2D输入测试 (batch_size=2, seq_len=3)
        NdArray batchInput = NdArray.of(Shape.of(2, 3));
        batchInput.set(1, 0, 0); batchInput.set(2, 0, 1); batchInput.set(3, 0, 2);
        batchInput.set(4, 1, 0); batchInput.set(5, 1, 1); batchInput.set(6, 1, 2);
        
        Variable batchOutput = model.forwardWithLogits(new Variable(batchInput));
        
        Shape outputShape = batchOutput.getValue().getShape();
        assert outputShape.getDimNum() == 3 : "批次输出维度错误";
        assert outputShape.getDimension(0) == 2 : "batch维度错误";
        assert outputShape.getDimension(1) == 3 : "序列长度错误";
        assert outputShape.getDimension(2) == 1000 : "词汇表维度错误";
        
        System.out.println("✓ 批次序列前向传播成功");
    }
    
    /**
     * 测试文本生成
     */
    private static void testTextGeneration() {
        System.out.println("4. 文本生成测试");
        
        Qwen3Model model = Qwen3Model.createTinyModel("generation_test");
        
        // 测试下一个token预测
        NdArray inputSequence = NdArray.of(new float[]{1, 2, 3});
        int nextToken = model.predictNextToken(inputSequence);
        assert nextToken >= 0 && nextToken < 1000 : "预测token超出范围";
        System.out.println("✓ 下一个token预测: " + nextToken);
        
        // 测试序列生成
        NdArray generatedSequence = model.generate(inputSequence, 8);
        assert generatedSequence.getShape().getDimension(1) == 8 : "生成序列长度错误";
        
        System.out.println("✓ 生成序列: ");
        printSequence(generatedSequence);
        
        System.out.println("✓ 文本生成测试通过\n");
    }
    
    /**
     * 测试各个组件
     */
    private static void testComponents() {
        System.out.println("5. 组件单元测试");
        
        testRMSNorm();
        testSiLULayer();
        testRotaryPositionalEmbedding();
        
        System.out.println("✓ 组件单元测试通过\n");
    }
    
    /**
     * 测试RMSNorm层
     */
    private static void testRMSNorm() {
        RMSNorm rmsNorm = new RMSNorm("test_rms", 256, 1e-6f);
        
        // 测试2D输入
        NdArray input2D = NdArray.likeRandomN(Shape.of(2, 256));
        Variable output2D = rmsNorm.layerForward(new Variable(input2D));
        assert output2D.getValue().getShape().equals(input2D.getShape()) : "RMSNorm 2D输出形状错误";
        
        // 测试3D输入
        NdArray input3D = NdArray.likeRandomN(Shape.of(2, 5, 256));
        Variable output3D = rmsNorm.layerForward(new Variable(input3D));
        assert output3D.getValue().getShape().equals(input3D.getShape()) : "RMSNorm 3D输出形状错误";
        
        System.out.println("✓ RMSNorm测试成功");
    }
    
    /**
     * 测试SiLU层
     */
    private static void testSiLULayer() {
        SiLULayer siluLayer = new SiLULayer("test_silu", Shape.of(2, 5, 256));
        
        NdArray input = NdArray.likeRandomN(Shape.of(2, 5, 256));
        Variable output = siluLayer.layerForward(new Variable(input));
        assert output.getValue().getShape().equals(input.getShape()) : "SiLU输出形状错误";
        
        // 验证SiLU函数性质（正数输入应该产生正数输出）
        NdArray positiveInput = NdArray.ones(Shape.of(2, 2));
        Variable positiveOutput = siluLayer.layerForward(new Variable(positiveInput));
        NdArray flattenedArray = positiveOutput.getValue().flatten();
        
        // 获取底层数据
        float[] outputData;
        if (flattenedArray instanceof io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) {
            outputData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) flattenedArray).buffer;
        } else {
            // 对于非 NdArrayCpu 实现，使用 get 方法
            io.leavesfly.tinyai.ndarr.Shape shape = flattenedArray.getShape();
            outputData = new float[shape.size()];
            for (int i = 0; i < shape.size(); i++) {
                outputData[i] = flattenedArray.get(i);
            }
        }
        
        for (float val : outputData) {
            assert val > 0 : "SiLU对正数输入产生了非正数输出";
        }
        
        System.out.println("✓ SiLU测试成功");
    }
    
    /**
     * 测试旋转位置编码
     */
    private static void testRotaryPositionalEmbedding() {
        RotaryPositionalEmbedding rope = new RotaryPositionalEmbedding("test_rope", 64, 512, 10000.0f);
        
        NdArray input = NdArray.likeRandomN(Shape.of(2, 8, 5, 64)); // (batch, heads, seq_len, head_dim)
        Variable output = rope.layerForward(new Variable(input));
        assert output.getValue().getShape().equals(input.getShape()) : "RoPE输出形状错误";
        
        System.out.println("✓ RoPE测试成功");
    }
    
    /**
     * 打印序列（用于调试）
     */
    private static void printSequence(NdArray sequence) {
        Shape shape = sequence.getShape();
        if (shape.getDimNum() == 2) {
            int batchSize = shape.getDimension(0);
            int seqLen = shape.getDimension(1);
            
            for (int b = 0; b < batchSize; b++) {
                System.out.print("  Batch " + b + ": [");
                for (int s = 0; s < seqLen; s++) {
                    if (s > 0) System.out.print(", ");
                    System.out.print((int) sequence.get(b, s));
                }
                System.out.println("]");
            }
        } else {
            System.out.println("  形状: " + shape);
        }
    }
    
    /**
     * 性能基准测试
     */
    public static void benchmarkTest() {
        System.out.println("=== 性能基准测试 ===");
        
        Qwen3Model model = Qwen3Model.createTinyModel("benchmark");
        NdArray input = NdArray.of(new float[]{1, 2, 3, 4, 5});
        
        // 预热
        for (int i = 0; i < 10; i++) {
            model.forwardWithLogits(new Variable(input));
        }
        
        // 计时测试
        long startTime = System.currentTimeMillis();
        int iterations = 100;
        
        for (int i = 0; i < iterations; i++) {
            model.forwardWithLogits(new Variable(input));
        }
        
        long endTime = System.currentTimeMillis();
        double avgTime = (endTime - startTime) / (double) iterations;
        
        System.out.println("平均前向传播时间: " + avgTime + " ms");
        System.out.println("吞吐量: " + (1000.0 / avgTime) + " inferences/sec");
    }
}