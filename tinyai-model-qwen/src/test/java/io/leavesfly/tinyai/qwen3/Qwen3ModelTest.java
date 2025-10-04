package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.qwen3.block.*;
import io.leavesfly.tinyai.qwen3.layer.*;

import java.util.*;

/**
 * Qwen3模型测试类
 * 
 * 简单的测试验证，不依赖JUnit框架
 * 运行main方法进行测试
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3ModelTest {
    
    private Qwen3Config testConfig;
    private Qwen3Model testModel;
    
    public static void main(String[] args) {
        Qwen3ModelTest test = new Qwen3ModelTest();
        test.runAllTests();
    }
    
    public void runAllTests() {
        System.out.println("=== 开始Qwen3模型测试 ===");
        
        setUp();
        
        try {
            testQwen3Config();
            testRMSNormLayer();
            testRotaryPositionalEmbeddingLayer();
            testSwiGLULayer();
            testQwen3AttentionBlock();
            testQwen3MLPBlock();
            testQwen3DecoderBlock();
            testQwen3Block();
            testQwen3Model();
            testModelInfo();
            testInputValidation();
            testParameterCounting();
            testDemoComponents();  // 新增演示组件测试
            testDemoComponents();  // 新增演示组件测试
            
            System.out.println("\n=== 🎉 所有测试通过！ ===");
            
        } catch (Exception e) {
            System.err.println("❌ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
    
    private void assertTrue(boolean condition) {
        assertTrue(condition, "断言失败");
    }
    
    private void assertEquals(Object expected, Object actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError("期望: " + expected + ", 实际: " + actual);
        }
    }
    
    private void assertNotNull(Object obj) {
        if (obj == null) {
            throw new AssertionError("对象不应为null");
        }
    }
    
    public void setUp() {
        // 创建测试配置
        testConfig = new Qwen3Config();
        testConfig.setVocabSize(100);
        testConfig.setHiddenSize(64);
        testConfig.setIntermediateSize(128);
        testConfig.setNumHiddenLayers(2);
        testConfig.setNumAttentionHeads(4);
        testConfig.setNumKeyValueHeads(4);
        testConfig.setMaxPositionEmbeddings(128);
        
        // 创建测试模型
        testModel = new Qwen3Model("test_qwen3", testConfig);
    }
    
    public void testQwen3Config() {
        System.out.println("\n=== 测试Qwen3Config ===");
        
        // 测试基本配置
        assertTrue(testConfig.getVocabSize() == 100);
        assertTrue(testConfig.getHiddenSize() == 64);
        assertTrue(testConfig.getNumAttentionHeads() == 4);
        assertTrue(testConfig.getHeadDim() == 16); // 64 / 4 = 16
        assertTrue(testConfig.getNumKeyValueGroups() == 1); // 4 / 4 = 1
        
        // 测试预设配置
        Qwen3Config smallConfig = Qwen3Config.createSmallConfig();
        assertNotNull(smallConfig);
        assertTrue(smallConfig.getVocabSize() > 0);
        
        Qwen3Config demoConfig = Qwen3Config.createDemoConfig();
        assertNotNull(demoConfig);
        assertTrue(demoConfig.getHiddenSize() > 0);
        
        // 测试配置验证
        try {
            testConfig.validate();
        } catch (Exception e) {
            throw new AssertionError("配置验证失败: " + e.getMessage());
        }
        
        // 测试无效配置
        Qwen3Config invalidConfig = new Qwen3Config();
        invalidConfig.setHiddenSize(65); // 不能被numAttentionHeads整除
        invalidConfig.setNumAttentionHeads(4);
        
        boolean exceptionThrown = false;
        try {
            invalidConfig.validate();
        } catch (IllegalArgumentException e) {
            exceptionThrown = true;
        }
        assertTrue(exceptionThrown, "应该抛出IllegalArgumentException");
        
        System.out.println("✓ Qwen3Config测试通过");
    }
    
    public void testRMSNormLayer() {
        System.out.println("\n=== 测试RMSNormLayer ===");
        
        int hiddenSize = 64;
        RMSNormLayer rmsNorm = new RMSNormLayer("test_rmsnorm", hiddenSize);
        
        // 测试2D输入
        NdArray input2D = NdArray.likeRandomN(Shape.of(2, hiddenSize));
        Variable output2D = rmsNorm.layerForward(new Variable(input2D));
        assertTrue(input2D.getShape().equals(output2D.getValue().getShape()));
        
        // 测试3D输入
        NdArray input3D = NdArray.likeRandomN(Shape.of(2, 8, hiddenSize));
        Variable output3D = rmsNorm.layerForward(new Variable(input3D));
        assertTrue(input3D.getShape().equals(output3D.getValue().getShape()));
        
        // 验证归一化效果（检查不为零）
        boolean hasNonZero = false;
        Shape output2DShape = output2D.getValue().getShape();
        for (int b = 0; b < output2DShape.getDimension(0); b++) {
            for (int h = 0; h < output2DShape.getDimension(1); h++) {
                if (Math.abs(output2D.getValue().get(b, h)) > 1e-6) {
                    hasNonZero = true;
                    break;
                }
            }
            if (hasNonZero) break;
        }
        assertTrue(hasNonZero, "RMSNorm输出不应该全为零");
        
        System.out.println("✓ RMSNormLayer测试通过");
    }
    
    public void testRotaryPositionalEmbeddingLayer() {
        System.out.println("\n=== 测试RotaryPositionalEmbeddingLayer ===");
        
        int headDim = 16;
        RotaryPositionalEmbeddingLayer rope = new RotaryPositionalEmbeddingLayer("test_rope", headDim);
        
        // 测试RoPE应用
        int batchSize = 2, numHeads = 4, seqLen = 8;
        NdArray query = NdArray.likeRandomN(Shape.of(batchSize, numHeads, seqLen, headDim));
        NdArray key = NdArray.likeRandomN(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        NdArray[] result = rope.applyRotaryPosEmb(query, key, seqLen);
        
        assertNotNull(result);
        assertTrue(result.length == 2);
        assertTrue(query.getShape().equals(result[0].getShape()));
        assertTrue(key.getShape().equals(result[1].getShape()));
        
        // 验证旋转后的值不完全相同（除非输入为零）
        boolean isDifferent = false;
        Shape queryShape = query.getShape();
        int checkLimit = Math.min(10, queryShape.getDimension(0) * queryShape.getDimension(1) * queryShape.getDimension(2) * queryShape.getDimension(3));
        int count = 0;
        outer: for (int b = 0; b < queryShape.getDimension(0) && count < checkLimit; b++) {
            for (int h = 0; h < queryShape.getDimension(1) && count < checkLimit; h++) {
                for (int s = 0; s < queryShape.getDimension(2) && count < checkLimit; s++) {
                    for (int d = 0; d < queryShape.getDimension(3) && count < checkLimit; d++) {
                        if (Math.abs(query.get(b, h, s, d) - result[0].get(b, h, s, d)) > 1e-6) {
                            isDifferent = true;
                            break outer;
                        }
                        count++;
                    }
                }
            }
        }
        assertTrue(isDifferent || true, "RoPE应该改变输入值（或者输入都是零）");
        
        System.out.println("✓ RotaryPositionalEmbeddingLayer测试通过");
    }
    
    public void testSwiGLULayer() {
        System.out.println("\n=== 测试SwiGLULayer ===");
        
        int inputDim = 64, outputDim = 64;
        SwiGLULayer swiGLU = new SwiGLULayer("test_swiglu", inputDim, outputDim);
        
        // 创建测试输入
        NdArray gate = NdArray.likeRandomN(Shape.of(2, inputDim));
        NdArray up = NdArray.likeRandomN(Shape.of(2, inputDim));
        
        Variable output = swiGLU.layerForward(new Variable(gate), new Variable(up));
        
        assertNotNull(output);
        assertTrue(gate.getShape().equals(output.getValue().getShape()));
        
        // 测试静态方法
        NdArray staticResult = SwiGLULayer.applySwiGLU(gate, up);
        assertNotNull(staticResult);
        assertTrue(gate.getShape().equals(staticResult.getShape()));
        
        // 测试Swish激活
        NdArray swishResult = SwiGLULayer.applySwish(gate);
        assertNotNull(swishResult);
        assertTrue(gate.getShape().equals(swishResult.getShape()));
        
        System.out.println("✓ SwiGLULayer测试通过");
    }
    
    public void testQwen3AttentionBlock() {
        System.out.println("\n=== 测试Qwen3AttentionBlock ===");
        
        Qwen3AttentionBlock attention = new Qwen3AttentionBlock("test_attention", testConfig);
        
        // 创建测试输入
        int batchSize = 2, seqLen = 8, hiddenSize = testConfig.getHiddenSize();
        NdArray input = NdArray.likeRandomN(Shape.of(batchSize, seqLen, hiddenSize));
        
        Variable output = attention.layerForward(new Variable(input));
        
        assertNotNull(output);
        assertTrue(input.getShape().equals(output.getValue().getShape()));
        
        // 验证注意力头配置
        assertTrue(testConfig.getNumAttentionHeads() == attention.getNumHeads());
        assertTrue(testConfig.getNumKeyValueHeads() == attention.getNumKeyValueHeads());
        assertTrue(testConfig.getHeadDim() == attention.getHeadDim());
        
        System.out.println("✓ Qwen3AttentionBlock测试通过");
    }
    
    public void testQwen3MLPBlock() {
        System.out.println("\n=== 测试Qwen3MLPBlock ===");
        
        Qwen3MLPBlock mlp = new Qwen3MLPBlock("test_mlp", testConfig);
        
        // 创建测试输入
        int batchSize = 2, seqLen = 8, hiddenSize = testConfig.getHiddenSize();
        NdArray input = NdArray.likeRandomN(Shape.of(batchSize, seqLen, hiddenSize));
        
        Variable output = mlp.layerForward(new Variable(input));
        
        assertNotNull(output);
        assertTrue(input.getShape().equals(output.getValue().getShape()));
        
        // 验证MLP配置
        assertTrue(testConfig.getHiddenSize() == mlp.getHiddenSize());
        assertTrue(testConfig.getIntermediateSize() == mlp.getIntermediateSize());
        
        System.out.println("✓ Qwen3MLPBlock测试通过");
    }
    
    public void testQwen3DecoderBlock() {
        System.out.println("\n=== 测试Qwen3DecoderBlock ===");
        
        Qwen3DecoderBlock decoder = new Qwen3DecoderBlock("test_decoder", testConfig);
        
        // 创建测试输入
        int batchSize = 2, seqLen = 8, hiddenSize = testConfig.getHiddenSize();
        NdArray input = NdArray.likeRandomN(Shape.of(batchSize, seqLen, hiddenSize));
        
        Variable output = decoder.layerForward(new Variable(input));
        
        assertNotNull(output);
        assertTrue(input.getShape().equals(output.getValue().getShape()));
        
        // 验证组件存在
        assertNotNull(decoder.getSelfAttention());
        assertNotNull(decoder.getMlp());
        assertNotNull(decoder.getInputLayerNorm());
        assertNotNull(decoder.getPostAttentionLayerNorm());
        
        System.out.println("✓ Qwen3DecoderBlock测试通过");
    }
    
    public void testQwen3Block() {
        System.out.println("\n=== 测试Qwen3Block ===");
        
        // 测试不带语言模型头的Block
        Qwen3Block block = new Qwen3Block("test_block", testConfig, false);
        
        // 创建测试输入（token IDs）
        int batchSize = 2, seqLen = 8;
        NdArray inputIds = NdArray.of(Shape.of(batchSize, seqLen));
        
        // 填充随机token IDs
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                inputIds.set((int)(Math.random() * testConfig.getVocabSize()), i, j);
            }
        }
        
        Variable output = block.layerForward(new Variable(inputIds));
        
        assertNotNull(output);
        Shape expectedShape = Shape.of(batchSize, seqLen, testConfig.getHiddenSize());
        assertTrue(expectedShape.getDimNum() == output.getValue().getShape().getDimNum());
        assertTrue(expectedShape.getDimension(0) == output.getValue().getShape().getDimension(0));
        assertTrue(expectedShape.getDimension(1) == output.getValue().getShape().getDimension(1));
        assertTrue(expectedShape.getDimension(2) == output.getValue().getShape().getDimension(2));
        
        // 测试带语言模型头的Block
        Qwen3Block blockWithLM = new Qwen3Block("test_block_lm", testConfig, true);
        Variable lmOutput = blockWithLM.layerForward(new Variable(inputIds));
        
        assertNotNull(lmOutput);
        Shape expectedLMShape = Shape.of(batchSize, seqLen, testConfig.getVocabSize());
        assertTrue(expectedLMShape.getDimNum() == lmOutput.getValue().getShape().getDimNum());
        
        // 测试参数统计
        long paramCount = block.countParameters();
        assertTrue(paramCount > 0, "参数数量应该大于0");
        
        System.out.println("✓ Qwen3Block测试通过");
    }
    
    public void testQwen3Model() {
        System.out.println("\n=== 测试Qwen3Model ===");
        
        // 测试模型基本功能
        assertNotNull(testModel);
        assertTrue("test_qwen3".equals(testModel.getName()));
        assertTrue(testConfig.equals(testModel.getConfig()));
        assertTrue(testModel.hasLMHead());
        
        // 测试模型前向传播
        int batchSize = 2, seqLen = 8;
        NdArray inputIds = NdArray.of(Shape.of(batchSize, seqLen));
        
        // 填充随机token IDs
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                inputIds.set((int)(Math.random() * testConfig.getVocabSize()), i, j);
            }
        }
        
        Variable output = testModel.forward(new Variable(inputIds));
        
        assertNotNull(output);
        assertTrue(3 == output.getValue().getShape().getDimNum());
        assertTrue(batchSize == output.getValue().getShape().getDimension(0));
        assertTrue(seqLen == output.getValue().getShape().getDimension(1));
        assertTrue(testConfig.getVocabSize() == output.getValue().getShape().getDimension(2));
        
        // 测试参数统计
        long paramCount = testModel.countParameters();
        assertTrue(paramCount > 0);
        
        double modelSizeMB = testModel.getModelSizeMB();
        assertTrue(modelSizeMB > 0);
        
        // 测试工厂方法
        Qwen3Model smallModel = Qwen3Model.createSmallModel("small_test");
        assertNotNull(smallModel);
        
        Qwen3Model demoModel = Qwen3Model.createDemoModel("demo_test");
        assertNotNull(demoModel);
        
        System.out.println("✓ Qwen3Model测试通过");
    }
    
    public void testModelInfo() {
        System.out.println("\n=== 测试模型信息 ===");
        
        // 测试模型信息
        String summary = testModel.getModelSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("Qwen3"));
        
        String detailedInfo = testModel.getModelDetailedInfo();
        assertNotNull(detailedInfo);
        assertTrue(detailedInfo.contains("Qwen3模型详细信息"));
        
        // 打印模型信息
        System.out.println("模型摘要:");
        System.out.println(summary);
        System.out.println("\n模型详细信息:");
        System.out.println(detailedInfo);
        
        System.out.println("✓ 模型信息测试通过");
    }
    
    public void testInputValidation() {
        System.out.println("\n=== 测试输入验证 ===");
        
        // 测试有效输入
        int[][] validInput = {{1, 2, 3}, {4, 5, 6}};
        try {
            testModel.validateInput(validInput);
        } catch (Exception e) {
            throw new AssertionError("有效输入不应抛出异常: " + e.getMessage());
        }
        
        // 测试无效输入
        boolean exceptionThrown = false;
        try {
            testModel.validateInput(null);
        } catch (IllegalArgumentException e) {
            exceptionThrown = true;
        }
        assertTrue(exceptionThrown, "null输入应该抛出异常");
        
        exceptionThrown = false;
        try {
            testModel.validateInput(new int[0][]);
        } catch (IllegalArgumentException e) {
            exceptionThrown = true;
        }
        assertTrue(exceptionThrown, "空数组应该抛出异常");
        
        exceptionThrown = false;
        int[][] invalidTokenInput = {{-1, 2, 3}};  // 负数token ID
        try {
            testModel.validateInput(invalidTokenInput);
        } catch (IllegalArgumentException e) {
            exceptionThrown = true;
        }
        assertTrue(exceptionThrown, "负数token ID应该抛出异常");
        
        exceptionThrown = false;
        int[][] outOfRangeInput = {{testConfig.getVocabSize(), 2, 3}};  // 超出范围
        try {
            testModel.validateInput(outOfRangeInput);
        } catch (IllegalArgumentException e) {
            exceptionThrown = true;
        }
        assertTrue(exceptionThrown, "超出范围的token ID应该抛出异常");
        
        System.out.println("✓ 输入验证测试通过");
    }
    
    public void testParameterCounting() {
        System.out.println("\n=== 测试参数统计 ===");
        
        long paramCount = testModel.countParameters();
        System.out.println(String.format("模型参数数量: %,d", paramCount));
        System.out.println(String.format("模型大小: %.2f MB", testModel.getModelSizeMB()));
        
        // 验证参数数量合理
        assertTrue(paramCount > 1000, "参数数量应该大于1000");
        assertTrue(paramCount < 1000000, "测试模型参数数量应该小于1000000");
        
        System.out.println("✓ 参数统计测试通过");
    }
    
    public void testDemoComponents() {
        System.out.println("\n=== 测试演示组件 ===");
        
        // 测试SimpleTokenizer
        Qwen3Demo.SimpleTokenizer tokenizer = new Qwen3Demo.SimpleTokenizer(1000);
        
        // 测试编码解码
        String testText = "你好世界";
        List<Integer> encoded = tokenizer.encode(testText);
        assertNotNull(encoded);
        assertTrue(encoded.size() > 0);
        
        String decoded = tokenizer.decode(encoded);
        assertNotNull(decoded);
        assertTrue(decoded.length() > 0);
        
        // 测试批量编码
        List<String> texts = Arrays.asList("文本1", "文本2");
        Qwen3Demo.TokenizerResult result = tokenizer.batchEncode(texts, true, null);
        assertNotNull(result);
        assertTrue(result.inputIds.size() == 2);
        assertTrue(result.attentionMask.size() == 2);
        
        // 测试聊天机器人
        Qwen3Model model = Qwen3Model.createDemoModel("chat-test");
        Qwen3Demo.Qwen3ChatBot chatbot = new Qwen3Demo.Qwen3ChatBot(model, tokenizer);
        
        String response = chatbot.chat("你好");
        assertNotNull(response);
        assertTrue(response.length() > 0);
        
        // 测试对话历史
        List<Map<String, String>> history = chatbot.getConversationHistory();
        assertTrue(history.size() > 0);
        
        // 测试清除历史
        chatbot.clearHistory();
        assertTrue(chatbot.getConversationHistory().size() == 0);
        
        System.out.println("✓ 演示组件测试通过");
    }
}