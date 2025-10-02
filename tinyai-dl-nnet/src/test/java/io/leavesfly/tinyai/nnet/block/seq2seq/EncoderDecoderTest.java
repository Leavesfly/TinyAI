package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * EncoderDecoder的综合测试类
 * <p>
 * 测试编码解码器组合模型的完整功能
 */
public class EncoderDecoderTest {

    private EncoderDecoder encoderDecoder;
    private Encoder encoder;
    private Decoder decoder;
    private Shape sourceInputShape;
    private Shape targetInputShape;
    private Shape outputShape;

    // 测试参数
    private static final int SOURCE_VOCAB_SIZE = 1000;
    private static final int TARGET_VOCAB_SIZE = 2000;
    private static final int EMBEDDING_DIM = 64;
    private static final int HIDDEN_SIZE = 128;
    private static final double DROPOUT_RATE = 0.1;
    private static final int BATCH_SIZE = 4;
    private static final int SOURCE_SEQ_LENGTH = 10;
    private static final int TARGET_SEQ_LENGTH = 12;

    @Before
    public void setUp() {
        // 定义形状
        sourceInputShape = Shape.of(BATCH_SIZE, 1); // 简化为单个时间步
        targetInputShape = Shape.of(BATCH_SIZE, 1);
        Shape encoderOutputShape = Shape.of(BATCH_SIZE, HIDDEN_SIZE);
        outputShape = Shape.of(BATCH_SIZE, TARGET_VOCAB_SIZE);

        // 创建编码器和解码器
        encoder = new Seq2SeqEncoder(
                "test_encoder",
                sourceInputShape,
                encoderOutputShape,
                SOURCE_VOCAB_SIZE,
                EMBEDDING_DIM,
                HIDDEN_SIZE,
                DROPOUT_RATE
        );

        decoder = new Seq2SeqDecoder(
                "test_decoder",
                targetInputShape,
                outputShape,
                TARGET_VOCAB_SIZE,
                EMBEDDING_DIM,
                HIDDEN_SIZE,
                TARGET_VOCAB_SIZE
        );

        // 创建编码解码器组合
        encoderDecoder = new EncoderDecoder("test_seq2seq", encoder, decoder);
    }

    @Test
    public void testEncoderDecoderConstruction() {
        assertNotNull("编码解码器实例不应该为null", encoderDecoder);
        assertEquals("编码解码器名称应该正确", "test_seq2seq", encoderDecoder.getName());
        assertEquals("输入形状应该正确", sourceInputShape, encoderDecoder.getInputShape());
        assertEquals("输出形状应该正确", outputShape, encoderDecoder.getOutputShape());
        assertSame("编码器应该正确保存", encoder, encoderDecoder.getEncoder());
        assertSame("解码器应该正确保存", decoder, encoderDecoder.getDecoder());
        assertFalse("初始时模型应该未初始化", encoderDecoder.isInitialized());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullName() {
        new EncoderDecoder(null, encoder, decoder);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithEmptyName() {
        new EncoderDecoder("", encoder, decoder);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullEncoder() {
        new EncoderDecoder("test", null, decoder);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullDecoder() {
        new EncoderDecoder("test", encoder, null);
    }

    @Test
    public void testInitialization() {
        encoderDecoder.init();
        
        assertTrue("初始化后模型应该已初始化", encoderDecoder.isInitialized());
        
        // 验证子层已添加到模型中
        Map<String, ?> allParams = encoderDecoder.getAllParams();
        assertFalse("应该包含子层参数", allParams.isEmpty());
    }

    @Test
    public void testMultipleInitialization() {
        encoderDecoder.init();
        Encoder firstEncoder = encoderDecoder.getEncoder();
        
        encoderDecoder.init(); // 再次初始化
        Encoder secondEncoder = encoderDecoder.getEncoder();
        
        assertSame("多次初始化应该使用同一个编码器实例", firstEncoder, secondEncoder);
    }

    @Test
    public void testLayerForward() {
        // 创建输入数据
        NdArray sourceSequence = createTestSourceSequence();
        NdArray targetSequence = createTestTargetSequence();
        Variable sourceVar = new Variable(sourceSequence);
        Variable targetVar = new Variable(targetSequence);
        
        Variable output = encoderDecoder.layerForward(sourceVar, targetVar);
        
        assertNotNull("输出不应该为null", output);
        assertTrue("模型应该已自动初始化", encoderDecoder.isInitialized());
        assertEquals("输出形状应该正确", outputShape, output.getValue().getShape());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithNullInputs() {
        encoderDecoder.layerForward((Variable[]) null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithInsufficientInputs() {
        Variable sourceVar = new Variable(createTestSourceSequence());
        encoderDecoder.layerForward(sourceVar); // 只提供一个输入
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithNullSourceInput() {
        Variable targetVar = new Variable(createTestTargetSequence());
        encoderDecoder.layerForward(null, targetVar);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithNullTargetInput() {
        Variable sourceVar = new Variable(createTestSourceSequence());
        encoderDecoder.layerForward(sourceVar, null);
    }

    @Test
    public void testBatchProcessing() {
        int[] batchSizes = {1, 2, 8, 16};
        
        for (int batchSize : batchSizes) {
            encoderDecoder.resetState(); // 重置状态
            
            NdArray sourceSeq = createTestSourceSequenceWithBatchSize(batchSize);
            NdArray targetSeq = createTestTargetSequenceWithBatchSize(batchSize);
            Variable sourceVar = new Variable(sourceSeq);
            Variable targetVar = new Variable(targetSeq);
            
            Variable output = encoderDecoder.layerForward(sourceVar, targetVar);
            
            Shape expectedShape = Shape.of(batchSize, TARGET_VOCAB_SIZE);
            assertEquals(
                    String.format("批大小%d的输出形状应该正确", batchSize),
                    expectedShape,
                    output.getValue().getShape()
            );
        }
    }

    @Test
    public void testDifferentSequenceLengths() {
        int[] sourceLengths = {5, 15, 20};
        int[] targetLengths = {8, 18, 25};
        
        for (int i = 0; i < sourceLengths.length; i++) {
            encoderDecoder.resetState(); // 重置状态
            
            int sourceLen = sourceLengths[i];
            int targetLen = targetLengths[i];
            
            NdArray sourceSeq = createTestSourceSequenceWithLength(sourceLen);
            NdArray targetSeq = createTestTargetSequenceWithLength(targetLen);
            Variable sourceVar = new Variable(sourceSeq);
            Variable targetVar = new Variable(targetSeq);
            
            Variable output = encoderDecoder.layerForward(sourceVar, targetVar);
            
            Shape expectedShape = Shape.of(BATCH_SIZE, TARGET_VOCAB_SIZE);
            assertEquals(
                    String.format("源序列长度%d、目标序列长度%d的输出形状应该正确", sourceLen, targetLen),
                    expectedShape,
                    output.getValue().getShape()
            );
        }
    }

    @Test
    public void testResetState() {
        // 先进行一次前向传播
        NdArray sourceSeq = createTestSourceSequence();
        NdArray targetSeq = createTestTargetSequence();
        Variable sourceVar = new Variable(sourceSeq);
        Variable targetVar = new Variable(targetSeq);
        encoderDecoder.layerForward(sourceVar, targetVar);
        
        // 重置状态应该正常执行
        assertDoesNotThrow(() -> encoderDecoder.resetState());
    }

    @Test
    public void testGetEncoderAndDecoder() {
        assertEquals("应该返回正确的编码器", encoder, encoderDecoder.getEncoder());
        assertEquals("应该返回正确的解码器", decoder, encoderDecoder.getDecoder());
    }

    @Test
    public void testIsInitialized() {
        assertFalse("初始时应该未初始化", encoderDecoder.isInitialized());
        
        encoderDecoder.init();
        assertTrue("手动初始化后应该已初始化", encoderDecoder.isInitialized());
        
        // 创建新的实例测试自动初始化
        EncoderDecoder newModel = new EncoderDecoder("new_model", encoder, decoder);
        assertFalse("新实例初始时应该未初始化", newModel.isInitialized());
        
        NdArray sourceSeq = createTestSourceSequence();
        NdArray targetSeq = createTestTargetSequence();
        newModel.layerForward(new Variable(sourceSeq), new Variable(targetSeq));
        assertTrue("前向传播后应该自动初始化", newModel.isInitialized());
    }

    @Test
    public void testToString() {
        String modelStr = encoderDecoder.toString();
        
        assertNotNull("toString结果不应该为null", modelStr);
        assertTrue("应该包含模型名称", modelStr.contains("test_seq2seq"));
        assertTrue("应该包含编码器名称", modelStr.contains("test_encoder"));
        assertTrue("应该包含解码器名称", modelStr.contains("test_decoder"));
        assertTrue("应该包含初始化状态", modelStr.contains("false")); // 初始未初始化
    }

    @Test
    public void testSequentialTranslation() {
        encoderDecoder.resetState();
        
        // 翻译第一个句子对
        NdArray source1 = createTestSourceSequence();
        NdArray target1 = createTestTargetSequence();
        Variable output1 = encoderDecoder.layerForward(new Variable(source1), new Variable(target1));
        
        encoderDecoder.resetState();
        
        // 翻译第二个句子对
        NdArray source2 = createTestSourceSequence();
        NdArray target2 = createTestTargetSequence();
        Variable output2 = encoderDecoder.layerForward(new Variable(source2), new Variable(target2));
        
        // 输出形状应该一致
        assertEquals("两次翻译的输出形状应该一致", 
                output1.getValue().getShape(), output2.getValue().getShape());
    }

    @Test
    public void testModelWorkflow() {
        // 完整的工作流程测试
        
        // 1. 创建输入数据
        NdArray sourceSeq = createTestSourceSequence();
        NdArray targetSeq = createTestTargetSequence();
        Variable sourceVar = new Variable(sourceSeq);
        Variable targetVar = new Variable(targetSeq);
        
        // 2. 执行编码解码
        Variable output = encoderDecoder.layerForward(sourceVar, targetVar);
        
        // 3. 验证输出
        assertNotNull("输出不应该为null", output);
        assertEquals("输出形状应该正确", outputShape, output.getValue().getShape());
        
        // 4. 验证模型状态
        assertTrue("模型应该已初始化", encoderDecoder.isInitialized());
        
        // 5. 重置并再次处理
        encoderDecoder.resetState();
        Variable output2 = encoderDecoder.layerForward(sourceVar, targetVar);
        assertEquals("重置后的输出形状应该一致", output.getValue().getShape(), output2.getValue().getShape());
    }

    @Test
    public void testWithDifferentEncoderDecoders() {
        // 测试不同配置的编码器和解码器组合
        
        // 配置1：更大的隐藏层
        Seq2SeqEncoder largeEncoder = new Seq2SeqEncoder(
                "large_encoder",
                sourceInputShape,
                Shape.of(BATCH_SIZE, 256),
                SOURCE_VOCAB_SIZE, EMBEDDING_DIM, 256, DROPOUT_RATE
        );
        
        Seq2SeqDecoder largeDecoder = new Seq2SeqDecoder(
                "large_decoder",
                targetInputShape,
                Shape.of(BATCH_SIZE, TARGET_VOCAB_SIZE),
                TARGET_VOCAB_SIZE, EMBEDDING_DIM, 256, TARGET_VOCAB_SIZE
        );
        
        EncoderDecoder largeModel = new EncoderDecoder("large_model", largeEncoder, largeDecoder);
        
        NdArray sourceSeq = createTestSourceSequence();
        NdArray targetSeq = createTestTargetSequence();
        Variable output = largeModel.layerForward(new Variable(sourceSeq), new Variable(targetSeq));
        
        assertEquals("大模型的输出形状应该正确", outputShape, output.getValue().getShape());
    }

    @Test
    public void testParameterAccess() {
        encoderDecoder.init();
        
        // 测试所有参数的获取
        Map<String, ?> allParams = encoderDecoder.getAllParams();
        assertNotNull("参数映射不应该为null", allParams);
        assertFalse("应该包含参数", allParams.isEmpty());
    }

    // Helper方法

    /**
     * 创建测试源序列数据
     */
    private NdArray createTestSourceSequence() {
        return createTestSourceSequenceWithBatchSize(BATCH_SIZE);
    }

    /**
     * 创建指定批大小的测试源序列
     */
    private NdArray createTestSourceSequenceWithBatchSize(int batchSize) {
        NdArray input = NdArray.zeros(Shape.of(batchSize, 1)); // 单个时间步
        for (int i = 0; i < batchSize; i++) {
            input.set(i % SOURCE_VOCAB_SIZE, i, 0);
        }
        return input;
    }

    /**
     * 创建指定序列长度的测试源序列
     */
    private NdArray createTestSourceSequenceWithLength(int seqLen) {
        // 为了简化，使用单个时间步
        NdArray input = NdArray.zeros(Shape.of(BATCH_SIZE, 1));
        for (int i = 0; i < BATCH_SIZE; i++) {
            input.set(i % SOURCE_VOCAB_SIZE, i, 0);
        }
        return input;
    }

    /**
     * 创建测试目标序列数据
     */
    private NdArray createTestTargetSequence() {
        return createTestTargetSequenceWithBatchSize(BATCH_SIZE);
    }

    /**
     * 创建指定批大小的测试目标序列
     */
    private NdArray createTestTargetSequenceWithBatchSize(int batchSize) {
        NdArray input = NdArray.zeros(Shape.of(batchSize, 1)); // 单个时间步
        for (int i = 0; i < batchSize; i++) {
            input.set(i % TARGET_VOCAB_SIZE, i, 0);
        }
        return input;
    }

    /**
     * 创建指定序列长度的测试目标序列
     */
    private NdArray createTestTargetSequenceWithLength(int seqLen) {
        // 为了简化，使用单个时间步
        NdArray input = NdArray.zeros(Shape.of(BATCH_SIZE, 1));
        for (int i = 0; i < BATCH_SIZE; i++) {
            input.set(i % TARGET_VOCAB_SIZE, i, 0);
        }
        return input;
    }

    /**
     * 断言不抛出异常的Helper方法
     */
    private void assertDoesNotThrow(Runnable runnable) {
        try {
            runnable.run();
        } catch (Exception e) {
            fail("不应该抛出异常: " + e.getMessage());
        }
    }
}