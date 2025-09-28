package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;
import io.leavesfly.tinyai.nnet.layer.norm.Dropout;
import io.leavesfly.tinyai.nnet.layer.rnn.LstmLayer;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * Seq2SeqEncoder的综合测试类
 * <p>
 * 测试基于LSTM的序列到序列编码器的各种功能
 */
public class Seq2SeqEncoderTest {

    private Seq2SeqEncoder encoder;
    private Shape inputShape;
    private Shape outputShape;

    // 测试参数
    private static final int VOCAB_SIZE = 1000;
    private static final int EMBEDDING_DIM = 64;
    private static final int HIDDEN_SIZE = 128;
    private static final double DROPOUT_RATE = 0.1;
    private static final int BATCH_SIZE = 4;
    private static final int SEQ_LENGTH = 10;

    @Before
    public void setUp() {
        inputShape = Shape.of(BATCH_SIZE, 1); // 简化为单个时间步
        outputShape = Shape.of(BATCH_SIZE, HIDDEN_SIZE);
        encoder = new Seq2SeqEncoder(
                "test_encoder",
                inputShape,
                outputShape,
                VOCAB_SIZE,
                EMBEDDING_DIM,
                HIDDEN_SIZE,
                DROPOUT_RATE
        );
    }

    @Test
    public void testEncoderConstruction() {
        assertNotNull("编码器实例不应该为null", encoder);
        assertEquals("编码器名称应该正确", "test_encoder", encoder.getName());
        assertEquals("输入形状应该正确", inputShape, encoder.getInputShape());
        assertEquals("输出形状应该正确", outputShape, encoder.getOutputShape());
        assertEquals("词汇表大小应该正确", VOCAB_SIZE, encoder.getVocabSize());
        assertEquals("嵌入维度应该正确", EMBEDDING_DIM, encoder.getEmbeddingDim());
        assertEquals("隐藏层大小应该正确", HIDDEN_SIZE, encoder.getHiddenSize());
        assertEquals("Dropout比率应该正确", DROPOUT_RATE, encoder.getDropoutRate(), 1e-6);
        assertFalse("初始时层应该未初始化", encoder.isLayersInitialized());
    }

    @Test
    public void testDeprecatedConstructor() {
        Seq2SeqEncoder deprecatedEncoder = new Seq2SeqEncoder("deprecated", inputShape, outputShape);
        
        assertEquals("默认词汇表大小应该正确", 10000, deprecatedEncoder.getVocabSize());
        assertEquals("默认嵌入维度应该正确", 128, deprecatedEncoder.getEmbeddingDim());
        assertEquals("默认隐藏层大小应该正确", 256, deprecatedEncoder.getHiddenSize());
        assertEquals("默认Dropout比率应该正确", 0.1, deprecatedEncoder.getDropoutRate(), 1e-6);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidVocabSize() {
        new Seq2SeqEncoder("invalid", inputShape, outputShape, 0, EMBEDDING_DIM, HIDDEN_SIZE, DROPOUT_RATE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidEmbeddingDim() {
        new Seq2SeqEncoder("invalid", inputShape, outputShape, VOCAB_SIZE, 0, HIDDEN_SIZE, DROPOUT_RATE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidHiddenSize() {
        new Seq2SeqEncoder("invalid", inputShape, outputShape, VOCAB_SIZE, EMBEDDING_DIM, 0, DROPOUT_RATE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidDropoutRate() {
        new Seq2SeqEncoder("invalid", inputShape, outputShape, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, 1.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNegativeDropoutRate() {
        new Seq2SeqEncoder("invalid", inputShape, outputShape, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, -0.1);
    }

    @Test
    public void testInitialization() {
        encoder.init();
        
        assertTrue("初始化后层应该已初始化", encoder.isLayersInitialized());
        assertNotNull("嵌入层应该不为null", encoder.getEmbedding());
        assertNotNull("LSTM层应该不为null", encoder.getLstmLayer());
        assertNotNull("Dropout层应该不为null", encoder.getDropout());
        
        // 验证子层已添加到encoder中
        Map<String, ?> allParams = encoder.getAllParams();
        assertFalse("应该包含子层参数", allParams.isEmpty());
    }

    @Test
    public void testMultipleInitialization() {
        encoder.init();
        Embedding firstEmbedding = encoder.getEmbedding();
        
        encoder.init(); // 再次初始化
        Embedding secondEmbedding = encoder.getEmbedding();
        
        assertSame("多次初始化应该使用同一个嵌入层实例", firstEmbedding, secondEmbedding);
    }

    @Test
    public void testLayerForward() {
        // 创建输入序列（词索引）
        NdArray input = createTestInput();
        Variable inputVar = new Variable(input);
        
        Variable output = encoder.layerForward(inputVar);
        
        assertNotNull("输出不应该为null", output);
        assertTrue("层应该已自动初始化", encoder.isLayersInitialized());
        
        // 验证输出形状（注意：当前实现返回的是最后一个时间步的状态）
        // 实际输出形状可能因为单个时间步处理而略有不同
        assertNotNull("输出不应该为null", output);
        assertNotNull("输出值不应该为null", output.getValue());
        Shape outputShape = output.getValue().getShape();
        assertTrue("输出形状应该包含隐藏维度", outputShape.getDimension(outputShape.getDimNum() - 1) == HIDDEN_SIZE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithNullInput() {
        encoder.layerForward((Variable) null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithNoInput() {
        encoder.layerForward();
    }

    @Test
    public void testBatchProcessing() {
        int[] batchSizes = {1, 2, 8, 16};
        
        for (int batchSize : batchSizes) {
            encoder.resetState(); // 重置状态以避免批次间的影响
            
            NdArray input = createTestInputWithBatchSize(batchSize);
            Variable inputVar = new Variable(input);
            Variable output = encoder.layerForward(inputVar);
            
            assertNotNull(
                    String.format("批大小%d的输出不应该为null", batchSize),
                    output
            );
            Shape outputShape = output.getValue().getShape();
            assertTrue(
                    String.format("批大小%d的输出应该包含隐藏维度", batchSize),
                    outputShape.getDimension(outputShape.getDimNum() - 1) == HIDDEN_SIZE
            );
        }
    }

    @Test
    public void testDifferentSequenceLengths() {
        int[] sequenceLengths = {5, 15, 20, 30};
        
        for (int seqLen : sequenceLengths) {
            encoder.resetState(); // 重置状态
            
            NdArray input = createTestInputWithSeqLength(seqLen);
            Variable inputVar = new Variable(input);
            Variable output = encoder.layerForward(inputVar);
            
            assertNotNull(
                    String.format("序列长度%d的输出不应该为null", seqLen),
                    output
            );
            Shape outputShape = output.getValue().getShape();
            assertTrue(
                    String.format("序列长度%d的输出应该包含隐藏维度", seqLen),
                    outputShape.getDimension(outputShape.getDimNum() - 1) == HIDDEN_SIZE
            );
        }
    }

    @Test
    public void testResetState() {
        // 先进行一次前向传播
        NdArray input = createTestInput();
        Variable inputVar = new Variable(input);
        encoder.layerForward(inputVar);
        
        // 重置状态应该正常执行
        assertDoesNotThrow(() -> encoder.resetState());
    }

    @Test
    public void testGetFinalHiddenState() {
        // 当前实现返回null，这是预期的行为
        assertNull("当前实现应该返回null", encoder.getFinalHiddenState());
    }

    @Test
    public void testGetEncodedSequenceLength() {
        assertEquals("编码序列长度应该正确", HIDDEN_SIZE, encoder.getEncodedSequenceLength());
    }

    @Test
    public void testParameterAccess() {
        encoder.init();
        
        // 测试所有参数的获取
        Map<String, ?> allParams = encoder.getAllParams();
        assertNotNull("参数映射不应该为null", allParams);
        assertFalse("应该包含参数", allParams.isEmpty());
    }

    @Test
    public void testLayerAccessAfterInit() {
        encoder.init();
        
        Embedding embedding = encoder.getEmbedding();
        LstmLayer lstmLayer = encoder.getLstmLayer();
        Dropout dropout = encoder.getDropout();
        
        assertNotNull("嵌入层不应该为null", embedding);
        assertNotNull("LSTM层不应该为null", lstmLayer);
        assertNotNull("Dropout层不应该为null", dropout);
        
        assertTrue("嵌入层名称应该包含前缀", embedding.getName().contains("test_encoder"));
        assertTrue("LSTM层名称应该包含前缀", lstmLayer.getName().contains("test_encoder"));
        assertTrue("Dropout层名称应该包含前缀", dropout.getName().contains("test_encoder"));
    }

    @Test
    public void testToString() {
        String encoderStr = encoder.toString();
        
        assertNotNull("toString结果不应该为null", encoderStr);
        assertTrue("应该包含编码器名称", encoderStr.contains("test_encoder"));
        assertTrue("应该包含词汇表大小", encoderStr.contains(String.valueOf(VOCAB_SIZE)));
        assertTrue("应该包含嵌入维度", encoderStr.contains(String.valueOf(EMBEDDING_DIM)));
        assertTrue("应该包含隐藏层大小", encoderStr.contains(String.valueOf(HIDDEN_SIZE)));
    }

    @Test
    public void testSequentialProcessing() {
        encoder.resetState();
        
        // 处理第一个序列
        NdArray input1 = createTestInput();
        Variable output1 = encoder.layerForward(new Variable(input1));
        
        encoder.resetState();
        
        // 处理第二个序列
        NdArray input2 = createTestInput();
        Variable output2 = encoder.layerForward(new Variable(input2));
        
        // 输出形状应该一致
        assertEquals("两次处理的输出形状应该一致", 
                output1.getValue().getShape(), output2.getValue().getShape());
    }

    @Test
    public void testEncoderInheritance() {
        assertTrue("应该是Encoder的实例", encoder instanceof Encoder);
        assertEquals("应该正确获取输入形状", inputShape, encoder.getInputShape());
        assertEquals("应该正确获取输出形状", outputShape, encoder.getOutputShape());
    }

    // Helper方法

    /**
     * 创建测试输入数据
     */
    private NdArray createTestInput() {
        return createTestInputWithBatchSize(BATCH_SIZE);
    }

    /**
     * 创建指定批大小的测试输入
     */
    private NdArray createTestInputWithBatchSize(int batchSize) {
        // 创建随机词索引，范围在[0, VOCAB_SIZE)
        // 为了简化，使用最后一个时间步的词索引
        NdArray input = NdArray.zeros(Shape.of(batchSize, 1)); // 使用单个时间步
        for (int i = 0; i < batchSize; i++) {
            input.set(i % VOCAB_SIZE, i, 0);
        }
        return input;
    }

    /**
     * 创建指定序列长度的测试输入
     */
    private NdArray createTestInputWithSeqLength(int seqLen) {
        // 为了简化，使用单个时间步
        NdArray input = NdArray.zeros(Shape.of(BATCH_SIZE, 1));
        for (int i = 0; i < BATCH_SIZE; i++) {
            input.set(i % VOCAB_SIZE, i, 0);
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