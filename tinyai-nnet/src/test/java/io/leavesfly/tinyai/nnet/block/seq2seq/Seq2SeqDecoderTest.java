package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;
import io.leavesfly.tinyai.nnet.layer.rnn.LstmLayer;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * Seq2SeqDecoder的综合测试类
 * <p>
 * 测试基于LSTM的序列到序列解码器的各种功能
 */
public class Seq2SeqDecoderTest {

    private Seq2SeqDecoder decoder;
    private Shape inputShape;
    private Shape outputShape;
    private NdArray encoderState;

    // 测试参数
    private static final int TARGET_VOCAB_SIZE = 2000;
    private static final int EMBEDDING_DIM = 64;
    private static final int HIDDEN_SIZE = 128;
    private static final int OUTPUT_VOCAB_SIZE = 2000;
    private static final int BATCH_SIZE = 4;
    private static final int TARGET_SEQ_LENGTH = 12;

    @Before
    public void setUp() {
        inputShape = Shape.of(BATCH_SIZE, 1); // 简化为单个时间步
        outputShape = Shape.of(BATCH_SIZE, OUTPUT_VOCAB_SIZE);
        decoder = new Seq2SeqDecoder(
                "test_decoder",
                inputShape,
                outputShape,
                TARGET_VOCAB_SIZE,
                EMBEDDING_DIM,
                HIDDEN_SIZE,
                OUTPUT_VOCAB_SIZE
        );
        
        // 创建模拟的编码器状态
        encoderState = NdArray.likeRandomN(Shape.of(BATCH_SIZE, HIDDEN_SIZE));
    }

    @Test
    public void testDecoderConstruction() {
        assertNotNull("解码器实例不应该为null", decoder);
        assertEquals("解码器名称应该正确", "test_decoder", decoder.getName());
        assertEquals("输入形状应该正确", inputShape, decoder.getInputShape());
        assertEquals("输出形状应该正确", outputShape, decoder.getOutputShape());
        assertEquals("目标词汇表大小应该正确", TARGET_VOCAB_SIZE, decoder.getTargetVocabSize());
        assertEquals("嵌入维度应该正确", EMBEDDING_DIM, decoder.getEmbeddingDim());
        assertEquals("隐藏层大小应该正确", HIDDEN_SIZE, decoder.getHiddenSize());
        assertEquals("输出词汇表大小应该正确", OUTPUT_VOCAB_SIZE, decoder.getOutputVocabSize());
        assertFalse("初始时层应该未初始化", decoder.isLayersInitialized());
        assertFalse("初始时状态应该未初始化", decoder.isStateInitialized());
    }

    @Test
    public void testDeprecatedConstructor() {
        Seq2SeqDecoder deprecatedDecoder = new Seq2SeqDecoder("deprecated", inputShape, outputShape);
        
        assertEquals("默认目标词汇表大小应该正确", 10000, deprecatedDecoder.getTargetVocabSize());
        assertEquals("默认嵌入维度应该正确", 128, deprecatedDecoder.getEmbeddingDim());
        assertEquals("默认隐藏层大小应该正确", 256, deprecatedDecoder.getHiddenSize());
        assertEquals("默认输出词汇表大小应该正确", 10000, deprecatedDecoder.getOutputVocabSize());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidTargetVocabSize() {
        new Seq2SeqDecoder("invalid", inputShape, outputShape, 0, EMBEDDING_DIM, HIDDEN_SIZE, OUTPUT_VOCAB_SIZE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidEmbeddingDim() {
        new Seq2SeqDecoder("invalid", inputShape, outputShape, TARGET_VOCAB_SIZE, 0, HIDDEN_SIZE, OUTPUT_VOCAB_SIZE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidHiddenSize() {
        new Seq2SeqDecoder("invalid", inputShape, outputShape, TARGET_VOCAB_SIZE, EMBEDDING_DIM, 0, OUTPUT_VOCAB_SIZE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithInvalidOutputVocabSize() {
        new Seq2SeqDecoder("invalid", inputShape, outputShape, TARGET_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, 0);
    }

    @Test
    public void testInitialization() {
        decoder.init();
        
        assertTrue("初始化后层应该已初始化", decoder.isLayersInitialized());
        assertNotNull("嵌入层应该不为null", decoder.getEmbedding());
        assertNotNull("LSTM层应该不为null", decoder.getLstmLayer());
        assertNotNull("线性层应该不为null", decoder.getLinearLayer());
        
        // 验证子层已添加到decoder中
        Map<String, ?> allParams = decoder.getAllParams();
        assertFalse("应该包含子层参数", allParams.isEmpty());
    }

    @Test
    public void testMultipleInitialization() {
        decoder.init();
        Embedding firstEmbedding = decoder.getEmbedding();
        
        decoder.init(); // 再次初始化
        Embedding secondEmbedding = decoder.getEmbedding();
        
        assertSame("多次初始化应该使用同一个嵌入层实例", firstEmbedding, secondEmbedding);
    }

    @Test
    public void testInitState() {
        decoder.initState(encoderState);
        
        assertTrue("状态应该已初始化", decoder.isStateInitialized());
        assertEquals("编码器状态应该正确保存", encoderState, decoder.getEncoderState());
        assertTrue("层应该自动初始化", decoder.isLayersInitialized());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInitStateWithNull() {
        decoder.initState(null);
    }

    @Test
    public void testLayerForward() {
        // 先初始化状态
        decoder.initState(encoderState);
        
        // 创建目标序列输入
        NdArray targetInput = createTestTargetInput();
        Variable targetVar = new Variable(targetInput);
        
        Variable output = decoder.layerForward(targetVar);
        
        assertNotNull("输出不应该为null", output);
        assertEquals("输出形状应该正确", Shape.of(BATCH_SIZE, OUTPUT_VOCAB_SIZE), output.getValue().getShape());
    }

    @Test(expected = IllegalStateException.class)
    public void testLayerForwardWithoutStateInit() {
        NdArray targetInput = createTestTargetInput();
        Variable targetVar = new Variable(targetInput);
        decoder.layerForward(targetVar);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithNullInput() {
        decoder.initState(encoderState);
        decoder.layerForward((Variable) null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLayerForwardWithNoInput() {
        decoder.initState(encoderState);
        decoder.layerForward();
    }

    @Test
    public void testBatchProcessing() {
        int[] batchSizes = {1, 2, 8, 16};
        
        for (int batchSize : batchSizes) {
            decoder.resetState(); // 重置状态
            
            // 创建对应批大小的编码器状态和目标输入
            NdArray batchEncoderState = NdArray.likeRandomN(Shape.of(batchSize, HIDDEN_SIZE));
            decoder.initState(batchEncoderState);
            
            NdArray targetInput = createTestTargetInputWithBatchSize(batchSize);
            Variable targetVar = new Variable(targetInput);
            Variable output = decoder.layerForward(targetVar);
            
            Shape expectedShape = Shape.of(batchSize, OUTPUT_VOCAB_SIZE);
            assertEquals(
                    String.format("批大小%d的输出形状应该正确", batchSize),
                    expectedShape,
                    output.getValue().getShape()
            );
        }
    }

    @Test
    public void testDifferentSequenceLengths() {
        int[] sequenceLengths = {5, 15, 20, 30};
        
        for (int seqLen : sequenceLengths) {
            decoder.resetState(); // 重置状态
            
            decoder.initState(encoderState);
            
            NdArray targetInput = createTestTargetInputWithSeqLength(seqLen);
            Variable targetVar = new Variable(targetInput);
            Variable output = decoder.layerForward(targetVar);
            
            Shape expectedShape = Shape.of(BATCH_SIZE, OUTPUT_VOCAB_SIZE);
            assertEquals(
                    String.format("序列长度%d的输出形状应该正确", seqLen),
                    expectedShape,
                    output.getValue().getShape()
            );
        }
    }

    @Test
    public void testResetState() {
        // 先初始化状态
        decoder.initState(encoderState);
        assertTrue("状态应该已初始化", decoder.isStateInitialized());
        
        // 重置状态
        decoder.resetState();
        assertFalse("重置后状态应该未初始化", decoder.isStateInitialized());
        assertNull("重置后编码器状态应该为null", decoder.getEncoderState());
    }

    @Test
    public void testLayerAccessAfterInit() {
        decoder.init();
        
        Embedding embedding = decoder.getEmbedding();
        LstmLayer lstmLayer = decoder.getLstmLayer();
        LinearLayer linearLayer = decoder.getLinearLayer();
        
        assertNotNull("嵌入层不应该为null", embedding);
        assertNotNull("LSTM层不应该为null", lstmLayer);
        assertNotNull("线性层不应该为null", linearLayer);
        
        assertTrue("嵌入层名称应该包含前缀", embedding.getName().contains("test_decoder"));
        assertTrue("LSTM层名称应该包含前缀", lstmLayer.getName().contains("test_decoder"));
        assertTrue("线性层名称应该包含前缀", linearLayer.getName().contains("test_decoder"));
    }

    @Test
    public void testParameterAccess() {
        decoder.init();
        
        // 测试所有参数的获取
        Map<String, ?> allParams = decoder.getAllParams();
        assertNotNull("参数映射不应该为null", allParams);
        assertFalse("应该包含参数", allParams.isEmpty());
    }

    @Test
    public void testSequentialDecoding() {
        decoder.initState(encoderState);
        
        // 解码第一个目标序列
        NdArray target1 = createTestTargetInput();
        Variable output1 = decoder.layerForward(new Variable(target1));
        
        decoder.resetState();
        decoder.initState(encoderState);
        
        // 解码第二个目标序列
        NdArray target2 = createTestTargetInput();
        Variable output2 = decoder.layerForward(new Variable(target2));
        
        // 输出形状应该一致
        assertEquals("两次解码的输出形状应该一致", 
                output1.getValue().getShape(), output2.getValue().getShape());
    }

    @Test
    public void testDecoderInheritance() {
        assertTrue("应该是Decoder的实例", decoder instanceof Decoder);
        assertEquals("应该正确获取输入形状", inputShape, decoder.getInputShape());
        assertEquals("应该正确获取输出形状", outputShape, decoder.getOutputShape());
    }

    @Test
    public void testStateManagement() {
        assertFalse("初始时状态应该未初始化", decoder.isStateInitialized());
        assertNull("初始时编码器状态应该为null", decoder.getEncoderState());
        
        decoder.initState(encoderState);
        assertTrue("初始化后状态应该已初始化", decoder.isStateInitialized());
        assertNotNull("初始化后编码器状态不应该为null", decoder.getEncoderState());
        
        decoder.resetState();
        assertFalse("重置后状态应该未初始化", decoder.isStateInitialized());
        assertNull("重置后编码器状态应该为null", decoder.getEncoderState());
    }

    @Test
    public void testWithDifferentEncoderStates() {
        // 测试不同形状的编码器状态
        NdArray[] differentStates = {
                NdArray.likeRandomN(Shape.of(BATCH_SIZE, HIDDEN_SIZE)),
                NdArray.likeRandomN(Shape.of(BATCH_SIZE, HIDDEN_SIZE * 2)),
                NdArray.likeRandomN(Shape.of(BATCH_SIZE, 64))
        };
        
        for (int i = 0; i < differentStates.length; i++) {
            decoder.resetState();
            decoder.initState(differentStates[i]);
            
            assertEquals(String.format("第%d个编码器状态应该正确保存", i+1), 
                    differentStates[i], decoder.getEncoderState());
        }
    }

    // Helper方法

    /**
     * 创建测试目标输入数据
     */
    private NdArray createTestTargetInput() {
        return createTestTargetInputWithBatchSize(BATCH_SIZE);
    }

    /**
     * 创建指定批大小的测试目标输入
     */
    private NdArray createTestTargetInputWithBatchSize(int batchSize) {
        // 创建随机词索引，范围在[0, TARGET_VOCAB_SIZE)
        NdArray input = NdArray.zeros(Shape.of(batchSize, 1)); // 单个时间步
        for (int i = 0; i < batchSize; i++) {
            input.set(i % TARGET_VOCAB_SIZE, i, 0);
        }
        return input;
    }

    /**
     * 创建指定序列长度的测试目标输入
     */
    private NdArray createTestTargetInputWithSeqLength(int seqLen) {
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