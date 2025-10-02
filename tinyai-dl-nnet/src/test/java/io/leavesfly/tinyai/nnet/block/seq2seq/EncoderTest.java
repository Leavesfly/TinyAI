package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Encoder抽象基类的测试
 * <p>
 * 测试编码器的基本功能和接口规范
 */
public class EncoderTest {

    private TestableEncoder encoder;
    private Shape inputShape;
    private Shape outputShape;

    @Before
    public void setUp() {
        inputShape = Shape.of(32, 50); // batch_size=32, seq_len=50
        outputShape = Shape.of(32, 256); // batch_size=32, hidden_dim=256
        encoder = new TestableEncoder("test_encoder", inputShape, outputShape);
    }

    @Test
    public void testEncoderConstruction() {
        assertNotNull("编码器实例不应该为null", encoder);
        assertEquals("编码器名称应该正确", "test_encoder", encoder.getName());
        assertEquals("输入形状应该正确", inputShape, encoder.getInputShape());
        assertEquals("输出形状应该正确", outputShape, encoder.getOutputShape());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullName() {
        new TestableEncoder(null, inputShape, outputShape);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithEmptyName() {
        new TestableEncoder("", inputShape, outputShape);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullInputShape() {
        new TestableEncoder("test", null, outputShape);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullOutputShape() {
        new TestableEncoder("test", inputShape, null);
    }

    @Test
    public void testResetState() {
        // 基类的resetState应该能正常执行
        assertDoesNotThrow(() -> encoder.resetState());
    }

    @Test
    public void testGetFinalHiddenState() {
        // 默认实现应该返回null
        assertNull("默认实现应该返回null", encoder.getFinalHiddenState());
    }

    @Test
    public void testGetEncodedSequenceLength() {
        // 测试从输出形状获取序列长度
        // encoder的outputShape是Shape.of(32, 256)，第1维度是256（隐藏维度）
        // 所以应该返回256
        assertEquals("应该从输出形状获取维度", 256, encoder.getEncodedSequenceLength());
        
        // 测试包含序列维度的输出形状
        Shape seqOutputShape = Shape.of(32, 50, 256); // 包含序列维度
        TestableEncoder seqEncoder = new TestableEncoder("seq_encoder", inputShape, seqOutputShape);
        assertEquals("序列长度应该正确", 50, seqEncoder.getEncodedSequenceLength());
    }

    @Test
    public void testGetEncodedSequenceLengthWithInvalidShape() {
        Shape invalidShape = Shape.of(32); // 只有一个维度
        TestableEncoder invalidEncoder = new TestableEncoder("invalid", invalidShape, invalidShape);
        assertEquals("无效形状应该返回-1", -1, invalidEncoder.getEncodedSequenceLength());
    }

    // Helper方法
    private void assertDoesNotThrow(Runnable runnable) {
        try {
            runnable.run();
        } catch (Exception e) {
            fail("不应该抛出异常: " + e.getMessage());
        }
    }

    /**
     * 测试用的Encoder实现类
     */
    private static class TestableEncoder extends Encoder {
        public TestableEncoder(String name, Shape inputShape, Shape outputShape) {
            super(name, inputShape, outputShape);
        }

        @Override
        public void init() {
            // 简单的测试实现
        }

        @Override
        public Variable layerForward(Variable... inputs) {
            // 返回一个符合输出形状的变量
            NdArray output = NdArray.zeros(outputShape);
            return new Variable(output);
        }
    }
}