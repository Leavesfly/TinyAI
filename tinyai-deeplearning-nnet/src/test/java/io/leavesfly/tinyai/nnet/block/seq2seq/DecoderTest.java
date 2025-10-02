package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Decoder抽象基类的测试
 * <p>
 * 测试解码器的基本功能和接口规范
 */
public class DecoderTest {

    private TestableDecoder decoder;
    private Shape inputShape;
    private Shape outputShape;
    private NdArray encoderState;

    @Before
    public void setUp() {
        inputShape = Shape.of(32, 40); // batch_size=32, target_seq_len=40
        outputShape = Shape.of(32, 40, 10000); // batch_size=32, target_seq_len=40, vocab_size=10000
        decoder = new TestableDecoder("test_decoder", inputShape, outputShape);
        
        // 创建模拟的编码器状态
        encoderState = NdArray.likeRandomN(Shape.of(32, 256));
    }

    @Test
    public void testDecoderConstruction() {
        assertNotNull("解码器实例不应该为null", decoder);
        assertEquals("解码器名称应该正确", "test_decoder", decoder.getName());
        assertEquals("输入形状应该正确", inputShape, decoder.getInputShape());
        assertEquals("输出形状应该正确", outputShape, decoder.getOutputShape());
        assertFalse("初始状态应该未初始化", decoder.isStateInitialized());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullName() {
        new TestableDecoder(null, inputShape, outputShape);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithEmptyName() {
        new TestableDecoder("", inputShape, outputShape);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullInputShape() {
        new TestableDecoder("test", null, outputShape);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructorWithNullOutputShape() {
        new TestableDecoder("test", inputShape, null);
    }

    @Test
    public void testInitState() {
        decoder.initState(encoderState);
        assertTrue("状态应该已初始化", decoder.isStateInitialized());
        assertEquals("编码器状态应该正确保存", encoderState, decoder.getEncoderState());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInitStateWithNull() {
        decoder.initState(null);
    }

    @Test
    public void testIsStateInitialized() {
        assertFalse("初始时状态应该未初始化", decoder.isStateInitialized());
        decoder.initState(encoderState);
        assertTrue("初始化后状态应该已初始化", decoder.isStateInitialized());
    }

    @Test
    public void testGetEncoderState() {
        assertNull("未初始化时应该返回null", decoder.getEncoderState());
        decoder.initState(encoderState);
        assertEquals("初始化后应该返回正确的状态", encoderState, decoder.getEncoderState());
    }

    @Test
    public void testResetState() {
        decoder.initState(encoderState);
        assertTrue("初始化后状态应该已初始化", decoder.isStateInitialized());
        
        decoder.resetState();
        assertFalse("重置后状态应该未初始化", decoder.isStateInitialized());
        assertNull("重置后编码器状态应该为null", decoder.getEncoderState());
    }

    @Test
    public void testGetCurrentHiddenState() {
        // 默认实现应该返回null
        assertNull("默认实现应该返回null", decoder.getCurrentHiddenState());
    }

    @Test
    public void testSetCurrentHiddenState() {
        NdArray hiddenState = NdArray.likeRandomN(Shape.of(32, 256));
        // 默认实现应该不抛出异常
        assertDoesNotThrow(() -> decoder.setCurrentHiddenState(hiddenState));
    }

    @Test(expected = IllegalStateException.class)
    public void testLayerForwardWithoutStateInitialization() {
        Variable input = new Variable(NdArray.likeRandomN(inputShape));
        decoder.layerForward(input);
    }

    @Test
    public void testLayerForwardWithStateInitialization() {
        decoder.initState(encoderState);
        Variable input = new Variable(NdArray.likeRandomN(inputShape));
        Variable output = decoder.layerForward(input);
        
        assertNotNull("输出不应该为null", output);
        assertEquals("输出形状应该正确", outputShape, output.getValue().getShape());
    }

    @Test
    public void testValidateForwardPreconditions() {
        // 未初始化状态时应该抛出异常
        try {
            decoder.validateForwardPreconditions();
            fail("应该抛出IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue("异常消息应该包含相关信息", e.getMessage().contains("状态尚未初始化"));
        }
        
        // 初始化状态后应该正常执行
        decoder.initState(encoderState);
        assertDoesNotThrow(() -> decoder.validateForwardPreconditions());
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
     * 测试用的Decoder实现类
     */
    private static class TestableDecoder extends Decoder {
        public TestableDecoder(String name, Shape inputShape, Shape outputShape) {
            super(name, inputShape, outputShape);
        }

        @Override
        public void initState(NdArray encoderOutput) {
            if (encoderOutput == null) {
                throw new IllegalArgumentException("编码器输出状态不能为null");
            }
            this.encoderState = encoderOutput;
            this.stateInitialized = true;
        }

        @Override
        public void init() {
            // 简单的测试实现
        }

        @Override
        public Variable layerForward(Variable... inputs) {
            validateForwardPreconditions();
            // 返回一个符合输出形状的变量
            NdArray output = NdArray.zeros(outputShape);
            return new Variable(output);
        }

        // 为了测试需要，将protected方法暴露为public
        @Override
        public void validateForwardPreconditions() {
            super.validateForwardPreconditions();
        }
    }
}