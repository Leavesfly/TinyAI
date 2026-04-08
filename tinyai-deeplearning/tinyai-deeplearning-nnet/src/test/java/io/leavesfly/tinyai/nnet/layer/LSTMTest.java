package io.leavesfly.tinyai.nnet.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.rnn.LSTM;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * LSTM层的单元测试
 */
public class LSTMTest {

    @Test
    public void testLSTMCreation() {
        LSTM lstm = new LSTM("lstm", 10, 20, true);

        assertEquals("lstm", lstm.getName());
        assertEquals(10, lstm.getInputSize());
        assertEquals(20, lstm.getHiddenSize());
    }

    @Test
    public void testLSTMForward() {
        LSTM lstm = new LSTM("lstm", 10, 20);

        // 创建输入 (batch=4, features=10)
        NdArray inputData = NdArray.randn(Shape.of(4, 10));
        Variable input = new Variable(inputData);

        // 前向传播
        Variable output = lstm.forward(input);

        // 验证输出形状 (batch=4, hidden=20)
        assertEquals(Shape.of(4, 20), output.getShape());
    }

    @Test
    public void testLSTMStateInitialization() {
        LSTM lstm = new LSTM("lstm", 10, 20);

        // 初始状态应该为null
        assertNull(lstm.getHiddenState());
        assertNull(lstm.getCellState());

        // 第一次前向传播后状态被初始化
        NdArray inputData = NdArray.randn(Shape.of(4, 10));
        Variable input = new Variable(inputData);
        lstm.forward(input);

        // 验证状态已初始化
        assertNotNull(lstm.getHiddenState());
        assertNotNull(lstm.getCellState());
        assertEquals(Shape.of(4, 20), lstm.getHiddenState().getShape());
        assertEquals(Shape.of(4, 20), lstm.getCellState().getShape());
    }

    @Test
    public void testLSTMResetState() {
        LSTM lstm = new LSTM("lstm", 10, 20);

        // 初始化状态
        NdArray inputData = NdArray.randn(Shape.of(4, 10));
        Variable input = new Variable(inputData);
        lstm.forward(input);

        assertNotNull(lstm.getHiddenState());

        // 重置状态
        lstm.resetState();

        assertNull(lstm.getHiddenState());
        assertNull(lstm.getCellState());
    }

    @Test
    public void testLSTMSequentialProcessing() {
        LSTM lstm = new LSTM("lstm", 10, 20);

        int seqLen = 5;
        int batchSize = 4;

        // 处理序列
        Variable lastOutput = null;
        for (int t = 0; t < seqLen; t++) {
            NdArray inputData = NdArray.randn(Shape.of(batchSize, 10));
            Variable input = new Variable(inputData);
            lastOutput = lstm.forward(input);
        }

        // 验证最后的输出形状
        assertNotNull(lastOutput);
        assertEquals(Shape.of(batchSize, 20), lastOutput.getShape());
    }

    @Test
    public void testLSTMParameterCount() {
        LSTM lstm = new LSTM("lstm", 10, 20, true);

        var params = lstm.namedParameters();

        // LSTM有12个参数：
        // 4个门 × (W_i, W_h, b) = 4 × 3 = 12
        assertEquals(12, params.size());

        // 验证参数名
        assertTrue(params.containsKey("W_ii"));
        assertTrue(params.containsKey("W_hi"));
        assertTrue(params.containsKey("b_i"));
        assertTrue(params.containsKey("W_if"));
        assertTrue(params.containsKey("W_hf"));
        assertTrue(params.containsKey("b_f"));
        assertTrue(params.containsKey("W_ig"));
        assertTrue(params.containsKey("W_hg"));
        assertTrue(params.containsKey("b_g"));
        assertTrue(params.containsKey("W_io"));
        assertTrue(params.containsKey("W_ho"));
        assertTrue(params.containsKey("b_o"));
    }

    @Test
    public void testLSTMWithoutBias() {
        LSTM lstm = new LSTM("lstm", 10, 20, false);

        var params = lstm.namedParameters();

        // 没有偏置时只有8个参数（4个门 × 2个权重）
        assertEquals(8, params.size());

        // 验证没有偏置参数
        assertFalse(params.containsKey("b_i"));
        assertFalse(params.containsKey("b_f"));
        assertFalse(params.containsKey("b_g"));
        assertFalse(params.containsKey("b_o"));
    }

    @Test
    public void testLSTMBuffers() {
        LSTM lstm = new LSTM("lstm", 10, 20);

        // 初始化前缓冲区应该存在但为null
        var buffers = lstm.namedBuffers();
//        assertEquals(2, buffers.size());

        // 前向传播后缓冲区应该有值
        NdArray inputData = NdArray.randn(Shape.of(4, 10));
        Variable input = new Variable(inputData);
        lstm.forward(input);

        buffers = lstm.namedBuffers();
        assertNotNull(buffers.get("hidden_state"));
        assertNotNull(buffers.get("cell_state"));
    }

    @Test
    public void testLSTMSetState() {
        LSTM lstm = new LSTM("lstm", 10, 20);

        // 设置自定义状态
        NdArray customHidden = NdArray.ones(Shape.of(4, 20));
        NdArray customCell = NdArray.zeros(Shape.of(4, 20));

        lstm.setHiddenState(customHidden);
        lstm.setCellState(customCell);

        assertEquals(customHidden, lstm.getHiddenState());
        assertEquals(customCell, lstm.getCellState());
    }

    @Test
    public void testLSTMGradientCheck() {
        LSTM lstm = new LSTM("lstm", 10, 20, true);
        NdArray inputData = NdArray.randn(Shape.of(4, 10));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(lstm, input);
    }

    @Test
    public void testLSTMWithoutBiasGradientCheck() {
        LSTM lstm = new LSTM("lstm", 10, 20, false);
        NdArray inputData = NdArray.randn(Shape.of(4, 10));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(lstm, input);
    }
}
