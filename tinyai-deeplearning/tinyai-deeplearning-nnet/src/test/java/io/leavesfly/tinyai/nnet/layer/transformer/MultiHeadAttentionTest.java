package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MultiHeadAttention层的单元测试
 */
public class MultiHeadAttentionTest {

    @Test
    public void testMultiHeadAttentionCreation() {
        MultiHeadAttention attention = new MultiHeadAttention("attn", 64, 4, 0.1f);

        assertEquals("attn", attention.getName());
        assertEquals(64, attention.getDModel());
        assertEquals(4, attention.getNumHeads());
    }

    @Test
    public void testMultiHeadAttentionForward() {
        MultiHeadAttention attention = new MultiHeadAttention("attn", 64, 4, 0.0f);

        // 创建输入 (batch=2, seq_len=10, d_model=64)
        NdArray inputData = NdArray.randn(Shape.of(2, 10, 64));
        Variable input = new Variable(inputData);

        // 前向传播（自注意力：Q=K=V）
        Variable output = attention.forward(input, input, input);

        // 验证输出形状
        assertEquals(Shape.of(2, 10, 64), output.getShape());
    }

    @Test
    public void testMultiHeadAttentionGradientCheck() {
        MultiHeadAttention attention = new MultiHeadAttention("attn", 64, 4, 0.0f);
        
        // 创建输入 (batch=2, seq_len=10, d_model=64)
        NdArray inputData = NdArray.randn(Shape.of(2, 10, 64));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性（自注意力：Q=K=V）
        GradientChecker.checkGraphConnectivity(attention, input, input, input);
    }
}

