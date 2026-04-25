package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * RotaryPositionEmbedding单元测试
 * 
 * @author leavesfly
 */
public class RotaryPositionEmbeddingTest {
    
    private RotaryPositionEmbedding rope;
    private final int dimModel = 64;
    private final int maxSeqLen = 128;
    private final float theta = 10000.0f;
    
    @BeforeEach
    public void setUp() {
        rope = new RotaryPositionEmbedding(dimModel, maxSeqLen, theta);
    }
    
    @Test
    public void testRoPECreation() {
        assertNotNull(rope, "RoPE不应为null");
    }
    
    @Test
    public void testComputeFrequencies() {
        // RoPE应该预计算频率
        // 这是内部实现,我们通过forward测试间接验证
        assertNotNull(rope, "RoPE创建后应正常");
    }
    
    @Test
    public void testForwardSinglePosition() {
        // 输入: [batch=1, seq_len=1, dim=64]
        float[] data = new float[dimModel];
        for (int i = 0; i < dimModel; i++) {
            data[i] = (float) i / dimModel; // 简单初始化
        }
        
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable posVar = new Variable(NdArray.of(new float[]{0}, Shape.of(1)));
        Variable output = rope.forward(inputVar, posVar);
        
        assertNotNull(output, "RoPE输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(1, shape[0], "batch维度应保持");
        assertEquals(1, shape[1], "seq_len维度应保持");
        assertEquals(dimModel, shape[2], "dim维度应保持");
    }
    
    @Test
    public void testForwardMultiplePositions() {
        // 输入: [batch=2, seq_len=4, dim=64]
        int batchSize = 2;
        int seqLen = 4;
        
        float[] data = new float[batchSize * seqLen * dimModel];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i / data.length;
        }
        
        NdArray input = NdArray.of(data, Shape.of(batchSize, seqLen, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable posVar = new Variable(NdArray.of(new float[]{0}, Shape.of(1)));
        Variable output = rope.forward(inputVar, posVar);
        
        assertNotNull(output, "RoPE输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应为" + batchSize);
        assertEquals(seqLen, shape[1], "seq_len维度应为" + seqLen);
        assertEquals(dimModel, shape[2], "dim维度应为" + dimModel);
    }
    
    @Test
    public void testPositionOffset() {
        // 测试不同position offset的影响
        float[] data = new float[dimModel];
        for (int i = 0; i < dimModel; i++) {
            data[i] = 1.0f;
        }
        
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));
        
        Variable pos1 = new Variable(NdArray.of(new float[]{0}, Shape.of(1)));
        Variable pos2 = new Variable(NdArray.of(new float[]{10}, Shape.of(1)));
        Variable output1 = rope.forward(new Variable(input), pos1);
        Variable output2 = rope.forward(new Variable(input), pos2);
        
        // 不同位置应产生不同的输出
        float[] result1 = output1.getValue().getArray();
        float[] result2 = output2.getValue().getArray();
        
        boolean different = false;
        for (int i = 0; i < result1.length; i++) {
            if (Math.abs(result1[i] - result2[i]) > 1e-6f) {
                different = true;
                break;
            }
        }
        assertTrue(different, "不同位置应产生不同的RoPE编码");
    }
    
    @Test
    public void testRoPERotation() {
        // RoPE应该对输入进行旋转变换
        float[] data = new float[dimModel];
        for (int i = 0; i < dimModel; i++) {
            data[i] = 1.0f; // 全1向量
        }
        
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable posVar = new Variable(NdArray.of(new float[]{5}, Shape.of(1)));
        Variable output = rope.forward(inputVar, posVar);
        
        float[] result = output.getValue().getArray();
        
        // 验证输出已经被修改(不再是全1)
        boolean modified = false;
        for (int i = 0; i < result.length; i++) {
            if (Math.abs(result[i] - 1.0f) > 1e-3f) {
                modified = true;
                break;
            }
        }
        assertTrue(modified, "RoPE应该修改输入向量");
    }
    
    @Test
    public void testMaxSeqLenBoundary() {
        // 测试最大序列长度边界
        float[] data = new float[dimModel];
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));

        // 在最大长度内应该正常工作
        Variable posVar = new Variable(NdArray.of(new float[]{maxSeqLen - 1}, Shape.of(1)));
        Variable output = rope.forward(new Variable(input), posVar);
        assertNotNull(output, "在最大序列长度内应正常工作");
    }

    /**
     * 回归测试：同一个 RoPE 实例被 Q、K 两种不同形状的输入先后调用后，
     * 反向传播必须都能正确完成。
     * <p>
     * 该测试精确复现历史 Bug：
     * {@code java.lang.IllegalArgumentException: 数据长度 16384 与形状大小 8192 不匹配}。
     * <br>
     * 根因：RotaryEmbedding Function 将 {@code inputShape} / {@code startPos} 缓存为实例字段，
     * 同一实例被 Q（[B, numHeads, L, D]）和 K（[B, numKVHeads, L, D]）先后 call() 会互相覆盖，
     * 导致反传 Q 分支时读到 K 的 shape，数据长度变成 shape size 的 numKVGroups 倍。
     * <br>
     * 修复后：每次 forward 都 new 一个独立的 Function 实例（共享 cos/sin 缓存），Q、K 互不干扰。
     */
    @Test
    public void testBackwardWithDifferentShapesGQA() {
        // 模拟 GQA 场景：numHeads=8, numKVHeads=2, numKVGroups=4
        int batchSize = 1;
        int numHeads = 8;
        int numKVHeads = 2;
        int seqLen = 16;
        int headDim = 16;

        // Q: [1, 8, 16, 16] = 2048
        float[] qData = new float[batchSize * numHeads * seqLen * headDim];
        for (int i = 0; i < qData.length; i++) {
            qData[i] = (i % 7) * 0.1f;
        }
        Variable q = new Variable(NdArray.of(qData, Shape.of(batchSize, numHeads, seqLen, headDim)));

        // K: [1, 2, 16, 16] = 512（形状比 Q 小 4 倍）
        float[] kData = new float[batchSize * numKVHeads * seqLen * headDim];
        for (int i = 0; i < kData.length; i++) {
            kData[i] = (i % 5) * 0.1f;
        }
        Variable k = new Variable(NdArray.of(kData, Shape.of(batchSize, numKVHeads, seqLen, headDim)));

        RotaryPositionEmbedding sharedRope = new RotaryPositionEmbedding(headDim, 64, 10000.0f);

        // 先对 Q 调用 RoPE，再对 K 调用 RoPE（模拟 MultiHeadAttention.forwardWithCache 的调用顺序）
        Variable startPos = new Variable(NdArray.of(new float[]{0}));
        startPos.setRequireGrad(false);
        Variable qOut = sharedRope.forward(q, startPos);
        Variable kOut = sharedRope.forward(k, startPos);

        assertArrayEquals(new int[]{batchSize, numHeads, seqLen, headDim},
                qOut.getValue().getShape().getShapeDims(), "Q 输出形状应保持");
        assertArrayEquals(new int[]{batchSize, numKVHeads, seqLen, headDim},
                kOut.getValue().getShape().getShapeDims(), "K 输出形状应保持");

        // 关键点：对 Q 的输出做反向传播
        // 修复前会抛 IllegalArgumentException: 数据长度 2048 与形状大小 512 不匹配
        qOut.backward();
        assertNotNull(q.getGrad(), "Q 的梯度不应为 null");
        assertArrayEquals(q.getValue().getShape().getShapeDims(),
                q.getGrad().getShape().getShapeDims(),
                "Q 的梯度形状必须与 Q 的输入形状一致");

        // 对 K 的输出也做反向传播
        kOut.backward();
        assertNotNull(k.getGrad(), "K 的梯度不应为 null");
        assertArrayEquals(k.getValue().getShape().getShapeDims(),
                k.getGrad().getShape().getShapeDims(),
                "K 的梯度形状必须与 K 的输入形状一致");
    }
}
