package io.leavesfly.tinyai.nnet.layer.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Embedding层的单元测试
 */
public class EmbeddingTest {

    @Test
    public void testEmbeddingCreation() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        assertEquals("emb", embedding.getName());
        assertEquals(1000, embedding.getNumEmbeddings());
        assertEquals(64, embedding.getEmbeddingDim());
    }

    @Test
    public void testEmbeddingForward1D() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        // 创建1D索引输入 (seq_len=10)
        NdArray indices = NdArray.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, Shape.of(10));
        Variable input = new Variable(indices);

        // 前向传播
        Variable output = embedding.forward(input);

        // 验证输出形状 (seq_len=10, embedding_dim=64)
        assertEquals(Shape.of(10, 64), output.getShape());
    }

    @Test
    public void testEmbeddingForward2D() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        // 创建2D索引输入 (batch=2, seq_len=5)
        NdArray indices = NdArray.of(new float[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        }, Shape.of(2, 5));
        Variable input = new Variable(indices);

        // 前向传播
        Variable output = embedding.forward(input);

        // 验证输出形状 (batch=2, seq_len=5, embedding_dim=64)
        assertEquals(Shape.of(2, 5, 64), output.getShape());
    }

    @Test
    public void testEmbeddingForward2DSeqLen1() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        // 2D输入且seqLen==1，输出应压缩为 (batch_size, embedding_dim)
        NdArray indices = NdArray.of(new float[]{3, 7}, Shape.of(2, 1));
        Variable input = new Variable(indices);

        Variable output = embedding.forward(input);

        // 验证输出形状被压缩为 (batch=2, embedding_dim=64)
        assertEquals(Shape.of(2, 64), output.getShape());
    }

    @Test
    public void testEmbeddingGradientCheck2D() {
        Embedding embedding = new Embedding("emb", 1000, 64);
        
        // 创建2D索引输入 (batch=2, seq_len=5)
        NdArray indices = NdArray.of(new float[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        }, Shape.of(2, 5));
        Variable input = new Variable(indices);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(embedding, input);
    }

    @Test
    public void testEmbeddingGradientCheck1D() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        // 1D索引输入的梯度检查
        NdArray indices = NdArray.of(new float[]{1, 2, 3, 4, 5}, Shape.of(5));
        Variable input = new Variable(indices);

        GradientChecker.checkGraphConnectivity(embedding, input);
    }

    @Test
    public void testEmbeddingInvalidDimension() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        // 3D输入应抛出异常
        NdArray indices = NdArray.of(new float[]{1, 2, 3, 4}, Shape.of(2, 1, 2));
        Variable input = new Variable(indices);

        assertThrows(IllegalArgumentException.class, () -> embedding.forward(input));
    }

    @Test
    public void testEmbeddingIndexOutOfBoundsNegative() {
        Embedding embedding = new Embedding("emb", 100, 16);

        // 负索引应抛出异常
        NdArray indices = NdArray.of(new float[]{-1, 5}, Shape.of(2));
        Variable input = new Variable(indices);

        assertThrows(IndexOutOfBoundsException.class, () -> embedding.forward(input));
    }

    @Test
    public void testEmbeddingIndexOutOfBoundsExceedVocab() {
        Embedding embedding = new Embedding("emb", 100, 16);

        // 超出vocab大小的索引应抛出异常
        NdArray indices = NdArray.of(new float[]{5, 100}, Shape.of(2));
        Variable input = new Variable(indices);

        assertThrows(IndexOutOfBoundsException.class, () -> embedding.forward(input));
    }
}

