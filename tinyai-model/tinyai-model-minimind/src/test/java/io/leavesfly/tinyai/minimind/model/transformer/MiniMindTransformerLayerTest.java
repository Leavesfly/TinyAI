package io.leavesfly.tinyai.minimind.model.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.attention.KVCache;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.util.GradientChecker;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MiniMindTransformerLayer单元测试
 * 
 * @author leavesfly
 */
public class MiniMindTransformerLayerTest {
    
    private MiniMindTransformerLayer layer;
    private final int hiddenSize = 64;
    private final int numHeads = 4;
    private final int ffnHiddenSize = 128;
    private final int maxSeqLen = 128;
    private final float dropoutRate = 0.0f;
    private final float epsilon = 1e-5f;
    
    @BeforeEach
    public void setUp() {
        layer = new MiniMindTransformerLayer(
            "test_layer",
            hiddenSize,
            numHeads,
            ffnHiddenSize,
            maxSeqLen,
            dropoutRate,
            epsilon,
            "silu",
            true
        );
    }
    
    @Test
    public void testLayerCreation() {
        assertNotNull(layer, "TransformerLayer不应为null");
        assertEquals(hiddenSize, layer.getHiddenSize(), "隐藏层维度应匹配");
        assertEquals(ffnHiddenSize, layer.getFfnHiddenSize(), "FFN维度应匹配");
    }
    
    @Test
    public void testForwardBasic() {
        // 输入: [batch=1, seq_len=5, dim=64]
        int batchSize = 1;
        int seqLen = 5;
        
        float[] data = new float[batchSize * seqLen * hiddenSize];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i / data.length;
        }
        
        NdArray input = NdArray.of(data, Shape.of(batchSize, seqLen, hiddenSize));
        Variable inputVar = new Variable(input);
        
        Variable output = layer.forward(inputVar);
        
        assertNotNull(output, "输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应保持");
        assertEquals(seqLen, shape[1], "seq_len维度应保持");
        assertEquals(hiddenSize, shape[2], "hidden_size维度应保持");
    }
    
    @Test
    public void testForwardWithCache() {
        int batchSize = 1;
        int seqLen = 3;
        int headDim = hiddenSize / numHeads;
        
        // 创建KV Cache
        KVCache kvCache = new KVCache(batchSize, numHeads, headDim, maxSeqLen);
        
        // 第一次forward
        NdArray input1 = NdArray.of(Shape.of(batchSize, seqLen, hiddenSize));
        Variable inputVar1 = new Variable(input1);
        
        Variable output1 = layer.forwardWithCache(inputVar1, kvCache, 0);
        
        assertNotNull(output1, "第一次输出不应为null");
        assertEquals(seqLen, kvCache.getCurrentSeqLen(), "Cache长度应更新");
        
        // 第二次forward (增量)
        int newSeqLen = 1;
        NdArray input2 = NdArray.of(Shape.of(batchSize, newSeqLen, hiddenSize));
        Variable inputVar2 = new Variable(input2);
        
        Variable output2 = layer.forwardWithCache(inputVar2, kvCache, seqLen);
        
        assertNotNull(output2, "第二次输出不应为null");
        assertEquals(seqLen + newSeqLen, kvCache.getCurrentSeqLen(), "Cache长度应累加");
    }
    
    @Test
    public void testResidualConnection() {
        // Transformer层使用残差连接,输出应该不同于单纯的注意力输出
        int batchSize = 1;
        int seqLen = 3;
        
        float[] data = new float[batchSize * seqLen * hiddenSize];
        for (int i = 0; i < data.length; i++) {
            data[i] = 1.0f; // 全1向量
        }
        
        NdArray input = NdArray.of(data, Shape.of(batchSize, seqLen, hiddenSize));
        Variable inputVar = new Variable(input);
        
        Variable output = layer.forward(inputVar);
        
        assertNotNull(output, "输出不应为null");
        
        // 由于残差连接,输出应该被修改
        float[] outputData = output.getValue().getArray();
        assertNotNull(outputData, "输出数据不应为null");
        assertTrue(outputData.length > 0, "输出应有数据");
    }
    
    @Test
    public void testBatchProcessing() {
        // 测试批处理
        int batchSize = 3;
        int seqLen = 5;
        
        float[] data = new float[batchSize * seqLen * hiddenSize];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i / data.length;
        }
        
        NdArray input = NdArray.of(data, Shape.of(batchSize, seqLen, hiddenSize));
        Variable inputVar = new Variable(input);
        
        Variable output = layer.forward(inputVar);
        
        assertNotNull(output, "批处理输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应为" + batchSize);
        assertEquals(seqLen, shape[1], "seq_len维度应为" + seqLen);
        assertEquals(hiddenSize, shape[2], "hidden_size维度应为" + hiddenSize);
    }
    
    @Test
    public void testTrainingModeSwitch() {
        // 测试训练模式切换
        layer.setTraining(true);
        
        int batchSize = 1;
        int seqLen = 3;
        NdArray input = NdArray.of(Shape.of(batchSize, seqLen, hiddenSize));
        Variable inputVar = new Variable(input);
        
        Variable output1 = layer.forward(inputVar);
        assertNotNull(output1, "训练模式输出不应为null");
        
        // 切换到推理模式
        layer.setTraining(false);
        Variable output2 = layer.forward(inputVar);
        assertNotNull(output2, "推理模式输出不应为null");
    }
    
    @Test
    public void testGetAttention() {
        // 测试获取注意力层
        assertNotNull(layer.getAttention(), "应该能获取注意力层");
    }

    @Test
    public void testMiniMindTransformerLayerGradientCheck() {
        MiniMindTransformerLayer layer = new MiniMindTransformerLayer(
            "test_layer",
            hiddenSize,
            numHeads,
            ffnHiddenSize,
            maxSeqLen,
            dropoutRate,
            epsilon,
            "silu",
            true
        );
        
        // 创建输入 (batch=1, seq_len=5, dim=64)
        NdArray inputData = NdArray.randn(Shape.of(1, 5, hiddenSize));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(layer, input);
    }
}
