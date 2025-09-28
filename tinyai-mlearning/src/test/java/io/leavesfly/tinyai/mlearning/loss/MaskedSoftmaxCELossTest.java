package io.leavesfly.tinyai.mlearning.loss;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MaskedSoftmaxCELoss单元测试
 */
public class MaskedSoftmaxCELossTest {
    
    private MaskedSoftmaxCELoss lossFunction;
    
    @BeforeEach
    public void setUp() {
        lossFunction = new MaskedSoftmaxCELoss();
    }
    
    @Test
    public void testBasicLoss() {
        // 创建简单的测试数据
        float[][][] predictData = {
            {{0.1f, 0.2f, 0.3f, 0.4f}, {0.2f, 0.3f, 0.4f, 0.1f}}
        };
        float[][] labelData = {{3, 2}};
        
        Variable predict = new Variable(NdArray.of(predictData));
        Variable labels = new Variable(NdArray.of(labelData));
        
        Variable loss = lossFunction.loss(labels, predict);
        
        assertNotNull(loss);
        assertTrue(loss.getValue().getNumber().floatValue() > 0);
    }
    
    @Test
    public void testMaskedLoss() {
        // 测试包含填充的序列
        float[][][] predictData = {
            {{0.1f, 0.2f, 0.3f, 0.4f}, {0.2f, 0.3f, 0.4f, 0.1f}, {0.0f, 0.0f, 0.0f, 0.0f}}
        };
        float[][] labelData = {{3, 2, 0}};  // 最后一个0是填充
        
        Variable predict = new Variable(NdArray.of(predictData));
        Variable labels = new Variable(NdArray.of(labelData));
        
        Variable maskedLoss = lossFunction.maskedSoftmaxCrossEntropy(labels, predict, 0);
        
        assertNotNull(maskedLoss);
        assertTrue(maskedLoss.getValue().getNumber().floatValue() > 0);
    }
    
    @Test
    public void testLossStats() {
        float[][][] predictData = {
            {{0.1f, 0.2f, 0.3f, 0.4f}, {0.2f, 0.3f, 0.4f, 0.1f}, {0.3f, 0.4f, 0.1f, 0.2f}}
        };
        float[][] labelData = {{3, 2, 0}};  // 最后一个0是填充
        
        Variable predict = new Variable(NdArray.of(predictData));
        Variable labels = new Variable(NdArray.of(labelData));
        
        MaskedSoftmaxCELoss.LossStats stats = lossFunction.computeLossStats(labels, predict, 0);
        
        assertNotNull(stats);
        assertEquals(2, stats.validTokens);  // 应该有2个有效tokens
        assertTrue(stats.totalLoss > 0);
        assertTrue(stats.averageLoss > 0);
    }
    
    @Test
    public void testCustomMask() {
        float[][][] predictData = {
            {{0.1f, 0.2f, 0.3f, 0.4f}, {0.2f, 0.3f, 0.4f, 0.1f}, {0.3f, 0.4f, 0.1f, 0.2f}}
        };
        float[][] labelData = {{3, 2, 1}};
        float[][] maskData = {{1.0f, 1.0f, 0.0f}};  // 最后一个位置被掩码
        
        Variable predict = new Variable(NdArray.of(predictData));
        Variable labels = new Variable(NdArray.of(labelData));
        Variable mask = new Variable(NdArray.of(maskData));
        
        Variable maskedLoss = lossFunction.maskedSoftmaxCrossEntropyWithMask(labels, predict, mask);
        
        assertNotNull(maskedLoss);
        assertTrue(maskedLoss.getValue().getNumber().floatValue() > 0);
    }
    
    @Test
    public void testLossStatsToString() {
        MaskedSoftmaxCELoss.LossStats stats = new MaskedSoftmaxCELoss.LossStats(10.5f, 2.1f, 5);
        String statsString = stats.toString();
        
        assertNotNull(statsString);
        assertTrue(statsString.contains("总损失"));
        assertTrue(statsString.contains("平均损失"));
        assertTrue(statsString.contains("有效tokens"));
    }
}