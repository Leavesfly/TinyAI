package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Loss 包相关类的单元测试
 * 
 * 测试 Loss、MeanSquaredLoss、SoftmaxCrossEntropy、MaskedSoftmaxCELoss、Classify 相关功能
 * 
 * @author TinyDL
 * @version 1.0
 */
public class LossTest {

    private Variable testLabel;
    private Variable testPredict;
    private Variable testLabelClassify;
    private Variable testPredictClassify;

    @Before
    public void setUp() {
        // 回归测试数据
        testLabel = new Variable(NdArray.of(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}}));
        testPredict = new Variable(NdArray.of(new float[][]{{1.1f, 1.9f}, {2.9f, 4.1f}}));

        // 分类测试数据
        testLabelClassify = new Variable(NdArray.of(new float[][]{{0.0f, 1.0f}})); // 类别标签
        testPredictClassify = new Variable(NdArray.of(new float[][]{
            {0.1f, 0.9f, 0.0f}, // 第一个样本的预测概率
            {0.8f, 0.1f, 0.1f}  // 第二个样本的预测概率
        }));
    }

    @Test
    public void testMeanSquaredLossCreation() {
        // 测试 MeanSquaredLoss 创建
        MeanSquaredLoss mse = new MeanSquaredLoss();
        assertNotNull(mse);
    }

    @Test
    public void testMeanSquaredLossComputation() {
        // 测试均方误差损失计算
        MeanSquaredLoss mse = new MeanSquaredLoss();
        Variable loss = mse.loss(testLabel, testPredict);
        
        assertNotNull(loss);
        assertTrue("MSE损失应该是正值", loss.getValue().getNumber().floatValue() >= 0);
        
        // 预测值与标签相同时，损失应该为0
        Variable sameLoss = mse.loss(testLabel, testLabel);
        assertEquals("相同值的MSE损失应为0", 0.0f, sameLoss.getValue().getNumber().floatValue(), 1e-6f);
    }

    @Test
    public void testSoftmaxCrossEntropyCreation() {
        // 测试 SoftmaxCrossEntropy 创建
        SoftmaxCrossEntropy softmaxCE = new SoftmaxCrossEntropy();
        assertNotNull(softmaxCE);
    }

    @Test
    public void testSoftmaxCrossEntropyComputation() {
        // 测试 Softmax 交叉熵损失计算
        SoftmaxCrossEntropy softmaxCE = new SoftmaxCrossEntropy();
        
        // 创建简单的分类测试数据
        Variable label = new Variable(NdArray.of(new float[][]{{1.0f}})); // 类别1
        Variable predict = new Variable(NdArray.of(new float[][]{{0.1f, 0.9f, 0.0f}})); // 预测概率
        
        Variable loss = softmaxCE.loss(label, predict);
        
        assertNotNull(loss);
        assertTrue("交叉熵损失应该是正值", loss.getValue().getNumber().floatValue() >= 0);
    }

    @Test
    public void testClassifyAccuracyRate() {
        // 测试分类准确率计算
        Classify classify = new Classify();
        
        // 创建测试数据：标签为[0, 1]，预测为[[0.9, 0.1], [0.2, 0.8]]
        Variable label = new Variable(NdArray.of(new float[][]{{0.0f}, {1.0f}}));
        Variable predict = new Variable(NdArray.of(new float[][]{
            {0.9f, 0.1f}, // 预测类别0（正确）
            {0.2f, 0.8f}  // 预测类别1（正确）
        }));
        
        float accuracy = classify.accuracyRate(label, predict);
        
        assertTrue("准确率应在0-1之间", accuracy >= 0.0f && accuracy <= 1.0f);
        assertEquals("完全正确预测的准确率应为1.0", 1.0f, accuracy, 1e-6f);
    }

    @Test
    public void testClassifyAccuracyRatePartial() {
        // 测试部分正确的准确率
        Classify classify = new Classify();
        
        // 创建测试数据：标签为[0, 1]，预测为[[0.1, 0.9], [0.2, 0.8]]
        Variable label = new Variable(NdArray.of(new float[][]{{0.0f}, {1.0f}}));
        Variable predict = new Variable(NdArray.of(new float[][]{
            {0.1f, 0.9f}, // 预测类别1（错误，应为0）
            {0.2f, 0.8f}  // 预测类别1（正确）
        }));
        
        float accuracy = classify.accuracyRate(label, predict);
        
        assertEquals("50%正确的准确率应为0.5", 0.5f, accuracy, 1e-6f);
    }

    @Test
    public void testMaskedSoftmaxCELossCreation() {
        // 测试 MaskedSoftmaxCELoss 创建
        MaskedSoftmaxCELoss maskedLoss = new MaskedSoftmaxCELoss();
        assertNotNull(maskedLoss);
    }

    @Test
    public void testMaskedSoftmaxCELossBasic() {
        // 测试掩码 Softmax 交叉熵损失的基本功能
        MaskedSoftmaxCELoss maskedLoss = new MaskedSoftmaxCELoss();
        
        // 创建序列数据：标签包含填充(0)
        Variable sequenceLabel = new Variable(NdArray.of(new float[][]{
            {1.0f, 2.0f, 0.0f}, // 序列1：有效token是1,2，填充是0
            {2.0f, 0.0f, 0.0f}  // 序列2：有效token是2，填充是0,0
        }));
        
        // 预测值：(batch_size=2, seq_len=3, vocab_size=3)
        Variable sequencePredict = new Variable(NdArray.of(new float[][][]{
            {{0.1f, 0.8f, 0.1f}, {0.1f, 0.1f, 0.8f}, {0.3f, 0.3f, 0.4f}}, // 序列1的预测
            {{0.1f, 0.1f, 0.8f}, {0.3f, 0.3f, 0.4f}, {0.3f, 0.3f, 0.4f}}  // 序列2的预测
        }));
        
        Variable loss = maskedLoss.loss(sequenceLabel, sequencePredict);
        
        assertNotNull(loss);
        assertTrue("掩码损失应该是正值", loss.getValue().getNumber().floatValue() >= 0);
    }

    @Test
    public void testMaskedSoftmaxCELossWithCustomPadToken() {
        // 测试使用自定义填充标记的掩码损失
        MaskedSoftmaxCELoss maskedLoss = new MaskedSoftmaxCELoss();
        
        // 使用-1作为填充标记
        Variable sequenceLabel = new Variable(NdArray.of(new float[][]{
            {1.0f, 2.0f, -1.0f},
            {2.0f, -1.0f, -1.0f}
        }));
        
        Variable sequencePredict = new Variable(NdArray.of(new float[][][]{
            {{0.1f, 0.8f, 0.1f}, {0.1f, 0.1f, 0.8f}, {0.3f, 0.3f, 0.4f}},
            {{0.1f, 0.1f, 0.8f}, {0.3f, 0.3f, 0.4f}, {0.3f, 0.3f, 0.4f}}
        }));
        
        Variable loss = maskedLoss.maskedSoftmaxCrossEntropy(sequenceLabel, sequencePredict, -1);
        
        assertNotNull(loss);
        assertTrue("自定义填充标记的掩码损失应该是正值", loss.getValue().getNumber().floatValue() >= 0);
    }

    @Test
    public void testMaskedSoftmaxCELossStats() {
        // 测试掩码损失统计信息
        MaskedSoftmaxCELoss maskedLoss = new MaskedSoftmaxCELoss();
        
        Variable sequenceLabel = new Variable(NdArray.of(new float[][]{
            {1.0f, 2.0f, 0.0f},
            {2.0f, 0.0f, 0.0f}
        }));
        
        Variable sequencePredict = new Variable(NdArray.of(new float[][][]{
            {{0.1f, 0.8f, 0.1f}, {0.1f, 0.1f, 0.8f}, {0.3f, 0.3f, 0.4f}},
            {{0.1f, 0.1f, 0.8f}, {0.3f, 0.3f, 0.4f}, {0.3f, 0.3f, 0.4f}}
        }));
        
        MaskedSoftmaxCELoss.LossStats stats = maskedLoss.computeLossStats(sequenceLabel, sequencePredict, 0);
        
        assertNotNull(stats);
        assertTrue("总损失应该是正值", stats.totalLoss >= 0);
        assertTrue("平均损失应该是正值", stats.averageLoss >= 0);
        assertEquals("有效token数量应为3", 3, stats.validTokens);
        
        // 测试toString方法
        String statsString = stats.toString();
        assertNotNull(statsString);
        assertTrue("统计信息字符串应包含损失信息", statsString.contains("总损失"));
    }

    @Test
    public void testLossAbstractClass() {
        // 测试 Loss 抽象类
        Loss customLoss = new Loss() {
            @Override
            public Variable loss(Variable y, Variable predict) {
                // 简单的自定义损失：相减后取平方
                return predict.sub(y).pow(2.0f).sum();
            }
        };
        
        Variable loss = customLoss.loss(testLabel, testPredict);
        assertNotNull(loss);
        assertTrue("自定义损失应该是正值", loss.getValue().getNumber().floatValue() >= 0);
    }

    @Test
    public void testLossFunctionComparison() {
        // 比较不同损失函数的特性
        MeanSquaredLoss mse = new MeanSquaredLoss();
        
        // 测试完全正确的预测
        Variable perfectPredict = testLabel;
        Variable mseLoss = mse.loss(testLabel, perfectPredict);
        assertEquals("完美预测的MSE应为0", 0.0f, mseLoss.getValue().getNumber().floatValue(), 1e-6f);
        
        // 测试不同的预测误差
        Variable smallErrorPredict = new Variable(NdArray.of(new float[][]{{1.01f, 2.01f}, {3.01f, 4.01f}}));
        Variable largeErrorPredict = new Variable(NdArray.of(new float[][]{{1.1f, 2.1f}, {3.1f, 4.1f}}));
        
        Variable smallError = mse.loss(testLabel, smallErrorPredict);
        Variable largeError = mse.loss(testLabel, largeErrorPredict);
        
        assertTrue("大误差的损失应该大于小误差", 
                   largeError.getValue().getNumber().floatValue() > 
                   smallError.getValue().getNumber().floatValue());
    }

    @Test
    public void testEdgeCases() {
        // 测试边界情况
        
        // 测试单个样本
        Variable singleLabel = new Variable(NdArray.of(new float[][]{{1.0f}}));
        Variable singlePredict = new Variable(NdArray.of(new float[][]{{1.1f}}));
        
        MeanSquaredLoss mse = new MeanSquaredLoss();
        Variable singleLoss = mse.loss(singleLabel, singlePredict);
        assertNotNull(singleLoss);
        
        // 测试零值
        Variable zeroLabel = new Variable(NdArray.of(new float[][]{{0.0f, 0.0f}}));
        Variable zeroPredict = new Variable(NdArray.of(new float[][]{{0.0f, 0.0f}}));
        
        Variable zeroLoss = mse.loss(zeroLabel, zeroPredict);
        assertEquals("零值预测的MSE应为0", 0.0f, zeroLoss.getValue().getNumber().floatValue(), 1e-6f);
    }
}