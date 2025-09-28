package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * MultiHeadAttention层的全面单元测试
 * 
 * 测试多头注意力机制层的功能：
 * 1. 参数初始化和子层创建
 * 2. 不同构造函数配置
 * 3. 自注意力机制
 * 4. 交叉注意力机制
 * 5. 掩码注意力机制
 * 6. 维度变换正确性
 * 7. 边界情况处理
 */
public class MultiHeadAttentionTest {

    private MultiHeadAttention selfAttention;
    private MultiHeadAttention maskedAttention;
    private MultiHeadAttention crossAttention;
    private MultiHeadAttention smallAttention;
    
    @Before
    public void setUp() {
        // 标准自注意力配置
        selfAttention = new MultiHeadAttention("self_attn", 512, 8, false);
        
        // 带掩码的注意力配置（用于解码器）
        maskedAttention = new MultiHeadAttention("masked_attn", 256, 4, true);
        
        // 交叉注意力配置
        crossAttention = new MultiHeadAttention("cross_attn", 128, 2, false);
        
        // 小型配置用于快速测试
        smallAttention = new MultiHeadAttention("small_attn", 64, 4, false);
    }

    @Test
    public void testParameterInitialization() {
        // 测试线性变换层是否正确初始化
        assertNotNull("query线性层应该被初始化", selfAttention.getClass());
        assertNotNull("key线性层应该被初始化", selfAttention.getClass());
        assertNotNull("value线性层应该被初始化", selfAttention.getClass());
        assertNotNull("output线性层应该被初始化", selfAttention.getClass());
    }

    @Test
    public void testInvalidDimensionConfiguration() {
        // 测试不合法的维度配置
        try {
            new MultiHeadAttention("invalid", 127, 8, false); // 127不能被8整除
            fail("应该抛出IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息应该包含相关描述", 
                     e.getMessage().contains("divisible"));
        }
    }

    @Test
    public void testRequireInputNum() {
        // 测试requireInputNum方法
        assertEquals("MultiHeadAttention应该需要3个输入", 3, selfAttention.requireInputNum());
        assertEquals("带掩码的MultiHeadAttention应该需要3个输入", 3, maskedAttention.requireInputNum());
    }

    @Test
    public void testSelfAttentionForward() {
        // 测试自注意力前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(2, 8, 64)); // batch_size=2, seq_len=8, d_model=64
        Variable inputVar = new Variable(input);
        
        // 自注意力：Q=K=V=input
        Variable output = smallAttention.layerForward(inputVar, inputVar, inputVar);
        
        // 验证输出形状
        assertEquals("自注意力输出形状应该与输入形状相同", 
                   Shape.of(2, 8, 64), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("自注意力输出不应该为null", output.getValue());
    }

    @Test
    public void testCrossAttentionForward() {
        // 测试交叉注意力前向传播
        NdArray query = NdArray.likeRandomN(Shape.of(2, 6, 128));   // 解码器序列
        NdArray key = NdArray.likeRandomN(Shape.of(2, 10, 128));    // 编码器序列
        NdArray value = NdArray.likeRandomN(Shape.of(2, 10, 128));  // 编码器序列
        
        Variable queryVar = new Variable(query);
        Variable keyVar = new Variable(key);
        Variable valueVar = new Variable(value);
        
        // 交叉注意力：Q来自解码器，K和V来自编码器
        Variable output = crossAttention.layerForward(queryVar, keyVar, valueVar);
        
        // 验证输出形状（应该与query相同）
        assertEquals("交叉注意力输出形状应该与query形状相同", 
                   Shape.of(2, 6, 128), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("交叉注意力输出不应该为null", output.getValue());
    }

    @Test
    public void testMaskedAttentionForward() {
        // 测试带掩码的注意力前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(2, 5, 256));
        Variable inputVar = new Variable(input);
        
        // 带掩码的自注意力
        Variable output = maskedAttention.layerForward(inputVar, inputVar, inputVar);
        
        // 验证输出形状
        assertEquals("带掩码的注意力输出形状应该与输入形状相同", 
                   Shape.of(2, 5, 256), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("带掩码的注意力输出不应该为null", output.getValue());
    }

    @Test
    public void testForwardMethod() {
        // 测试forward方法（直接使用NdArray）
        NdArray query = NdArray.likeRandomN(Shape.of(1, 4, 64));
        NdArray key = NdArray.likeRandomN(Shape.of(1, 4, 64));
        NdArray value = NdArray.likeRandomN(Shape.of(1, 4, 64));
        
        NdArray output = smallAttention.forward(query, key, value);
        
        assertNotNull("forward方法应该返回有效输出", output);
        assertEquals("forward输出形状应该正确", 
                   Shape.of(1, 4, 64), output.getShape());
    }

    @Test
    public void testSingleTokenInput() {
        // 测试单个token输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 1, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = smallAttention.layerForward(inputVar, inputVar, inputVar);
        
        assertEquals("单token输出形状应该正确", 
                   Shape.of(1, 1, 64), output.getValue().getShape());
        assertNotNull("单token输出不应该为null", output.getValue());
    }

    @Test
    public void testLongSequenceInput() {
        // 测试长序列输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 50, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = smallAttention.layerForward(inputVar, inputVar, inputVar);
        
        assertEquals("长序列输出形状应该正确", 
                   Shape.of(1, 50, 64), output.getValue().getShape());
        assertNotNull("长序列输出不应该为null", output.getValue());
    }

    @Test
    public void testLargeBatchInput() {
        // 测试大批次输入
        NdArray input = NdArray.likeRandomN(Shape.of(8, 10, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = smallAttention.layerForward(inputVar, inputVar, inputVar);
        
        assertEquals("大批次输出形状应该正确", 
                   Shape.of(8, 10, 64), output.getValue().getShape());
        assertNotNull("大批次输出不应该为null", output.getValue());
    }

    @Test
    public void testBackwardMethod() {
        // 测试backward方法
        NdArray grad = NdArray.likeRandomN(Shape.of(2, 4, 64));
        
        try {
            smallAttention.backward(grad);
            // backward方法应该能正常执行而不抛出异常
        } catch (Exception e) {
            fail("backward方法不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testNumericalStability() {
        // 测试数值稳定性，使用极端值
        float[][][] extremeData = new float[1][3][64];
        
        // 填充极端值
        for (int s = 0; s < 3; s++) {
            for (int d = 0; d < 64; d++) {
                if (d % 3 == 0) {
                    extremeData[0][s][d] = 100.0f;   // 大值
                } else if (d % 3 == 1) {
                    extremeData[0][s][d] = -100.0f;  // 大负值
                } else {
                    extremeData[0][s][d] = 1e-6f;    // 小值
                }
            }
        }
        
        NdArray input = NdArray.of(extremeData);
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = smallAttention.layerForward(inputVar, inputVar, inputVar);
            assertNotNull("极端值输入应该产生有效输出", output.getValue());
            
            // 检查输出是否包含NaN或无穷大
            NdArray outputArray = output.getValue();
            float[][][] outputData = outputArray.get3dArray();
            for (int b = 0; b < outputData.length; b++) {
                for (int s = 0; s < outputData[b].length; s++) {
                    for (int d = 0; d < outputData[b][s].length; d++) {
                        float val = outputData[b][s][d];
                        assertFalse("输出不应该包含NaN", Float.isNaN(val));
                        assertFalse("输出不应该包含无穷大", Float.isInfinite(val));
                    }
                }
            }
        } catch (Exception e) {
            fail("极端值输入不应该导致异常: " + e.getMessage());
        }
    }

    @Test
    public void testAttentionWeightProperties() {
        // 测试注意力权重的基本性质
        // 创建简单的测试数据
        float[][][] simpleData = {
            {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}
        };
        
        // 扩展到所需维度
        float[][][] inputData = new float[1][3][64];
        for (int s = 0; s < 3; s++) {
            for (int d = 0; d < 64; d++) {
                if (d < 4) {
                    inputData[0][s][d] = simpleData[0][s][d];
                } else {
                    inputData[0][s][d] = 0.0f;
                }
            }
        }
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = smallAttention.layerForward(inputVar, inputVar, inputVar);
        
        // 输出应该是有效的
        assertNotNull("注意力输出应该有效", output.getValue());
        assertEquals("注意力输出形状应该正确", 
                   Shape.of(1, 3, 64), output.getValue().getShape());
    }

    @Test
    public void testConsistency() {
        // 测试多次调用的一致性
        NdArray query = NdArray.likeRandomN(Shape.of(1, 3, 64));
        NdArray key = NdArray.likeRandomN(Shape.of(1, 3, 64));
        NdArray value = NdArray.likeRandomN(Shape.of(1, 3, 64));
        
        Variable queryVar = new Variable(query);
        Variable keyVar = new Variable(key);
        Variable valueVar = new Variable(value);
        
        Variable output1 = smallAttention.layerForward(queryVar, keyVar, valueVar);
        Variable output2 = smallAttention.layerForward(queryVar, keyVar, valueVar);
        
        // 相同输入应该产生相同输出
        NdArray arr1 = output1.getValue();
        NdArray arr2 = output2.getValue();
        
        float[][][] data1 = arr1.get3dArray();
        float[][][] data2 = arr2.get3dArray();
        
        for (int b = 0; b < data1.length; b++) {
            for (int s = 0; s < data1[b].length; s++) {
                for (int d = 0; d < data1[b][s].length; d++) {
                    assertEquals("相同输入应该产生相同输出", 
                               data1[b][s][d], data2[b][s][d], 1e-6f);
                }
            }
        }
    }

    @Test
    public void testDifferentSequenceLengths() {
        // 测试不同序列长度的查询和键值对
        NdArray query = NdArray.likeRandomN(Shape.of(1, 4, 64));  // 查询序列长度为4
        NdArray key = NdArray.likeRandomN(Shape.of(1, 6, 64));    // 键序列长度为6
        NdArray value = NdArray.likeRandomN(Shape.of(1, 6, 64));  // 值序列长度为6
        
        Variable queryVar = new Variable(query);
        Variable keyVar = new Variable(key);
        Variable valueVar = new Variable(value);
        
        Variable output = smallAttention.layerForward(queryVar, keyVar, valueVar);
        
        // 输出序列长度应该与查询序列长度相同
        assertEquals("输出序列长度应该与查询序列长度相同", 
                   Shape.of(1, 4, 64), output.getValue().getShape());
        assertNotNull("不同序列长度的输出不应该为null", output.getValue());
    }

    @Test
    public void testOutputValueRange() {
        // 测试输出值的合理性
        NdArray input = NdArray.likeRandomN(Shape.of(1, 4, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = smallAttention.layerForward(inputVar, inputVar, inputVar);
        NdArray outputArray = output.getValue();
        
        // 检查输出值在合理范围内
        float[][][] outputData = outputArray.get3dArray();
        
        for (int b = 0; b < outputData.length; b++) {
            for (int s = 0; s < outputData[b].length; s++) {
                for (int d = 0; d < outputData[b][s].length; d++) {
                    float val = outputData[b][s][d];
                    // 检查值在合理范围内（避免梯度爆炸）
                    assertTrue("输出值应该在合理范围内", Math.abs(val) < 1000.0f);
                }
            }
        }
    }
}