package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * TransformerEncoderLayer层的全面单元测试
 * 
 * 测试Transformer编码器层的功能：
 * 1. 参数初始化和子层创建
 * 2. 前向传播计算
 * 3. 残差连接和层归一化
 * 4. 不同配置参数
 * 5. 边界情况处理
 */
public class TransformerEncoderLayerTest {

    private TransformerEncoderLayer encoderLayer512;
    private TransformerEncoderLayer encoderLayerSmall;
    private TransformerEncoderLayer encoderLayerDefault;
    
    @Before
    public void setUp() {
        // 标准配置：512维度，8个注意力头，2048隐藏层
        encoderLayer512 = new TransformerEncoderLayer("enc512", 512, 8, 2048, 0.1);
        
        // 小型配置用于快速测试：64维度，4个注意力头，256隐藏层
        encoderLayerSmall = new TransformerEncoderLayer("enc_small", 64, 4, 256, 0.0);
        
        // 使用默认参数的配置
        encoderLayerDefault = new TransformerEncoderLayer("enc_default", 128, 8);
    }

    @Test
    public void testParameterInitialization() {
        // 测试子层是否正确初始化
        MultiHeadAttention selfAttention = encoderLayerSmall.getSelfAttention();
        LayerNorm layerNorm1 = encoderLayerSmall.getLayerNorm1();
        FeedForward feedForward = encoderLayerSmall.getFeedForward();
        LayerNorm layerNorm2 = encoderLayerSmall.getLayerNorm2();
        
        assertNotNull("自注意力层应该被初始化", selfAttention);
        assertNotNull("第一个层归一化层应该被初始化", layerNorm1);
        assertNotNull("前馈网络层应该被初始化", feedForward);
        assertNotNull("第二个层归一化层应该被初始化", layerNorm2);
    }

    @Test
    public void testRequireInputNum() {
        // 测试requireInputNum方法
        assertEquals("TransformerEncoderLayer应该需要1个输入", 1, encoderLayer512.requireInputNum());
        assertEquals("小型编码器层应该需要1个输入", 1, encoderLayerSmall.requireInputNum());
    }

    @Test
    public void testForwardPropagation() {
        // 测试基本的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(2, 8, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = encoderLayerSmall.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", 
                   Shape.of(2, 8, 64), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testForwardMethod() {
        // 测试forward方法（直接使用NdArray）
        NdArray input = NdArray.likeRandomN(Shape.of(1, 10, 64));
        
        NdArray output = encoderLayerSmall.forward(input);
        
        assertNotNull("forward方法应该返回有效输出", output);
        assertEquals("forward输出形状应该正确", input.getShape(), output.getShape());
    }

    @Test
    public void testSingleTokenInput() {
        // 测试单个token输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 1, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = encoderLayerSmall.layerForward(inputVar);
        
        assertEquals("单token输出形状应该正确", 
                   Shape.of(1, 1, 64), output.getValue().getShape());
        assertNotNull("单token输出不应该为null", output.getValue());
    }

    @Test
    public void testLongSequenceInput() {
        // 测试长序列输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 50, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = encoderLayerSmall.layerForward(inputVar);
        
        assertEquals("长序列输出形状应该正确", 
                   Shape.of(1, 50, 64), output.getValue().getShape());
        assertNotNull("长序列输出不应该为null", output.getValue());
    }

    @Test
    public void testLargeBatchInput() {
        // 测试大批次输入
        NdArray input = NdArray.likeRandomN(Shape.of(8, 15, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = encoderLayerSmall.layerForward(inputVar);
        
        assertEquals("大批次输出形状应该正确", 
                   Shape.of(8, 15, 64), output.getValue().getShape());
        assertNotNull("大批次输出不应该为null", output.getValue());
    }

    @Test
    public void testBackwardMethod() {
        // 测试backward方法
        NdArray grad = NdArray.likeRandomN(Shape.of(2, 5, 64));
        
        try {
            encoderLayerSmall.backward(grad);
            // backward方法应该能正常执行而不抛出异常
        } catch (Exception e) {
            fail("backward方法不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testDefaultConstructor() {
        // 测试默认构造函数
        assertNotNull("默认构造函数应该正常工作", encoderLayerDefault.getSelfAttention());
        assertNotNull("默认构造函数应该正常工作", encoderLayerDefault.getFeedForward());
        
        // 测试默认构造函数的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(1, 5, 128));
        Variable inputVar = new Variable(input);
        
        Variable output = encoderLayerDefault.layerForward(inputVar);
        assertEquals("默认配置输出形状应该正确", 
                   Shape.of(1, 5, 128), output.getValue().getShape());
    }

    @Test
    public void testNumericalStability() {
        // 测试数值稳定性，使用极端值
        float[][][] extremeData = new float[1][3][64];
        
        // 填充极端值
        for (int s = 0; s < 3; s++) {
            for (int d = 0; d < 64; d++) {
                if (d % 3 == 0) {
                    extremeData[0][s][d] = 10.0f;   // 较大值
                } else if (d % 3 == 1) {
                    extremeData[0][s][d] = -10.0f;  // 较大负值
                } else {
                    extremeData[0][s][d] = 1e-6f;   // 小值
                }
            }
        }
        
        NdArray input = NdArray.of(extremeData);
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = encoderLayerSmall.layerForward(inputVar);
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
    public void testOutputValueRange() {
        // 测试输出值的合理性
        NdArray input = NdArray.likeRandomN(Shape.of(2, 4, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = encoderLayerSmall.layerForward(inputVar);
        NdArray outputArray = output.getValue();
        
        // 检查输出值在合理范围内
        float[][][] outputData = outputArray.get3dArray();
        
        for (int b = 0; b < outputData.length; b++) {
            for (int s = 0; s < outputData[b].length; s++) {
                for (int d = 0; d < outputData[b][s].length; d++) {
                    float val = outputData[b][s][d];
                    // 检查值在合理范围内（避免梯度爆炸）
                    assertTrue("输出值应该在合理范围内", Math.abs(val) < 100.0f);
                }
            }
        }
    }

    @Test
    public void testConsistency() {
        // 测试多次调用的一致性
        NdArray input = NdArray.likeRandomN(Shape.of(1, 3, 64));
        Variable inputVar = new Variable(input);
        
        Variable output1 = encoderLayerSmall.layerForward(inputVar);
        Variable output2 = encoderLayerSmall.layerForward(inputVar);
        
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
    public void testResidualConnection() {
        // 测试残差连接的效果
        // 使用全零输入测试残差连接
        NdArray zeroInput = NdArray.zeros(Shape.of(1, 2, 64));
        Variable zeroInputVar = new Variable(zeroInput);
        
        Variable output = encoderLayerSmall.layerForward(zeroInputVar);
        
        // 由于有残差连接和层归一化，输出不应该是全零
        // （注意：层归一化可能会使零输入保持为零，这里主要测试不会出错）
        assertNotNull("残差连接输出应该有效", output.getValue());
        assertEquals("残差连接输出形状应该正确", 
                   Shape.of(1, 2, 64), output.getValue().getShape());
    }
}