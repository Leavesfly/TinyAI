package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * FeedForward层的全面单元测试
 * 
 * 测试前馈神经网络层的功能：
 * 1. 参数初始化和子层创建
 * 2. 不同构造函数
 * 3. 前向传播计算
 * 4. 维度变换正确性
 * 5. 边界情况处理
 */
public class FeedForwardTest {

    private FeedForward feedForward512;
    private FeedForward feedForwardCustom;
    private FeedForward feedForwardSmall;
    
    @Before
    public void setUp() {
        // 标准配置：512维度，默认隐藏层为4倍
        feedForward512 = new FeedForward("ff512", 512);
        
        // 自定义配置：256维度，隐藏层1024
        feedForwardCustom = new FeedForward("ff_custom", 256, 1024);
        
        // 小型配置：用于快速测试
        feedForwardSmall = new FeedForward("ff_small", 64, 128);
    }

    @Test
    public void testParameterInitialization() {
        // 测试子层是否正确初始化
        LinearLayer firstLinear = feedForward512.getFirstLinear();
        ReLuLayer activation = feedForward512.getActivation();
        LinearLayer secondLinear = feedForward512.getSecondLinear();
        
        assertNotNull("第一个线性层应该被初始化", firstLinear);
        assertNotNull("激活函数层应该被初始化", activation);
        assertNotNull("第二个线性层应该被初始化", secondLinear);
        
        // 验证层名称
        assertEquals("第一个线性层名称应该正确", "ff512_linear1", firstLinear.getName());
        assertEquals("激活函数层名称应该正确", "ff512_relu", activation.getName());
        assertEquals("第二个线性层名称应该正确", "ff512_linear2", secondLinear.getName());
    }

    @Test
    public void testDefaultConstructor() {
        // 测试默认构造函数（隐藏层为4倍dModel）
        FeedForward defaultFF = new FeedForward("default", 128);
        
        assertNotNull("默认构造函数应该正常工作", defaultFF.getFirstLinear());
        assertNotNull("默认构造函数应该正常工作", defaultFF.getSecondLinear());
    }

    @Test
    public void testCustomDimensions() {
        // 测试自定义维度配置
        LinearLayer firstLinear = feedForwardCustom.getFirstLinear();
        LinearLayer secondLinear = feedForwardCustom.getSecondLinear();
        
        assertNotNull("自定义维度的第一个线性层应该存在", firstLinear);
        assertNotNull("自定义维度的第二个线性层应该存在", secondLinear);
    }

    @Test
    public void testRequireInputNum() {
        // 测试requireInputNum方法
        assertEquals("FeedForward应该需要1个输入", 1, feedForward512.requireInputNum());
        assertEquals("自定义FeedForward应该需要1个输入", 1, feedForwardCustom.requireInputNum());
    }

    @Test
    public void testForwardPropagation() {
        // 测试基本的前向传播
        float[][][] inputData = new float[2][8][64]; // batch_size=2, seq_len=8, d_model=64
        
        // 填充测试数据
        for (int b = 0; b < 2; b++) {
            for (int s = 0; s < 8; s++) {
                for (int d = 0; d < 64; d++) {
                    inputData[b][s][d] = (float) (Math.random() - 0.5);
                }
            }
        }
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = feedForwardSmall.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 8, 64), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testForwardMethod() {
        // 测试forward方法（直接使用NdArray）
        NdArray input = NdArray.likeRandomN(Shape.of(3, 10, 64));
        
        NdArray output = feedForwardSmall.forward(input);
        
        assertNotNull("forward方法应该返回有效输出", output);
        assertEquals("forward输出形状应该正确", input.getShape(), output.getShape());
    }

    @Test
    public void testSingleBatchInput() {
        // 测试单批次输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 5, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = feedForwardSmall.layerForward(inputVar);
        
        assertEquals("单批次输出形状应该正确", Shape.of(1, 5, 64), output.getValue().getShape());
        assertNotNull("单批次输出不应该为null", output.getValue());
    }

    @Test
    public void testLongSequenceInput() {
        // 测试长序列输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 100, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = feedForwardSmall.layerForward(inputVar);
        
        assertEquals("长序列输出形状应该正确", Shape.of(1, 100, 64), output.getValue().getShape());
        assertNotNull("长序列输出不应该为null", output.getValue());
    }

    @Test
    public void testLargeBatchInput() {
        // 测试大批次输入
        NdArray input = NdArray.likeRandomN(Shape.of(16, 20, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = feedForwardSmall.layerForward(inputVar);
        
        assertEquals("大批次输出形状应该正确", Shape.of(16, 20, 64), output.getValue().getShape());
        assertNotNull("大批次输出不应该为null", output.getValue());
    }

    @Test
    public void testBackwardMethod() {
        // 测试backward方法
        NdArray grad = NdArray.likeRandomN(Shape.of(2, 5, 64));
        
        try {
            feedForwardSmall.backward(grad);
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
                    extremeData[0][s][d] = 1000.0f;  // 大值
                } else if (d % 3 == 1) {
                    extremeData[0][s][d] = -1000.0f; // 大负值
                } else {
                    extremeData[0][s][d] = 1e-8f;    // 小值
                }
            }
        }
        
        NdArray input = NdArray.of(extremeData);
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = feedForwardSmall.layerForward(inputVar);
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
        
        Variable output = feedForwardSmall.layerForward(inputVar);
        NdArray outputArray = output.getValue();
        
        // 由于使用了ReLU激活，输出应该都是非负的（经过第一个线性层和ReLU后）
        // 但第二个线性层可能产生负值，所以我们只检查输出是否在合理范围内
        float[][][] outputData = outputArray.get3dArray();
        boolean hasPositiveValues = false;
        
        for (int b = 0; b < outputData.length; b++) {
            for (int s = 0; s < outputData[b].length; s++) {
                for (int d = 0; d < outputData[b][s].length; d++) {
                    float val = outputData[b][s][d];
                    if (val > 0) {
                        hasPositiveValues = true;
                    }
                    // 检查值在合理范围内（避免梯度爆炸）
                    assertTrue("输出值应该在合理范围内", Math.abs(val) < 1000.0f);
                }
            }
        }
        
        // 至少应该有一些正值（考虑到ReLU的作用）
        assertTrue("输出应该包含一些正值", hasPositiveValues);
    }

    @Test
    public void testConsistency() {
        // 测试多次调用的一致性
        NdArray input = NdArray.likeRandomN(Shape.of(1, 3, 64));
        Variable inputVar = new Variable(input);
        
        Variable output1 = feedForwardSmall.layerForward(inputVar);
        Variable output2 = feedForwardSmall.layerForward(inputVar);
        
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
}