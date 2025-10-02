package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * PositionalEncoding层的全面单元测试
 * 
 * 测试位置编码层的功能：
 * 1. 参数初始化和位置编码生成
 * 2. 不同配置参数
 * 3. 前向传播计算
 * 4. 位置编码的数学正确性
 * 5. 不同序列长度处理
 * 6. 边界情况处理
 */
public class PositionalEncodingTest {

    private PositionalEncoding posEnc512;
    private PositionalEncoding posEnc256;
    private PositionalEncoding posEncSmall;
    private PositionalEncoding posEncWithDropout;
    
    @Before
    public void setUp() {
        // 标准配置：512维度，最大序列长度1000
        posEnc512 = new PositionalEncoding("pe512", 512, 1000, 0.0);
        
        // 较小配置：256维度，最大序列长度500
        posEnc256 = new PositionalEncoding("pe256", 256, 500, 0.0);
        
        // 小型配置用于快速测试：64维度，最大序列长度100
        posEncSmall = new PositionalEncoding("pe_small", 64, 100, 0.0);
        
        // 带dropout的配置
        posEncWithDropout = new PositionalEncoding("pe_dropout", 128, 200, 0.1);
    }

    @Test
    public void testParameterInitialization() {
        // 测试位置编码矩阵是否正确初始化
        // 由于位置编码是在init()方法中生成的，我们通过前向传播来验证
        NdArray input = NdArray.likeRandomN(Shape.of(1, 10, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = posEncSmall.layerForward(inputVar);
        
        assertNotNull("位置编码输出不应该为null", output.getValue());
        assertEquals("位置编码输出形状应该与输入相同", 
                   input.getShape(), output.getValue().getShape());
    }

    @Test
    public void testRequireInputNum() {
        // 测试requireInputNum方法
        assertEquals("PositionalEncoding应该需要1个输入", 1, posEnc512.requireInputNum());
        assertEquals("带dropout的PositionalEncoding应该需要1个输入", 1, posEncWithDropout.requireInputNum());
    }

    @Test
    public void testForwardPropagation() {
        // 测试基本的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(2, 20, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = posEncSmall.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", 
                   Shape.of(2, 20, 64), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testForwardMethod() {
        // 测试forward方法（直接使用NdArray）
        NdArray input = NdArray.likeRandomN(Shape.of(1, 15, 64));
        
        NdArray output = posEncSmall.forward(input);
        
        assertNotNull("forward方法应该返回有效输出", output);
        assertEquals("forward输出形状应该正确", input.getShape(), output.getShape());
    }

    @Test
    public void testSingleTokenInput() {
        // 测试单个token输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 1, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = posEncSmall.layerForward(inputVar);
        
        assertEquals("单token输出形状应该正确", 
                   Shape.of(1, 1, 64), output.getValue().getShape());
        assertNotNull("单token输出不应该为null", output.getValue());
    }

    @Test
    public void testMaxSequenceLengthInput() {
        // 测试最大序列长度输入
        NdArray input = NdArray.likeRandomN(Shape.of(1, 100, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = posEncSmall.layerForward(inputVar);
        
        assertEquals("最大序列长度输出形状应该正确", 
                   Shape.of(1, 100, 64), output.getValue().getShape());
        assertNotNull("最大序列长度输出不应该为null", output.getValue());
    }

    @Test
    public void testLargeBatchInput() {
        // 测试大批次输入
        NdArray input = NdArray.likeRandomN(Shape.of(8, 30, 64));
        Variable inputVar = new Variable(input);
        
        Variable output = posEncSmall.layerForward(inputVar);
        
        assertEquals("大批次输出形状应该正确", 
                   Shape.of(8, 30, 64), output.getValue().getShape());
        assertNotNull("大批次输出不应该为null", output.getValue());
    }

    @Test
    public void testBackwardMethod() {
        // 测试backward方法
        NdArray grad = NdArray.likeRandomN(Shape.of(2, 10, 64));
        
        try {
            posEncSmall.backward(grad);
            // backward方法应该能正常执行而不抛出异常
        } catch (Exception e) {
            fail("backward方法不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testPositionalEncodingAddition() {
        // 测试位置编码是否正确添加到输入上
        // 使用全零输入，这样输出就是纯位置编码
        NdArray zeroInput = NdArray.zeros(Shape.of(1, 5, 64));
        Variable zeroInputVar = new Variable(zeroInput);
        
        Variable output = posEncSmall.layerForward(zeroInputVar);
        
        // 输出应该不等于零输入（因为添加了位置编码）
        NdArray outputArray = output.getValue();
        boolean hasNonZeroValues = false;
        
        float[][][] outputData = outputArray.get3dArray();
        for (int b = 0; b < outputData.length; b++) {
            for (int s = 0; s < outputData[b].length; s++) {
                for (int d = 0; d < outputData[b][s].length; d++) {
                    if (Math.abs(outputData[b][s][d]) > 1e-6f) {
                        hasNonZeroValues = true;
                        break;
                    }
                }
                if (hasNonZeroValues) break;
            }
            if (hasNonZeroValues) break;
        }
        
        assertTrue("位置编码应该添加非零值", hasNonZeroValues);
    }

    @Test
    public void testPositionalEncodingConsistency() {
        // 测试相同位置的位置编码应该一致
        NdArray input1 = NdArray.zeros(Shape.of(1, 3, 64));
        NdArray input2 = NdArray.zeros(Shape.of(2, 3, 64));  // 不同批次大小但相同序列长度
        
        Variable inputVar1 = new Variable(input1);
        Variable inputVar2 = new Variable(input2);
        
        Variable output1 = posEncSmall.layerForward(inputVar1);
        Variable output2 = posEncSmall.layerForward(inputVar2);
        
        // 第一个批次的位置编码应该与第二个批次的第一个样本相同
        float[][][] data1 = output1.getValue().get3dArray();
        float[][][] data2 = output2.getValue().get3dArray();
        
        for (int s = 0; s < 3; s++) {
            for (int d = 0; d < 64; d++) {
                assertEquals("相同位置的位置编码应该一致", 
                           data1[0][s][d], data2[0][s][d], 1e-6f);
                assertEquals("不同批次相同位置的位置编码应该一致", 
                           data2[0][s][d], data2[1][s][d], 1e-6f);
            }
        }
    }

    @Test
    public void testPositionalEncodingMathematicalProperties() {
        // 测试位置编码的数学性质
        // 位置编码使用sin和cos函数，应该在[-1, 1]范围内
        NdArray zeroInput = NdArray.zeros(Shape.of(1, 10, 64));
        Variable zeroInputVar = new Variable(zeroInput);
        
        Variable output = posEncSmall.layerForward(zeroInputVar);
        NdArray outputArray = output.getValue();
        
        float[][][] outputData = outputArray.get3dArray();
        for (int b = 0; b < outputData.length; b++) {
            for (int s = 0; s < outputData[b].length; s++) {
                for (int d = 0; d < outputData[b][s].length; d++) {
                    float val = outputData[b][s][d];
                    assertTrue("位置编码值应该在合理范围内", Math.abs(val) <= 2.0f);
                    assertFalse("位置编码不应该包含NaN", Float.isNaN(val));
                    assertFalse("位置编码不应该包含无穷大", Float.isInfinite(val));
                }
            }
        }
    }

    @Test
    public void testDifferentDimensions() {
        // 测试不同维度的位置编码
        NdArray input256 = NdArray.likeRandomN(Shape.of(1, 10, 256));
        NdArray input512 = NdArray.likeRandomN(Shape.of(1, 10, 512));
        
        Variable inputVar256 = new Variable(input256);
        Variable inputVar512 = new Variable(input512);
        
        Variable output256 = posEnc256.layerForward(inputVar256);
        Variable output512 = posEnc512.layerForward(inputVar512);
        
        assertEquals("256维输出形状应该正确", 
                   Shape.of(1, 10, 256), output256.getValue().getShape());
        assertEquals("512维输出形状应该正确", 
                   Shape.of(1, 10, 512), output512.getValue().getShape());
        
        assertNotNull("256维输出不应该为null", output256.getValue());
        assertNotNull("512维输出不应该为null", output512.getValue());
    }

    @Test
    public void testWithDropout() {
        // 测试带dropout的位置编码
        NdArray input = NdArray.likeRandomN(Shape.of(2, 15, 128));
        Variable inputVar = new Variable(input);
        
        Variable output = posEncWithDropout.layerForward(inputVar);
        
        assertEquals("带dropout的输出形状应该正确", 
                   Shape.of(2, 15, 128), output.getValue().getShape());
        assertNotNull("带dropout的输出不应该为null", output.getValue());
    }

    @Test
    public void testSequenceLengthVariations() {
        // 测试不同序列长度的处理
        int[] seqLengths = {1, 5, 10, 25, 50};
        
        for (int seqLen : seqLengths) {
            if (seqLen <= 100) { // 确保不超过最大序列长度
                NdArray input = NdArray.likeRandomN(Shape.of(1, seqLen, 64));
                Variable inputVar = new Variable(input);
                
                Variable output = posEncSmall.layerForward(inputVar);
                
                assertEquals("序列长度" + seqLen + "的输出形状应该正确", 
                           Shape.of(1, seqLen, 64), output.getValue().getShape());
                assertNotNull("序列长度" + seqLen + "的输出不应该为null", output.getValue());
            }
        }
    }

    @Test
    public void testInputPreservation() {
        // 测试输入值是否被正确保留并与位置编码相加
        float[][][] knownInput = {{{1.0f, 2.0f, 3.0f, 4.0f}}};
        
        // 扩展到所需维度
        float[][][] inputData = new float[1][1][64];
        for (int d = 0; d < 64; d++) {
            if (d < 4) {
                inputData[0][0][d] = knownInput[0][0][d];
            } else {
                inputData[0][0][d] = 0.0f;
            }
        }
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = posEncSmall.layerForward(inputVar);
        
        // 验证输出包含了原始输入加上位置编码
        assertNotNull("输出应该有效", output.getValue());
        assertEquals("输出形状应该正确", Shape.of(1, 1, 64), output.getValue().getShape());
        
        // 使用全零输入来获取纯位置编码
        NdArray zeroInput = NdArray.zeros(Shape.of(1, 1, 64));
        Variable zeroInputVar = new Variable(zeroInput);
        Variable posOutput = posEncSmall.layerForward(zeroInputVar);
        
        // 验证输出 = 输入 + 位置编码
        float[][][] outputData = output.getValue().get3dArray();
        float[][][] posData = posOutput.getValue().get3dArray();
        
        for (int d = 0; d < 4; d++) {
            float expected = inputData[0][0][d] + posData[0][0][d];
            assertEquals("输出应该等于输入加位置编码", 
                       expected, outputData[0][0][d], 1e-6f);
        }
    }

    @Test
    public void testConsistency() {
        // 测试多次调用的一致性
        NdArray input = NdArray.likeRandomN(Shape.of(1, 5, 64));
        Variable inputVar = new Variable(input);
        
        Variable output1 = posEncSmall.layerForward(inputVar);
        Variable output2 = posEncSmall.layerForward(inputVar);
        
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