package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Conv2d卷积操作的单元测试
 * 
 * @author TinyAI Team
 */
public class Conv2dTest {

    private boolean originalTrainMode;

    @Before
    public void setUp() {
        originalTrainMode = Config.train;
        Config.train = true; // 启用训练模式以构建计算图
    }

    @After
    public void tearDown() {
        Config.train = originalTrainMode;
    }

    // =============================================================================
    // 基础功能测试
    // =============================================================================

    @Test
    public void testBasicConv2d() {
        // 测试基本的卷积操作
        Conv2d conv = new Conv2d(1, 0); // stride=1, padding=0
        
        // 输入: [1, 1, 3, 3]
        float[][][][] inputData = {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}};
        NdArray input = NdArray.of(inputData);
        
        // 卷积核: [1, 1, 2, 2]
        float[][][][] kernelData = {{{{1, 0}, {0, 1}}}};
        NdArray kernel = NdArray.of(kernelData);
        
        // 前向传播
        NdArray output = conv.forward(input, kernel);
        
        // 验证输出形状: (3-2)/1+1 = 2
        assertEquals(Shape.of(1, 1, 2, 2), output.getShape());
        
        // 验证输出值
        // output[0,0,0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
        // output[0,0,0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
        // output[0,0,1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
        // output[0,0,1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
        assertEquals(6f, output.get(0, 0, 0, 0), 1e-6);
        assertEquals(8f, output.get(0, 0, 0, 1), 1e-6);
        assertEquals(12f, output.get(0, 0, 1, 0), 1e-6);
        assertEquals(14f, output.get(0, 0, 1, 1), 1e-6);
    }

    @Test
    public void testConv2dWithPadding() {
        // 测试带padding的卷积
        Conv2d conv = new Conv2d(1, 1); // stride=1, padding=1
        
        // 输入: [1, 1, 3, 3]
        float[][][][] inputData = {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}};
        NdArray input = NdArray.of(inputData);
        
        // 卷积核: [1, 1, 3, 3]
        float[][][][] kernelData = {{{{1, 0, -1}, {1, 0, -1}, {1, 0, -1}}}};
        NdArray kernel = NdArray.of(kernelData);
        
        // 前向传播
        NdArray output = conv.forward(input, kernel);
        
        // 验证输出形状: (3+2*1-3)/1+1 = 3 (same padding)
        assertEquals(Shape.of(1, 1, 3, 3), output.getShape());
        
        // padding后输入变为5x5，中心是原始3x3
        assertNotNull(output);
    }

    @Test
    public void testConv2dWithStride() {
        // 测试带stride的卷积（下采样）
        Conv2d conv = new Conv2d(2, 0); // stride=2, padding=0
        
        // 输入: [1, 1, 4, 4]
        float[][][][] inputData = {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}};
        NdArray input = NdArray.of(inputData);
        
        // 卷积核: [1, 1, 2, 2]
        float[][][][] kernelData = {{{{1, 1}, {1, 1}}}};
        NdArray kernel = NdArray.of(kernelData);
        
        // 前向传播
        NdArray output = conv.forward(input, kernel);
        
        // 验证输出形状: (4-2)/2+1 = 2
        assertEquals(Shape.of(1, 1, 2, 2), output.getShape());
        
        // output[0,0,0,0] = 1+2+5+6 = 14
        // output[0,0,0,1] = 3+4+7+8 = 22
        // output[0,0,1,0] = 9+10+13+14 = 46
        // output[0,0,1,1] = 11+12+15+16 = 54
        assertEquals(14f, output.get(0, 0, 0, 0), 1e-6);
        assertEquals(22f, output.get(0, 0, 0, 1), 1e-6);
        assertEquals(46f, output.get(0, 0, 1, 0), 1e-6);
        assertEquals(54f, output.get(0, 0, 1, 1), 1e-6);
    }

    @Test
    public void testConv2dMultiChannel() {
        // 测试多通道卷积
        Conv2d conv = new Conv2d(1, 0); // stride=1, padding=0
        
        // 输入: [1, 2, 3, 3] - 2个输入通道
        NdArray input = NdArray.zeros(Shape.of(1, 2, 3, 3));
        
        // 通道0全部为1
        for (int h = 0; h < 3; h++) {
            for (int w = 0; w < 3; w++) {
                input.set(1f, 0, 0, h, w);
            }
        }
        
        // 通道1全部为2
        for (int h = 0; h < 3; h++) {
            for (int w = 0; w < 3; w++) {
                input.set(2f, 0, 1, h, w);
            }
        }
        
        // 卷积核: [1, 2, 2, 2] - 1个输出通道，2个输入通道
        NdArray kernel = NdArray.zeros(Shape.of(1, 2, 2, 2));
        
        // 第一个输入通道的kernel全为1
        for (int kh = 0; kh < 2; kh++) {
            for (int kw = 0; kw < 2; kw++) {
                kernel.set(1f, 0, 0, kh, kw);
            }
        }
        
        // 第二个输入通道的kernel全为0.5
        for (int kh = 0; kh < 2; kh++) {
            for (int kw = 0; kw < 2; kw++) {
                kernel.set(0.5f, 0, 1, kh, kw);
            }
        }
        
        // 前向传播
        NdArray output = conv.forward(input, kernel);
        
        // 验证输出形状
        assertEquals(Shape.of(1, 1, 2, 2), output.getShape());
        
        // 每个位置的值 = (1*1*4) + (2*0.5*4) = 4 + 4 = 8
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                assertEquals(8f, output.get(0, 0, h, w), 1e-6);
            }
        }
    }

    @Test
    public void testConv2dMultiOutputChannel() {
        // 测试多输出通道卷积
        Conv2d conv = new Conv2d(1, 0);
        
        // 输入: [1, 1, 3, 3]
        NdArray input = NdArray.ones(Shape.of(1, 1, 3, 3));
        
        // 卷积核: [2, 1, 2, 2] - 2个输出通道
        NdArray kernel = NdArray.zeros(Shape.of(2, 1, 2, 2));
        
        // 第一个输出通道的kernel全为1
        for (int kh = 0; kh < 2; kh++) {
            for (int kw = 0; kw < 2; kw++) {
                kernel.set(1f, 0, 0, kh, kw);
            }
        }
        
        // 第二个输出通道的kernel全为2
        for (int kh = 0; kh < 2; kh++) {
            for (int kw = 0; kw < 2; kw++) {
                kernel.set(2f, 1, 0, kh, kw);
            }
        }
        
        // 前向传播
        NdArray output = conv.forward(input, kernel);
        
        // 验证输出形状
        assertEquals(Shape.of(1, 2, 2, 2), output.getShape());
        
        // 第一个输出通道的值应该全为4 (1*4)
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                assertEquals(4f, output.get(0, 0, h, w), 1e-6);
            }
        }
        
        // 第二个输出通道的值应该全为8 (2*4)
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                assertEquals(8f, output.get(0, 1, h, w), 1e-6);
            }
        }
    }

    @Test
    public void testConv2dBatchProcessing() {
        // 测试批处理
        Conv2d conv = new Conv2d(1, 0);
        
        // 输入: [2, 1, 3, 3] - batch_size=2
        NdArray input = NdArray.zeros(Shape.of(2, 1, 3, 3));
        
        // 第一个batch全为1
        for (int h = 0; h < 3; h++) {
            for (int w = 0; w < 3; w++) {
                input.set(1f, 0, 0, h, w);
            }
        }
        
        // 第二个batch全为2
        for (int h = 0; h < 3; h++) {
            for (int w = 0; w < 3; w++) {
                input.set(2f, 1, 0, h, w);
            }
        }
        
        // 卷积核: [1, 1, 2, 2]
        NdArray kernel = NdArray.ones(Shape.of(1, 1, 2, 2));
        
        // 前向传播
        NdArray output = conv.forward(input, kernel);
        
        // 验证输出形状
        assertEquals(Shape.of(2, 1, 2, 2), output.getShape());
        
        // 第一个batch的输出应该全为4
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                assertEquals(4f, output.get(0, 0, h, w), 1e-6);
            }
        }
        
        // 第二个batch的输出应该全为8
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                assertEquals(8f, output.get(1, 0, h, w), 1e-6);
            }
        }
    }

    // =============================================================================
    // 反向传播测试
    // =============================================================================

    @Test
    public void testConv2dBackward() {
        // 测试卷积的反向传播（计算图完整性）
        Conv2d conv = new Conv2d(1, 0);
        
        // 输入: [1, 1, 3, 3]
        float[][][][] inputData = {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}};
        NdArray inputNd = NdArray.of(inputData);
        
        // 卷积核: [1, 1, 2, 2]
        float[][][][] kernelData = {{{{1, 0}, {0, 1}}}};
        NdArray kernelNd = NdArray.of(kernelData);
        
        // 创建Variable
        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);
        
        // 前向传播
        Variable output = conv.call(input, kernel);
        
        // 反向传播
        output.backward();
        
        // 验证梯度存在
        assertNotNull(input.getGrad());
        assertNotNull(kernel.getGrad());
        
        // 验证梯度形状与输入形状相同
        assertEquals(input.getValue().getShape(), input.getGrad().getShape());
        assertEquals(kernel.getValue().getShape(), kernel.getGrad().getShape());
    }

    @Test
    public void testConv2dGradientValues() {
        // 测试卷积梯度的具体数值
        Conv2d conv = new Conv2d(1, 0);
        
        // 简单的输入: [1, 1, 2, 2]
        float[][][][] inputData = {{{{1, 2}, {3, 4}}}};
        NdArray inputNd = NdArray.of(inputData);
        
        // 简单的卷积核: [1, 1, 2, 2]
        float[][][][] kernelData = {{{{1, 1}, {1, 1}}}};
        NdArray kernelNd = NdArray.of(kernelData);
        
        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);
        
        // 前向传播: output = 1+2+3+4 = 10 (单个值)
        Variable output = conv.call(input, kernel);
        
        // 验证前向结果
        assertEquals(10f, output.getValue().get(0, 0, 0, 0), 1e-6);
        
        // 反向传播
        output.backward();
        
        // 输入梯度应该等于kernel的值（因为输出梯度为1）
        float[][] expectedInputGrad = {{1, 1}, {1, 1}};
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedInputGrad[i][j], 
                    input.getGrad().get(0, 0, i, j), 1e-6);
            }
        }
        
        // kernel梯度应该等于输入的值（因为输出梯度为1）
        float[][] expectedKernelGrad = {{1, 2}, {3, 4}};
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedKernelGrad[i][j], 
                    kernel.getGrad().get(0, 0, i, j), 1e-6);
            }
        }
    }

    @Test
    public void testConv2dChainRule() {
        // 测试卷积在计算图中的链式法则
        Conv2d conv = new Conv2d(1, 0);
        
        NdArray inputNd = NdArray.ones(Shape.of(1, 1, 3, 3));
        NdArray kernelNd = NdArray.ones(Shape.of(1, 1, 2, 2));
        
        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);
        
        // 构建链式计算: conv -> sum
        Variable convOut = conv.call(input, kernel);
        Variable sum = convOut.sum();
        
        // 验证前向传播
        // conv输出: [1, 1, 2, 2]，每个值都是4
        // sum: 4*4 = 16
        assertEquals(16f, sum.getValue().getNumber().floatValue(), 1e-6);
        
        // 反向传播
        sum.backward();
        
        // 验证梯度传播
        assertNotNull(input.getGrad());
        assertNotNull(kernel.getGrad());
    }

    // =============================================================================
    // 边界条件和异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testConv2dInvalidInputDimension() {
        // 测试错误的输入维度
        Conv2d conv = new Conv2d(1, 0);
        
        // 3D输入而不是4D
        NdArray input = NdArray.ones(Shape.of(3, 3, 3));
        NdArray kernel = NdArray.ones(Shape.of(1, 1, 2, 2));
        
        conv.forward(input, kernel); // 应该抛出异常
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConv2dInvalidKernelDimension() {
        // 测试错误的kernel维度
        Conv2d conv = new Conv2d(1, 0);
        
        NdArray input = NdArray.ones(Shape.of(1, 1, 3, 3));
        NdArray kernel = NdArray.ones(Shape.of(2, 2)); // 2D而不是4D
        
        conv.forward(input, kernel); // 应该抛出异常
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConv2dChannelMismatch() {
        // 测试输入通道数与kernel不匹配
        Conv2d conv = new Conv2d(1, 0);
        
        // 输入有2个通道
        NdArray input = NdArray.ones(Shape.of(1, 2, 3, 3));
        
        // 但kernel只针对1个输入通道
        NdArray kernel = NdArray.ones(Shape.of(1, 1, 2, 2));
        
        conv.forward(input, kernel); // 应该抛出异常
    }

    @Test
    public void testConv2dMinimalSize() {
        // 测试最小尺寸的卷积
        Conv2d conv = new Conv2d(1, 0);
        
        // 输入和kernel都是最小尺寸
        NdArray input = NdArray.ones(Shape.of(1, 1, 1, 1));
        NdArray kernel = NdArray.ones(Shape.of(1, 1, 1, 1));
        
        NdArray output = conv.forward(input, kernel);
        
        // 输出应该是1x1
        assertEquals(Shape.of(1, 1, 1, 1), output.getShape());
        assertEquals(1f, output.get(0, 0, 0, 0), 1e-6);
    }

    @Test
    public void testConv2dLargeKernel() {
        // 测试大卷积核
        Conv2d conv = new Conv2d(1, 2); // padding=2以保持尺寸
        
        // 输入: [1, 1, 5, 5]
        NdArray input = NdArray.ones(Shape.of(1, 1, 5, 5));
        
        // 大卷积核: [1, 1, 5, 5]
        NdArray kernel = NdArray.ones(Shape.of(1, 1, 5, 5));
        
        NdArray output = conv.forward(input, kernel);
        
        // 验证输出形状: (5+2*2-5)/1+1 = 5
        assertEquals(Shape.of(1, 1, 5, 5), output.getShape());
    }

    @Test
    public void testConv2dZeroInput() {
        // 测试全零输入
        Conv2d conv = new Conv2d(1, 0);
        
        NdArray input = NdArray.zeros(Shape.of(1, 1, 3, 3));
        NdArray kernel = NdArray.ones(Shape.of(1, 1, 2, 2));
        
        NdArray output = conv.forward(input, kernel);
        
        // 输出应该全为0
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                assertEquals(0f, output.get(0, 0, h, w), 1e-6);
            }
        }
    }

    @Test
    public void testConv2dZeroKernel() {
        // 测试全零kernel
        Conv2d conv = new Conv2d(1, 0);
        
        NdArray input = NdArray.ones(Shape.of(1, 1, 3, 3));
        NdArray kernel = NdArray.zeros(Shape.of(1, 1, 2, 2));
        
        NdArray output = conv.forward(input, kernel);
        
        // 输出应该全为0
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                assertEquals(0f, output.get(0, 0, h, w), 1e-6);
            }
        }
    }

    // =============================================================================
    // Variable算子集成测试
    // =============================================================================

    @Test
    public void testConv2dVariableOperator() {
        // 测试Variable.conv2d()算子
        NdArray inputNd = NdArray.ones(Shape.of(1, 1, 3, 3));
        NdArray kernelNd = NdArray.ones(Shape.of(1, 1, 2, 2));
        
        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);
        
        // 使用Variable算子
        Variable output = input.conv2d(kernel, 1, 0);
        
        // 验证输出
        assertNotNull(output);
        assertEquals(Shape.of(1, 1, 2, 2), output.getValue().getShape());
        
        // 验证计算图连接
        assertNotNull(output.getCreator());
        assertTrue(output.getCreator() instanceof Conv2d);
    }

    @Test
    public void testConv2dVariableOperatorDefaultParams() {
        // 测试Variable.conv2d()默认参数
        NdArray inputNd = NdArray.ones(Shape.of(1, 1, 3, 3));
        NdArray kernelNd = NdArray.ones(Shape.of(1, 1, 2, 2));
        
        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);
        
        // 使用默认参数 (stride=1, padding=0)
        Variable output = input.conv2d(kernel);
        
        assertNotNull(output);
        assertEquals(Shape.of(1, 1, 2, 2), output.getValue().getShape());
    }

    @Test
    public void testConv2dInComputationGraph() {
        // 测试卷积在复杂计算图中的表现
        NdArray inputNd = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        inputNd = inputNd.reshape(Shape.of(1, 1, 2, 2));
        
        NdArray kernelNd = NdArray.ones(Shape.of(2, 1, 2, 2));
        
        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);
        
        // 构建计算图: input -> conv2d -> relu -> sum
        Variable conv = input.conv2d(kernel, 1, 0);
        
        // 验证卷积输出
        assertEquals(Shape.of(1, 2, 1, 1), conv.getValue().getShape());
        
        // 验证计算图完整性
        conv.backward();
        assertNotNull(input.getGrad());
        assertNotNull(kernel.getGrad());
    }

    // =============================================================================
    // 性能和数值稳定性测试
    // =============================================================================

    @Test
    public void testConv2dNumericalStability() {
        // 测试数值稳定性（大数值）
        Conv2d conv = new Conv2d(1, 0);
        
        NdArray input = NdArray.of(new float[][]{{1000f, 2000f}, {3000f, 4000f}});
        input = input.reshape(Shape.of(1, 1, 2, 2));
        
        NdArray kernel = NdArray.of(new float[][]{{0.001f, 0.002f}, {0.003f, 0.004f}});
        kernel = kernel.reshape(Shape.of(1, 1, 2, 2));
        
        NdArray output = conv.forward(input, kernel);
        
        // 验证计算正确性
        float expected = 1000*0.001f + 2000*0.002f + 3000*0.003f + 4000*0.004f;
        assertEquals(expected, output.get(0, 0, 0, 0), 1e-3);
    }

    @Test
    public void testConv2dRequireInputNum() {
        // 测试requireInputNum方法
        Conv2d conv = new Conv2d(1, 0);
        assertEquals(2, conv.requireInputNum());
    }

    // =============================================================================
    // 修复验证测试 - 验证kernel梯度在多通道场景下的正确性
    // =============================================================================

    @Test
    public void testKernelGradientMultiOutputChannel() {
        // 验证多输出通道时kernel梯度的数值正确性
        // 此测试可暴露之前 im2col^T @ yGradFlat 矩阵乘法方向错误的bug
        Conv2d conv = new Conv2d(1, 0);

        // 输入: [1, 1, 3, 3]，值为1-9
        float[][][][] inputData = {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}};
        NdArray inputNd = NdArray.of(inputData);

        // 卷积核: [2, 1, 2, 2]，两个输出通道
        // 第一个通道全1，第二个通道全2
        float[][][][] kernelData = {
            {{{1, 1}, {1, 1}}},
            {{{2, 2}, {2, 2}}}
        };
        NdArray kernelNd = NdArray.of(kernelData);

        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);

        // 前向传播 -> sum，使所有输出元素的梯度为1
        Variable output = conv.call(input, kernel);
        Variable sum = output.sum();
        sum.backward();

        // 验证kernel梯度形状
        NdArray kernelGrad = kernel.getGrad();
        assertEquals(Shape.of(2, 1, 2, 2), kernelGrad.getShape());

        // 手工计算kernel梯度:
        // output shape: [1, 2, 2, 2]
        // yGrad = ones([1, 2, 2, 2])（来自sum的反向传播）
        //
        // 对于kernel[oc, ic, kh, kw]的梯度 = sum over (b, oh, ow) of input[b, ic, oh*s+kh, ow*s+kw] * yGrad[b, oc, oh, ow]
        // yGrad全为1，所以:
        // kernelGrad[oc, 0, kh, kw] = sum over (oh, ow) of input[0, 0, oh+kh, ow+kw]
        // 与oc无关（因为yGrad全为1），所以两个输出通道的kernel梯度应该相同
        //
        // kernelGrad[*, 0, 0, 0] = input[0,0,0,0]+input[0,0,0,1]+input[0,0,1,0]+input[0,0,1,1] = 1+2+4+5 = 12
        // kernelGrad[*, 0, 0, 1] = input[0,0,0,1]+input[0,0,0,2]+input[0,0,1,1]+input[0,0,1,2] = 2+3+5+6 = 16
        // kernelGrad[*, 0, 1, 0] = input[0,0,1,0]+input[0,0,1,1]+input[0,0,2,0]+input[0,0,2,1] = 4+5+7+8 = 24
        // kernelGrad[*, 0, 1, 1] = input[0,0,1,1]+input[0,0,1,2]+input[0,0,2,1]+input[0,0,2,2] = 5+6+8+9 = 28
        float[][] expectedKernelGrad = {{12, 16}, {24, 28}};
        for (int oc = 0; oc < 2; oc++) {
            for (int kh = 0; kh < 2; kh++) {
                for (int kw = 0; kw < 2; kw++) {
                    assertEquals(
                        "kernelGrad[" + oc + ",0," + kh + "," + kw + "] mismatch",
                        expectedKernelGrad[kh][kw],
                        kernelGrad.get(oc, 0, kh, kw), 1e-5);
                }
            }
        }
    }

    @Test
    public void testKernelGradientWithPadding() {
        // 验证带padding时kernel梯度的正确性
        Conv2d conv = new Conv2d(1, 1);

        // 输入: [1, 1, 2, 2]
        float[][][][] inputData = {{{{1, 2}, {3, 4}}}};
        NdArray inputNd = NdArray.of(inputData);

        // 卷积核: [1, 1, 2, 2]
        float[][][][] kernelData = {{{{1, 1}, {1, 1}}}};
        NdArray kernelNd = NdArray.of(kernelData);

        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);

        // padding=1时，输出尺寸 = (2+2*1-2)/1+1 = 3
        Variable output = conv.call(input, kernel);
        assertEquals(Shape.of(1, 1, 3, 3), output.getValue().getShape());

        Variable sum = output.sum();
        sum.backward();

        // 验证kernel梯度
        // 带padding的输入（0填充后）:
        // 0 0 0 0
        // 0 1 2 0
        // 0 3 4 0
        // 0 0 0 0
        //
        // kernelGrad[0,0,kh,kw] = sum over (oh,ow) of padded_input[oh+kh, ow+kw]
        // 9个输出位置(3x3)，每个位置取2x2窗口
        // kernelGrad[0,0,0,0] = sum of padded[oh,ow] for oh=0..2, ow=0..2
        //   = 0+0+0 + 0+1+2 + 0+3+4 = 10
        // kernelGrad[0,0,0,1] = sum of padded[oh,ow+1] for oh=0..2, ow=0..2
        //   = 0+0+0 + 1+2+0 + 3+4+0 = 10
        // kernelGrad[0,0,1,0] = sum of padded[oh+1,ow] for oh=0..2, ow=0..2
        //   = 0+1+2 + 0+3+4 + 0+0+0 = 10
        // kernelGrad[0,0,1,1] = sum of padded[oh+1,ow+1] for oh=0..2, ow=0..2
        //   = 1+2+0 + 3+4+0 + 0+0+0 = 10
        NdArray kernelGrad = kernel.getGrad();
        for (int kh = 0; kh < 2; kh++) {
            for (int kw = 0; kw < 2; kw++) {
                assertEquals(
                    "kernelGrad[0,0," + kh + "," + kw + "] mismatch",
                    10f, kernelGrad.get(0, 0, kh, kw), 1e-5);
            }
        }
    }

    @Test
    public void testInputGradientWithStride() {
        // 验证带stride时输入梯度的正确性
        Conv2d conv = new Conv2d(2, 0);

        // 输入: [1, 1, 4, 4]
        NdArray inputNd = NdArray.ones(Shape.of(1, 1, 4, 4));

        // 卷积核: [1, 1, 2, 2]，值为[1,2,3,4]
        float[][][][] kernelData = {{{{1, 2}, {3, 4}}}};
        NdArray kernelNd = NdArray.of(kernelData);

        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);

        // stride=2时，输出尺寸 = (4-2)/2+1 = 2
        Variable output = conv.call(input, kernel);
        assertEquals(Shape.of(1, 1, 2, 2), output.getValue().getShape());

        Variable sum = output.sum();
        sum.backward();

        // 验证输入梯度形状
        NdArray inputGrad = input.getGrad();
        assertEquals(Shape.of(1, 1, 4, 4), inputGrad.getShape());

        // stride=2时，每个输入位置最多只被一个输出窗口覆盖（无重叠）
        // 输出位置(0,0)覆盖输入[0:2, 0:2]，梯度=kernel值
        // 输出位置(0,1)覆盖输入[0:2, 2:4]，梯度=kernel值
        // 输出位置(1,0)覆盖输入[2:4, 0:2]，梯度=kernel值
        // 输出位置(1,1)覆盖输入[2:4, 2:4]，梯度=kernel值
        float[][] expectedGrad = {
            {1, 2, 1, 2},
            {3, 4, 3, 4},
            {1, 2, 1, 2},
            {3, 4, 3, 4}
        };
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                assertEquals(
                    "inputGrad[0,0," + h + "," + w + "] mismatch",
                    expectedGrad[h][w], inputGrad.get(0, 0, h, w), 1e-5);
            }
        }
    }

    @Test
    public void testGradientMultiChannelMultiBatch() {
        // 验证多通道+多batch场景下梯度的形状和非零性
        Conv2d conv = new Conv2d(1, 0);

        // 输入: [2, 3, 4, 4]
        NdArray inputNd = NdArray.likeRandomN(Shape.of(2, 3, 4, 4));
        // 卷积核: [2, 3, 3, 3]
        NdArray kernelNd = NdArray.likeRandomN(Shape.of(2, 3, 3, 3));

        Variable input = new Variable(inputNd);
        Variable kernel = new Variable(kernelNd);

        Variable output = conv.call(input, kernel);
        Variable sum = output.sum();
        sum.backward();

        // 验证梯度形状
        assertEquals(Shape.of(2, 3, 4, 4), input.getGrad().getShape());
        assertEquals(Shape.of(2, 3, 3, 3), kernel.getGrad().getShape());

        // 验证梯度非零（随机输入下梯度不应全为0）
        float inputGradSum = input.getGrad().sum().getNumber().floatValue();
        float kernelGradSum = kernel.getGrad().sum().getNumber().floatValue();
        assertTrue("Input gradient should not be all zeros",
            Math.abs(inputGradSum) > 1e-6);
        assertTrue("Kernel gradient should not be all zeros",
            Math.abs(kernelGradSum) > 1e-6);
    }
}
