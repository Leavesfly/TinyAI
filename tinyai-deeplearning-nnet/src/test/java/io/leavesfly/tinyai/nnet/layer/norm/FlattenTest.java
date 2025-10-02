package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Flatten层的单元测试
 * 
 * 测试打平层的基本功能：
 * 1. 形状变换计算
 * 2. 前向传播
 * 3. 反向传播形状恢复
 * 4. 边界情况处理
 */
public class FlattenTest {

    private Flatten flatten2D;
    private Flatten flatten3D;
    private Flatten flatten4D;
    
    @Before
    public void setUp() {
        // 创建不同维度输入的Flatten层
        flatten2D = new Flatten("flatten2d", Shape.of(3, 4));
        flatten3D = new Flatten("flatten3d", Shape.of(2, 3, 4));
        flatten4D = new Flatten("flatten4d", Shape.of(2, 3, 4, 5));
    }

    @Test
    public void testOutputShapeCalculation() {
        // 测试输出形状计算是否正确
        
        // 2D输入: (3, 4) -> (3, 4)
        assertTrue("2D输入的输出形状应该保持不变", 
                   Shape.of(3, 4).equals(flatten2D.getOutputShape()));
        
        // 3D输入: (2, 3, 4) -> (2, 12)
        assertTrue("3D输入应该打平为(batch_size, features)", 
                   Shape.of(2, 12).equals(flatten3D.getOutputShape()));
        
        // 4D输入: (2, 3, 4, 5) -> (2, 60)
        assertTrue("4D输入应该打平为(batch_size, features)", 
                   Shape.of(2, 60).equals(flatten4D.getOutputShape()));
    }

    @Test
    public void test2DForwardPass() {
        // 测试2D输入的前向传播（应该保持不变）
        float[][] inputData = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = flatten2D.layerForward(inputVar);
        
        // 验证形状
        assertTrue("2D输入输出形状应该不变", 
                   Shape.of(3, 4).equals(output.getValue().getShape()));
        
        // 验证数据内容保持不变
        float[][] expectedData = inputData;
        float[][] actualData = output.getValue().getMatrix();
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals("2D输入数据应该保持不变", 
                            expectedData[i][j], actualData[i][j], 0.001f);
            }
        }
    }

    @Test
    public void test3DForwardPass() {
        // 测试3D输入的前向传播
        // 创建3D输入数据 (2, 3, 4)
        NdArray input = NdArray.likeRandomN(Shape.of(2, 3, 4));
        Variable inputVar = new Variable(input);
        
        Variable output = flatten3D.layerForward(inputVar);
        
        // 验证输出形状
        assertTrue("3D输入应该打平为2D", 
                   Shape.of(2, 12).equals(output.getValue().getShape()));
        
        // 验证数据不为null
        assertNotNull("输出数据不应该为null", output.getValue());
    }

    @Test
    public void test4DForwardPass() {
        // 测试4D输入的前向传播（典型的卷积到全连接转换）
        // 创建4D输入数据 (batch=1, channels=2, height=3, width=4)
        NdArray input = NdArray.likeRandomN(Shape.of(1, 2, 3, 4));
        Variable inputVar = new Variable(input);
        
        Flatten flatten = new Flatten("test4d", Shape.of(1, 2, 3, 4));
        Variable output = flatten.layerForward(inputVar);
        
        // 验证输出形状：(1, 2*3*4) = (1, 24)
        assertTrue("4D输入应该打平为(batch_size, features)", 
                   Shape.of(1, 24).equals(output.getValue().getShape()));
        
        // 验证数据不为null
        assertNotNull("输出数据不应该为null", output.getValue());
    }

    @Test
    public void testBatchProcessing() {
        // 测试批量处理
        // 创建批量的4D输入 (batch=3, channels=2, height=2, width=2)
        NdArray input = NdArray.likeRandomN(Shape.of(3, 2, 2, 2));
        Variable inputVar = new Variable(input);
        
        Flatten flatten = new Flatten("batch_test", Shape.of(3, 2, 2, 2));
        Variable output = flatten.layerForward(inputVar);
        
        // 验证输出形状：(3, 2*2*2) = (3, 8)
        assertTrue("批量输入应该正确处理", 
                   Shape.of(3, 8).equals(output.getValue().getShape()));
        
        // 验证批量大小保持不变
        assertEquals("批量大小应该保持不变", 
                    3, output.getValue().getShape().getDimension(0));
    }

    @Test
    public void testNullInputShape() {
        // 测试null输入形状的处理
        Flatten nullFlatten = new Flatten("null_test", null);
        
        // 应该使用默认输出形状
        assertTrue("null输入形状应该使用默认值", 
                   Shape.of(-1, 1).equals(nullFlatten.getOutputShape()));
    }

    @Test
    public void testEmptyShape() {
        // 测试空形状的处理
        Flatten emptyFlatten = new Flatten("empty_test", Shape.of());
        
        // 应该能正常处理
        assertTrue("空形状应该使用默认值", 
                   Shape.of(-1, 1).equals(emptyFlatten.getOutputShape()));
    }

    @Test
    public void testSingleDimensionInput() {
        // 测试单维度输入
        Flatten singleDim = new Flatten("single", Shape.of(5));
        
        // 输出应该是(5, 1)
        assertTrue("单维度输入应该正确处理", 
                   Shape.of(5, 1).equals(singleDim.getOutputShape()));
    }

    @Test
    public void testForwardBackwardConsistency() {
        // 测试前向和反向传播的一致性
        NdArray input = NdArray.likeRandomN(Shape.of(2, 3, 4));
        Variable inputVar = new Variable(input);
        
        // 前向传播
        Variable output = flatten3D.layerForward(inputVar);
        
        // 模拟反向传播的梯度形状
        NdArray outputGrad = NdArray.likeRandomN(output.getValue().getShape());
        
        // 通过forward方法验证
        NdArray forwardResult = flatten3D.forward(input);
        assertTrue("layerForward和forward应该产生相同形状的结果", 
                   output.getValue().getShape().equals(forwardResult.getShape()));
    }

    @Test
    public void testRequiredInputNumber() {
        // 测试所需输入数量
        assertEquals("Flatten层应该只需要1个输入", 
                    1, flatten2D.requireInputNum());
        assertEquals("Flatten层应该只需要1个输入", 
                    1, flatten3D.requireInputNum());
        assertEquals("Flatten层应该只需要1个输入", 
                    1, flatten4D.requireInputNum());
    }

    @Test
    public void testLargeInputShape() {
        // 测试大形状输入
        Flatten largeFlatten = new Flatten("large", Shape.of(2, 10, 20, 30));
        
        // 输出应该是(2, 10*20*30) = (2, 6000)
        assertTrue("大形状输入应该正确计算", 
                   Shape.of(2, 6000).equals(largeFlatten.getOutputShape()));
    }

    @Test
    public void testDataIntegrity() {
        // 测试数据完整性（确保打平过程中数据不丢失）
        // 创建已知数据的小输入
        float[][][] inputData = {
            {{1, 2}, {3, 4}},  // batch 0: 2x2
            {{5, 6}, {7, 8}}   // batch 1: 2x2
        };
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Flatten testFlatten = new Flatten("integrity", Shape.of(2, 2, 2));
        Variable output = testFlatten.layerForward(inputVar);
        
        // 验证形状
        assertTrue("输出形状应该是(2, 4)", 
                   Shape.of(2, 4).equals(output.getValue().getShape()));
        
        // 验证数据内容（第一个batch应该是[1,2,3,4]，第二个batch应该是[5,6,7,8]）
        float[][] outputData = output.getValue().getMatrix();
        
        float[] expectedBatch0 = {1, 2, 3, 4};
        float[] expectedBatch1 = {5, 6, 7, 8};
        
        for (int i = 0; i < 4; i++) {
            assertEquals("第一个batch的数据应该正确", 
                        expectedBatch0[i], outputData[0][i], 0.001f);
            assertEquals("第二个batch的数据应该正确", 
                        expectedBatch1[i], outputData[1][i], 0.001f);
        }
    }

    @Test
    public void testInitialization() {
        try {
            flatten2D.init();
            assertTrue("Flatten层初始化不应该抛出异常", true);
        } catch (Exception e) {
            fail("Flatten层初始化不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testGetParams() {
        assertTrue("Flatten层不应该有参数", flatten2D.getParams().isEmpty());
    }
}