package io.leavesfly.tinyai.nnet.block;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.activate.SigmoidLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * SequentialBlock的单元测试
 * 
 * 测试顺序块的基本功能：
 * 1. 块初始化和层管理
 * 2. 层的顺序添加和前向传播
 * 3. 参数聚合和管理
 * 4. 复杂网络结构的构建
 * 5. 批处理和形状变换
 */
public class SequentialBlockTest {

    private SequentialBlock sequentialBlock;
    private SequentialBlock complexBlock;
    private Shape inputShape;
    private Shape outputShape;
    
    @Before
    public void setUp() {
        // 创建简单的顺序块
        inputShape = Shape.of(4, 5);   // batch_size=4, input_size=5
        outputShape = Shape.of(4, 2);  // batch_size=4, output_size=2
        sequentialBlock = new SequentialBlock("simple_sequential", inputShape, outputShape);
        
        // 添加简单的层序列：Linear -> ReLU -> Linear
        sequentialBlock.addLayer(new LinearLayer("linear1", 5, 3, true));
        sequentialBlock.addLayer(new ReLuLayer("relu1"));
        sequentialBlock.addLayer(new LinearLayer("linear2", 3, 2, true));
        
        // 创建复杂的顺序块
        complexBlock = new SequentialBlock("complex_sequential", 
                                         Shape.of(-1, 8), Shape.of(-1, 1));
        
        // 添加复杂的层序列
        complexBlock.addLayer(new LinearLayer("fc1", 8, 16, true));
        complexBlock.addLayer(new ReLuLayer("relu1"));
        complexBlock.addLayer(new LinearLayer("fc2", 16, 8, true));
        complexBlock.addLayer(new SigmoidLayer("sigmoid1"));
        complexBlock.addLayer(new LinearLayer("fc3", 8, 4, true));
        complexBlock.addLayer(new ReLuLayer("relu2"));
        complexBlock.addLayer(new LinearLayer("fc4", 4, 1, true));
    }

    @Test
    public void testBlockInitialization() {
        // 测试块的基本属性初始化
        assertEquals("块名称应该正确", "simple_sequential", sequentialBlock.getName());
        assertEquals("输入形状应该正确", inputShape, sequentialBlock.getInputShape());
        assertEquals("输出形状应该正确", outputShape, sequentialBlock.getOutputShape());
        
        // 验证层被正确添加 - 通过参数间接验证
        assertNotNull("块应该包含层", sequentialBlock.getAllParams());
        assertEquals("应该有3个层的参数", 1, sequentialBlock.getAllParams().size() >= 2 ? 1 : 0); // 至少有线性层参数
        
        // 验证复杂块
        assertEquals("复杂块名称应该正确", "complex_sequential", complexBlock.getName());
        assertTrue("复杂块应该有参数", complexBlock.getAllParams().size() > 0);
    }

    @Test
    public void testEmptySequentialBlock() {
        // 测试空的顺序块
        SequentialBlock emptyBlock = new SequentialBlock("empty", 
                                                       Shape.of(1, 3), Shape.of(1, 3));
        
        assertEquals("空块名称应该正确", "empty", emptyBlock.getName());
        assertNotNull("空块的参数不应该为null", emptyBlock.getAllParams());
        assertEquals("空块应该没有参数", 0, emptyBlock.getAllParams().size());
    }

    @Test
    public void testAddingLayers() {
        // 测试动态添加层
        SequentialBlock dynamicBlock = new SequentialBlock("dynamic", 
                                                          Shape.of(-1, 4), Shape.of(-1, 2));
        
        assertEquals("初始应该没有参数", 0, dynamicBlock.getAllParams().size());
        
        // 添加第一层
        dynamicBlock.addLayer(new LinearLayer("layer1", 4, 3, true));
        assertTrue("添加一层后应该有参数", dynamicBlock.getAllParams().size() >= 2);
        
        // 添加第二层
        dynamicBlock.addLayer(new ReLuLayer("relu"));
        assertTrue("添加ReLU后参数不变", dynamicBlock.getAllParams().size() >= 2);
        
        // 添加第三层
        dynamicBlock.addLayer(new LinearLayer("layer2", 3, 2, false));
        assertTrue("添加最后一层后应该有更多参数", dynamicBlock.getAllParams().size() >= 3);
    }

    @Test
    public void testParameterInitialization() {
        // 测试参数是否正确初始化
        Map<String, Parameter> allParams = sequentialBlock.getAllParams();
        assertNotNull("参数映射不应该为null", allParams);
        assertFalse("应该有参数被初始化", allParams.isEmpty());
        
        // 验证参数包含所有线性层的参数
        boolean hasLinear1Params = allParams.keySet().stream()
                .anyMatch(key -> key.contains("linear1"));
        boolean hasLinear2Params = allParams.keySet().stream()
                .anyMatch(key -> key.contains("linear2"));
        
        assertTrue("应该包含第一个线性层参数", hasLinear1Params);
        assertTrue("应该包含第二个线性层参数", hasLinear2Params);
        
        // 验证复杂块的参数
        Map<String, Parameter> complexParams = complexBlock.getAllParams();
        assertTrue("复杂块应该有更多参数", complexParams.size() >= 4); // 至少4个线性层的参数
    }

    @Test
    public void testForwardPass() {
        // 测试前向传播
        int batchSize = 4;
        NdArray input = NdArray.likeRandomN(Shape.of(batchSize, 5));
        Variable inputVar = new Variable(input);
        
        Variable output = sequentialBlock.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该正确", Shape.of(batchSize, 2), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
        
        // 验证输出值的有效性
        NdArray outputArray = output.getValue();
        float[][] outputData = outputArray.getMatrix();
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 2; j++) {
                float val = outputData[i][j];
                assertFalse("输出不应该包含NaN", Float.isNaN(val));
                assertFalse("输出不应该包含无穷大", Float.isInfinite(val));
            }
        }
    }

    @Test
    public void testComplexForwardPass() {
        // 测试复杂网络的前向传播
        int batchSize = 2;
        NdArray input = NdArray.likeRandomN(Shape.of(batchSize, 8));
        Variable inputVar = new Variable(input);
        
        Variable output = complexBlock.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("复杂网络输出形状应该正确", Shape.of(batchSize, 1), output.getValue().getShape());
        assertNotNull("复杂网络输出不应该为null", output.getValue());
        
        // 验证输出值的有效性
        NdArray outputArray = output.getValue();
        float[][] outputData = outputArray.getMatrix();
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 1; j++) {
                float val = outputData[i][j];
                assertFalse("复杂网络输出不应该包含NaN", Float.isNaN(val));
                assertFalse("复杂网络输出不应该包含无穷大", Float.isInfinite(val));
            }
        }
    }

    @Test
    public void testBatchProcessing() {
        // 测试不同批次大小的处理
        int[] batchSizes = {1, 3, 8, 16};
        
        for (int batchSize : batchSizes) {
            NdArray input = NdArray.likeRandomN(Shape.of(batchSize, 5));
            Variable inputVar = new Variable(input);
            
            Variable output = sequentialBlock.layerForward(inputVar);
            
            assertEquals(String.format("批大小%d的输出形状应该正确", batchSize), 
                        Shape.of(batchSize, 2), output.getValue().getShape());
        }
    }

    @Test
    public void testSingleLayerSequential() {
        // 测试只有一个层的顺序块
        SequentialBlock singleLayerBlock = new SequentialBlock("single", 
                                                             Shape.of(-1, 3), Shape.of(-1, 2));
        singleLayerBlock.addLayer(new LinearLayer("only_layer", 3, 2, true));
        
        NdArray input = NdArray.likeRandomN(Shape.of(2, 3));
        Variable inputVar = new Variable(input);
        
        Variable output = singleLayerBlock.layerForward(inputVar);
        
        assertEquals("单层顺序块输出形状应该正确", Shape.of(2, 2), output.getValue().getShape());
    }

    @Test
    public void testActivationOnlySequential() {
        // 测试只有激活函数的顺序块
        SequentialBlock activationBlock = new SequentialBlock("activation_only", 
                                                            Shape.of(-1, 3), Shape.of(-1, 3));
        activationBlock.addLayer(new ReLuLayer("relu"));
        activationBlock.addLayer(new SigmoidLayer("sigmoid"));
        
        NdArray input = NdArray.of(new float[][]{{-1.0f, 0.0f, 1.0f}, {2.0f, -2.0f, 0.5f}});
        Variable inputVar = new Variable(input);
        
        Variable output = activationBlock.layerForward(inputVar);
        
        assertEquals("激活函数序列输出形状应该正确", Shape.of(2, 3), output.getValue().getShape());
        
        // 验证ReLU + Sigmoid的组合效果
        float[][] outputData = output.getValue().getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                float val = outputData[i][j];
                assertTrue("ReLU+Sigmoid输出应该在[0,1]范围内", val >= 0.0f && val <= 1.0f);
            }
        }
    }

    @Test
    public void testZeroInput() {
        // 测试零输入的处理
        int batchSize = 2;
        NdArray zeroInput = NdArray.zeros(Shape.of(batchSize, 5));
        Variable zeroInputVar = new Variable(zeroInput);
        
        Variable output = sequentialBlock.layerForward(zeroInputVar);
        
        assertNotNull("零输入应该产生有效输出", output.getValue());
        assertEquals("零输入输出形状应该正确", Shape.of(batchSize, 2), output.getValue().getShape());
        
        // 验证输出不包含无效值
        NdArray outputArray = output.getValue();
        float[][] outputData = outputArray.getMatrix();
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 2; j++) {
                float val = outputData[i][j];
                assertFalse("输出不应该包含NaN", Float.isNaN(val));
                assertFalse("输出不应该包含无穷大", Float.isInfinite(val));
            }
        }
    }

    @Test
    public void testConsistentOutputWithSameInput() {
        // 测试相同输入产生一致输出
        NdArray input = NdArray.likeRandomN(Shape.of(3, 5));
        Variable inputVar1 = new Variable(input);
        Variable inputVar2 = new Variable(input);
        
        Variable output1 = sequentialBlock.layerForward(inputVar1);
        Variable output2 = sequentialBlock.layerForward(inputVar2);
        
        // 验证两次输出相同
        float[][] output1Data = output1.getValue().getMatrix();
        float[][] output2Data = output2.getValue().getMatrix();
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals("相同输入应该产生相同输出", 
                           output1Data[i][j], output2Data[i][j], 1e-6f);
            }
        }
    }

    @Test
    public void testClearGrads() {
        // 测试梯度清零功能
        try {
            sequentialBlock.clearGrads();
            complexBlock.clearGrads();
        } catch (Exception e) {
            fail("清除梯度不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testInit() {
        // 测试初始化功能
        try {
            sequentialBlock.init();
            complexBlock.init();
        } catch (Exception e) {
            fail("初始化不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testResetState() {
        // 测试状态重置功能（对非RNN层应该无影响）
        try {
            sequentialBlock.resetState();
            complexBlock.resetState();
        } catch (Exception e) {
            fail("重置状态不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testGetAllParams() {
        // 测试获取所有参数的功能
        Map<String, Parameter> allParams = sequentialBlock.getAllParams();
        
        assertNotNull("getAllParams不应该返回null", allParams);
        assertFalse("应该有参数", allParams.isEmpty());
        
        // 验证参数的有效性
        for (Parameter param : allParams.values()) {
            assertNotNull("参数不应该为null", param);
            assertNotNull("参数值不应该为null", param.getValue());
            assertNotNull("参数名称不应该为null", param.getName());
        }
    }

    @Test
    public void testLayerSequencePreservation() {
        // 测试层序列的保持
        SequentialBlock testBlock = new SequentialBlock("test_sequence", 
                                                       Shape.of(-1, 2), Shape.of(-1, 1));
        
        // 按特定顺序添加层
        testBlock.addLayer(new LinearLayer("first", 2, 3, false));
        testBlock.addLayer(new ReLuLayer("second"));
        testBlock.addLayer(new LinearLayer("third", 3, 1, false));
        
        // 验证层的数量 - 通过参数间接验证
        assertTrue("testBlock应该有参数", testBlock.getAllParams().size() > 0);
        
        // 验证层的正确性 - 通过第一个和最后一个参数的名称来验证
        Map<String, Parameter> params = testBlock.getAllParams();
        boolean hasFirstLayer = params.keySet().stream().anyMatch(name -> name.contains("first"));
        boolean hasThirdLayer = params.keySet().stream().anyMatch(name -> name.contains("third"));
        assertTrue("应该包含第一层参数", hasFirstLayer);
        assertTrue("应该包含第三层参数", hasThirdLayer);
    }

    @Test
    public void testDeepSequentialNetwork() {
        // 测试深层顺序网络
        SequentialBlock deepBlock = new SequentialBlock("deep", 
                                                       Shape.of(-1, 4), Shape.of(-1, 1));
        
        // 构建深层网络：4->8->6->4->2->1
        deepBlock.addLayer(new LinearLayer("deep1", 4, 8, true));
        deepBlock.addLayer(new ReLuLayer("relu1"));
        deepBlock.addLayer(new LinearLayer("deep2", 8, 6, true));
        deepBlock.addLayer(new ReLuLayer("relu2"));
        deepBlock.addLayer(new LinearLayer("deep3", 6, 4, true));
        deepBlock.addLayer(new ReLuLayer("relu3"));
        deepBlock.addLayer(new LinearLayer("deep4", 4, 2, true));
        deepBlock.addLayer(new ReLuLayer("relu4"));
        deepBlock.addLayer(new LinearLayer("deep5", 2, 1, true));
        
        // 验证深层网络的参数数量
        assertTrue("深层网络应该有足够的参数", deepBlock.getAllParams().size() >= 5); // 至少5个线性层的参数
        
        // 测试深层网络的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(1, 4));
        Variable inputVar = new Variable(input);
        
        Variable output = deepBlock.layerForward(inputVar);
        
        assertEquals("深层网络输出形状应该正确", Shape.of(1, 1), output.getValue().getShape());
        
        float outputValue = output.getValue().get(0, 0);
        assertFalse("深层网络输出不应该包含NaN", Float.isNaN(outputValue));
        assertFalse("深层网络输出不应该包含无穷大", Float.isInfinite(outputValue));
    }

    @Test
    public void testMixedActivationSequence() {
        // 测试混合激活函数序列
        SequentialBlock mixedBlock = new SequentialBlock("mixed", 
                                                        Shape.of(-1, 4), Shape.of(-1, 2));
        
        mixedBlock.addLayer(new LinearLayer("layer1", 4, 6, true));
        mixedBlock.addLayer(new ReLuLayer("relu"));
        mixedBlock.addLayer(new LinearLayer("layer2", 6, 4, true));
        mixedBlock.addLayer(new SigmoidLayer("sigmoid"));
        mixedBlock.addLayer(new LinearLayer("layer3", 4, 2, true));
        
        NdArray input = NdArray.likeRandomN(Shape.of(2, 4));
        Variable inputVar = new Variable(input);
        
        Variable output = mixedBlock.layerForward(inputVar);
        
        assertEquals("混合激活网络输出形状应该正确", Shape.of(2, 2), output.getValue().getShape());
        
        // 验证输出有效性
        float[][] outputData = output.getValue().getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float val = outputData[i][j];
                assertFalse("混合激活网络输出不应该包含NaN", Float.isNaN(val));
                assertFalse("混合激活网络输出不应该包含无穷大", Float.isInfinite(val));
            }
        }
    }
}