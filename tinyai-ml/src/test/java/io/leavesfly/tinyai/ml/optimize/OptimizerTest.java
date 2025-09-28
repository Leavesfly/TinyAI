package io.leavesfly.tinyai.ml.optimize;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Optimizer 包相关类的单元测试
 * 
 * 测试 Optimizer、SGD、Adam 相关功能
 * 
 * @author TinyDL
 * @version 1.0
 */
public class OptimizerTest {

    private TestModel testModel;
    private Parameter testParameter;

    @Before
    public void setUp() {
        testModel = new TestModel();
        
        // 创建测试参数
        NdArray value = NdArray.of(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
        NdArray grad = NdArray.of(new float[][]{{0.1f, 0.2f}, {0.3f, 0.4f}});
        
        testParameter = new Parameter(value);
        testParameter.setGrad(grad);
        
        testModel.addParameter("test_param", testParameter);
    }

    @Test
    public void testSGDCreation() {
        // 测试 SGD 优化器创建
        float learningRate = 0.01f;
        SGD sgd = new SGD(testModel, learningRate);
        
        assertNotNull(sgd);
    }

    @Test
    public void testSGDUpdateOne() {
        // 测试 SGD 单个参数更新
        float learningRate = 0.1f;
        SGD sgd = new SGD(testModel, learningRate);
        
        // 保存更新前的值
        float[][] originalValues = testParameter.getValue().getMatrix();
        float[][] gradValues = testParameter.getGrad().getMatrix();
        
        // 执行参数更新
        sgd.updateOne(testParameter);
        
        // 验证参数更新：value = value - lr * grad
        float[][] updatedValues = testParameter.getValue().getMatrix();
        
        for (int i = 0; i < originalValues.length; i++) {
            for (int j = 0; j < originalValues[i].length; j++) {
                float expected = originalValues[i][j] - learningRate * gradValues[i][j];
                assertEquals("参数更新不正确", expected, updatedValues[i][j], 1e-6f);
            }
        }
    }

    @Test
    public void testSGDUpdateAll() {
        // 测试 SGD 更新所有参数
        float learningRate = 0.05f;
        SGD sgd = new SGD(testModel, learningRate);
        
        // 添加更多参数用于测试
        Parameter param2 = new Parameter(NdArray.of(new float[][]{{5.0f}}));
        param2.setGrad(NdArray.of(new float[][]{{0.5f}}));
        testModel.addParameter("param2", param2);
        
        // 保存原始值
        float originalValue1 = testParameter.getValue().getMatrix()[0][0];
        float originalValue2 = param2.getValue().getMatrix()[0][0];
        
        // 执行所有参数更新
        sgd.update();
        
        // 验证所有参数都被更新
        assertNotEquals(originalValue1, testParameter.getValue().getMatrix()[0][0]);
        assertNotEquals(originalValue2, param2.getValue().getMatrix()[0][0]);
    }

    @Test
    public void testAdamCreation() {
        // 测试 Adam 优化器创建
        Adam adam = new Adam(testModel);
        assertNotNull(adam);
        
        // 测试带参数的构造函数
        Adam adamWithParams = new Adam(testModel, 0.001f, 0.9f, 0.999f, 1e-8f);
        assertNotNull(adamWithParams);
    }

    @Test
    public void testAdamUpdateOne() {
        // 测试 Adam 单个参数更新
        Adam adam = new Adam(testModel, 0.001f, 0.9f, 0.999f, 1e-8f);
        
        // 保存更新前的值
        float[][] originalValues = testParameter.getValue().getMatrix();
        
        // 执行参数更新（需要调用update来增加时间步）
        adam.update();
        
        // 验证参数已被更新
        float[][] updatedValues = testParameter.getValue().getMatrix();
        
        boolean hasChanged = false;
        for (int i = 0; i < originalValues.length; i++) {
            for (int j = 0; j < originalValues[i].length; j++) {
                if (Math.abs(originalValues[i][j] - updatedValues[i][j]) > 1e-9) {
                    hasChanged = true;
                    break;
                }
            }
        }
        assertTrue("Adam应该更新参数值", hasChanged);
    }

    @Test
    public void testAdamMultipleUpdates() {
        // 测试 Adam 多次更新
        Adam adam = new Adam(testModel, 0.01f, 0.9f, 0.999f, 1e-8f);
        
        // 记录初始值
        float initialValue = testParameter.getValue().getMatrix()[0][0];
        
        // 执行多次更新
        for (int i = 0; i < 5; i++) {
            adam.update();
        }
        
        // 验证参数持续更新
        float finalValue = testParameter.getValue().getMatrix()[0][0];
        assertNotEquals("多次更新后参数应该改变", initialValue, finalValue);
    }

    @Test
    public void testAdamMomentumAndVelocity() {
        // 测试 Adam 的动量和速度缓存功能
        Adam adam = new Adam(testModel, 0.001f, 0.9f, 0.999f, 1e-8f);
        
        // 第一次更新
        adam.update();
        float valueAfterFirst = testParameter.getValue().getMatrix()[0][0];
        
        // 改变梯度
        testParameter.setGrad(NdArray.of(new float[][]{{0.2f, 0.4f}, {0.6f, 0.8f}}));
        
        // 第二次更新
        adam.update();
        float valueAfterSecond = testParameter.getValue().getMatrix()[0][0];
        
        // 验证更新有效
        assertNotEquals("第二次更新应该改变参数", valueAfterFirst, valueAfterSecond);
    }

    @Test
    public void testOptimizerAbstractClass() {
        // 测试 Optimizer 抽象类的基本功能
        Optimizer customOptimizer = new Optimizer(testModel) {
            @Override
            public void updateOne(Parameter parameter) {
                // 自定义更新：简单地将参数设为0
                NdArray zeros = NdArray.zeros(parameter.getValue().getShape());
                parameter.setValue(zeros);
            }
        };
        
        // 验证参数不为0
        assertNotEquals(0.0f, testParameter.getValue().getMatrix()[0][0]);
        
        // 执行自定义更新
        customOptimizer.update();
        
        // 验证参数被设为0
        assertEquals(0.0f, testParameter.getValue().getMatrix()[0][0], 1e-9f);
    }

    @Test
    public void testOptimizerPerformanceComparison() {
        // 比较不同优化器的性能特征
        
        // 创建相同的初始参数
        Parameter sgdParam = new Parameter(NdArray.of(new float[][]{{1.0f}}));
        sgdParam.setGrad(NdArray.of(new float[][]{{0.1f}}));
        
        Parameter adamParam = new Parameter(NdArray.of(new float[][]{{1.0f}}));
        adamParam.setGrad(NdArray.of(new float[][]{{0.1f}}));
        
        TestModel sgdModel = new TestModel();
        TestModel adamModel = new TestModel();
        sgdModel.addParameter("param", sgdParam);
        adamModel.addParameter("param", adamParam);
        
        SGD sgd = new SGD(sgdModel, 0.1f);
        Adam adam = new Adam(adamModel, 0.1f, 0.9f, 0.999f, 1e-8f);
        
        // 执行相同次数的更新
        for (int i = 0; i < 3; i++) {
            sgd.update();
            adam.update();
        }
        
        // 验证两个优化器都更新了参数（但更新量可能不同）
        assertNotEquals(1.0f, sgdParam.getValue().getMatrix()[0][0]);
        assertNotEquals(1.0f, adamParam.getValue().getMatrix()[0][0]);
        
        // Adam 和 SGD 的更新结果应该不同
        assertNotEquals("不同优化器应产生不同结果", 
                       sgdParam.getValue().getMatrix()[0][0], 
                       adamParam.getValue().getMatrix()[0][0]);
    }

    /**
     * 测试用的 Block 实现
     */
    private static class TestBlock extends Block {
        
        public TestBlock() {
            super("TestBlock", Shape.of(2, 2));
        }
        
        @Override
        public void init() {
            // 简单初始化
            alreadyInit = true;
        }
    }

    /**
     * 测试用的 Model 实现
     */
    private static class TestModel extends Model {
        
        private Map<String, Parameter> parameters = new HashMap<>();

        public TestModel() {
            super("TestModel", new TestBlock());
        }

        public Variable forward(Variable x) {
            return x; // 简单返回输入
        }

        public void addParameter(String name, Parameter parameter) {
            parameters.put(name, parameter);
        }

        @Override
        public Map<String, Parameter> getAllParams() {
            return parameters;
        }
    }
}