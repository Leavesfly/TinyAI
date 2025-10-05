package io.leavesfly.tinyai.ml.evaluator;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.loss.Classify;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Evaluator 包相关类的单元测试
 * <p>
 * 测试 AccuracyEval、RegressEval、Evaluator 相关功能
 *
 * @author TinyDL
 * @version 1.0
 */
public class EvaluatorTest {

    private TestModel testModel;
    private TestDataSet testDataSet;
    private TestClassifyLoss testClassifyLoss;
    private TestRegressionLoss testRegressionLoss;

    @Before
    public void setUp() {
        testModel = new TestModel();
        testDataSet = new TestDataSet();
        testClassifyLoss = new TestClassifyLoss();
        testRegressionLoss = new TestRegressionLoss();
    }

    @Test
    public void testAccuracyEvalCreation() {
        // 测试 AccuracyEval 创建
        AccuracyEval accuracyEval = new AccuracyEval(testClassifyLoss, testModel, testDataSet);

        assertNotNull(accuracyEval);
    }

    @Test
    public void testAccuracyEvalEvaluate() {
        // 测试 AccuracyEval 评估功能
        AccuracyEval accuracyEval = new AccuracyEval(testClassifyLoss, testModel, testDataSet);

        // 捕获输出
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(outputStream));

        try {
            accuracyEval.evaluate();

            String output = outputStream.toString();
            assertTrue("输出应包含准确率信息", output.contains("avg-accuracy rate is"));
        } finally {
            System.setOut(originalOut);
        }
    }

    @Test
    public void testRegressEvalCreation() {
        // 测试 RegressEval 创建
        RegressEval regressEval = new RegressEval(testRegressionLoss, testModel, testDataSet);

        assertNotNull(regressEval);
    }

    @Test
    public void testRegressEvalEvaluate() {
        // 测试 RegressEval 评估功能
        RegressEval regressEval = new RegressEval(testRegressionLoss, testModel, testDataSet);

        // 捕获输出
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(outputStream));

        try {
            regressEval.evaluate();

            String output = outputStream.toString();
            assertTrue("输出应包含损失信息", output.contains("Test dataset model's avg loss is"));
        } finally {
            System.setOut(originalOut);
        }
    }

    @Test
    public void testEvaluatorAbstractClass() {
        // 测试 Evaluator 抽象类的基本功能
        Evaluator customEvaluator = new Evaluator() {
            @Override
            public void evaluate() {
                // 自定义评估逻辑
                System.out.println("Custom evaluation");
            }
        };

        // 设置模型和数据集
        customEvaluator.model = testModel;
        customEvaluator.dataSet = testDataSet;

        assertSame(testModel, customEvaluator.model);
        assertSame(testDataSet, customEvaluator.dataSet);

        // 捕获输出
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(outputStream));

        try {
            customEvaluator.evaluate();

            String output = outputStream.toString();
            assertTrue("输出应包含自定义评估信息", output.contains("Custom evaluation"));
        } finally {
            System.setOut(originalOut);
        }
    }

    @Test
    public void testMultipleEvaluations() {
        // 测试多次评估的一致性
        AccuracyEval accuracyEval = new AccuracyEval(testClassifyLoss, testModel, testDataSet);

        // 执行多次评估应该都能正常工作
        for (int i = 0; i < 3; i++) {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            PrintStream originalOut = System.out;
            System.setOut(new PrintStream(outputStream));

            try {
                accuracyEval.evaluate();

                String output = outputStream.toString();
                assertFalse("输出不应为空", output.trim().isEmpty());
                assertTrue("输出应包含准确率信息", output.contains("avg-accuracy rate is"));
            } finally {
                System.setOut(originalOut);
            }
        }
    }

    /**
     * 测试用的 Block 实现
     */
    private static class TestBlock extends Block implements java.io.Serializable {

        private static final long serialVersionUID = 1L;

        public TestBlock() {
            super("TestBlock", Shape.of(1, 2));
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

        public TestModel() {
            super("TestModel", new TestBlock());
        }

        @Override
        public Variable forward(Variable... inputs) {
            // 简单的前向传播：返回输入的复制
            return inputs[0];
        }
    }

    /**
     * 测试用的 DataSet 实现
     */
    private static class TestDataSet extends DataSet {

        private TestDataSet testDataSet;

        public TestDataSet() {
            super(2);
            this.testDataSet = this;
        }

        @Override
        public List<Batch> getBatches() {
            List<Batch> batches = new ArrayList<>();

            // 创建测试批次
            NdArray[] xs = {
                    NdArray.of(new float[][]{{1.0f, 2.0f}}),
                    NdArray.of(new float[][]{{3.0f, 4.0f}})
            };
            NdArray[] ys = {
                    NdArray.of(new float[][]{{0.0f}}),
                    NdArray.of(new float[][]{{1.0f}})
            };

            batches.add(new Batch(xs, ys));
            return batches;
        }

        @Override
        public void doPrepare() {
            // 测试数据不需要特殊准备
        }

        @Override
        public void shuffle() {
            // 测试数据不需要打乱
        }

        public Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validaRation) {
            if (Math.abs(trainRatio + testRatio + validaRation - 1.0f) > 1e-6) {
                throw new RuntimeException("splitDataset parameters error!");
            }

            Map<String, DataSet> result = new java.util.HashMap<>();
            result.put("TRAIN", this);
            result.put("TEST", this);
            result.put("VALIDATION", this);
            return result;
        }

        @Override
        public DataSet getTestDataSet() {
            return this;
        }

        @Override
        public int getSize() {
            return 2;
        }
    }

    /**
     * 测试用的分类损失函数
     */
    private static class TestClassifyLoss extends Classify {

        public float accuracyRate(Variable y, Variable predictY) {
            // 返回固定的准确率用于测试
            return 0.85f;
        }
    }

    /**
     * 测试用的回归损失函数
     */
    private static class TestRegressionLoss extends Loss {

        @Override
        public Variable loss(Variable y, Variable predictY) {
            // 简单的回归损失计算
            return new Variable(NdArray.of(0.25f));
        }
    }
}