//package io.leavesfly.tinyai.func;
//
//import io.leavesfly.tinyai.ndarr.NDArray;
//import org.junit.Test;
//import static org.junit.Assert.*;
//
///**
// * MathFunction类的单元测试
// *
// * @author TinyAI Team
// * @version 1.0
// */
//public class MathFunctionTest {
//
//    @Test
//    public void testSigmoidFunction() {
//        // 测试Sigmoid函数的特殊值
//        assertEquals(0.5, MathFunction.SIGMOID.apply(0.0), 1e-10);
//        assertTrue(MathFunction.SIGMOID.apply(1.0) > 0.5);
//        assertTrue(MathFunction.SIGMOID.apply(-1.0) < 0.5);
//
//        // 测试函数值域 [0, 1]
//        assertTrue("Sigmoid should be less than or equal to 1 for large positive values", MathFunction.SIGMOID.apply(100.0) <= 1.0);
//        assertTrue("Sigmoid should be greater than 0 for large negative values", MathFunction.SIGMOID.apply(-100.0) > 0.0);
//        assertTrue("Sigmoid should be close to 1 for large positive values", MathFunction.SIGMOID.apply(100.0) > 0.99);
//    }
//
//    @Test
//    public void testTanhFunction() {
//        // 测试Tanh函数的特殊值
//        assertEquals(0.0, MathFunction.TANH.apply(0.0), 1e-10);
//        assertTrue(MathFunction.TANH.apply(1.0) > 0);
//        assertTrue(MathFunction.TANH.apply(-1.0) < 0);
//
//        // 测试函数值域 [-1, 1]
//        assertTrue("Tanh should be less than or equal to 1 for large positive values", MathFunction.TANH.apply(100.0) <= 1.0);
//        assertTrue("Tanh should be greater than or equal to -1 for large negative values", MathFunction.TANH.apply(-100.0) >= -1.0);
//        assertTrue("Tanh should be close to 1 for large positive values", MathFunction.TANH.apply(100.0) > 0.99);
//        assertTrue("Tanh should be close to -1 for large negative values", MathFunction.TANH.apply(-100.0) < -0.99);
//    }
//
//    @Test
//    public void testReLUFunction() {
//        // 测试ReLU函数
//        assertEquals(0.0, MathFunction.RELU.apply(-5.0), 1e-10);
//        assertEquals(0.0, MathFunction.RELU.apply(0.0), 1e-10);
//        assertEquals(5.0, MathFunction.RELU.apply(5.0), 1e-10);
//        assertEquals(100.0, MathFunction.RELU.apply(100.0), 1e-10);
//    }
//
//    @Test
//    public void testSigmoidDerivative() {
//        // 测试Sigmoid导数
//        double x = 0.0;
//        double expected = 0.25; // sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5
//        assertEquals(expected, MathFunction.SIGMOID.derivative(x), 1e-10);
//    }
//
//    @Test
//    public void testReLUDerivative() {
//        // 测试ReLU导数
//        assertEquals(0.0, MathFunction.RELU.derivative(-1.0), 1e-10);
//        assertEquals(1.0, MathFunction.RELU.derivative(1.0), 1e-10);
//        assertEquals(0.0, MathFunction.RELU.derivative(0.0), 1e-10);
//    }
//
//    @Test
//    public void testApplyFunctionToArray() {
//        // 创建测试数组
//        NDArray array = new NDArray(2, 2);
//        array.set(-2.0, 0, 0);
//        array.set(-1.0, 0, 1);
//        array.set(1.0, 1, 0);
//        array.set(2.0, 1, 1);
//
//        // 应用ReLU函数
//        NDArray result = MathFunction.applyFunction(array, MathFunction.RELU);
//
//        assertEquals(0.0, result.get(0, 0), 1e-10);
//        assertEquals(0.0, result.get(0, 1), 1e-10);
//        assertEquals(1.0, result.get(1, 0), 1e-10);
//        assertEquals(2.0, result.get(1, 1), 1e-10);
//    }
//
//    @Test
//    public void testApplyFunctionToArray3D() {
//        // 测试三维数组
//        NDArray array = new NDArray(2, 1, 2);
//        array.set(-1.0, 0, 0, 0);
//        array.set(1.0, 0, 0, 1);
//        array.set(-2.0, 1, 0, 0);
//        array.set(2.0, 1, 0, 1);
//
//        NDArray result = MathFunction.applyFunction(array, MathFunction.RELU);
//
//        assertEquals(0.0, result.get(0, 0, 0), 1e-10);
//        assertEquals(1.0, result.get(0, 0, 1), 1e-10);
//        assertEquals(0.0, result.get(1, 0, 0), 1e-10);
//        assertEquals(2.0, result.get(1, 0, 1), 1e-10);
//    }
//
//    @Test
//    public void testStatistics() {
//        // 创建测试数组
//        NDArray array = new NDArray(2, 2);
//        array.set(1.0, 0, 0);
//        array.set(2.0, 0, 1);
//        array.set(3.0, 1, 0);
//        array.set(4.0, 1, 1);
//
//        MathFunction.Statistics stats = MathFunction.calculateStatistics(array);
//
//        // 验证统计量
//        assertEquals(2.5, stats.getMean(), 1e-10);    // (1+2+3+4)/4 = 2.5
//        assertEquals(10.0, stats.getSum(), 1e-10);    // 1+2+3+4 = 10
//        assertEquals(1.0, stats.getMin(), 1e-10);     // min = 1
//        assertEquals(4.0, stats.getMax(), 1e-10);     // max = 4
//
//        // 验证方差 = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4 = 1.25
//        assertEquals(1.25, stats.getVariance(), 1e-10);
//    }
//
//    @Test
//    public void testStatisticsWithSingleValue() {
//        NDArray array = new NDArray(1, 1);
//        array.set(5.0, 0, 0);
//
//        MathFunction.Statistics stats = MathFunction.calculateStatistics(array);
//
//        assertEquals(5.0, stats.getMean(), 1e-10);
//        assertEquals(5.0, stats.getSum(), 1e-10);
//        assertEquals(5.0, stats.getMin(), 1e-10);
//        assertEquals(5.0, stats.getMax(), 1e-10);
//        assertEquals(0.0, stats.getVariance(), 1e-10);
//        assertEquals(0.0, stats.getStandardDeviation(), 1e-10);
//    }
//
//    @Test
//    public void testCreateFunctionChart() {
//        // 测试创建图表（不显示，只测试创建过程）
//        org.jfree.chart.JFreeChart chart = MathFunction.createFunctionChart(
//            x -> x * x,  // 简单的平方函数
//            -2.0, 2.0, 10,
//            "测试函数"
//        );
//
//        assertNotNull(chart);
//        assertEquals("测试函数", chart.getTitle().getText());
//    }
//
//    @Test
//    public void testDemo() {
//        // 测试demo方法不抛出异常
//        try {
//            // 注意：这里不实际显示图表，只测试逻辑部分
//            System.out.println("测试MathFunction.demo()方法...");
//
//            // 创建测试数组
//            NDArray array = new NDArray(2, 3);
//            array.set(1.0, 0, 0);
//            array.set(-0.5, 0, 1);
//            array.set(2.0, 0, 2);
//            array.set(-1.0, 1, 0);
//            array.set(0.5, 1, 1);
//            array.set(1.5, 1, 2);
//
//            // 应用不同的激活函数
//            NDArray sigmoidResult = MathFunction.applyFunction(array, MathFunction.SIGMOID);
//            NDArray reluResult = MathFunction.applyFunction(array, MathFunction.RELU);
//            NDArray tanhResult = MathFunction.applyFunction(array, MathFunction.TANH);
//
//            // 计算统计信息
//            MathFunction.Statistics stats = MathFunction.calculateStatistics(array);
//
//            // 验证结果不为空
//            assertNotNull(sigmoidResult);
//            assertNotNull(reluResult);
//            assertNotNull(tanhResult);
//            assertNotNull(stats);
//
//            System.out.println("MathFunction.demo()测试通过！");
//        } catch (Exception e) {
//            fail("Demo方法不应该抛出异常: " + e.getMessage());
//        }
//    }
//}