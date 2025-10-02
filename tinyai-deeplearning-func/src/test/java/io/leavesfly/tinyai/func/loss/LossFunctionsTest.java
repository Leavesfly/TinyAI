package io.leavesfly.tinyai.func.loss;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 损失函数的单元测试
 *
 * @author TinyDL
 */
public class LossFunctionsTest {

    private boolean originalTrainMode;

    @Before
    public void setUp() {
        originalTrainMode = Config.train;
        Config.train = true;
    }

    @After
    public void tearDown() {
        Config.train = originalTrainMode;
    }

    @Test
    public void testMeanSE() {
        MeanSE mseFunc = new MeanSE();

        // 测试均方误差损失
        NdArray predict = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        NdArray target = NdArray.of(new float[][]{{1, 1, 1}, {1, 1, 1}});

        NdArray result = mseFunc.forward(predict, target);

        // 验证MSE结果
        float expectedMse = (0 + 1 + 4 + 9 + 16 + 25) / 6.0f; // (0+1+4+9+16+25)/6 = 55/6
        assertEquals(expectedMse, result.getNumber().floatValue(), 1e-6);

        // 测试反向传播
        Variable pred = new Variable(predict, "pred");
        Variable targ = new Variable(target, "targ");
        Variable loss = mseFunc.call(pred, targ);

        loss.backward();

        assertNotNull(pred.getGrad());
        assertNotNull(targ.getGrad());
    }

    @Test
    public void testSoftmaxCE() {
        SoftmaxCE softmaxCEFunc = new SoftmaxCE();

        // 测试softmax交叉熵损失
        NdArray predict = NdArray.of(new float[][]{{2, 1, 3}, {1, 3, 2}});
        NdArray label = NdArray.of(new float[][]{{2}, {1}}); // 类别标签

        NdArray result = softmaxCEFunc.forward(predict, label);

        // 验证结果是标量
        assertTrue(result.getShape().size() == 1);
        assertTrue(result.getNumber().floatValue() > 0); // 损失应该为正

        // 测试反向传播
        Variable pred = new Variable(predict, "pred");
        Variable lab = new Variable(label, "lab");
        Variable loss = softmaxCEFunc.call(pred, lab);

        loss.backward();

        assertNotNull(pred.getGrad());
        assertEquals(pred.getValue().getShape(), pred.getGrad().getShape());
    }

    @Test
    public void testSigmoidCE() {
        SigmoidCE sigmoidCEFunc = new SigmoidCE();

        // 测试sigmoid交叉熵损失
        NdArray predict = NdArray.of(new float[][]{{0.1f, 0.9f}, {0.8f, 0.2f}});
        NdArray label = NdArray.of(new float[][]{{0, 1}, {1, 0}});

        NdArray result = sigmoidCEFunc.forward(predict, label);

        // 验证结果
        assertTrue(result.getNumber().floatValue() > 0); // 损失应该为正

        // 测试反向传播
        Variable pred = new Variable(predict, "pred");
        Variable lab = new Variable(label, "lab");
        Variable loss = sigmoidCEFunc.call(pred, lab);

        loss.backward();

        assertNotNull(pred.getGrad());
        assertEquals(pred.getValue().getShape(), pred.getGrad().getShape());
    }
}