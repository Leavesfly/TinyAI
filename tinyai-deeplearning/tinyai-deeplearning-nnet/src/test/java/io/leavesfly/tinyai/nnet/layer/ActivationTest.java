package io.leavesfly.tinyai.nnet.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.activation.Sigmoid;
import io.leavesfly.tinyai.nnet.layer.activation.SoftMax;
import io.leavesfly.tinyai.nnet.layer.activation.Tanh;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 激活函数层的功能测试
 */
public class ActivationTest {

    @Test
    public void testReLU() {
        Variable input = new Variable(NdArray.of(new float[]{-1.0f, 0.0f, 2.0f}, Shape.of(1, 3)));
        Variable output = new ReLU().forward(input);
        assertArrayEquals(new float[]{0.0f, 0.0f, 2.0f}, output.getValue().getArray(), 1e-6f);
    }

    @Test
    public void testSigmoidRange() {
        Variable input = new Variable(NdArray.of(new float[]{-1.0f, 0.0f, 1.0f}, Shape.of(1, 3)));
        float[] out = new Sigmoid().forward(input).getValue().getArray();

        for (float v : out) {
            assertTrue(v > 0 && v < 1);
        }
        assertEquals(0.5f, out[1], 1e-6f);
        assertTrue(out[0] < out[2]);
    }

    @Test
    public void testTanhOutput() {
        Variable input = new Variable(NdArray.of(new float[]{-1.0f, 0.0f, 1.0f}, Shape.of(1, 3)));
        float[] out = new Tanh().forward(input).getValue().getArray();

        assertEquals((float) Math.tanh(-1.0f), out[0], 1e-5f);
        assertEquals(0.0f, out[1], 1e-6f);
        assertEquals((float) Math.tanh(1.0f), out[2], 1e-5f);
    }

    @Test
    public void testSoftmaxNormalized() {
        Variable input = new Variable(NdArray.of(new float[]{
            1.0f, 2.0f, 3.0f,
            0.0f, 0.0f, 0.0f
        }, Shape.of(2, 3)));

        float[] out = new SoftMax().forward(input).getValue().getArray();

        float sumRow0 = out[0] + out[1] + out[2];
        float sumRow1 = out[3] + out[4] + out[5];

        assertEquals(1.0f, sumRow0, 1e-5f);
        assertEquals(1.0f, sumRow1, 1e-5f);

        assertTrue(out[2] > out[1]);
        assertTrue(out[1] > out[0]);
    }

    @Test
    public void testTrainEvalConsistency() {
        Variable input = new Variable(NdArray.of(new float[]{-2.0f, 1.0f}, Shape.of(1, 2)));
        ReLU relu = new ReLU();

        relu.eval();
        float[] evalOut = relu.forward(input).getValue().getArray();

        relu.train();
        float[] trainOut = relu.forward(input).getValue().getArray();

        assertArrayEquals(evalOut, trainOut, 1e-6f);
    }

    @Test
    public void testReLUGradientCheck() {
        ReLU relu = new ReLU();
        Variable input = new Variable(NdArray.randn(Shape.of(32, 64)));
        GradientChecker.checkGraphConnectivity(relu, input);
    }

    @Test
    public void testSigmoidGradientCheck() {
        Sigmoid sigmoid = new Sigmoid();
        Variable input = new Variable(NdArray.randn(Shape.of(32, 64)));
        GradientChecker.checkGraphConnectivity(sigmoid, input);
    }

    @Test
    public void testTanhGradientCheck() {
        Tanh tanh = new Tanh();
        Variable input = new Variable(NdArray.randn(Shape.of(32, 64)));
        GradientChecker.checkGraphConnectivity(tanh, input);
    }

    @Test
    public void testSoftMaxGradientCheck() {
        SoftMax softmax = new SoftMax();
        Variable input = new Variable(NdArray.randn(Shape.of(32, 64)));
        GradientChecker.checkGraphConnectivity(softmax, input);
    }
}

