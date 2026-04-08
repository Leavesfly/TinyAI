package io.leavesfly.tinyai.nnet.container;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sequential容器的单元测试
 */
public class SequentialTest {

    @Test
    public void testSequentialCreation() {
        Sequential model = new Sequential("test");
        assertEquals("test", model.getName());
        assertTrue(model.isEmpty());
        assertEquals(0, model.size());
    }

    @Test
    public void testSequentialAdd() {
        Sequential model = new Sequential("test")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));

        assertEquals(3, model.size());
        assertFalse(model.isEmpty());
    }

    @Test
    public void testSequentialForward() {
        Sequential model = new Sequential("test")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));

        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);

        Variable output = model.forward(input);

        assertEquals(Shape.of(32, 10), output.getShape());
    }

    @Test
    public void testSequentialGet() {
        Linear fc1 = new Linear("fc1", 128, 64);
        ReLU relu = new ReLU();
        Linear fc2 = new Linear("fc2", 64, 10);

        Sequential model = new Sequential("test")
            .add(fc1)
            .add(relu)
            .add(fc2);

        assertEquals(fc1, model.get(0));
        assertEquals(relu, model.get(1));
        assertEquals(fc2, model.get(2));
    }

    @Test
    public void testSequentialNamedParameters() {
        Sequential model = new Sequential("test")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));

        var params = model.namedParameters();

        // Linear层有weight和bias，ReLU没有参数
        // 所以应该有4个参数：0.weight, 0.bias, 2.weight, 2.bias
        assertEquals(4, params.size());
        assertTrue(params.containsKey("0.weight"));
        assertTrue(params.containsKey("0.bias"));
        assertTrue(params.containsKey("2.weight"));
        assertTrue(params.containsKey("2.bias"));
    }

    @Test
    public void testSequentialTrainEval() {
        Sequential model = new Sequential("test")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU());

        assertTrue(model.isTraining());

        model.eval();
        assertFalse(model.isTraining());
        assertFalse(model.get(0).isTraining());

        model.train();
        assertTrue(model.isTraining());
        assertTrue(model.get(0).isTraining());
    }

    @Test
    public void testSequentialToString() {
        Sequential model = new Sequential("test")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU());

        String str = model.toString();
        assertTrue(str.contains("Sequential"));
        assertTrue(str.contains("test"));
    }

    @Test
    public void testSequentialEmptyForwardThrows() {
        Sequential model = new Sequential("test");

        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);

        assertThrows(IllegalStateException.class, () -> {
            model.forward(input);
        });
    }

    @Test
    public void testSequentialGradientCheck() {
        Sequential model = new Sequential("test")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));

        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(model, input);
    }
}
