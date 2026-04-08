package io.leavesfly.tinyai.nnet.integration;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.BatchNorm1d;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * V2模块的集成测试
 */
public class IntegrationTest {

    private Sequential buildModel() {
        return new Sequential("model")
            .add(new Linear("fc1", 4, 3))
            .add(new BatchNorm1d("bn1", 3))
            .add(new ReLU())
            // p=0 避免随机性，便于断言
            .add(new Dropout("dropout", 0.0f));
    }

    @Test
    public void testStateDictSyncsBuffersAcrossModules() {
        Sequential source = buildModel();
        NdArray input = NdArray.ones(Shape.of(2, 4));

        source.train();
        source.forward(new Variable(input)); // 触发BN统计量更新

        Map<String, NdArray> state = source.stateDict();

        Sequential target = buildModel();
        target.loadStateDict(state, true);

        BatchNorm1d srcBn = (BatchNorm1d) source.get(1);
        BatchNorm1d tgtBn = (BatchNorm1d) target.get(1);

        assertArrayEquals(srcBn.getRunningMean().getArray(),
            tgtBn.getRunningMean().getArray(), 1e-6f);
        assertArrayEquals(srcBn.getRunningVar().getArray(),
            tgtBn.getRunningVar().getArray(), 1e-6f);
    }

    @Test
    public void testForwardConsistencyAfterLoadStateDict() {
        Sequential source = buildModel();

        // 设置可复现的参数
        Linear srcLinear = (Linear) source.get(0);
        srcLinear.getWeight().setData(NdArray.of(new float[]{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
        }, Shape.of(3, 4)));
        srcLinear.getBias().setData(NdArray.of(new float[]{0.1f, -0.1f, 0.2f}, Shape.of(3)));

        BatchNorm1d srcBn = (BatchNorm1d) source.get(1);
        srcBn.getGamma().setData(NdArray.of(new float[]{1, 1, 1}, Shape.of(3)));
        srcBn.getBeta().setData(NdArray.of(new float[]{0, 0, 0}, Shape.of(3)));

        NdArray input = NdArray.ones(Shape.of(2, 4));

        // 先跑一次训练模式，便于同步running stats
        source.train();
        source.forward(new Variable(input));

        Map<String, NdArray> state = source.stateDict();

        Sequential target = buildModel();
        target.loadStateDict(state, true);

        source.eval();
        target.eval();

        Variable outSrc = source.forward(new Variable(input));
        Variable outTgt = target.forward(new Variable(input));

        assertArrayEquals(outSrc.getValue().getArray(), outTgt.getValue().getArray(), 1e-5f);
    }
}

