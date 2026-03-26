package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 小于比较
 */
public class LessThan extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].lt(inputs[1]);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray grad0 = NdArray.zeros(inputs[0].getValue().getShape());
        NdArray grad1 = NdArray.zeros(inputs[1].getValue().getShape());
        return Arrays.asList(grad0, grad1);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}

