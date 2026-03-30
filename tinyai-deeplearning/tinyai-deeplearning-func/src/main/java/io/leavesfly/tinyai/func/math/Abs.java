package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 绝对值运算
 * <p>
 * forward: y = |x|
 * backward: dy/dx = sgn(x) * gy
 */
public class Abs extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].abs();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // dx = gy * sgn(x)
        // sgn(x) = x / |x| (当 x != 0), 0 (当 x = 0)
        // 实际上 NdArray 可能没有 sgn 函数，但我们可以用 gt/lt 组合或者 div
        NdArray x = inputs[0].getValue();

        NdArray zeros = NdArray.zeros(x.getShape());
        NdArray maskPos = x.gt(zeros);
        NdArray maskNeg = x.lt(zeros);
        NdArray sgn = maskPos.sub(maskNeg);

        return Collections.singletonList(yGrad.mul(sgn));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

