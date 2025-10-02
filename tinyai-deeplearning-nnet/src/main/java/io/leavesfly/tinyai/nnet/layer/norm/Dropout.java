package io.leavesfly.tinyai.nnet.layer.norm;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.util.Config;

import java.util.Collections;
import java.util.List;

/**
 * 抑制过拟合，采用的随机丢弃一些参数权重的方法
 */
public class Dropout extends Layer {
    private float ration;
    private NdArray mask;

    public Dropout(String _name, float _ration, Shape _xInputShape) {
        super(_name, _xInputShape, _xInputShape);
        ration = _ration;
    }

    public Dropout(String _name, float _ration) {
        super(_name, null, null);
        ration = _ration;
    }


    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {

        return this.call(inputs[0]);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        if (Config.train) {
            mask = NdArray.likeRandom(0, 1, x.getShape()).gt(x.like(ration));
            return x.mul(mask);
        }
        return x.mulNum(1.0 - ration);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.mul(mask));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
