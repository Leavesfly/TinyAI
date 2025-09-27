package io.leavesfly.tinyai.nnet.layer.norm;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.Collections;
import java.util.List;

/**
 * 将矩阵打平
 * todo
 */
public class Flatten extends Layer {

    public Flatten(String _name, Shape _inputShape) {
        super(_name, _inputShape);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        return null;
    }
}
