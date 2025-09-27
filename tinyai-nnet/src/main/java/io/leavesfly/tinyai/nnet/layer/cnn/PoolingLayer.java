package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.Collections;
import java.util.List;

/**
 * 池化层
 * 支持最大池化、平均池化和自适应池化
 * <p>
 * todo
 */
public class PoolingLayer extends Layer {


    public PoolingLayer(String _name, Shape _inputShape) {
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
