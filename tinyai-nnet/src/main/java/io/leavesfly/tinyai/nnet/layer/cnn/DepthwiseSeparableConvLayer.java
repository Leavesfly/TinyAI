package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * 深度可分离卷积层（简化版本）
 * 实现MobileNet中的Depthwise Separable Convolution
 * 包含深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）
 * todo
 */
public class DepthwiseSeparableConvLayer extends Layer {


    public DepthwiseSeparableConvLayer(String _name, Shape _inputShape) {
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