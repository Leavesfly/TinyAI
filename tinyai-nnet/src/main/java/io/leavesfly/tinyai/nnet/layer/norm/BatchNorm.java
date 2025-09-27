package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.Arrays;
import java.util.List;

/**
 * 批量归一化层
 * 实现Batch Normalization算法，提高训练稳定性和收敛速度
 * //todo
 */
public class BatchNorm extends Layer {


    public BatchNorm(String _name, Shape _inputShape) {
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