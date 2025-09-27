package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 卷积层实现类
 * <p>
 * 实现了标准的卷积操作，支持步长、填充、偏置等参数。
 * 使用Im2Col技术将卷积操作转换为矩阵乘法，提高计算效率。
 * todo
 */
public class ConvLayer extends Layer {


    public ConvLayer(String _name, Shape _inputShape) {
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