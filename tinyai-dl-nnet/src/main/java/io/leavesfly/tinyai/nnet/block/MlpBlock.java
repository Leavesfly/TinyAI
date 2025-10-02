package io.leavesfly.tinyai.nnet.block;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.activate.SigmoidLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.util.Config;


import java.util.Objects;

/**
 * 多层全连接的BP神经网络
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * MlpBlock是一个多层感知机块，由多个全连接层和激活函数层组成，
 * 可以构建任意深度的前馈神经网络。
 */
public class MlpBlock extends Block {
    /**
     * 激活函数类型配置
     */
    Config.ActiveFunc activeFunc;

    /**
     * 构造函数，创建一个多层感知机块
     *
     * @param _name       块的名称
     * @param batchSize   批处理大小
     * @param _activeFunc 激活函数类型
     * @param layerSizes  各层的大小数组，从输入层到输出层
     */
    public MlpBlock(String _name, int batchSize, Config.ActiveFunc _activeFunc, int... layerSizes) {

        super(_name, Shape.of(batchSize, layerSizes[0]), Shape.of(-1, layerSizes[layerSizes.length - 1]));

        activeFunc = _activeFunc;

        for (int i = 1; i < layerSizes.length - 1; i++) {
            Layer layer = new LinearLayer("layer" + i, layerSizes[i - 1], layerSizes[i], true);
            addLayer(layer);
            if (!Objects.isNull(activeFunc) && Config.ActiveFunc.ReLU.name().equals(activeFunc.name())) {
                addLayer(new ReLuLayer("ReLU"));
            } else {
                addLayer(new SigmoidLayer("Sigmoid"));
            }
        }
        Layer layer = new LinearLayer("layer" + (layerSizes.length - 1), layerSizes[(layerSizes.length - 2)]
                , layerSizes[(layerSizes.length - 1)], true);
        addLayer(layer);
    }

    @Override
    public void init() {

    }
}