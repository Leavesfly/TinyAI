package io.leavesfly.tinyai.modality.cv;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.block.SequentialBlock;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.PoolingLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.AffineLayer;
import io.leavesfly.tinyai.nnet.layer.norm.BatchNorm;
import io.leavesfly.tinyai.nnet.layer.norm.Dropout;
import io.leavesfly.tinyai.nnet.layer.norm.Flatten;

/**
 * 增强的深度卷积神经网络实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * SimpleConvNet类实现了增强的深度卷积神经网络，包含多个卷积层、池化层、全连接层和正则化层的深度架构。
 * 支持批量归一化、残差连接和灵活配置，适用于图像分类等计算机视觉任务。
 * <p>
 * todo
 */
public class SimpleConvNet extends SequentialBlock {

    /**
     * 构造函数，创建一个顺序块
     *
     * @param _name         块的名称
     * @param _xInputShape  输入数据的形状
     * @param _yOutputShape 输出数据的形状
     */
    public SimpleConvNet(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }


}