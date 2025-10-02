package io.leavesfly.tinyai.nnet.block;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;

/**
 * 顺序块，用于按顺序组合多个层
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * SequentialBlock是一个简单的顺序块实现，用于按顺序组合多个神经网络层。
 * 层会按照添加的顺序依次执行前向传播。
 */
public class SequentialBlock extends Block {

    /**
     * 构造函数，创建一个顺序块
     *
     * @param _name         块的名称
     * @param _xInputShape  输入数据的形状
     * @param _yOutputShape 输出数据的形状
     */
    public SequentialBlock(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }

    @Override
    public void init() {
    }


}