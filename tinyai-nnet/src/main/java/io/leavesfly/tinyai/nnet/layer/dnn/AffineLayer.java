package io.leavesfly.tinyai.nnet.layer.dnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;


/**
 * 仿射层（全连接层）
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * AffineLayer与LinearLayer功能基本相同，都是实现全连接层的功能。
 * 该层执行以下计算：y = x * W + b
 * 其中W是权重矩阵，b是偏置项（可选）。
 */
public class AffineLayer extends Layer {
    /**
     * 权重参数矩阵
     * 形状: (input_size, output_size)
     */
    private Parameter wParam;

    /**
     * 偏置参数向量
     * 形状: (1, output_size)
     */
    private Parameter bParam;

    /**
     * 是否需要偏置项
     */
    private boolean needBias;

    /**
     * 构造一个仿射层实例
     *
     * @param _name       层名称
     * @param _inputShape 输入形状
     * @param hiddenCol   输出维度（列数）
     * @param _needBias   是否需要偏置项
     */
    public AffineLayer(String _name, Shape _inputShape, int hiddenCol, boolean _needBias) {
        super(_name, _inputShape, Shape.of(_inputShape.getRow(), hiddenCol));
        needBias = _needBias;
        //初始化
        init();
    }

    /**
     * 初始化仿射层的参数
     * 使用Xavier初始化方法初始化权重矩阵
     */
    @Override
    public void init() {

        if (!alreadyInit) {
            NdArray initWeight = NdArray.likeRandomN(Shape.of(inputShape.getColumn(), outputShape.getColumn()))
                    .mulNum(Math.sqrt((double) 1 / inputShape.getColumn()));

            wParam = new Parameter(initWeight);
            wParam.setName("w");
            addParam(wParam.getName(), wParam);

            if (needBias) {
                bParam = new Parameter(NdArray.zeros(Shape.of(1, outputShape.getColumn())));
                bParam.setName("b");
                addParam(bParam.getName(), bParam);
            }
            alreadyInit = true;
        }
    }

    /**
     * 仿射层的前向传播方法
     *
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 仿射变换后的输出变量
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        return x.linear(wParam, bParam);
    }

}