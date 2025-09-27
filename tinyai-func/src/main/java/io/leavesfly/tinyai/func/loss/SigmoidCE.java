package io.leavesfly.tinyai.func.loss;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;

import java.util.Arrays;
import java.util.List;


/**
 * Sigmoid交叉熵损失函数
 * <p>
 * 用于多分类问题的损失函数，结合了Sigmoid激活函数和交叉熵损失。
 */
public class SigmoidCE extends Function {

    private NdArray sigmoid;

    /**
     * 前向传播计算Sigmoid交叉熵损失
     * <p>
     * 计算公式：Loss = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
     * 其中σ(x)为Sigmoid函数，y为真实标签
     *
     * @param inputs 输入的NdArray数组，包含预测值和真实标签
     * @return Sigmoid交叉熵损失值
     * @throws RuntimeException 当预测值和标签列数不一致时抛出异常
     */
    @Override
    public NdArray forward(NdArray... inputs) {

        NdArray predict = inputs[0];
        NdArray labelY = inputs[1];
        sigmoid = predict.sigmoid();

        if (predict.getShape().getColumn() != labelY.getShape().getColumn()) {
            throw new RuntimeException("predict和label的列数不一致！");
        }
        NdArray other = sigmoid.like(1f).sub(sigmoid);

        float loss = crossEntropyError(NdArrayUtil.merge(1, other, sigmoid), labelY);
        return NdArray.of(loss);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于Sigmoid交叉熵损失函数，梯度计算公式为：
     * ∂Loss/∂x = (σ(x) - y) / n
     * 其中σ(x)为Sigmoid函数，y为真实标签，n为批次大小
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        //要求 预测值只有一个，支持二分类
        NdArray predict = inputs[0].getValue();
        NdArray label = inputs[1].getValue();
        int batchSize = predict.getShape().getRow();

        NdArray xGrad = sigmoid.sub(label).mul(yGrad.broadcastTo(label.getShape())).divNum(batchSize);

        return Arrays.asList(xGrad, label.like(1));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * Sigmoid交叉熵损失函数需要两个输入参数：预测值和真实标签。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }

    /**
     * 计算交叉熵误差
     * <p>
     * 计算预测值与真实标签之间的交叉熵误差。
     *
     * @param predict 预测值
     * @param labelY  真实标签
     * @return 交叉熵误差值
     */
    public static float crossEntropyError(NdArray predict, NdArray labelY) {

        if (labelY.getShape().getColumn() != 1) {
            //说明labelY进行了 one-hot编码
            labelY = labelY.argMax(1);
        }

        int batchSize = predict.getShape().getRow();
        int[] colSlices = NdArrayUtil.toInt(labelY.transpose().getMatrix()[0]);
        NdArray subPredict = predict.getItem(NdArrayUtil.getSeq(batchSize), colSlices);

        float crossEntropyError = subPredict.add(subPredict.like(1e-7)).log().sum().getNumber().floatValue() / (float) batchSize;
        return -crossEntropyError;
    }
}
