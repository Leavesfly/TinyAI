package io.leavesfly.tinyai.func.loss;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * Softmax交叉熵损失函数
 * <p>
 * 用于多分类问题的损失函数，结合了Softmax激活函数和交叉熵损失。
 */
public class SoftmaxCE extends Function {
    /**
     * 前向传播计算Softmax交叉熵损失
     * <p>
     * 计算公式：Loss = -Σ(yi*log(σ(xi)))
     * 其中σ(x)为Softmax函数，y为真实标签
     *
     * @param inputs 输入的NdArray数组，包含预测值和真实标签
     * @return Softmax交叉熵损失值
     */
    @Override
    public NdArray forward(NdArray... inputs) {

        NdArray predict = inputs[0];
        NdArray labelY = inputs[1];

        int row = predict.getShape().getRow();

        // log-sum-exp for numerical stability
        NdArray rowMax = predict.max(1); // shape: [row,1]
        NdArray stabilized = predict.sub(rowMax.broadcastTo(predict.getShape()));
        NdArray logSumExp = rowMax.add(stabilized.exp().sumTo(Shape.of(row, 1)).log());

        int[] colSlices = NdArrayUtil.toInt(labelY.transpose().getMatrix()[0]);
        NdArray logProb = predict.sub(logSumExp.broadcastTo(predict.getShape()));

        // 过滤 ignore_index(-100)：只对有效 label 计算损失
        int validCount = 0;
        float sum = 0.0f;
        float[][] logProbMatrix = logProb.getMatrix();
        for (int i = 0; i < row; i++) {
            if (colSlices[i] >= 0) {
                sum += logProbMatrix[i][colSlices[i]];
                validCount++;
            }
        }

        // 避免除零
        float divisor = validCount > 0 ? (float) validCount : 1.0f;
        return NdArray.of(-sum / divisor);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于Softmax交叉熵损失函数，梯度计算公式为：
     * ∂Loss/∂x = (σ(x) - y) / n
     * 其中σ(x)为Softmax函数，y为真实标签，n为批次大小
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray predict = inputs[0].getValue();
        NdArray label = inputs[1].getValue();

        int batchSize = predict.getShape().getRow();
        int numClasses = predict.getShape().getColumn();

        // softmax（数值稳定版本）
        NdArray max = predict.max(1);
        NdArray stabilized = predict.sub(max.broadcastTo(predict.getShape()));
        NdArray exp = stabilized.exp();
        NdArray softmax = exp.div(exp.sumTo(Shape.of(batchSize, 1)).broadcastTo(predict.getShape()));

        // one-hot labels，ignore_index(-100) 的行梯度为 0
        int[] labelIndices = NdArrayUtil.toInt(label.transpose().getMatrix()[0]);
        float[][] oneHotData = new float[batchSize][numClasses];
        int validCount = 0;
        for (int i = 0; i < batchSize; i++) {
            int labelIndex = labelIndices[i];
            if (labelIndex >= 0 && labelIndex < numClasses) {
                oneHotData[i][labelIndex] = 1.0f;
                validCount++;
            }
            // labelIndex < 0 (如 -100) 时，oneHot 全零，梯度 = softmax - 0 = softmax
            // 但 ignore 的行不应贡献梯度，所以需要将这些行的 softmax 也置零
        }
        NdArray oneHot = NdArray.of(oneHotData);

        float divisor = validCount > 0 ? (float) validCount : 1.0f;
        float scale = yGrad.getNumber().floatValue() / divisor;

        // 计算梯度：(softmax - oneHot) * scale
        // 对于 ignore_index 的行，将梯度置零
        float[][] softmaxMatrix = softmax.getMatrix();
        float[][] gradData = new float[batchSize][numClasses];
        for (int i = 0; i < batchSize; i++) {
            if (labelIndices[i] >= 0) {
                for (int j = 0; j < numClasses; j++) {
                    gradData[i][j] = (softmaxMatrix[i][j] - oneHotData[i][j]) * scale;
                }
            }
            // labelIndices[i] < 0 时 gradData[i] 保持全零
        }
        NdArray gradPredict = NdArray.of(gradData);

        return Arrays.asList(gradPredict, label.like(0));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * Softmax交叉熵损失函数需要两个输入参数：预测值和真实标签。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
