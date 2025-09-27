package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * 位置编码层实现
 * <p>
 * 位置编码用于给输入序列添加位置信息，因为Transformer本身没有循环或卷积结构来获取位置信息。
 * 使用正弦和余弦函数来生成位置编码：
 * <p>
 * PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 * <p>
 * 其中 pos 是位置，i 是维度
 */
public class PositionalEncoding extends Layer {

    private int maxSeqLength;
    private NdArray posEncoding;
    private boolean dropout;
    private double dropoutRate;

    /**
     * 构造位置编码层
     *
     * @param name         层名称
     * @param dModel       模型维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate  dropout比率
     */
    public PositionalEncoding(String name, int dModel, int maxSeqLength, double dropoutRate) {
        super(name, Shape.of(-1, maxSeqLength, dModel), Shape.of(-1, maxSeqLength, dModel));
        this.maxSeqLength = maxSeqLength;
        this.dropoutRate = dropoutRate;
        this.dropout = dropoutRate > 0.0;
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            int dModel = inputShape.getDimension(2);

            // 创建位置编码矩阵
            posEncoding = NdArray.zeros(Shape.of(maxSeqLength, dModel));

            // 计算位置编码
            for (int pos = 0; pos < maxSeqLength; pos++) {
                for (int i = 0; i < dModel; i++) {
                    double angle = pos / Math.pow(10000, (double) (2 * (i / 2)) / dModel);
                    if (i % 2 == 0) {
                        // 偶数维度使用sin
                        posEncoding.set((float) Math.sin(angle), pos, i);
                    } else {
                        // 奇数维度使用cos
                        posEncoding.set((float) Math.cos(angle), pos, i);
                    }
                }
            }

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        NdArray inputData = input.getValue();

        // 获取序列长度
        int seqLength = inputData.getShape().getDimension(1);

        // 截取对应长度的位置编码
        // 截取对应长度的位置编码
        NdArray posEnc = posEncoding.subNdArray(0, seqLength, 0, inputData.getShape().getDimension(2));

        // 扩展位置编码以匹配batch size
        int batchSize = inputData.getShape().getDimension(0);
        NdArray expandedPosEnc = NdArray.zeros(Shape.of(batchSize, seqLength, inputData.getShape().getDimension(2)));

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                for (int k = 0; k < inputData.getShape().getDimension(2); k++) {
                    expandedPosEnc.set(posEnc.get(j, k), i, j, k);
                }
            }
        }

        // 添加位置编码到输入
        Variable result = input.add(new Variable(expandedPosEnc));

        // 应用dropout（如果需要）
        if (dropout) {
            // 简单的dropout实现，在实际训练中需要考虑训练/推理模式
            // 这里暂时跳过dropout的实现
        }

        return result;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 位置编码的反向传播只是传递梯度，因为位置编码是固定的
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}