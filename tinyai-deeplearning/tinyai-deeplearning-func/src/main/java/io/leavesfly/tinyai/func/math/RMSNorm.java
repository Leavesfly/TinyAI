package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * RMS归一化算子
 * <p>
 * Root Mean Square Layer Normalization
 * RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
 * <p>
 * 相比LayerNorm更高效（无需均值计算），用于LLaMA、Qwen等现代LLM
 */
public class RMSNorm extends Function {

    private final int[] normalizedShape;
    private final float eps;
    private Shape inputShape;
    private NdArray normFactor; // 归一化因子：1 / sqrt(mean(x²) + eps)

    public RMSNorm(int[] normalizedShape, float eps) {
        this.normalizedShape = normalizedShape;
        this.eps = eps;
    }

    public RMSNorm(int[] normalizedShape) {
        this(normalizedShape, 1e-6f);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        NdArray weight = inputs[1]; // 可学习的缩放权重

        inputShape = x.getShape();
        int[] inputDims = inputShape.getShapeDims();

        // 计算RMS: sqrt(mean(x²) + eps)
        // 在normalizedShape指定的维度上计算
        NdArray xSquared = x.mul(x);
        
        // 计算均值（在归一化维度上）
        NdArray meanXSquared = computeMean(xSquared, normalizedShape);
        
        // 计算RMS: sqrt(mean + eps)
        NdArray epsArray = NdArray.of(eps);
        NdArray rms = meanXSquared.add(epsArray.broadcastTo(meanXSquared.getShape())).sqrt();
        
        // 保存归一化因子用于反向传播
        normFactor = rms;

        // 归一化: x / rms
        NdArray normalized = x.div(rms.broadcastTo(inputShape));

        // 应用权重: normalized * weight
        return normalized.mul(weight.broadcastTo(inputShape));
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 完整梯度计算：
        // dX = weight / rms * (dY - normalized * mean(dY * normalized))
        // dWeight = sum(dY * normalized)

        NdArray x = inputs[0].getValue();
        NdArray weight = inputs[1].getValue();

        NdArray rmsBroadcast = normFactor.broadcastTo(inputShape);
        NdArray weightBroadcast = weight.broadcastTo(inputShape);

        // normalized = x / rms
        NdArray normalized = x.div(rmsBroadcast);

        // dWeight: sum(dY * normalized) 在归一化维度上
        NdArray dWeight = sumOverNormalizedDims(yGrad.mul(normalized), normalizedShape);

        // dX 完整公式：weight / rms * (dY - normalized * mean(dY * normalized))
        // 1. 计算 dY * normalized 在归一化维度上的均值
        NdArray dYTimesNorm = yGrad.mul(normalized);
        NdArray meanDYTimesNorm = computeMean(dYTimesNorm, normalizedShape);

        // 2. 修正项: normalized * mean(dY * normalized)
        NdArray correction = normalized.mul(meanDYTimesNorm.broadcastTo(inputShape));

        // 3. dX = weight / rms * (dY - correction)
        NdArray dX = weightBroadcast.div(rmsBroadcast).mul(yGrad.sub(correction));

        return java.util.Arrays.asList(dX, dWeight);
    }

    /**
     * 计算均值（在指定维度上）
     */
    private NdArray computeMean(NdArray x, int[] normalizedShape) {
        // 简化实现：假设normalizedShape是最后几个维度
        // 完整实现需要更复杂的维度处理
        
        int[] inputDims = x.getShape().getShapeDims();
        int startDim = inputDims.length - normalizedShape.length;
        
        // 计算需要求和的维度大小
        int size = 1;
        for (int i = startDim; i < inputDims.length; i++) {
            size *= inputDims[i];
        }
        
        // 求和并除以大小
        NdArray sum = x.sum();
        return sum.divNum((float) size);
    }

    /**
     * 在归一化维度上求和
     */
    private NdArray sumOverNormalizedDims(NdArray x, int[] normalizedShape) {
        // 简化实现：对所有维度求和，然后reshape
        // 完整实现需要更精确的维度处理
        return x.sumTo(Shape.of(normalizedShape));
    }

    @Override
    public int requireInputNum() {
        return 2; // x 和 weight
    }
}

