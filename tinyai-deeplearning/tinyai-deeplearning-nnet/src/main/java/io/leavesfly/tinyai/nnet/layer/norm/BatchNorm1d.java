package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的BatchNorm1d层
 * <p>
 * Batch Normalization 在批次维度上进行归一化，
 * 适用于全连接层和1D卷积层。
 * <p>
 * 训练模式：
 * - 使用当前批次的均值和方差进行归一化
 * - 更新移动平均统计量（用于推理）
 * <p>
 * 推理模式：
 * - 使用固定的移动平均统计量
 * <p>
 * 公式：
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * @author leavesfly
 * @version 2.0
 */
public class BatchNorm1d extends Module {

    private Parameter gamma;           // 缩放参数 (num_features,)
    private Parameter beta;            // 平移参数 (num_features,)

    private final int numFeatures;     // 特征维度
    private final float eps;           // 数值稳定项
    private final float momentum;      // 移动平均动量
    private final boolean affine;      // 是否使用可学习参数
    private final boolean trackRunningStats;  // 是否跟踪统计量

    /** 缓存的 eps 常量 Variable，避免每次 forward 都创建新对象，且不参与梯度计算 */
    private transient Variable cachedEpsVar;

    /**
     * 构造函数
     *
     * @param name              层名称
     * @param numFeatures       特征维度
     * @param eps               数值稳定项（默认1e-5）
     * @param momentum          移动平均动量（默认0.1）
     * @param affine            是否使用可学习参数（默认true）
     * @param trackRunningStats 是否跟踪统计量（默认true）
     */
    public BatchNorm1d(String name, int numFeatures, float eps, float momentum,
                       boolean affine, boolean trackRunningStats) {
        super(name);
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        // 创建可学习参数
        if (affine) {
            NdArray gammaData = NdArray.of(Shape.of(numFeatures));
            NdArray betaData = NdArray.of(Shape.of(numFeatures));
            this.gamma = registerParameter("gamma", new Parameter(gammaData));
            this.beta = registerParameter("beta", new Parameter(betaData));
        } else {
            this.gamma = null;
            this.beta = null;
        }

        // 创建移动平均缓冲区
        if (trackRunningStats) {
            NdArray runningMean = NdArray.of(new float[numFeatures], Shape.of(numFeatures));
            NdArray runningVar = NdArray.of(new float[numFeatures], Shape.of(numFeatures));
            // 初始化 running_var 为 1
            for (int i = 0; i < numFeatures; i++) {
                runningVar.getArray()[i] = 1.0f;
            }
            registerBuffer("running_mean", runningMean);
            registerBuffer("running_var", runningVar);

            // 记录处理的批次数（可选）
            registerBuffer("num_batches_tracked", NdArray.of(new float[]{0}, Shape.of(1)));
        }

        // 初始化 eps 常量缓存（使用 1D shape 以兼容广播）
        this.cachedEpsVar = new Variable(NdArray.of(new float[]{eps}, Shape.of(1)));
        this.cachedEpsVar.setRequireGrad(false);

        // 初始化参数
        init();
    }

    /**
     * 构造函数（使用默认参数）
     *
     * @param name        层名称
     * @param numFeatures 特征维度
     */
    public BatchNorm1d(String name, int numFeatures) {
        this(name, numFeatures, 1e-5f, 0.1f, true, true);
    }

    @Override
    public void resetParameters() {
        if (affine) {
            // gamma初始化为1
            Initializers.ones(gamma.data());
            // beta初始化为0
            Initializers.zeros(beta.data());
        }
        resetRunningStats();
    }

    /**
     * 重置移动平均统计量
     */
    public void resetRunningStats() {
        if (trackRunningStats) {
            NdArray runningMean = getBuffer("running_mean");
            NdArray runningVar = getBuffer("running_var");

            // running_mean 初始化为 0
            float[] meanData = runningMean.getArray();
            for (int i = 0; i < meanData.length; i++) {
                meanData[i] = 0.0f;
            }

            // running_var 初始化为 1
            float[] varData = runningVar.getArray();
            for (int i = 0; i < varData.length; i++) {
                varData[i] = 1.0f;
            }

            // 重置批次计数
            NdArray numBatches = getBuffer("num_batches_tracked");
            if (numBatches != null) {
                numBatches.getArray()[0] = 0;
            }
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable的形状属性，完全停留在Variable层面
        int[] dims = x.sizes();

        // 输入形状检查：(batch_size, num_features) 或 (batch_size, num_features, length)
        if (dims.length < 2 || dims[1] != numFeatures) {
            throw new IllegalArgumentException(
                    String.format("Expected input with %d features, but got shape with dims[1]=%d",
                            numFeatures, dims.length > 1 ? dims[1] : -1));
        }

        int batchSize = dims[0];

        if (isTraining()) {
            // 训练模式：使用批次统计量
            return forwardTraining(x, batchSize);
        } else {
            // 推理模式：使用移动平均统计量
            return forwardInference(x);
        }
    }

    /**
     * 训练模式的前向传播
     *
     * @param x         输入变量
     * @param batchSize 批次大小
     * @return 归一化后的输出
     */
    private Variable forwardTraining(Variable x, int batchSize) {
        // 1. 计算批次统计量
        Variable batchMean = x.mean(0, false);  // (num_features,)
        Variable batchVar = x.var(0, false);    // (num_features,)

        // 2. 更新移动平均统计量
        if (trackRunningStats) {
            updateRunningStats(batchMean.getValue(), batchVar.getValue());
        }

        // 3. 归一化
        Variable normalized = normalize(x, batchMean, batchVar);

        // 4. 应用缩放和平移
        return applyAffineTransform(normalized);
    }

    /**
     * 推理模式的前向传播
     *
     * @param x 输入变量
     * @return 归一化后的输出
     */
    private Variable forwardInference(Variable x) {
        if (!trackRunningStats) {
            throw new IllegalStateException(
                    "Cannot use BatchNorm1d in eval mode without trackRunningStats=true");
        }

        // 使用移动平均统计量
        NdArray runningMean = getBuffer("running_mean");
        NdArray runningVar = getBuffer("running_var");

        Variable mean = new Variable(runningMean);
        Variable var = new Variable(runningVar);

        // 归一化
        Variable normalized = normalize(x, mean, var);

        // 应用缩放和平移
        return applyAffineTransform(normalized);
    }

    /**
     * 归一化操作
     *
     * @param x    输入
     * @param mean 均值
     * @param var  方差
     * @return 归一化结果
     */
    private Variable normalize(Variable x, Variable mean, Variable var) {
        // normalized = (x - mean) / sqrt(var + eps)
        Variable centered = x.sub(mean);
        Variable std = var.add(cachedEpsVar).sqrt();
        return centered.div(std);
    }

    /**
     * 应用仿射变换
     *
     * @param normalized 归一化后的输入
     * @return 变换后的输出
     */
    private Variable applyAffineTransform(Variable normalized) {
        if (affine) {
            // y = gamma * normalized + beta
            return normalized.mul(gamma).add(beta);
        } else {
            return normalized;
        }
    }

    /**
     * 更新移动平均统计量
     *
     * @param batchMean 当前批次均值
     * @param batchVar  当前批次方差
     */
    private void updateRunningStats(NdArray batchMean, NdArray batchVar) {
        NdArray runningMean = getBuffer("running_mean");
        NdArray runningVar = getBuffer("running_var");

        float[] runningMeanData = runningMean.getArray();
        float[] runningVarData = runningVar.getArray();
        float[] batchMeanData = batchMean.getArray();
        float[] batchVarData = batchVar.getArray();

        // 移动平均更新：
        // running_stat = (1 - momentum) * running_stat + momentum * batch_stat
        for (int i = 0; i < numFeatures; i++) {
            runningMeanData[i] = (1 - momentum) * runningMeanData[i] + momentum * batchMeanData[i];
            runningVarData[i] = (1 - momentum) * runningVarData[i] + momentum * batchVarData[i];
        }

        // 更新批次计数
        NdArray numBatches = getBuffer("num_batches_tracked");
        if (numBatches != null) {
            numBatches.getArray()[0] += 1;
        }
    }

    /**
     * 获取gamma参数
     *
     * @return gamma参数
     */
    public Parameter getGamma() {
        return gamma;
    }

    /**
     * 获取beta参数
     *
     * @return beta参数
     */
    public Parameter getBeta() {
        return beta;
    }

    /**
     * 获取移动平均均值
     *
     * @return running_mean
     */
    public NdArray getRunningMean() {
        return getBuffer("running_mean");
    }

    /**
     * 获取移动平均方差
     *
     * @return running_var
     */
    public NdArray getRunningVar() {
        return getBuffer("running_var");
    }

    /**
     * 获取已处理的批次数
     *
     * @return 批次数
     */
    public long getNumBatchesTracked() {
        NdArray numBatches = getBuffer("num_batches_tracked");
        return numBatches != null ? (long) numBatches.getArray()[0] : 0;
    }

    @Override
    public String toString() {
        return "BatchNorm1d{" +
                "name='" + name + '\'' +
                ", numFeatures=" + numFeatures +
                ", eps=" + eps +
                ", momentum=" + momentum +
                ", affine=" + affine +
                ", trackRunningStats=" + trackRunningStats +
                ", training=" + isTraining() +
                '}';
    }
}
