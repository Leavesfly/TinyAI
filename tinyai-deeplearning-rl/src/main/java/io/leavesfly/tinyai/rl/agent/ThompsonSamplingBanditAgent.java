package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Random;

/**
 * 汤普森采样（Thompson Sampling）多臂老虎机智能体
 * <p>
 * 汤普森采样是一种基于贝叶斯方法的算法：
 * - 为每个臂维护一个奖励分布的后验分布
 * - 每次选择时，从每个臂的后验分布中采样一个值
 * - 选择采样值最大的臂
 * <p>
 * 实现方式：
 * - 假设奖励服从正态分布
 * - 使用正态-正态共轭先验
 * - 后验分布也是正态分布，参数可以递增更新
 * <p>
 * 算法特点：
 * - 理论上有很好的悔恨界保证
 * - 自适应地平衡探索和利用
 * - 对环境变化有较好的适应性
 * - 计算复杂度相对较高
 *
 * @author leavesfly
 */
public class ThompsonSamplingBanditAgent extends BanditAgent {

    /**
     * 随机数生成器
     */
    private final Random random;

    /**
     * 每个臂的先验均值
     */
    private float[] priorMeans;

    /**
     * 每个臂的先验精度（方差的倒数）
     */
    private float[] priorPrecisions;

    /**
     * 观测噪声的精度（假设已知）
     */
    private float noisePrecision;

    /**
     * 每个臂的后验均值
     */
    private float[] posteriorMeans;

    /**
     * 每个臂的后验精度
     */
    private float[] posteriorPrecisions;

    /**
     * 构造函数，使用默认参数
     *
     * @param name    智能体名称
     * @param numArms 臂的数量
     */
    public ThompsonSamplingBanditAgent(String name, int numArms) {
        this(name, numArms, 0.0f, 1.0f, 1.0f);
    }

    /**
     * 完整构造函数
     *
     * @param name           智能体名称
     * @param numArms        臂的数量
     * @param priorMean      先验均值
     * @param priorPrecision 先验精度
     * @param noisePrecision 观测噪声精度
     */
    public ThompsonSamplingBanditAgent(String name, int numArms, float priorMean, float priorPrecision, float noisePrecision) {
        super(name, numArms);
        this.random = new Random();
        this.noisePrecision = noisePrecision;

        // 初始化先验分布参数
        this.priorMeans = new float[numArms];
        this.priorPrecisions = new float[numArms];
        this.posteriorMeans = new float[numArms];
        this.posteriorPrecisions = new float[numArms];

        for (int i = 0; i < numArms; i++) {
            this.priorMeans[i] = priorMean;
            this.priorPrecisions[i] = priorPrecision;
            this.posteriorMeans[i] = priorMean;
            this.posteriorPrecisions[i] = priorPrecision;
        }
    }

    @Override
    public Variable selectAction(Variable state) {
        int armIndex = selectArm();
        return new Variable(NdArray.of(new float[]{armIndex}, Shape.of(1)));
    }

    @Override
    public int selectArm() {
        // 从每个臂的后验分布中采样
        float[] samples = new float[actionDim];
        for (int i = 0; i < actionDim; i++) {
            samples[i] = sampleFromPosterior(i);
        }

        // 选择采样值最大的臂
        int bestArm = 0;
        for (int i = 1; i < actionDim; i++) {
            if (samples[i] > samples[bestArm]) {
                bestArm = i;
            }
        }

        return bestArm;
    }

    /**
     * 从指定臂的后验分布中采样
     *
     * @param armIndex 臂索引
     * @return 采样值
     */
    private float sampleFromPosterior(int armIndex) {
        float mean = posteriorMeans[armIndex];
        float precision = posteriorPrecisions[armIndex];
        float variance = 1.0f / precision;
        float stdDev = (float) Math.sqrt(variance);

        // 从正态分布采样
        return mean + stdDev * (float) random.nextGaussian();
    }

    /**
     * 更新后验分布参数
     */
    @Override
    protected void updateStatistics(int armIndex, float reward) {
        super.updateStatistics(armIndex, reward);

        // 贝叶斯更新：正态-正态共轭
        float oldMean = posteriorMeans[armIndex];
        float oldPrecision = posteriorPrecisions[armIndex];

        // 更新后验精度
        posteriorPrecisions[armIndex] = oldPrecision + noisePrecision;

        // 更新后验均值
        posteriorMeans[armIndex] = (oldPrecision * oldMean + noisePrecision * reward) / posteriorPrecisions[armIndex];
    }

    /**
     * 获取指定臂的后验均值
     *
     * @param armIndex 臂索引
     * @return 后验均值
     */
    public float getPosteriorMean(int armIndex) {
        return posteriorMeans[armIndex];
    }

    /**
     * 获取指定臂的后验方差
     *
     * @param armIndex 臂索引
     * @return 后验方差
     */
    public float getPosteriorVariance(int armIndex) {
        return 1.0f / posteriorPrecisions[armIndex];
    }

    /**
     * 获取指定臂的后验标准差
     *
     * @param armIndex 臂索引
     * @return 后验标准差
     */
    public float getPosteriorStdDev(int armIndex) {
        return (float) Math.sqrt(getPosteriorVariance(armIndex));
    }

    /**
     * 获取所有臂的后验均值
     *
     * @return 后验均值数组
     */
    public float[] getAllPosteriorMeans() {
        return posteriorMeans.clone();
    }

    /**
     * 获取所有臂的后验方差
     *
     * @return 后验方差数组
     */
    public float[] getAllPosteriorVariances() {
        float[] variances = new float[actionDim];
        for (int i = 0; i < actionDim; i++) {
            variances[i] = getPosteriorVariance(i);
        }
        return variances;
    }

    /**
     * 获取噪声精度
     *
     * @return 噪声精度
     */
    public float getNoisePrecision() {
        return noisePrecision;
    }

    /**
     * 设置噪声精度
     *
     * @param noisePrecision 新的噪声精度
     */
    public void setNoisePrecision(float noisePrecision) {
        this.noisePrecision = Math.max(0.001f, noisePrecision);
    }

    /**
     * 重置为先验分布
     */
    @Override
    public void reset() {
        super.reset();
        System.arraycopy(priorMeans, 0, posteriorMeans, 0, actionDim);
        System.arraycopy(priorPrecisions, 0, posteriorPrecisions, 0, actionDim);
    }

    /**
     * 设置随机种子（用于实验重现）
     *
     * @param seed 随机种子
     */
    public void setSeed(long seed) {
        random.setSeed(seed);
    }

    @Override
    public void printStatus() {
        super.printStatus();
        System.out.println("噪声精度: " + String.format("%.4f", noisePrecision));

        System.out.println("各臂后验分布:");
        for (int i = 0; i < actionDim; i++) {
            System.out.println("  臂 " + i + ": 后验均值=" + String.format("%.4f", posteriorMeans[i]) + ", 后验方差=" + String.format("%.4f", getPosteriorVariance(i)) + ", 后验精度=" + String.format("%.4f", posteriorPrecisions[i]));
        }
    }

    /**
     * 获取算法描述信息
     *
     * @return 算法描述
     */
    public String getAlgorithmDescription() {
        return String.format("汤普森采样算法 (噪声精度=%.4f)", noisePrecision);
    }

    /**
     * 获取采样信息
     *
     * @return 采样信息
     */
    public String getSamplingInfo() {
        StringBuilder info = new StringBuilder();
        info.append("汤普森采样信息:\n");

        float[] samples = new float[actionDim];
        for (int i = 0; i < actionDim; i++) {
            samples[i] = sampleFromPosterior(i);
        }

        int bestArm = 0;
        for (int i = 1; i < actionDim; i++) {
            if (samples[i] > samples[bestArm]) {
                bestArm = i;
            }
        }

        info.append(String.format("  推荐选择臂 %d (采样值=%.4f)\n", bestArm, samples[bestArm]));
        info.append("  各臂采样值:\n");

        for (int i = 0; i < actionDim; i++) {
            info.append(String.format("    臂 %d: 采样值=%.4f, 后验μ=%.4f, 后验σ=%.4f\n", i, samples[i], posteriorMeans[i], getPosteriorStdDev(i)));
        }

        return info.toString();
    }

    /**
     * 获取当前后验分布的置信区间
     *
     * @param armIndex        臂索引
     * @param confidenceLevel 置信水平（如0.95）
     * @return 置信区间 [下界, 上界]
     */
    public float[] getConfidenceInterval(int armIndex, float confidenceLevel) {
        float mean = posteriorMeans[armIndex];
        float stdDev = getPosteriorStdDev(armIndex);

        // 使用正态分布的分位数（近似）
        float z = getZScore(confidenceLevel);
        float margin = z * stdDev;

        return new float[]{mean - margin, mean + margin};
    }

    /**
     * 获取指定置信水平对应的Z分数（近似）
     *
     * @param confidenceLevel 置信水平
     * @return Z分数
     */
    private float getZScore(float confidenceLevel) {
        // 常用置信水平的Z分数
        if (confidenceLevel >= 0.99f) return 2.576f;
        if (confidenceLevel >= 0.95f) return 1.96f;
        if (confidenceLevel >= 0.90f) return 1.645f;
        if (confidenceLevel >= 0.80f) return 1.282f;
        return 1.0f; // 默认值
    }
}