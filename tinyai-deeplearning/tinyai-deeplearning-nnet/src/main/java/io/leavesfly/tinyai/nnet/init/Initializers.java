package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import java.util.Random;

/**
 * 参数初始化工具类
 * &lt;p&gt;
 * 提供静态方法实现常用的参数初始化策略，
 * 包括零初始化、常量初始化、均匀分布、正态分布、Xavier初始化、Kaiming初始化等。
 *
 * @author leavesfly
 * @version 2.0
 */
public class Initializers {

    /**
     * 共享的 Random 实例
     * 使用静态实例避免每次初始化都创建新的 Random 对象
     * 提高性能和可重复性
     */
    private static final Random SHARED_RANDOM = new Random();

    /**
     * 获取共享的 Random 实例
     *
     * @return 共享的 Random 实例
     */
    public static Random getSharedRandom() {
        return SHARED_RANDOM;
    }

    /**
     * 零初始化
     * &lt;p&gt;
     * 将张量所有元素设置为0
     *
     * @param tensor 需要初始化的张量
     */
    public static void zeros(NdArray tensor) {
        new ZerosInitializer().initialize(tensor);
    }

    /**
     * 全一初始化
     * &lt;p&gt;
     * 将张量所有元素设置为1
     *
     * @param tensor 需要初始化的张量
     */
    public static void ones(NdArray tensor) {
        new OnesInitializer().initialize(tensor);
    }

    /**
     * 常量初始化
     * &lt;p&gt;
     * 将张量所有元素设置为指定常量
     *
     * @param tensor 需要初始化的张量
     * @param value  常量值
     */
    public static void constant(NdArray tensor, float value) {
        new ConstantInitializer(value).initialize(tensor);
    }

    /**
     * 均匀分布初始化
     * &lt;p&gt;
     * 从均匀分布 U(a, b) 中采样初始化张量
     *
     * @param tensor 需要初始化的张量
     * @param a      均匀分布下界
     * @param b      均匀分布上界
     */
    public static void uniform(NdArray tensor, float a, float b) {
        new UniformInitializer(a, b).initialize(tensor);
    }

    /**
     * 正态分布初始化
     * &lt;p&gt;
     * 从正态分布 N(mean, std²) 中采样初始化张量
     *
     * @param tensor 需要初始化的张量
     * @param mean   均值
     * @param std    标准差
     */
    public static void normal(NdArray tensor, float mean, float std) {
        new NormalInitializer(mean, std).initialize(tensor);
    }

    /**
     * Xavier均匀初始化（Glorot均匀初始化）
     * &lt;p&gt;
     * 适用于Sigmoid、Tanh等激活函数
     * &lt;p&gt;
     * 从均匀分布 U(-a, a) 中采样，其中：
     * a = gain * sqrt(6 / (fan_in + fan_out))
     *
     * @param tensor 需要初始化的张量
     * @param gain   增益系数（默认为1.0）
     */
    public static void xavierUniform(NdArray tensor, float gain) {
        new XavierUniformInitializer(gain).initialize(tensor);
    }

    /**
     * Xavier均匀初始化（默认gain=1.0）
     *
     * @param tensor 需要初始化的张量
     */
    public static void xavierUniform(NdArray tensor) {
        xavierUniform(tensor, 1.0f);
    }

    /**
     * Xavier正态初始化（Glorot正态初始化）
     * &lt;p&gt;
     * 适用于Sigmoid、Tanh等激活函数
     * &lt;p&gt;
     * 从正态分布 N(0, std²) 中采样，其中：
     * std = gain * sqrt(2 / (fan_in + fan_out))
     *
     * @param tensor 需要初始化的张量
     * @param gain   增益系数（默认为1.0）
     */
    public static void xavierNormal(NdArray tensor, float gain) {
        new XavierNormalInitializer(gain).initialize(tensor);
    }

    /**
     * Xavier正态初始化（默认gain=1.0）
     *
     * @param tensor 需要初始化的张量
     */
    public static void xavierNormal(NdArray tensor) {
        xavierNormal(tensor, 1.0f);
    }

    /**
     * Kaiming均匀初始化（He均匀初始化）
     * &lt;p&gt;
     * 适用于ReLU及其变体激活函数
     * &lt;p&gt;
     * 从均匀分布 U(-bound, bound) 中采样，其中：
     * bound = sqrt(6 / ((1 + a²) * fan))
     * &lt;p&gt;
     * fan根据mode选择fan_in或fan_out
     *
     * @param tensor      需要初始化的张量
     * @param a           leaky_relu的负斜率（对于ReLU使用0）
     * @param mode        "fan_in"或"fan_out"
     * @param nonlinearity 非线性函数类型（"relu"、"leaky_relu"等）
     */
    public static void kaimingUniform(NdArray tensor, float a, String mode, String nonlinearity) {
        new KaimingUniformInitializer(a, mode, nonlinearity).initialize(tensor);
    }

    /**
     * Kaiming均匀初始化（默认参数：a=0, mode="fan_in", nonlinearity="relu"）
     *
     * @param tensor 需要初始化的张量
     */
    public static void kaimingUniform(NdArray tensor) {
        kaimingUniform(tensor, 0, "fan_in", "relu");
    }

    /**
     * Kaiming正态初始化（He正态初始化）
     * &lt;p&gt;
     * 适用于ReLU及其变体激活函数
     * &lt;p&gt;
     * 从正态分布 N(0, std²) 中采样，其中：
     * std = sqrt(2 / ((1 + a²) * fan))
     * &lt;p&gt;
     * fan根据mode选择fan_in或fan_out
     *
     * @param tensor      需要初始化的张量
     * @param a           leaky_relu的负斜率（对于ReLU使用0）
     * @param mode        "fan_in"或"fan_out"
     * @param nonlinearity 非线性函数类型（"relu"、"leaky_relu"等）
     */
    public static void kaimingNormal(NdArray tensor, float a, String mode, String nonlinearity) {
        new KaimingNormalInitializer(a, mode, nonlinearity).initialize(tensor);
    }

    /**
     * Kaiming正态初始化（默认参数：a=0, mode="fan_in", nonlinearity="relu"）
     *
     * @param tensor 需要初始化的张量
     */
    public static void kaimingNormal(NdArray tensor) {
        kaimingNormal(tensor, 0, "fan_in", "relu");
    }

    /**
     * 正交初始化
     * &lt;p&gt;
     * 生成正交矩阵，常用于RNN的权重初始化
     * &lt;p&gt;
     * 注意：当前实现为简化版本，使用Xavier初始化代替
     *
     * @param tensor 需要初始化的张量
     * @param gain   增益系数
     */
    public static void orthogonal(NdArray tensor, float gain) {
        new OrthogonalInitializer(gain).initialize(tensor);
    }

    /**
     * 正交初始化（默认gain=1.0）
     *
     * @param tensor 需要初始化的张量
     */
    public static void orthogonal(NdArray tensor) {
        orthogonal(tensor, 1.0f);
    }

    // ==================== 内部辅助方法 ====================

    /**
     * 计算fan_in和fan_out
     * &lt;p&gt;
     * fan_in：输入单元数
     * fan_out：输出单元数
     * &lt;p&gt;
     * 对于2D张量（权重矩阵）：
     * - fan_in = 列数（输入维度）
     * - fan_out = 行数（输出维度）
     * &lt;p&gt;
     * 对于4D张量（卷积核）：
     * - fan_in = kernel_size * kernel_size * in_channels
     * - fan_out = kernel_size * kernel_size * out_channels
     *
     * @param shape 张量形状
     * @return [fan_in, fan_out]
     */
    static int[] calculateFanInAndFanOut(Shape shape) {
        int dimNum = shape.getDimNum();

        if (dimNum < 2) {
            throw new IllegalArgumentException("张量至少需要2个维度才能计算fan_in和fan_out");
        }

        if (dimNum == 2) {
            // 2D张量：[out_features, in_features]
            int fanOut = shape.getDimension(0);
            int fanIn = shape.getDimension(1);
            return new int[]{fanIn, fanOut};
        } else if (dimNum >= 3) {
            // 多维张量（如卷积核）：[out_channels, in_channels, height, width, ...]
            int numInputFmaps = shape.getDimension(1);
            int numOutputFmaps = shape.getDimension(0);
            int receptiveFieldSize = 1;
            for (int i = 2; i < dimNum; i++) {
                receptiveFieldSize *= shape.getDimension(i);
            }
            int fanIn = numInputFmaps * receptiveFieldSize;
            int fanOut = numOutputFmaps * receptiveFieldSize;
            return new int[]{fanIn, fanOut};
        }

        throw new IllegalArgumentException("不支持的张量维度：" + dimNum);
    }

    /**
     * 根据mode获取fan值
     *
     * @param fanIn  fan_in值
     * @param fanOut fan_out值
     * @param mode   模式（"fan_in"、"fan_out"或"fan_avg"）
     * @return fan值
     */
    static int getFan(int fanIn, int fanOut, String mode) {
        if ("fan_in".equals(mode)) {
            return fanIn;
        } else if ("fan_out".equals(mode)) {
            return fanOut;
        } else if ("fan_avg".equals(mode)) {
            return (fanIn + fanOut) / 2;
        } else {
            throw new IllegalArgumentException("不支持的mode: " + mode + "，应为'fan_in'、'fan_out'或'fan_avg'");
        }
    }

    /**
     * 计算Kaiming初始化的增益
     *
     * @param a            负斜率参数
     * @param nonlinearity 非线性函数类型
     * @return 增益系数
     */
    static float calculateGain(float a, String nonlinearity) {
        if ("linear".equals(nonlinearity) || "sigmoid".equals(nonlinearity)) {
            return 1.0f;
        } else if ("tanh".equals(nonlinearity)) {
            return 5.0f / 3;
        } else if ("relu".equals(nonlinearity)) {
            return (float) Math.sqrt(2.0);
        } else if ("leaky_relu".equals(nonlinearity)) {
            return (float) Math.sqrt(2.0 / (1 + a * a));
        } else {
            // 默认使用1.0
            return 1.0f;
        }
    }
}