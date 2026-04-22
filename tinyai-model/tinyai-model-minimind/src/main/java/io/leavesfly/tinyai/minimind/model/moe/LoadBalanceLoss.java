package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.Serializable;

/**
 * Load Balance Loss - 负载均衡损失
 * <p>
 * 确保专家使用均衡,避免某些专家过载而其他专家闲置
 * <p>
 * 核心公式:
 * L_balance = α · importance_loss + β · load_loss
 * <p>
 * importance_loss = num_experts · Σ(importance_i · load_i)
 * load_loss = CV(load) = std(load) / mean(load)
 * <p>
 * 其中:
 * - importance_i: 专家i的重要性(所有样本的权重和)
 * - load_i: 专家i的负载(被选中的次数占比)
 * - CV: 变异系数(Coefficient of Variation)
 *
 * @author leavesfly
 * @since 2024
 */
public class LoadBalanceLoss implements Serializable {

    private static final long serialVersionUID = 1L;

    private final float importanceCoef;  // 重要性损失系数
    private final float loadCoef;        // 负载损失系数

    /**
     * 构造函数
     *
     * @param importanceCoef 重要性损失系数(默认0.01)
     * @param loadCoef       负载损失系数(默认0.01)
     */
    public LoadBalanceLoss(float importanceCoef, float loadCoef) {
        this.importanceCoef = importanceCoef;
        this.loadCoef = loadCoef;
    }

    /**
     * 默认构造函数
     */
    public LoadBalanceLoss() {
        this(0.01f, 0.01f);
    }

    /**
     * 计算负载均衡损失
     *
     * @param stats      负载均衡统计
     * @param numExperts 专家数量
     * @return 负载均衡损失
     */
    public float computeLoss(MoEBlock.LoadBalanceStats stats, int numExperts) {
        float[] importance = stats.getImportance();
        float[] load = stats.getLoad();

        // 1. Switch Transformer 标准的辅助损失
        // Auxiliary Loss = numExperts * sum(fraction_i * probability_i)
        // 其中 fraction_i 是分配给专家 i 的 token 比例
        // probability_i 是路由到专家 i 的平均概率
        float auxiliaryLoss = 0.0f;
        for (int i = 0; i < numExperts; i++) {
            auxiliaryLoss += load[i] * importance[i];
        }
        auxiliaryLoss *= numExperts;

        // 2. Load Loss: 变异系数CV(load)
        float loadLoss = coefficientOfVariation(load);

        // 3. 总损失：使用辅助损失替代原来的 importance loss
        float totalLoss = importanceCoef * auxiliaryLoss + loadCoef * loadLoss;

        return totalLoss;
    }

    /**
     * 计算负载均衡损失（Variable 版本，保留计算图）
     * <p>
     * 修复说明（计算图断裂修复）：
     * 原版 {@link #computeLoss(MoEBlock.LoadBalanceStats, int)} 返回 float，
     * 导致 gate_linear 无法通过辅助损失获得梯度（虽然 MoE 主路径在本次修复后已能回传，
     * 但负载均衡损失本身对 gate 训练稳定性至关重要）。
     * <p>
     * 可微分部分：Auxiliary Loss = numExperts * Σ(load_i * importance_i)
     *   - importance_i：对 allWeightsVar 沿 batch 维求平均得到的每专家平均权重，可微；
     *   - load_i：每个专家被选中的 token 比例，由离散 Top-K 选择产生，本身不可导，
     *             以常量形式参与乘法（不需要梯度）。
     * <p>
     * CV(load) 部分同样是离散统计量，以常量形式加到总损失上，不参与梯度。
     *
     * @param allWeightsVar Router softmax 后的权重 Variable，形状 [B, numExperts]（可为 null，此时退化为常量）
     * @param stats         负载均衡统计（float[] 形式，含 importance/load 快照）
     * @param numExperts    专家数量
     * @return 负载均衡损失 Variable（若 allWeightsVar 为 null，返回不可导的常量 Variable）
     */
    public Variable computeLossVar(Variable allWeightsVar,
                                   MoEBlock.LoadBalanceStats stats,
                                   int numExperts) {
        float[] load = stats.getLoad();
        float cvLoad = coefficientOfVariation(load);

        // 1) CV(load) 部分：完全离散，包装为不可导常量
        Variable cvLoadVar = new Variable(
                NdArray.of(new float[]{loadCoef * cvLoad}, Shape.of(1)));
        cvLoadVar.setRequireGrad(false);

        if (allWeightsVar == null) {
            // 无法构造可微的 auxiliary loss，退化为 float 版本的常量返回
            float fallback = computeLoss(stats, numExperts);
            Variable fallbackVar = new Variable(NdArray.of(new float[]{fallback}, Shape.of(1)));
            fallbackVar.setRequireGrad(false);
            return fallbackVar;
        }

        int[] dims = allWeightsVar.getShape().getShapeDims();
        int batchSize = dims[0];
        if (dims.length != 2 || dims[1] != numExperts) {
            // 形状不符，退化为常量
            float fallback = computeLoss(stats, numExperts);
            Variable fallbackVar = new Variable(NdArray.of(new float[]{fallback}, Shape.of(1)));
            fallbackVar.setRequireGrad(false);
            return fallbackVar;
        }

        // 2) importanceVar = mean over batch of allWeightsVar，形状 [numExperts]
        //    用 sumTo 得 [1, numExperts]，再除以 batchSize，然后 reshape 到 [numExperts]
        Variable importanceSum = allWeightsVar.sumTo(Shape.of(1, numExperts));
        Variable batchSizeConst = new Variable(
                NdArray.of(new float[]{(float) batchSize}, Shape.of(1)));
        batchSizeConst.setRequireGrad(false);
        Variable batchSizeBroadcast = batchSizeConst.broadcastTo(Shape.of(1, numExperts));
        Variable importanceAvg = importanceSum.div(batchSizeBroadcast);
        Variable importanceVar = importanceAvg.reshape(Shape.of(numExperts));

        // 3) loadVar：常量（不可导），形状 [numExperts]
        Variable loadConst = new Variable(NdArray.of(load.clone(), Shape.of(numExperts)));
        loadConst.setRequireGrad(false);

        // 4) auxLoss = numExperts * sum(importanceVar * loadConst)
        Variable product = importanceVar.mul(loadConst);
        Variable auxSum = product.sumTo(Shape.of(1));
        Variable numExpertsConst = new Variable(
                NdArray.of(new float[]{(float) numExperts}, Shape.of(1)));
        numExpertsConst.setRequireGrad(false);
        Variable auxLoss = auxSum.mul(numExpertsConst);

        // 5) 加权系数 importanceCoef
        Variable importanceCoefConst = new Variable(
                NdArray.of(new float[]{importanceCoef}, Shape.of(1)));
        importanceCoefConst.setRequireGrad(false);
        Variable weightedAux = auxLoss.mul(importanceCoefConst);

        // 6) 总损失 = importanceCoef * auxLoss + loadCoef * cvLoad（CV 部分为常量）
        return weightedAux.add(cvLoadVar);
    }

    /**
     * 计算变异系数 CV = std / mean
     */
    private float coefficientOfVariation(float[] values) {
        int n = values.length;
        if (n == 0) return 0.0f;

        // 计算均值
        float mean = 0.0f;
        for (float v : values) {
            mean += v;
        }
        mean /= n;

        if (mean == 0.0f) return 0.0f;

        // 计算标准差
        float variance = 0.0f;
        for (float v : values) {
            variance += (v - mean) * (v - mean);
        }
        variance /= n;
        float std = (float) Math.sqrt(variance);

        // 变异系数
        return std / mean;
    }

    /**
     * 获取重要性系数
     */
    public float getImportanceCoef() {
        return importanceCoef;
    }

    /**
     * 获取负载系数
     */
    public float getLoadCoef() {
        return loadCoef;
    }

    @Override
    public String toString() {
        return String.format("LoadBalanceLoss(importance_coef=%.4f, load_coef=%.4f)",
                importanceCoef, loadCoef);
    }
}