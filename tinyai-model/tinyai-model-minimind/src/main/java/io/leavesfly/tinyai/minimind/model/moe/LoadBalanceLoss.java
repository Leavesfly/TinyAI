package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.Serializable;

/**
 * Load Balance Loss - 负载均衡损失（对标 Python MOEFeedForward.aux_loss）
 * <p>
 * Python 公式:
 * load = one_hot(topk_idx, num_experts).float().mean(0)
 * aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
 * <p>
 * 其中:
 * - load: 每个专家被选中的频率（one-hot 求均值）
 * - scores.mean(0): 每个专家的平均路由概率
 * - router_aux_loss_coef: 配置中的系数
 *
 * @author leavesfly
 * @since 2024
 */
public class LoadBalanceLoss implements Serializable {

    private static final long serialVersionUID = 1L;

    private final float routerAuxLossCoef;  // 对标 Python router_aux_loss_coef

    /**
     * 构造函数（对标 Python）
     *
     * @param routerAuxLossCoef 路由辅助损失系数（默认 5e-4）
     */
    public LoadBalanceLoss(float routerAuxLossCoef) {
        this.routerAuxLossCoef = routerAuxLossCoef;
    }

    /**
     * 兼容旧接口的构造函数
     */
    public LoadBalanceLoss(float importanceCoef, float loadCoef) {
        // 旧接口：取 importanceCoef 作为 routerAuxLossCoef
        this.routerAuxLossCoef = importanceCoef;
    }

    /**
     * 默认构造函数
     */
    public LoadBalanceLoss() {
        this(5e-4f);
    }

    /**
     * 计算负载均衡损失（float 版本，对标 Python）
     * <p>
     * Python: (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
     */
    public float computeLoss(MoEBlock.LoadBalanceStats stats, int numExperts) {
        float[] importance = stats.getImportance();  // scores.mean(0)
        float[] load = stats.getLoad();               // one_hot mean(0)

        // aux_loss = sum(load_i * importance_i) * numExperts * routerAuxLossCoef
        float auxLoss = 0.0f;
        for (int i = 0; i < numExperts; i++) {
            auxLoss += load[i] * importance[i];
        }
        auxLoss *= numExperts * routerAuxLossCoef;

        return auxLoss;
    }

    /**
     * 计算负载均衡损失（Variable 版本，保留计算图，对标 Python）
     * <p>
     * 可微分部分: importance = allWeightsVar.mean(dim=0)
     * load 部分为离散操作，以常量形式参与
     * aux_loss = (load * importance).sum() * numExperts * routerAuxLossCoef
     */
    public Variable computeLossVar(Variable allWeightsVar,
                                   MoEBlock.LoadBalanceStats stats,
                                   int numExperts) {
        float[] load = stats.getLoad();

        if (allWeightsVar == null) {
            float fallback = computeLoss(stats, numExperts);
            Variable fallbackVar = new Variable(NdArray.of(new float[]{fallback}, Shape.of(1)));
            fallbackVar.setRequireGrad(false);
            return fallbackVar;
        }

        int[] dims = allWeightsVar.getShape().getShapeDims();
        int batchSize = dims[0];
        if (dims.length != 2 || dims[1] != numExperts) {
            float fallback = computeLoss(stats, numExperts);
            Variable fallbackVar = new Variable(NdArray.of(new float[]{fallback}, Shape.of(1)));
            fallbackVar.setRequireGrad(false);
            return fallbackVar;
        }

        // importanceVar = mean over batch of allWeightsVar，形状 [numExperts]
        Variable importanceSum = allWeightsVar.sumTo(Shape.of(1, numExperts));
        Variable batchSizeConst = new Variable(
                NdArray.of(new float[]{(float) batchSize}, Shape.of(1)));
        batchSizeConst.setRequireGrad(false);
        Variable batchSizeBroadcast = batchSizeConst.broadcastTo(Shape.of(1, numExperts));
        Variable importanceAvg = importanceSum.div(batchSizeBroadcast);
        Variable importanceVar = importanceAvg.reshape(Shape.of(numExperts));

        // loadVar：常量
        Variable loadConst = new Variable(NdArray.of(load.clone(), Shape.of(numExperts)));
        loadConst.setRequireGrad(false);

        // auxLoss = (load * importance).sum() * numExperts * routerAuxLossCoef
        Variable product = importanceVar.mul(loadConst);
        Variable auxSum = product.sumTo(Shape.of(1));
        Variable coefConst = new Variable(
                NdArray.of(new float[]{(float) numExperts * routerAuxLossCoef}, Shape.of(1)));
        coefConst.setRequireGrad(false);
        return auxSum.mul(coefConst);
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
     * 获取路由辅助损失系数
     */
    public float getRouterAuxLossCoef() {
        return routerAuxLossCoef;
    }

    @Override
    public String toString() {
        return String.format("LoadBalanceLoss(router_aux_loss_coef=%.4f)",
                routerAuxLossCoef);
    }
}