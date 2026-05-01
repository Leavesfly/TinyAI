package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.Map;

/**
 * GRPO 参考模型快照（Reference Policy Snapshot）
 *
 * <p>GRPO（arXiv:2501.12948）的 KL 约束项 {@code β · KL(π_new || π_ref)} 需要一个冻结的
 * 参考策略 π_ref。本类封装参考模型的生命周期管理：
 * <ol>
 *   <li><b>快照</b>：构造时从当前 policy 深拷贝 state_dict，所有参数 {@code requireGrad=false}</li>
 *   <li><b>推理</b>：只做 forward，梯度不会流回参考模型（也不污染 policy 计算图）</li>
 *   <li><b>同步</b>：训练过程中按周期（默认每 epoch）将参考模型同步到最新 policy，
 *       模拟 PPO 多 epoch refresh 行为</li>
 * </ol>
 *
 * <p>实现要点：
 * <ul>
 *   <li>利用 {@link io.leavesfly.tinyai.nnet.core.Module#stateDict()} 获取深拷贝的参数字典</li>
 *   <li>利用 {@link io.leavesfly.tinyai.nnet.core.Module#loadStateDict(Map, boolean)} 加载到参考模型</li>
 *   <li>通过 {@link io.leavesfly.tinyai.nnet.core.Module#requiresGrad(boolean)} 冻结参考模型参数</li>
 *   <li>参考模型与 policy 使用相同架构但独立实例，避免内存引用污染</li>
 * </ul>
 *
 * <p>线程安全：本类非线程安全。调用 {@link #syncFrom(DeepSeekR1Model)} 与 {@link #forwardLogits(Variable)}
 * 必须串行化。
 *
 * @author leavesfly
 * @version 1.0
 */
public final class GRPOReferenceModel {

    /**
     * 参考模型实例（独立的 DeepSeekR1Model），参数已冻结，仅作 forward 使用
     */
    private final DeepSeekR1Model referenceModel;

    /**
     * 模型配置（与 policy 必须完全一致）
     */
    private final DeepSeekR1Config config;

    /**
     * 构造参考模型快照
     *
     * <p>使用给定 policy 的 config 创建同架构的新模型实例，然后从 policy 深拷贝参数值。
     * 构造完成后，参考模型处于 <b>frozen</b> 状态，requireGrad 全部为 false。
     *
     * @param policy 当前策略模型（作为快照源）
     * @param config R1 配置（必须与 policy 完全一致）
     */
    public GRPOReferenceModel(DeepSeekR1Model policy, DeepSeekR1Config config) {
        if (policy == null) {
            throw new IllegalArgumentException("policy must not be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        this.config = config;
        // 1. 创建同架构新模型实例
        this.referenceModel = new DeepSeekR1Model(policy.getName() + "_ref", config);
        // 2. 从 policy 深拷贝当前参数（stateDict 返回的是值的副本）
        syncFrom(policy);
    }

    /**
     * 将参考模型的参数同步为 policy 的当前参数值
     *
     * <p>使用 state_dict 机制完成深拷贝，完成后重新 freeze 所有参数，
     * 防止参考模型参数被反向传播误修改。
     *
     * @param policy 当前策略模型
     */
    public void syncFrom(DeepSeekR1Model policy) {
        if (policy == null) {
            throw new IllegalArgumentException("policy must not be null");
        }
        // stateDict() 返回的是参数值的深拷贝副本
        Map<String, NdArray> snapshot = policy.getModule().stateDict();
        referenceModel.getModule().loadStateDict(snapshot, true);
        // 加载后必须重新 freeze：loadStateDict 只换 value，不改 requireGrad 标志
        freeze();
    }

    /**
     * 冻结参考模型的所有参数（requireGrad=false）
     *
     * <p>这是参考模型的<b>不变式</b>：任何会改变 requireGrad 的操作后（如构造、同步），
     * 必须调用本方法保证参数无法进入反向传播。
     */
    public void freeze() {
        for (Parameter p : referenceModel.getModule().parameters(true)) {
            if (p != null) {
                p.setRequireGrad(false);
            }
        }
    }

    /**
     * 参考模型前向传播，返回 logits
     *
     * <p>虽然参考模型参数已冻结，为了保险起见，调用方应再对返回的 Variable 调用 {@link Variable#detach()}
     * 彻底切断与 policy 计算图的潜在联系。
     *
     * @param inputIds 输入 token ids [batchSize, seqLen]
     * @return logits [batchSize, seqLen, vocabSize]，已 detach
     */
    public Variable forwardLogits(Variable inputIds) {
        if (inputIds == null) {
            throw new IllegalArgumentException("inputIds must not be null");
        }
        DeepSeekR1Model.ReasoningResult result = referenceModel.performReasoning(inputIds);
        // 无论内部参数 requireGrad 如何，强制 detach，确保 ref_logits 完全脱离计算图
        return result.logits.detach();
    }

    /**
     * 获取参考模型实例（主要供测试/诊断使用，生产代码应只通过 forwardLogits 访问）
     */
    public DeepSeekR1Model getReferenceModel() {
        return referenceModel;
    }

    /**
     * 获取模型配置
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }
}
