package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.activation.GELU;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;

/**
 * GPT-3 Transformer块实现（V2 API）
 *
 * GPT-3的核心创新：并行注意力和前馈网络计算
 * 与GPT-2的串行架构不同，GPT-3同时计算注意力和MLP，然后合并结果
 *
 * 并行架构流程：
 * <pre>
 * input -> split
 *           |-> LayerNorm1 -> GPT3Attention -> attn_output  (RoPE + 稀疏注意力)
 *           |-> LayerNorm2 -> FeedForward  -> mlp_output
 *          merge -> input + attn_output + mlp_output -> output
 * </pre>
 *
 * 串行兼容模式（parallelAttention=false）：
 * <pre>
 * input -> LayerNorm1 -> Attention -> Add(input)
 *       -> LayerNorm2 -> FeedForward -> Add -> output
 * </pre>
 *
 * 注意力特性（由 GPT3Config 控制）：
 * - useRotaryEmbedding=true  : 启用 RoPE 旋转位置编码
 * - sparseAttention=true     : 启用局部窗口 + 步长全局稀疏注意力
 * - 两者可独立开启或组合使用
 *
 * @author leavesfly
 * @version 2.0 - 接入 GPT3Attention（RoPE + 稀疏注意力 + KV Cache）
 */
public class GPT3TransformerBlock extends Module {

    private final GPT3Config config;

    // 注意力分支
    private final LayerNorm layerNorm1;       // 注意力分支的 Pre-LayerNorm
    private final GPT3Attention attention;    // 增强注意力（支持 RoPE / 稀疏注意力 / KV Cache）
    private final Dropout attnDropout;        // 注意力输出残差 Dropout

    // 前馈分支
    private final LayerNorm layerNorm2;       // MLP 分支的 Pre-LayerNorm
    private final Linear ffnLinear1;          // FFN 第一层：dModel -> dFF
    private final GELU activation;            // GELU 激活函数
    private final Linear ffnLinear2;          // FFN 第二层：dFF -> dModel
    private final Dropout mlpDropout;         // MLP 输出残差 Dropout

    /**
     * 构造 GPT-3 Transformer 块
     *
     * @param name   块名称
     * @param config GPT-3 配置（包含注意力模式控制开关）
     */
    public GPT3TransformerBlock(String name, GPT3Config config) {
        super(name);
        this.config = config;

        int dModel = config.getNEmbd();
        int dFF = config.getNInner();
        float residDropout = (float) config.getResidPdrop();

        // 初始化注意力分支
        this.layerNorm1 = new LayerNorm("ln1", dModel, (float) config.getLayerNormEpsilon());
        // GPT3Attention 根据 config 中的开关自动启用 RoPE / 稀疏注意力
        this.attention = new GPT3Attention("attn", config);
        this.attnDropout = new Dropout("attn_dropout", residDropout);

        // 初始化前馈分支
        this.layerNorm2 = new LayerNorm("ln2", dModel, (float) config.getLayerNormEpsilon());
        this.ffnLinear1 = new Linear("ffn_fc1", dModel, dFF, true);
        this.activation = new GELU("gelu");
        this.ffnLinear2 = new Linear("ffn_fc2", dFF, dModel, true);
        this.mlpDropout = new Dropout("mlp_dropout", residDropout);

        // 注册所有子模块
        registerModule("ln1", layerNorm1);
        registerModule("attn", attention);
        registerModule("attn_dropout", attnDropout);
        registerModule("ln2", layerNorm2);
        registerModule("ffn_fc1", ffnLinear1);
        registerModule("gelu", activation);
        registerModule("ffn_fc2", ffnLinear2);
        registerModule("mlp_dropout", mlpDropout);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];  // (batch_size, seq_len, n_embd)

        if (config.isParallelAttention()) {
            return forwardParallel(x);
        } else {
            return forwardSequential(x);
        }
    }

    /**
     * 带 KV Cache 的前向传播（用于增量推理加速）
     *
     * @param x        输入张量 (batch, newSeqLen, dModel)
     * @param cache    KV缓存（null 表示不使用）
     * @param startPos 当前 Token 在完整序列中的起始位置
     * @return 输出张量 (batch, newSeqLen, dModel)
     */
    public Variable forwardWithCache(Variable x, GPT3KVCache cache, int startPos) {
        if (config.isParallelAttention()) {
            return forwardParallelWithCache(x, cache, startPos);
        } else {
            return forwardSequentialWithCache(x, cache, startPos);
        }
    }

    /**
     * GPT-3 并行前向传播
     * 注意力分支和 MLP 分支同时对同一输入计算，最后将结果合并到残差中
     */
    private Variable forwardParallel(Variable x) {
        // 注意力分支：Pre-LayerNorm -> GPT3Attention（含 RoPE/稀疏掩码）
        Variable attnInput = layerNorm1.forward(x);
        Variable attnOutput = attention.forward(attnInput);
        attnOutput = attnDropout.forward(attnOutput);

        // MLP 分支：Pre-LayerNorm -> Linear -> GELU -> Linear
        Variable mlpInput = layerNorm2.forward(x);
        Variable mlpOutput = ffnLinear1.forward(mlpInput);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);

        // 并行残差合并：x + attn + mlp
        return x.add(attnOutput).add(mlpOutput);
    }

    /**
     * 带 KV Cache 的并行前向传播
     */
    private Variable forwardParallelWithCache(Variable x, GPT3KVCache cache, int startPos) {
        Variable attnInput = layerNorm1.forward(x);
        Variable attnOutput = attention.forwardWithCache(attnInput, cache, startPos);
        attnOutput = attnDropout.forward(attnOutput);

        Variable mlpInput = layerNorm2.forward(x);
        Variable mlpOutput = ffnLinear1.forward(mlpInput);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);

        return x.add(attnOutput).add(mlpOutput);
    }

    /**
     * GPT-2 风格的串行前向传播（兼容模式）
     * 先计算注意力，再计算 MLP，结果顺序叠加到残差
     */
    private Variable forwardSequential(Variable x) {
        // 第一子层：Pre-LayerNorm -> Attention -> Residual
        Variable normalized1 = layerNorm1.forward(x);
        Variable attnOutput = attention.forward(normalized1);
        attnOutput = attnDropout.forward(attnOutput);
        Variable residual1 = x.add(attnOutput);

        // 第二子层：Pre-LayerNorm -> MLP -> Residual
        Variable normalized2 = layerNorm2.forward(residual1);
        Variable mlpOutput = ffnLinear1.forward(normalized2);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);

        return residual1.add(mlpOutput);
    }

    /**
     * 带 KV Cache 的串行前向传播
     */
    private Variable forwardSequentialWithCache(Variable x, GPT3KVCache cache, int startPos) {
        Variable normalized1 = layerNorm1.forward(x);
        Variable attnOutput = attention.forwardWithCache(normalized1, cache, startPos);
        attnOutput = attnDropout.forward(attnOutput);
        Variable residual1 = x.add(attnOutput);

        Variable normalized2 = layerNorm2.forward(residual1);
        Variable mlpOutput = ffnLinear1.forward(normalized2);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);

        return residual1.add(mlpOutput);
    }

    // ========================== Getter ==========================

    public LayerNorm getLayerNorm1() { return layerNorm1; }

    public GPT3Attention getAttention() { return attention; }

    public LayerNorm getLayerNorm2() { return layerNorm2; }

    public Linear getFfnLinear1() { return ffnLinear1; }

    public GELU getActivation() { return activation; }

    public Linear getFfnLinear2() { return ffnLinear2; }

    public GPT3Config getConfig() { return config; }

    @Override
    public String toString() {
        String mode = config.isParallelAttention() ? "Parallel" : "Sequential";
        String attnFeatures = attention.toString();
        return String.format(
                "GPT3TransformerBlock{name='%s', mode=%s, dModel=%d, numHeads=%d, dFF=%d, attn=%s}",
                name, mode, config.getNEmbd(), config.getNHead(), config.getNInner(), attnFeatures);
    }
}
