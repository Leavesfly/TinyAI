package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-3主体块实现（完全基于V2 Module）
 * 
 * 组成部分：
 * 1. Token嵌入层（GPT3TokenEmbedding - 独立实现）
 * 2. N个GPT3TransformerBlock（并行架构）
 * 3. 最终LayerNorm层
 * 4. 输出投影层（词汇表映射）
 * 
 * 架构流程：
 * <pre>
 * TokenIDs -> TokenEmbedding -> [TransformerBlock x N] -> LayerNorm -> OutputProjection -> Logits
 * </pre>
 * 
 * @author leavesfly
 * @version 2.0 - 完全基于V2 API，完全独立实现
 */
public class GPT3MainBlock extends Module {
    
    private final GPT3Config config;
    
    // 核心组件
    private GPT3TokenEmbedding tokenEmbedding;                // Token嵌入（独立实现）
    private List<GPT3TransformerBlock> transformerBlocks;         // Transformer块列表
    private LayerNorm finalLayerNorm;                             // 最终归一化
    private Linear outputProjection;                              // 输出投影（V2 API）
    
    /**
     * 构造GPT-3主体块
     * 
     * @param name 块名称
     * @param config GPT-3配置
     */
    public GPT3MainBlock(String name, GPT3Config config) {
        super(name);
        this.config = config;
        
        // 初始化所有组件
        initializeComponents();
    }
    
    /**
     * 初始化所有组件（V2 Module方式）
     */
    private void initializeComponents() {
        // 1. 初始化Token嵌入层（GPT-3独立实现，基于V2 Module）
        tokenEmbedding = new GPT3TokenEmbedding(name + "_token_embedding", config);
        registerModule("token_embedding", tokenEmbedding);
        
        // 2. 初始化Transformer块列表
        transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            GPT3TransformerBlock block = new GPT3TransformerBlock(
                name + "_transformer_" + i, 
                config
            );
            transformerBlocks.add(block);
            // 注册为子模块
            registerModule("transformer_" + i, block);
        }
        
        // 3. 初始化最终LayerNorm
        finalLayerNorm = new LayerNorm(
            name + "_final_ln", 
            config.getNEmbd(),
            (float) config.getLayerNormEpsilon()
        );
        registerModule("final_ln", finalLayerNorm);
        
        // 4. 初始化输出投影层（将隐藏状态映射到词汇表维度）
        outputProjection = new Linear(
            name + "_output_proj",
            config.getNEmbd(),
            config.getVocabSize(),
            false  // 通常不使用bias
        );
        registerModule("output_proj", outputProjection);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }

        Variable tokenIds = inputs[0];  // (batch_size, seq_len)

        // 验证输入形状
        validateInput(tokenIds);

        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);  // (batch_size, seq_len, n_embd)

        // 2. 通过所有 Transformer 块（支持梯度检查点）
        for (int i = 0; i < transformerBlocks.size(); i++) {
            GPT3TransformerBlock block = transformerBlocks.get(i);

            if (config.isGradientCheckpointing() && isTraining()) {
                // 梯度检查点：截断计算图，反向传播时重新计算本层前向
                // 实现原理：
                //   1. 正常执行前向传播（获得正确输出值）
                //   2. 将输出包装为新 Variable，断开与上游计算图的连接
                //   3. 反向传播到此处时，需重新执行本层前向以恢复中间激活
                // 当前实现：截断计算图（节省显存），牺牲梯度精确流通的能力
                // 完整实现需框架支持"重计算"钩子（参考 PyTorch checkpoint()）
                Variable blockOutput = block.forward(x);
                Variable checkpointVar = new Variable(blockOutput.getValue());
                checkpointVar.setRequireGrad(x.isRequireGrad());
                x = checkpointVar;
            } else {
                x = block.forward(x);
            }
        }

        // 3. 最终 LayerNorm
        x = finalLayerNorm.forward(x);  // (batch_size, seq_len, n_embd)

        // 4. 输出投影到词汇表维度
        Variable logits = outputProjection.forward(x);  // (batch_size, seq_len, vocab_size)

        return logits;
    }

    /**
     * 带 KV Cache 的前向传播（用于增量推理加速）
     *
     * @param tokenIds  输入 Token ID，Shape: (batch_size, newSeqLen)
     * @param kvCaches  每层 Transformer 块对应的 KV Cache 列表（长度 = nLayer）
     * @param startPos  当前输入在完整序列中的起始位置
     * @return logits，Shape: (batch_size, newSeqLen, vocab_size)
     */
    public Variable forwardWithCache(Variable tokenIds, List<GPT3KVCache> kvCaches, int startPos) {
        validateInput(tokenIds);

        // 1. Token 嵌入（位置编码基于 startPos 偏移）
        Variable x = tokenEmbedding.forwardWithStartPos(tokenIds, startPos);

        // 2. 逐层通过 Transformer 块（每层使用各自的 KV Cache）
        for (int i = 0; i < transformerBlocks.size(); i++) {
            GPT3KVCache cache = (kvCaches != null && i < kvCaches.size()) ? kvCaches.get(i) : null;
            x = transformerBlocks.get(i).forwardWithCache(x, cache, startPos);
        }

        // 3. 最终 LayerNorm + 输出投影
        x = finalLayerNorm.forward(x);
        return outputProjection.forward(x);
    }
    
    /**
     * 验证输入的有效性
     * 
     * @param tokenIds 输入的token IDs
     */
    private void validateInput(Variable tokenIds) {
        NdArray data = tokenIds.getValue();
        
        if (data.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", 
                    data.getShape())
            );
        }
        
        int seqLen = data.getShape().getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", seqLen, config.getNPositions())
            );
        }
    }
    
    // ==================== 公共方法 ====================
    
    /**
     * 获取模型参数数量（估算）
     * 
     * @return 参数数量
     */
    public long getParameterCount() {
        return config.estimateParameterCount();
    }
    
    /**
     * 打印模型网络架构信息（树形层次结构）
     */
    public void printArchitecture() {
        String sep  = "=".repeat(70);
        String dash = "-".repeat(70);
        String attnMode = config.isParallelAttention() ? "Parallel" : "Sequential(GPT-2 style)";
        String attnFeatures = buildAttnFeatures();

        System.out.println(sep);
        System.out.println(" GPT-3 网络架构");
        System.out.println(sep);

        // ── 全局超参 ──────────────────────────────────────────────────────────
        System.out.println("[超参配置]");
        System.out.printf("  %-20s %d\n",  "词汇表大小 (vocab):",   config.getVocabSize());
        System.out.printf("  %-20s %d\n",  "最大序列长度 (ctx):",   config.getNPositions());
        System.out.printf("  %-20s %d\n",  "嵌入维度 (d_model):",   config.getNEmbd());
        System.out.printf("  %-20s %d\n",  "注意力头数 (n_head):",  config.getNHead());
        System.out.printf("  %-20s %d\n",  "每头维度 (d_head):",    config.getNEmbd() / config.getNHead());
        System.out.printf("  %-20s %d\n",  "FFN维度 (d_ff):",       config.getNInner());
        System.out.printf("  %-20s %d\n",  "Transformer层数:",      config.getNLayer());
        System.out.printf("  %-20s %s\n",  "注意力模式:",            attnMode);
        System.out.printf("  %-20s %s\n",  "注意力特性:",            attnFeatures);
        System.out.printf("  %-20s %.2f / %.2f / %.2f\n",
                "Dropout(embd/attn/resid):",
                config.getEmbdPdrop(), config.getAttnPdrop(), config.getResidPdrop());
        System.out.printf("  %-20s %.1e\n", "LayerNorm epsilon:",   config.getLayerNormEpsilon());
        System.out.println();

        // ── 网络层次 ──────────────────────────────────────────────────────────
        System.out.println("[网络架构]");
        System.out.printf("GPT3MainBlock ('%s')\n", name);

        // 1. Token Embedding
        System.out.println("├─ [1] TokenEmbedding");
        System.out.printf("│    ├─ token_embedding  : Embedding(%d, %d)\n",
                config.getVocabSize(), config.getNEmbd());
        System.out.printf("│    ├─ position_embedding: Embedding(%d, %d)  [可学习位置编码]\n",
                config.getNPositions(), config.getNEmbd());
        System.out.printf("│    └─ dropout           : Dropout(p=%.3f)\n",
                config.getEmbdPdrop());

        // 2. Transformer Blocks（只展开第 0 块作为代表）
        int nLayer = config.getNLayer();
        System.out.printf("├─ [2] TransformerBlocks  x%d\n", nLayer);
        printTransformerBlockDetail("│    ", 0, nLayer);
        if (nLayer > 1) {
            System.out.printf("│    ├─ transformer_1  ... transformer_%d  (结构同上)\n", nLayer - 1);
        }

        // 3. Final LayerNorm
        System.out.printf("├─ [3] final_ln          : LayerNorm(%d, eps=%.1e)\n",
                config.getNEmbd(), config.getLayerNormEpsilon());

        // 4. Output Projection
        System.out.printf("└─ [4] output_proj       : Linear(%d → %d, bias=false)\n",
                config.getNEmbd(), config.getVocabSize());

        // ── 数据流 ────────────────────────────────────────────────────────────
        System.out.println();
        System.out.println("[数据流]");
        System.out.println("  TokenIDs (batch, seq)");
        System.out.println("    │");
        System.out.printf("    ├─ TokenEmbedding  → (batch, seq, %d)\n", config.getNEmbd());
        System.out.printf("    ├─ TransformerBlock x%d\n", nLayer);
        if (config.isParallelAttention()) {
            System.out.println("    │    ├─ [parallel] LayerNorm → Attention(RoPE+Sparse)");
            System.out.println("    │    ├─ [parallel] LayerNorm → FFN(Linear→GELU→Linear)");
            System.out.println("    │    └─ x = x + attn_out + mlp_out");
        } else {
            System.out.println("    │    ├─ LayerNorm → Attention → x = x + attn_out");
            System.out.println("    │    └─ LayerNorm → FFN → x = x + mlp_out");
        }
        System.out.printf("    ├─ FinalLayerNorm  → (batch, seq, %d)\n", config.getNEmbd());
        System.out.printf("    └─ OutputProjection→ (batch, seq, %d) [logits]\n", config.getVocabSize());

        // ── 参数统计 ──────────────────────────────────────────────────────────
        System.out.println();
        System.out.println(dash);
        System.out.printf(" 估算总参数量: %s\n", formatParamCount(getParameterCount()));
        System.out.println(sep);
    }

    /**
     * 展开一个 TransformerBlock 的内部结构（用于 printArchitecture 树形输出）
     *
     * @param prefix 缩进前缀
     * @param idx    展开块的索引
     * @param total  总块数
     */
    private void printTransformerBlockDetail(String prefix, int idx, int total) {
        String blockTag = total > 1
                ? String.format("transformer_%d  (代表，共 %d 块)", idx, total)
                : String.format("transformer_%d", idx);
        System.out.printf("%s├─ %s\n", prefix, blockTag);

        String inner = prefix + "│    ";
        if (config.isParallelAttention()) {
            // 并行注意力分支
            System.out.printf("%s├─ [attn branch]\n", inner);
            System.out.printf("%s│    ├─ ln1        : LayerNorm(%d)\n", inner, config.getNEmbd());
            System.out.printf("%s│    ├─ attention  : GPT3Attention(%s)\n", inner, buildAttnFeatures());
            System.out.printf("%s│    └─ dropout    : Dropout(p=%.3f)\n", inner, config.getResidPdrop());
            // 并行 MLP 分支
            System.out.printf("%s├─ [mlp branch]\n", inner);
            System.out.printf("%s│    ├─ ln2        : LayerNorm(%d)\n", inner, config.getNEmbd());
            System.out.printf("%s│    ├─ ffn_fc1    : Linear(%d → %d, bias=true)\n",
                    inner, config.getNEmbd(), config.getNInner());
            System.out.printf("%s│    ├─ gelu       : GELU\n", inner);
            System.out.printf("%s│    ├─ ffn_fc2    : Linear(%d → %d, bias=true)\n",
                    inner, config.getNInner(), config.getNEmbd());
            System.out.printf("%s│    └─ dropout    : Dropout(p=%.3f)\n", inner, config.getResidPdrop());
            System.out.printf("%s└─ residual merge : x = x + attn_out + mlp_out\n", inner);
        } else {
            // 串行：先注意力再 MLP
            System.out.printf("%s├─ [sub-layer 1: attention]\n", inner);
            System.out.printf("%s│    ├─ ln1        : LayerNorm(%d)\n", inner, config.getNEmbd());
            System.out.printf("%s│    ├─ attention  : GPT3Attention(%s)\n", inner, buildAttnFeatures());
            System.out.printf("%s│    ├─ dropout    : Dropout(p=%.3f)\n", inner, config.getResidPdrop());
            System.out.printf("%s│    └─ residual   : x = x + attn_out\n", inner);
            System.out.printf("%s└─ [sub-layer 2: ffn]\n", inner);
            System.out.printf("%s     ├─ ln2        : LayerNorm(%d)\n", inner, config.getNEmbd());
            System.out.printf("%s     ├─ ffn_fc1    : Linear(%d → %d, bias=true)\n",
                    inner, config.getNEmbd(), config.getNInner());
            System.out.printf("%s     ├─ gelu       : GELU\n", inner);
            System.out.printf("%s     ├─ ffn_fc2    : Linear(%d → %d, bias=true)\n",
                    inner, config.getNInner(), config.getNEmbd());
            System.out.printf("%s     ├─ dropout    : Dropout(p=%.3f)\n", inner, config.getResidPdrop());
            System.out.printf("%s     └─ residual   : x = x + mlp_out\n", inner);
        }
    }

    /**
     * 构建注意力特性描述字符串
     *
     * @return 注意力特性字符串，如 "RoPE+SparseAttn" 或 "Standard"
     */
    private String buildAttnFeatures() {
        StringBuilder sb = new StringBuilder();
        if (config.isUseRotaryEmbedding()) {
            sb.append("RoPE");
        }
        if (config.isSparseAttention()) {
            if (sb.length() > 0) sb.append("+");
            sb.append("SparseAttn(window=").append(config.getSparseLocalWindow()).append(")");
        }
        if (sb.length() == 0) {
            sb.append("Standard");
        }
        return sb.toString();
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2f B", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2f M", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    // ==================== Getter方法 ====================
    
    public GPT3TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    public List<GPT3TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
    
    public GPT3TransformerBlock getTransformerBlock(int index) {
        if (index < 0 || index >= transformerBlocks.size()) {
            throw new IndexOutOfBoundsException(
                String.format("Transformer块索引越界: %d (总共%d个块)", index, transformerBlocks.size())
            );
        }
        return transformerBlocks.get(index);
    }
    
    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }
    
    public Linear getOutputProjection() {
        return outputProjection;
    }
    
    public GPT3Config getConfig() {
        return config;
    }
    
    @Override
    public String toString() {
        return String.format("GPT3MainBlock{name='%s', nLayer=%d, nEmbd=%d, nHead=%d, params=%s}",
            name, config.getNLayer(), config.getNEmbd(), config.getNHead(), 
            formatParamCount(getParameterCount()));
    }
}
