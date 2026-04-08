package io.leavesfly.tinyai.minimind.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.minimind.model.transformer.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.transformer.attention.MultiHeadAttention;
import io.leavesfly.tinyai.minimind.model.transformer.TransformerBlock;
import io.leavesfly.tinyai.minimind.training.lora.LoRAConfig;
import io.leavesfly.tinyai.minimind.training.lora.LoRALinear;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * MiniMind 语言模型（统一标准模式和 MoE 模式）
 * <p>
 * 轻量级 GPT 风格语言模型,支持:
 * - 自回归文本生成（KV-Cache 增量推理）
 * - 标准 FFN 和 MoE FFN 两种架构（由配置决定）
 * - 预训练、SFT、LoRA 微调
 * - 负载均衡损失（MoE 模式）
 * <p>
 * 模型规模:
 * - Small: 26M 参数
 * - Medium: 108M 参数
 * - MoE: 145M 参数（4专家,每次激活2个）
 *
 * @author leavesfly
 * @version 1.0
 */
public class MiniMindModel extends Model {

    /**
     * 模型配置
     */
    private final MiniMindConfig config;

    /**
     * 模型主体
     */
    private final MiniMindBlock miniMindBlock;

    /**
     * 构造 MiniMind 模型
     *
     * @param name   模型名称
     * @param config 模型配置
     */
    public MiniMindModel(String name, MiniMindConfig config) {
        super(name, new MiniMindBlock(config));
        this.config = config;
        this.miniMindBlock = (MiniMindBlock) getModule();

        setDescription("MiniMind Language Model - " + config.getModelSize() + 
                      " with " + config.estimateParameters() + " parameters");
    }

    /**
     * 使用预设配置创建模型
     *
     * @param name       模型名称
     * @param modelSize  模型规模 ("small", "medium", "moe")
     * @return MiniMind 模型实例
     */
    public static MiniMindModel create(String name, String modelSize) {
        MiniMindConfig config;
        switch (modelSize.toLowerCase()) {
            case "small":
                config = MiniMindConfig.createSmallConfig();
                break;
            case "medium":
                config = MiniMindConfig.createMediumConfig();
                break;
            case "moe":
                config = MiniMindConfig.createMoEConfig();
                break;
            default:
                throw new IllegalArgumentException("Unknown model size: " + modelSize + 
                    ". Available: small, medium, moe");
        }
        return new MiniMindModel(name, config);
    }

    /**
     * 预测（单次前向传播）
     *
     * @param tokenIds Token IDs,形状 [batch_size, seq_len]
     * @return Logits,形状 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(Variable tokenIds) {
        return miniMindBlock.forward(tokenIds);
    }

    /**
     * 预测（从 NdArray）
     *
     * @param tokenIds Token IDs NdArray,形状 [batch_size, seq_len]
     * @return Logits NdArray,形状 [batch_size, seq_len, vocab_size]
     */
    public NdArray predict(NdArray tokenIds) {
        Variable result = miniMindBlock.forward(new Variable(tokenIds));
        return result.getValue();
    }

    /**
     * 带负载均衡损失的预测（MoE 模式）
     *
     * @param tokenIds Token IDs,形状 [batch_size, seq_len]
     * @return MoE 输出结果（标准模式下 balanceLoss 为 0）
     */
    public MiniMindBlock.MoEOutput predictWithLoss(NdArray tokenIds) {
        Variable input = new Variable(tokenIds);
        return miniMindBlock.forwardWithMoEOutput(input, null, 0);
    }

    /**
     * 生成文本（自回归生成）
     *
     * @param promptTokenIds 提示词 token IDs
     * @param maxNewTokens   最大生成 token 数量
     * @param temperature    温度参数（0.0 = 贪婪,1.0 = 随机）
     * @param topK           Top-K 采样参数（0 表示不使用）
     * @param topP           Top-P 采样参数（0.0 表示不使用）
     * @return 生成的完整 token IDs
     */
    public int[] generate(int[] promptTokenIds, int maxNewTokens, 
                         float temperature, int topK, float topP) {
        return generate(promptTokenIds, maxNewTokens, temperature, topK, topP, 1.2f);
    }

    /**
     * 生成文本（贪婪采样，MoE 兼容接口）
     */
    public int[] generate(int[] promptTokenIds, int maxNewTokens) {
        return generate(promptTokenIds, maxNewTokens, 0.0f, 0, 0.0f, 1.0f);
    }
    
    /**
     * 生成文本（带重复惩罚）
     */
    public int[] generate(int[] promptTokenIds, int maxNewTokens, 
                         float temperature, int topK, float topP, float repetitionPenalty) {
        miniMindBlock.setTraining(false);

        List<KVCache> kvCaches = miniMindBlock.createKVCaches(1);

        int[] outputTokens = new int[promptTokenIds.length + maxNewTokens];
        System.arraycopy(promptTokenIds, 0, outputTokens, 0, promptTokenIds.length);

        int currentLen = promptTokenIds.length;
        
        Set<Integer> generatedTokens = new HashSet<>();
        for (int id : promptTokenIds) {
            generatedTokens.add(id);
        }

        // 首次前向传播（处理完整提示词）
        NdArray promptNdArray = createTokenIdsArray(promptTokenIds);
        Variable promptVar = new Variable(promptNdArray);
        miniMindBlock.forwardWithCache(promptVar, kvCaches, 0);

        // 自回归生成
        for (int i = 0; i < maxNewTokens; i++) {
            int position = currentLen;

            int lastToken = outputTokens[currentLen - 1];
            NdArray tokenNdArray = NdArray.of(new float[]{lastToken}, Shape.of(1, 1));
            Variable tokenVar = new Variable(tokenNdArray);

            Variable logits = miniMindBlock.forwardGeneration(tokenVar, kvCaches, position);

            NdArray lastLogits = extractLastLogits(logits.getValue());

            if (repetitionPenalty != 1.0f) {
                lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens, repetitionPenalty);
            }

            int nextToken = sampleToken(lastLogits, temperature, topK, topP);

            outputTokens[currentLen] = nextToken;
            currentLen++;
            generatedTokens.add(nextToken);

            if (nextToken == config.getEosTokenId()) {
                break;
            }
        }

        int[] result = new int[currentLen];
        System.arraycopy(outputTokens, 0, result, 0, currentLen);

        miniMindBlock.clearKVCaches(kvCaches);

        return result;
    }
    
    /**
     * 应用重复惩罚
     */
    private NdArray applyRepetitionPenalty(NdArray logits, Set<Integer> generatedTokens, float penalty) {
        float[] logitsArray = logits.getArray().clone();
        
        for (int tokenId : generatedTokens) {
            if (tokenId >= 0 && tokenId < logitsArray.length) {
                if (logitsArray[tokenId] > 0) {
                    logitsArray[tokenId] /= penalty;
                } else {
                    logitsArray[tokenId] *= penalty;
                }
            }
        }
        
        return NdArray.of(logitsArray, logits.getShape());
    }

    /**
     * 创建 token IDs 的 NdArray
     */
    private NdArray createTokenIdsArray(int[] tokenIds) {
        float[] data = new float[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            data[i] = tokenIds[i];
        }
        return NdArray.of(data, Shape.of(1, tokenIds.length));
    }

    /**
     * 提取最后一个位置的 logits
     */
    private NdArray extractLastLogits(NdArray logits) {
        int[] shape = logits.getShape().getShapeDims();
        int batchSize = shape[0];
        int seqLen = shape[1];
        int vocabSize = shape[2];

        float[] logitsData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) logits).buffer;
        float[] lastLogits = new float[vocabSize];

        int offset = (batchSize - 1) * seqLen * vocabSize + (seqLen - 1) * vocabSize;
        
        // 边界检查：确保不会越界访问
        if (offset + vocabSize > logitsData.length) {
            throw new IllegalArgumentException(String.format(
                "提取last logits时越界: offset=%d, vocabSize=%d, arrayLength=%d",
                offset, vocabSize, logitsData.length));
        }
        
        System.arraycopy(logitsData, offset, lastLogits, 0, vocabSize);

        return NdArray.of(lastLogits, Shape.of(vocabSize));
    }

    /**
     * 采样下一个 token
     */
    private int sampleToken(NdArray logits, float temperature, int topK, float topP) {
        float[] logitsArray = logits.getArray();
        return TextSampler.sample(logitsArray, temperature, topK, topP);
    }

    /**
     * 设置训练模式
     */
    public void setTraining(boolean training) {
        miniMindBlock.setTraining(training);
    }

    /**
     * 获取模型配置
     */
    public MiniMindConfig getConfig() {
        return config;
    }

    /**
     * 获取模型主体
     */
    public MiniMindBlock getMiniMindBlock() {
        return miniMindBlock;
    }

    /**
     * 获取专家使用统计（MoE 模式）
     */
    public String getExpertUsageStats() {
        return miniMindBlock.getExpertUsageStats();
    }

    /**
     * 重置统计信息（MoE 模式）
     */
    public void resetStats() {
        miniMindBlock.resetStats();
    }

    /**
     * 打印模型信息
     */
    @Override
    public void printModelInfo() {
        miniMindBlock.printModelInfo();
        super.printModelInfo();
    }
    
    /**
     * 应用 LoRA 层注入
     * <p>
     * 将目标模块的 Linear 层替换为 LoRALinear 层（仅标准模式支持）
     * 
     * @param loraConfig LoRA 配置
     * @return 注入的 LoRA 层数量
     */
    public int applyLoRA(LoRAConfig loraConfig) {
        if (config.isUseMoE()) {
            System.out.println("⚠️ MoE 模式暂不支持 LoRA 注入");
            return 0;
        }

        int injectedCount = 0;
        List<String> targetModules = Arrays.asList(loraConfig.getTargetModules());
        
        for (TransformerBlock layer : miniMindBlock.getLayers()) {
            MultiHeadAttention attention = layer.getAttention();
            int hiddenSize = attention.getHiddenSize();
            
            if (targetModules.contains("queryProj") || targetModules.contains("query_proj")) {
                if (attention.getQueryProj() instanceof Linear) {
                    Linear originalLinear = (Linear) attention.getQueryProj();
                    LoRALinear loraLinear = createLoRALinear(
                        "query_proj_lora", originalLinear, hiddenSize, hiddenSize, loraConfig);
                    attention.setQueryProj(loraLinear);
                    injectedCount++;
                }
            }
            
            if (targetModules.contains("valueProj") || targetModules.contains("value_proj")) {
                if (attention.getValueProj() instanceof Linear) {
                    Linear originalLinear = (Linear) attention.getValueProj();
                    LoRALinear loraLinear = createLoRALinear(
                        "value_proj_lora", originalLinear, hiddenSize, hiddenSize, loraConfig);
                    attention.setValueProj(loraLinear);
                    injectedCount++;
                }
            }
        }
        
        if (injectedCount > 0) {
            System.out.println("✅ LoRA 层注入完成: " + injectedCount + " 个层");
        } else {
            System.out.println("⚠️ 未注入任何 LoRA 层");
        }
        
        return injectedCount;
    }
    
    /**
     * 从现有 Linear 层创建 LoRALinear
     */
    private LoRALinear createLoRALinear(String name, Linear originalLinear, 
                                        int inFeatures, int outFeatures, 
                                        LoRAConfig loraConfig) {
        Parameter originalWeight = originalLinear.getWeight();
        Parameter originalBias = originalLinear.getBias();
        
        return LoRALinear.fromLinear(
            name, originalWeight, originalBias,
            loraConfig.getRank(), loraConfig.getAlpha(), loraConfig.getDropout());
    }
    
    /**
     * 合并 LoRA 权重到原始权重，并将 LoRALinear 替换回普通 Linear 层
     * <p>
     * 保存模型前调用此方法，确保加载后的模型不再依赖 LoRA 结构，
     * 推理结果与合并前完全一致。
     *
     * @return 合并的 LoRA 层数量
     */
    public int mergeLoRA() {
        if (config.isUseMoE()) {
            System.out.println("⚠️ MoE 模式暂不支持 LoRA 合并");
            return 0;
        }

        int mergedCount = 0;

        for (TransformerBlock layer : miniMindBlock.getLayers()) {
            MultiHeadAttention attention = layer.getAttention();

            if (attention.getQueryProj() instanceof LoRALinear) {
                LoRALinear loraLinear = (LoRALinear) attention.getQueryProj();
                Linear mergedLinear = mergeLoRAToLinear("query_proj", loraLinear);
                attention.setQueryProj(mergedLinear);
                mergedCount++;
            }

            if (attention.getValueProj() instanceof LoRALinear) {
                LoRALinear loraLinear = (LoRALinear) attention.getValueProj();
                Linear mergedLinear = mergeLoRAToLinear("value_proj", loraLinear);
                attention.setValueProj(mergedLinear);
                mergedCount++;
            }
        }

        if (mergedCount > 0) {
            System.out.println("✅ LoRA 权重合并完成: " + mergedCount + " 个层已合并回 Linear");
        } else {
            System.out.println("⚠️ 未检测到需要合并的 LoRA 层");
        }

        return mergedCount;
    }

    /**
     * 将单个 LoRALinear 合并为普通 Linear 层
     */
    private Linear mergeLoRAToLinear(String name, LoRALinear loraLinear) {
        NdArray mergedWeight = loraLinear.mergeWeights();
        int inFeatures = loraLinear.inFeatures;
        int outFeatures = loraLinear.outFeatures;
        boolean hasBias = loraLinear.getOriginalBias() != null;

        Linear linear = new Linear(name, inFeatures, outFeatures, hasBias);

        // 将合并后的权重复制到新 Linear 层
        float[] src = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) mergedWeight).buffer;
        float[] dst = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) linear.getWeight().data()).buffer;
        System.arraycopy(src, 0, dst, 0, src.length);

        // 复制偏置
        if (hasBias) {
            float[] biasSrc = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) loraLinear.getOriginalBias().data()).buffer;
            float[] biasDst = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) linear.getBias().data()).buffer;
            System.arraycopy(biasSrc, 0, biasDst, 0, biasSrc.length);
        }

        return linear;
    }

    /**
     * 获取 LoRA 参数统计
     */
    public void printLoRAStats() {
        if (config.isUseMoE()) {
            System.out.println("⚠️ MoE 模式不支持 LoRA");
            return;
        }

        int loraParams = 0;
        int totalParams = 0;
        int loraLayers = 0;
        
        for (TransformerBlock layer : miniMindBlock.getLayers()) {
            MultiHeadAttention attention = layer.getAttention();
            
            if (attention.getQueryProj() instanceof LoRALinear) {
                LoRALinear lora = (LoRALinear) attention.getQueryProj();
                loraParams += lora.getLoRAParams();
                totalParams += lora.getOriginalParams();
                loraLayers++;
            }
            
            if (attention.getValueProj() instanceof LoRALinear) {
                LoRALinear lora = (LoRALinear) attention.getValueProj();
                loraParams += lora.getLoRAParams();
                totalParams += lora.getOriginalParams();
                loraLayers++;
            }
        }
        
        if (loraLayers > 0) {
            float ratio = (float) loraParams / totalParams * 100;
            System.out.println("=".repeat(60));
            System.out.println("LoRA 参数统计:");
            System.out.println("  LoRA 层数量: " + loraLayers);
            System.out.println("  LoRA 参数量: " + loraParams);
            System.out.println("  原始参数量: " + totalParams);
            System.out.println("  参数压缩比: " + String.format("%.2f%%", ratio));
            System.out.println("=".repeat(60));
        } else {
            System.out.println("⚠️ 未检测到 LoRA 层");
        }
    }
}