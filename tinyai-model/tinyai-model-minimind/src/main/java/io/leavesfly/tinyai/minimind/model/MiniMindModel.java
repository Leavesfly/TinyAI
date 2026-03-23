package io.leavesfly.tinyai.minimind.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.minimind.model.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.attention.MultiHeadAttention;
import io.leavesfly.tinyai.minimind.model.transformer.MiniMindTransformerLayer;
import io.leavesfly.tinyai.minimind.training.lora.LoRAConfig;
import io.leavesfly.tinyai.minimind.training.lora.LoRALinear;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * MiniMind 语言模型
 * <p>
 * 这是一个轻量级的 GPT 风格语言模型,支持:
 * - 自回归文本生成
 * - 增量推理（KV-Cache）
 * - 预训练和微调
 * <p>
 * 模型规模:
 * - Small: 26M 参数
 * - Medium: 108M 参数
 * - MoE: 145M 参数（含 MoE 机制）
 * <p>
 * 继承自 TinyAI Model 类,提供统一的模型接口
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

        // 设置模型描述
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
     * <p>
     * 输入 token IDs,输出词汇表上的概率分布
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
     * 生成文本（自回归生成）
     * <p>
     * 给定提示词 token IDs,生成指定长度的文本
     *
     * @param promptTokenIds 提示词 token IDs,形状 [1, prompt_len]
     * @param maxNewTokens   最大生成 token 数量
     * @param temperature    温度参数（控制随机性,0.0 = 贪婪,1.0 = 随机）
     * @param topK           Top-K 采样参数（0 表示不使用）
     * @param topP           Top-P 采样参数（0.0 表示不使用）
     * @return 生成的完整 token IDs,形状 [1, prompt_len + generated_len]
     */
    public int[] generate(int[] promptTokenIds, int maxNewTokens, 
                         float temperature, int topK, float topP) {
        return generate(promptTokenIds, maxNewTokens, temperature, topK, topP, 1.2f);
    }
    
    /**
     * 生成文本（带重复惩罚）
     */
    public int[] generate(int[] promptTokenIds, int maxNewTokens, 
                         float temperature, int topK, float topP, float repetitionPenalty) {
        // 设置为推理模式
        miniMindBlock.setTraining(false);

        // 创建 KV-Cache
        List<KVCache> kvCaches = miniMindBlock.createKVCaches(1);

        // 初始化输出序列
        int[] outputTokens = new int[promptTokenIds.length + maxNewTokens];
        System.arraycopy(promptTokenIds, 0, outputTokens, 0, promptTokenIds.length);

        int currentLen = promptTokenIds.length;
        
        // 记录已生成的 token 用于重复惩罚
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

            // 获取当前最后一个 token
            int lastToken = outputTokens[currentLen - 1];
            NdArray tokenNdArray = NdArray.of(new float[]{lastToken}, Shape.of(1, 1));
            Variable tokenVar = new Variable(tokenNdArray);

            // 前向传播（仅处理新 token）
            Variable logits = miniMindBlock.forwardGeneration(tokenVar, kvCaches, position);

            // 获取最后一个位置的 logits: [1, vocab_size]
            NdArray lastLogits = extractLastLogits(logits.getValue());

            // 应用重复惩罚
            if (repetitionPenalty != 1.0f) {
                lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens, repetitionPenalty);
            }

            // 采样下一个 token
            int nextToken = sampleToken(lastLogits, temperature, topK, topP);

            // 添加到输出序列
            outputTokens[currentLen] = nextToken;
            currentLen++;
            generatedTokens.add(nextToken);

            // 检查是否遇到结束符
            if (nextToken == config.getEosTokenId()) {
                break;
            }
        }

        // 截取有效部分
        int[] result = new int[currentLen];
        System.arraycopy(outputTokens, 0, result, 0, currentLen);

        // 清空缓存
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
     *
     * @param tokenIds Token IDs 数组
     * @return NdArray,形状 [1, seq_len]
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
     *
     * @param logits 完整 logits,形状 [batch, seq_len, vocab_size]
     * @return 最后位置的 logits,形状 [vocab_size]
     */
    private NdArray extractLastLogits(NdArray logits) {
        int[] shape = logits.getShape().getShapeDims();
        int batchSize = shape[0];
        int seqLen = shape[1];
        int vocabSize = shape[2];

        float[] logitsData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) logits).buffer;
        float[] lastLogits = new float[vocabSize];

        // 提取最后一个位置的 logits
        int offset = (batchSize - 1) * seqLen * vocabSize + (seqLen - 1) * vocabSize;
        System.arraycopy(logitsData, offset, lastLogits, 0, vocabSize);

        return NdArray.of(lastLogits, Shape.of(vocabSize));
    }

    /**
     * 采样下一个 token
     *
     * @param logits      Logits,形状 [vocab_size]
     * @param temperature 温度参数
     * @param topK        Top-K 参数
     * @param topP        Top-P 参数
     * @return 采样的 token ID
     */
    private int sampleToken(NdArray logits, float temperature, int topK, float topP) {
        float[] logitsArray = logits.getArray();

        // 应用温度
        if (temperature > 0 && temperature != 1.0f) {
            for (int i = 0; i < logitsArray.length; i++) {
                logitsArray[i] /= temperature;
            }
        }

        // Softmax 转换为概率
        float[] probs = softmax(logitsArray);

        // 贪婪采样（temperature = 0）
        if (temperature == 0.0f) {
            return argmax(probs);
        }

        // Top-K 采样
        if (topK > 0) {
            probs = applyTopK(probs, topK);
        }

        // Top-P 采样
        if (topP > 0 && topP < 1.0f) {
            probs = applyTopP(probs, topP);
        }

        // 多项式采样
        return multinomialSample(probs);
    }

    /**
     * Softmax 函数
     */
    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }

        float sum = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }

        // 添加 sum == 0 的保护，避免 NaN
        if (sum == 0.0f) {
            // 如果所有概率都为 0，则均匀分布
            for (int i = 0; i < probs.length; i++) {
                probs[i] = 1.0f / probs.length;
            }
        } else {
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
        }

        return probs;
    }

    /**
     * Argmax 函数
     */
    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Top-K 采样过滤
     */
    private float[] applyTopK(float[] probs, int k) {
        // 简化实现：保留前 K 个最大概率，其余置零
        float[] result = new float[probs.length];
        int[] indices = argsort(probs);

        for (int i = 0; i < Math.min(k, indices.length); i++) {
            result[indices[i]] = probs[indices[i]];
        }

        // 重新归一化
        float sum = 0.0f;
        for (float v : result) {
            sum += v;
        }
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    /**
     * Top-P 采样过滤
     */
    private float[] applyTopP(float[] probs, float p) {
        int[] indices = argsort(probs);
        float cumSum = 0.0f;
        float[] result = new float[probs.length];

        for (int idx : indices) {
            if (cumSum >= p) {
                break;
            }
            result[idx] = probs[idx];
            cumSum += probs[idx];
        }

        // 重新归一化
        float sum = 0.0f;
        for (float v : result) {
            sum += v;
        }
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    /**
     * 多项式采样
     */
    private int multinomialSample(float[] probs) {
        float rand = (float) Math.random();
        float cumSum = 0.0f;

        for (int i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) {
                return i;
            }
        }

        return probs.length - 1;
    }

    /**
     * 降序排序索引
     */
    private int[] argsort(float[] array) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }

        java.util.Arrays.sort(indices, (a, b) -> Float.compare(array[b], array[a]));

        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = indices[i];
        }

        return result;
    }

    /**
     * 设置训练模式
     *
     * @param training 是否为训练模式
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
     * 将目标模块的 Linear 层替换为 LoRALinear 层
     * 
     * @param loraConfig LoRA 配置
     * @return 注入的 LoRA 层数量
     */
    public int applyLoRA(LoRAConfig loraConfig) {
        int injectedCount = 0;
        List<String> targetModules = Arrays.asList(loraConfig.getTargetModules());
        
        // 遍历所有 Transformer 层
        for (MiniMindTransformerLayer layer : miniMindBlock.getLayers()) {
            MultiHeadAttention attention = layer.getAttention();
            int hiddenSize = attention.getHiddenSize();
            
            // 检查是否需要注入 queryProj
            if (targetModules.contains("queryProj") || targetModules.contains("query_proj")) {
                if (attention.getQueryProj() instanceof Linear) {
                    Linear originalLinear = (Linear) attention.getQueryProj();
                    LoRALinear loraLinear = createLoRALinear(
                        "query_proj_lora", 
                        originalLinear, 
                        hiddenSize, 
                        hiddenSize, 
                        loraConfig
                    );
                    attention.setQueryProj(loraLinear);
                    injectedCount++;
                }
            }
            
            // 检查是否需要注入 valueProj
            if (targetModules.contains("valueProj") || targetModules.contains("value_proj")) {
                if (attention.getValueProj() instanceof Linear) {
                    Linear originalLinear = (Linear) attention.getValueProj();
                    LoRALinear loraLinear = createLoRALinear(
                        "value_proj_lora", 
                        originalLinear, 
                        hiddenSize, 
                        hiddenSize, 
                        loraConfig
                    );
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
        // 获取原始权重
        Parameter originalWeight = originalLinear.getWeight();
        Parameter originalBias = originalLinear.getBias();
        
        // 创建 LoRALinear
        LoRALinear loraLinear = LoRALinear.fromLinear(
            name,
            originalWeight,
            originalBias,
            loraConfig.getRank(),
            loraConfig.getAlpha(),
            loraConfig.getDropout()
        );
        
        return loraLinear;
    }
    
    /**
     * 获取 LoRA 参数统计
     */
    public void printLoRAStats() {
        int loraParams = 0;
        int totalParams = 0;
        int loraLayers = 0;
        
        for (MiniMindTransformerLayer layer : miniMindBlock.getLayers()) {
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
