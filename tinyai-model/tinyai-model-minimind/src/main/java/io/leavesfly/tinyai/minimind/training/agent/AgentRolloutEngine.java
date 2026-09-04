package io.leavesfly.tinyai.minimind.training.agent;

import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;

import java.util.*;

/**
 * Agent 多轮 Rollout 引擎
 * <p>
 * 对标 Python minimind3 train_agent.py rollout_single (L178-L268) +
 * rollout_batch (L270-L312)。
 * <p>
 * Rollout 流程：
 * 1. 将用户消息编码为 prompt
 * 2. 模型生成响应
 * 3. 解析 &lt;tool_call&gt; 标签
 * 4. 执行工具获取结果
 * 5. 将工具结果注入上下文，继续下一轮
 * 6. 重复直到无工具调用或达到最大轮数
 *
 * @author TinyAI Team
 * @since 2025
 */
public class AgentRolloutEngine {

    /**
     * 单次 Rollout 的结果
     */
    public static class RolloutResult {
        /** 最终生成的文本 */
        private final String finalOutput;
        /** 所有轮次的输出列表 */
        private final List<String> turnOutputs;
        /** 是否因达到最大轮数而未完成 */
        private final boolean unfinished;
        /** 完整上下文（prompt + 所有轮次输出） */
        private final String fullContext;
        /**
         * 最后一轮<b>实际喂给模型</b>的 prompt token（已按模型窗口左截断）
         * <p>
         * 训练侧应当直接用这些 token，而不是对 {@link #fullContext} 重新编码：
         * 后者会受 tokenizer 自身 maxSeqLen 截断、BOS/EOS 拼接、BPE 跨边界合并
         * 三重影响，得到的序列与模型真正看到的并不一致。
         */
        private final List<Integer> finalPromptTokenIds;
        /** 最后一轮生成的 token（不包含 prompt），生成失败时为 null */
        private final List<Integer> finalCompletionTokenIds;

        public RolloutResult(String finalOutput, List<String> turnOutputs,
                             boolean unfinished, String fullContext) {
            this(finalOutput, turnOutputs, unfinished, fullContext, null, null);
        }

        public RolloutResult(String finalOutput, List<String> turnOutputs,
                             boolean unfinished, String fullContext,
                             List<Integer> finalPromptTokenIds,
                             List<Integer> finalCompletionTokenIds) {
            this.finalOutput = finalOutput;
            this.turnOutputs = turnOutputs;
            this.unfinished = unfinished;
            this.fullContext = fullContext;
            this.finalPromptTokenIds = finalPromptTokenIds;
            this.finalCompletionTokenIds = finalCompletionTokenIds;
        }

        public String getFinalOutput() { return finalOutput; }
        public List<String> getTurnOutputs() { return turnOutputs; }
        public boolean isUnfinished() { return unfinished; }
        public String getFullContext() { return fullContext; }

        /**
         * 最后一轮实际喂给模型的 prompt token；若本次 rollout 没有成功生成，返回 null
         */
        public List<Integer> getFinalPromptTokenIds() { return finalPromptTokenIds; }

        /**
         * 最后一轮生成的 token；若本次 rollout 没有成功生成，返回 null
         */
        public List<Integer> getFinalCompletionTokenIds() { return finalCompletionTokenIds; }

        /** prompt 与 completion 的 token 是否都可用（训练侧可直接消费） */
        public boolean hasTokenTrace() {
            return finalPromptTokenIds != null
                    && finalCompletionTokenIds != null
                    && !finalCompletionTokenIds.isEmpty();
        }
    }

    private final MiniMindModel model;
    private final MiniMindTokenizer tokenizer;
    /** 生成失败只打印一次，避免每个候选都刷屏 */
    private boolean generateFailureReported;

    public AgentRolloutEngine(MiniMindModel model, MiniMindTokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }

    /**
     * 单样本多轮工具调用 Rollout
     * <p>
     * 对标 Python rollout_single (L178-L268)
     *
     * @param messages   对话消息列表（会被修改，添加 assistant 和 tool 消息）
     * @param tools      可用工具名列表
     * @param maxTurns   最大工具交互轮数
     * @param maxGenLen  每轮最大生成 token 数
     * @param temperature 采样温度
     * @return Rollout 结果
     */
    public RolloutResult rolloutSingle(
            List<Map<String, String>> messages,
            List<String> tools,
            int maxTurns,
            int maxGenLen,
            float temperature) {

        List<String> allOutputs = new ArrayList<>();
        boolean unfinished = false;
        StringBuilder fullContext = new StringBuilder();
        Generation lastGeneration = null;

        for (int turn = 0; turn < maxTurns; turn++) {
            // 构建当前上下文（简化版 apply_chat_template）
            String context = buildChatContext(messages, tools);
            fullContext.setLength(0);
            fullContext.append(context);

            // 模型生成
            Generation generation = generateResponse(context, maxGenLen, temperature);
            lastGeneration = generation;
            String generated = generation.text;
            allOutputs.add(generated);
            fullContext.append(generated);

            // 解析工具调用
            List<Map<String, Object>> toolCalls = AgentTool.parseToolCalls(generated);

            if (toolCalls.isEmpty()) {
                // 没有工具调用，结束 rollout
                break;
            }

            // 判断是否是最后一轮
            unfinished = (turn == maxTurns - 1);

            // 添加 assistant 消息
            Map<String, String> assistantMsg = new HashMap<>();
            assistantMsg.put("role", "assistant");
            assistantMsg.put("content", generated);
            messages.add(assistantMsg);

            // 执行工具并添加结果消息
            for (Map<String, Object> call : toolCalls) {
                String name = (String) call.getOrDefault("name", "");
                @SuppressWarnings("unchecked")
                Map<String, Object> args = (Map<String, Object>) call.getOrDefault("arguments", Collections.emptyMap());

                String result;
                if (AgentTool.validateArgs(name, args)) {
                    result = AgentTool.executeTool(name, args);
                } else {
                    result = "{\"error\": \"invalid arguments\"}";
                }

                // 限制结果长度，防止过长
                if (result != null && result.length() > 2048) {
                    result = result.substring(0, 2048);
                }

                Map<String, String> toolMsg = new HashMap<>();
                toolMsg.put("role", "tool");
                toolMsg.put("content", result != null ? result : "{\"error\": \"tool not found\"}");
                messages.add(toolMsg);
            }
        }

        String finalOutput = allOutputs.isEmpty() ? "" : allOutputs.get(allOutputs.size() - 1);
        return new RolloutResult(finalOutput, allOutputs, unfinished, fullContext.toString(),
                lastGeneration == null ? null : lastGeneration.promptTokenIds,
                lastGeneration == null ? null : lastGeneration.completionTokenIds);
    }

    /**
     * 批量多轮工具调用 Rollout
     * <p>
     * 对标 Python rollout_batch (L270-L312)
     *
     * @param messagesBatch 批量对话消息
     * @param toolsBatch    批量工具列表
     * @param numGen        每个样本生成的候选数量
     * @param maxTurns      最大轮数
     * @param maxGenLen     每轮最大生成长度
     * @param temperature   采样温度
     * @return 所有 Rollout 结果（flattenSize = batchSize * numGen）
     */
    public List<RolloutResult> rolloutBatch(
            List<List<Map<String, String>>> messagesBatch,
            List<List<String>> toolsBatch,
            int numGen,
            int maxTurns,
            int maxGenLen,
            float temperature) {

        List<RolloutResult> results = new ArrayList<>();

        for (int i = 0; i < messagesBatch.size(); i++) {
            List<Map<String, String>> messages = messagesBatch.get(i);
            List<String> tools = toolsBatch.get(i);

            for (int g = 0; g < numGen; g++) {
                // 深拷贝 messages，每个候选独立修改
                List<Map<String, String>> msgsCopy = new ArrayList<>();
                for (Map<String, String> msg : messages) {
                    msgsCopy.add(new HashMap<>(msg));
                }

                RolloutResult result = rolloutSingle(
                        msgsCopy, tools, maxTurns, maxGenLen, temperature);
                results.add(result);
            }
        }

        return results;
    }

    // ========== 内部方法 ==========

    /**
     * 构建聊天上下文（简化版 apply_chat_template）
     * <p>
     * 将 messages 列表拼接为模型输入文本：
     * [System] tools 描述
     * [User] prompt
     * [Assistant] response (如果有)
     * [Tool] result (如果有)
     */
    private String buildChatContext(List<Map<String, String>> messages, List<String> tools) {
        StringBuilder sb = new StringBuilder();

        // 系统消息：可用工具描述
        if (tools != null && !tools.isEmpty()) {
            sb.append("You are a helpful assistant with tool calling capabilities. ");
            sb.append("Available tools: ").append(String.join(", ", tools)).append("\n");
        }

        // 拼接对话消息
        for (Map<String, String> msg : messages) {
            String role = msg.get("role");
            String content = msg.get("content");
            sb.append("[").append(role).append("] ").append(content).append("\n");
        }

        // 添加生成提示
        sb.append("[assistant] ");

        return sb.toString();
    }

    /**
     * 单次生成的结果：文本 + 实际使用的 token
     * <p>
     * 同时带回 token 是为了让训练侧能直接复用“模型真正看到的序列”，
     * 避开 文本 → token 的二次编码带来的截断/特殊 token 不一致。
     */
    private static final class Generation {
        final String text;
        /** 实际喂给模型的 prompt token（可能已左截断）；失败时为 null */
        final List<Integer> promptTokenIds;
        /** 生成的 token；失败或为空时为 null */
        final List<Integer> completionTokenIds;

        Generation(String text, List<Integer> promptTokenIds, List<Integer> completionTokenIds) {
            this.text = text;
            this.promptTokenIds = promptTokenIds;
            this.completionTokenIds = completionTokenIds;
        }

        static Generation empty() {
            return new Generation("", null, null);
        }
    }

    /**
     * 使用模型生成响应
     * <p>
     * 简化实现：encode → generate → decode
     * <p>
     * 上下文长度按模型的 {@code maxSeqLen} 做<b>左截断</b>，并预留 {@code maxGenLen} 个位置：
     * tokenizer 自己的 maxSeqLen 通常远大于模型的（尤其是字符级分词），直接把超长 prompt
     * 交给 {@code generate} 会让位置编码越界抛异常。
     * <p>
     * 编码时显式传 {@code addBos=false, addEos=false}：上下文已经以 {@code "[assistant] "}
     * 结尾，单参 {@code encode(text)} 会在其后补一个 EOS，等于告诉模型"本轮已结束"，
     * 生成质量会被直接带坏；同时也与 {@code AgentTrainer.encodeRollouts} 的编码口径保持一致。
     * <p>
     * 本方法<b>不负责</b>切换模型的训练/评估模式：{@code MiniMindModel.generate} 内部已经
     * 把 block 置为 eval，而调用方（如 {@code AgentTrainer}）会在 rollout 前后显式设置模式。
     * 早期实现在这里无条件 {@code setTraining(true)}，会把同一批次的第 2..N 个候选
     * 变成"带 dropout 的训练态生成"，破坏 rollout 分布的一致性。
     */
    private Generation generateResponse(String context, int maxGenLen, float temperature) {
        try {
            // 编码上下文（不补 BOS/EOS）
            List<Integer> contextTokens = tokenizer.encode(context, false, false);
            int[] promptIds = contextTokens.stream().mapToInt(Integer::intValue).toArray();

            // 左截断到模型能容纳的 prompt 预算，保留最靠近生成位置的上下文
            int budget = promptBudget(maxGenLen);
            if (promptIds.length > budget) {
                promptIds = Arrays.copyOfRange(promptIds, promptIds.length - budget, promptIds.length);
            }
            int promptLen = promptIds.length;
            if (promptLen == 0) {
                return Generation.empty();
            }

            // 生成（使用模型的 generate 方法）
            int[] result = model.generate(promptIds, maxGenLen, temperature, 0, 0.0f);

            List<Integer> promptTokenIds = new ArrayList<>(promptLen);
            for (int id : promptIds) {
                promptTokenIds.add(id);
            }

            // 提取新生成的部分
            if (result.length <= promptLen) {
                return new Generation("", promptTokenIds, null);
            }

            int[] generatedIds = Arrays.copyOfRange(result, promptLen, result.length);
            List<Integer> generatedList = new ArrayList<>();
            for (int id : generatedIds) {
                generatedList.add(id);
            }

            return new Generation(tokenizer.decode(generatedList), promptTokenIds, generatedList);
        } catch (Exception e) {
            // 生成失败会让整条 rollout 变成空 completion，进而被训练侧丢弃；
            // 静默吞掉会让"参数完全不更新"这类现象无从定位，因此首次失败必须打出来
            if (!generateFailureReported) {
                generateFailureReported = true;
                System.err.println("⚠️ AgentRollout: 生成失败，返回空字符串（后续同类失败不再重复打印）: "
                        + e.getClass().getSimpleName() + ": " + e.getMessage());
            }
            return Generation.empty();
        }
    }

    /**
     * prompt 可占用的最大 token 数：模型窗口减去本次要生成的长度，至少保留 1 个 prompt token
     */
    private int promptBudget(int maxGenLen) {
        int modelMaxSeqLen = model.getConfig().getMaxSeqLen();
        return Math.max(1, modelMaxSeqLen - Math.max(0, maxGenLen));
    }
}
