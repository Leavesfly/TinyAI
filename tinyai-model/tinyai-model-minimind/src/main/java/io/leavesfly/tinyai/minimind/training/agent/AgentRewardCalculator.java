package io.leavesfly.tinyai.minimind.training.agent;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Agent 多维度奖励计算器
 * <p>
 * 对标 Python minimind3 train_agent.py calculate_rewards (L333-L410)
 * + reward_utils.py compute_repetition_penalty。
 * <p>
 * 奖励计算逻辑：
 * 1. 无工具调用：长度分 + 思考块分 + 重复惩罚
 * 2. 有工具调用：工具对齐分 + GT验证分 + 未完成扣分 + 重复惩罚
 * 3. 标签不匹配扣分（所有场景）
 * 4. 总分 Clip 到 [-3.0, 3.0]
 *
 * @author TinyAI Team
 * @since 2025
 */
public class AgentRewardCalculator {

    /** 匹配数值的正则（整数或小数） */
    private static final Pattern NUMBER_PATTERN =
            Pattern.compile("(?<![\\w.])[-+]?\\d+(?:\\.\\d+)?(?![\\w.])");

    // ========== 重复惩罚（对标 Python reward_utils.py L59-L80） ==========

    /**
     * 基于 n-gram 重复率的惩罚分数
     * <p>
     * 对标 Python compute_repetition_penalty：
     * - 将文本切分为词级 token
     * - 统计 n-gram 中重复出现的比例
     * - 截断到 cap
     *
     * @param text 待评估文本
     * @param n    n-gram 窗口大小（默认 3）
     * @param cap  惩罚上限（默认 0.5）
     * @return [0, cap] 之间的惩罚值
     */
    public static float computeRepetitionPenalty(String text, int n, float cap) {
        if (text == null || text.isEmpty()) return 0.0f;

        // 提取词级 token（字母数字序列 + 标点）
        List<String> tokens = new ArrayList<>();
        Matcher m = Pattern.compile("\\w+|[^\\w\\s]").matcher(text.toLowerCase());
        while (m.find()) {
            tokens.add(m.group());
        }

        if (tokens.size() < n + 1) return 0.0f;

        // 构建 n-gram 集合
        List<String> grams = new ArrayList<>();
        Set<String> uniqueGrams = new HashSet<>();
        for (int i = 0; i <= tokens.size() - n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < i + n; j++) {
                if (j > i) sb.append("_");
                sb.append(tokens.get(j));
            }
            String gram = sb.toString();
            grams.add(gram);
            uniqueGrams.add(gram);
        }

        if (grams.isEmpty()) return 0.0f;

        float duplicateRatio = (grams.size() - uniqueGrams.size()) * cap * 2.0f / grams.size();
        return Math.min(cap, duplicateRatio);
    }

    /**
     * 重复惩罚（默认参数：3-gram, cap=0.5）
     */
    public static float computeRepetitionPenalty(String text) {
        return computeRepetitionPenalty(text, 3, 0.5f);
    }

    // ========== GT 验证（对标 Python validate_gt_in_text L316-L330） ==========

    /**
     * 验证文本中是否包含 ground truth 的数值
     * <p>
     * 对标 Python validate_gt_in_text：
     * - 字符串精确匹配（忽略大小写）
     * - 数值近似匹配（误差 < 1e-6）
     *
     * @param text   待验证文本
     * @param gtList ground truth 列表
     * @return 在文本中验证通过的 gt 集合
     */
    public static Set<String> validateGtInText(String text, List<String> gtList) {
        Set<String> verified = new HashSet<>();
        if (text == null || gtList == null || gtList.isEmpty()) return verified;

        String textLower = text.toLowerCase();
        String textNoComma = text.replace(",", "");

        // 提取文本中的所有数值
        List<Double> numsInText = new ArrayList<>();
        Matcher numMatcher = NUMBER_PATTERN.matcher(textNoComma);
        while (numMatcher.find()) {
            try {
                numsInText.add(Double.parseDouble(numMatcher.group()));
            } catch (NumberFormatException ignored) {
            }
        }

        for (String gt : gtList) {
            if (gt == null) continue;
            String gtTrimmed = gt.trim();
            if (gtTrimmed.isEmpty()) continue;

            // 1. 字符串精确匹配（忽略大小写）
            if (textLower.contains(gtTrimmed.toLowerCase())) {
                verified.add(gt);
                continue;
            }

            // 2. 数值近似匹配
            String gtNoComma = gtTrimmed.replace(",", "");
            try {
                double gtNum = Double.parseDouble(gtNoComma);
                for (double numInText : numsInText) {
                    if (Math.abs(gtNum - numInText) < 1e-6) {
                        verified.add(gt);
                        break;
                    }
                }
            } catch (NumberFormatException ignored) {
                // 非数值 gt，跳过数值匹配
            }
        }

        return verified;
    }

    // ========== 核心奖励计算（对标 Python calculate_rewards L333-L410） ==========

    /**
     * 批量计算 Agent 响应的奖励
     * <p>
     * 对标 Python calculate_rewards：
     * - 无工具调用：长度分 + 思考块分 + 重复惩罚
     * - 有工具调用：工具对齐分 + GT 验证分 + 未完成扣分 + 重复惩罚
     *
     * @param completions      生成的响应列表
     * @param gtBatch          每个样本的 ground truth 列表
     * @param toolsBatch       每个样本的可用工具名列表
     * @param numGenerations   每个样本生成的候选数量
     * @param turnOutputsBatch 每个候选的多轮输出
     * @param unfinishedBatch  每个候选是否未完成
     * @return 奖励数组，长度 = completions.size()
     */
    public static float[] calculateRewards(
            List<String> completions,
            List<List<String>> gtBatch,
            List<List<String>> toolsBatch,
            int numGenerations,
            List<List<String>> turnOutputsBatch,
            List<Boolean> unfinishedBatch) {

        float[] rewards = new float[completions.size()];

        for (int idx = 0; idx < completions.size(); idx++) {
            String response = completions.get(idx);
            float reward = 0.0f;
            int sampleIdx = idx / numGenerations;

            List<String> tools = toolsBatch.get(sampleIdx);
            List<String> turnOutputs = (turnOutputsBatch != null && idx < turnOutputsBatch.size())
                    ? turnOutputsBatch.get(idx) : Collections.singletonList(response);
            boolean unfinished = (unfinishedBatch != null && idx < unfinishedBatch.size())
                    ? unfinishedBatch.get(idx) : false;

            // 提取每轮的实际回答（去除 </think> 之前的思考内容）
            List<String> turnAnswers = new ArrayList<>();
            for (String turn : turnOutputs) {
                if (turn.contains("</think>")) {
                    String[] parts = turn.split("</think>", 2);
                    turnAnswers.add(parts[1].trim());
                } else {
                    turnAnswers.add(turn.trim());
                }
            }
            String answer = turnAnswers.isEmpty() ? response.trim()
                    : turnAnswers.get(turnAnswers.size() - 1);

            // 收集有效工具名
            Set<String> validNames = new HashSet<>(tools);

            // 解析所有轮次的工具调用
            List<Map<String, Object>> toolCalls = new ArrayList<>();
            for (String turnAnswer : turnAnswers) {
                toolCalls.addAll(AgentTool.parseToolCalls(turnAnswer));
            }

            // 标签不匹配扣分（对标 Python L368-L369）
            for (String turnText : turnAnswers) {
                int openCount = countOccurrences(turnText, "<tool_call>");
                int closeCount = countOccurrences(turnText, "</tool_call>");
                reward -= 0.5f * Math.abs(openCount - closeCount);
            }

            if (toolCalls.isEmpty()) {
                // ======== 无工具调用：格式奖励（对标 Python L371-L386） ========
                // 长度分
                int len = response.trim().length();
                reward += (len >= 5 && len <= 800) ? 0.5f : -0.5f;

                // 思考块分
                if (response.contains("</think>")) {
                    String[] parts = response.split("</think>", 2);
                    String thinkContent = parts[0].trim();
                    answer = parts[1].trim();

                    int thinkLen = thinkContent.length();
                    reward += (thinkLen >= 20 && thinkLen <= 300) ? 1.0f : -0.5f;

                    // 思考闭合分
                    int thinkCount = countOccurrences(response, "</think>");
                    reward += (thinkCount == 1) ? 0.25f : -0.25f;
                }

                // 重复惩罚
                reward -= computeRepetitionPenalty(answer);

            } else {
                // ======== 有工具调用：执行结果奖励（对标 Python L388-L409） ========
                List<String> gt = gtBatch.get(sampleIdx);

                // 统计有效工具调用数
                int validCallCount = 0;
                for (Map<String, Object> toolCall : toolCalls) {
                    String name = (String) toolCall.getOrDefault("name", "");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> args = (Map<String, Object>) toolCall.getOrDefault("arguments", Collections.emptyMap());
                    if (validNames.contains(name) && AgentTool.validateArgs(name, args)) {
                        validCallCount++;
                    }
                }

                // 工具对齐分（对标 Python L400-L401）
                int toolGap = Math.abs(validCallCount - gt.size())
                        + Math.max(0, toolCalls.size() - validCallCount);
                reward += (toolGap == 0) ? 0.5f : -0.5f * toolGap;

                // GT 验证分（对标 Python L403-L406）
                String finalText;
                if (unfinished) {
                    finalText = "";
                } else if (answer.contains("</tool_call>")) {
                    String[] parts = answer.split("</tool_call>");
                    finalText = parts[parts.length - 1];
                } else {
                    finalText = answer;
                }

                if (!gt.isEmpty()) {
                    Set<String> verified = validateGtInText(finalText, gt);
                    reward += 2.5f * verified.size() / gt.size();
                }

                // 未完成扣分（对标 Python L407）
                if (unfinished) {
                    reward -= 0.5f;
                }

                // 重复惩罚
                reward -= computeRepetitionPenalty(finalText.isEmpty() ? answer : finalText);
            }

            // 总分 Clip 到 [-3.0, 3.0]（对标 Python L386, L409）
            rewards[idx] = Math.max(-3.0f, Math.min(3.0f, reward));
        }

        return rewards;
    }

    /**
     * 计算子串出现次数
     */
    private static int countOccurrences(String text, String sub) {
        int count = 0;
        int idx = 0;
        while ((idx = text.indexOf(sub, idx)) != -1) {
            count++;
            idx += sub.length();
        }
        return count;
    }
}
