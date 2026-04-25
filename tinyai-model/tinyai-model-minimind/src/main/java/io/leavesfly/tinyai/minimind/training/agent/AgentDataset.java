package io.leavesfly.tinyai.minimind.training.agent;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Agent 强化学习数据集
 * <p>
 * 对标 Python minimind3 dataset/lm_dataset.py AgentRLDataset (L404-L430)。
 * <p>
 * 数据格式（JSONL）：
 * <pre>
 * {"conversations": [{"role":"user","content":"..."}], "tools": ["tool1","tool2"], "gt": ["expected_value"]}
 * </pre>
 * <p>
 * 简化格式（Demo 适用）：
 * <pre>
 * {"prompt": "...", "tools": ["tool1"], "gt": ["expected_value"]}
 * </pre>
 *
 * @author TinyAI Team
 * @since 2025
 */
public class AgentDataset {

    /**
     * Agent 训练样本
     */
    public static class AgentSample {
        private final List<Map<String, String>> messages;
        private final List<String> tools;
        private final List<String> gt;

        public AgentSample(List<Map<String, String>> messages, List<String> tools, List<String> gt) {
            this.messages = messages;
            this.tools = tools;
            this.gt = gt;
        }

        public List<Map<String, String>> getMessages() { return messages; }
        public List<String> getTools() { return tools; }
        public List<String> getGt() { return gt; }
    }

    /**
     * Agent 批次数据
     */
    public static class Batch {
        private final List<List<Map<String, String>>> messagesBatch;
        private final List<List<String>> toolsBatch;
        private final List<List<String>> gtBatch;

        public Batch(List<List<Map<String, String>>> messagesBatch,
                     List<List<String>> toolsBatch,
                     List<List<String>> gtBatch) {
            this.messagesBatch = messagesBatch;
            this.toolsBatch = toolsBatch;
            this.gtBatch = gtBatch;
        }

        public List<List<Map<String, String>>> getMessagesBatch() { return messagesBatch; }
        public List<List<String>> getToolsBatch() { return toolsBatch; }
        public List<List<String>> getGtBatch() { return gtBatch; }
        public int getBatchSize() { return messagesBatch.size(); }
    }

    private final int batchSize;
    private final List<AgentSample> samples;

    private int currentIndex;
    private List<AgentSample> shuffledSamples;

    public AgentDataset(int batchSize) {
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.currentIndex = 0;
    }

    // ========== 数据加载 ==========

    /**
     * 从 JSONL 文件加载 Agent 训练数据
     * <p>
     * 支持两种格式：
     * 1. conversations + tools + gt（完整格式）
     * 2. prompt + tools + gt（简化格式，自动转换为 conversations）
     */
    public void loadFromJsonl(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("Agent 数据文件不存在: " + filePath);
        }

        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        for (String line : lines) {
            if (line.trim().isEmpty()) continue;
            try {
                JSONObject json = new JSONObject(line);
                addSampleFromJson(json);
            } catch (Exception e) {
                System.err.println("跳过无效 Agent 数据行: " + e.getMessage());
            }
        }
        System.out.println("Agent 数据加载完成: " + samples.size() + " 条");
    }

    /**
     * 从 JSON 对象解析并添加样本
     */
    private void addSampleFromJson(JSONObject json) {
        // 解析 messages（conversations）
        List<Map<String, String>> messages = new ArrayList<>();
        if (json.has("conversations")) {
            JSONArray convs = json.getJSONArray("conversations");
            for (int i = 0; i < convs.length(); i++) {
                JSONObject msg = convs.getJSONObject(i);
                Map<String, String> m = new HashMap<>();
                m.put("role", msg.getString("role"));
                m.put("content", msg.getString("content"));
                messages.add(m);
            }
        } else if (json.has("prompt")) {
            // 简化格式：prompt → user message
            Map<String, String> userMsg = new HashMap<>();
            userMsg.put("role", "user");
            userMsg.put("content", json.getString("prompt"));
            messages.add(userMsg);
        }

        // 解析 tools
        List<String> tools = new ArrayList<>();
        if (json.has("tools")) {
            JSONArray toolsArray = json.getJSONArray("tools");
            for (int i = 0; i < toolsArray.length(); i++) {
                tools.add(toolsArray.getString(i));
            }
        }

        // 解析 gt（ground truth）
        List<String> gt = new ArrayList<>();
        if (json.has("gt")) {
            JSONArray gtArray = json.getJSONArray("gt");
            for (int i = 0; i < gtArray.length(); i++) {
                gt.add(String.valueOf(gtArray.get(i)));
            }
        }

        if (!messages.isEmpty()) {
            samples.add(new AgentSample(messages, tools, gt));
        }
    }

    /**
     * 手动添加样本
     */
    public void addSample(String prompt, List<String> tools, List<String> gt) {
        List<Map<String, String>> messages = new ArrayList<>();
        Map<String, String> userMsg = new HashMap<>();
        userMsg.put("role", "user");
        userMsg.put("content", prompt);
        messages.add(userMsg);
        samples.add(new AgentSample(messages, tools, gt));
    }

    // ========== 迭代控制 ==========

    /**
     * 准备数据集
     */
    public void prepare(boolean shuffle) {
        shuffledSamples = new ArrayList<>(samples);
        if (shuffle) {
            Collections.shuffle(shuffledSamples);
        }
        currentIndex = 0;
    }

    /**
     * 是否还有下一批数据
     */
    public boolean hasNext() {
        return currentIndex < shuffledSamples.size();
    }

    /**
     * 获取下一批数据
     */
    public Batch nextBatch() {
        if (!hasNext()) {
            throw new IllegalStateException("No more batches available");
        }

        int endIndex = Math.min(currentIndex + batchSize, shuffledSamples.size());
        List<AgentSample> batchSamples = shuffledSamples.subList(currentIndex, endIndex);
        currentIndex = endIndex;

        List<List<Map<String, String>>> messagesBatch = new ArrayList<>();
        List<List<String>> toolsBatch = new ArrayList<>();
        List<List<String>> gtBatch = new ArrayList<>();

        for (AgentSample sample : batchSamples) {
            // 深拷贝 messages，防止 rollout 修改原始数据
            List<Map<String, String>> msgsCopy = new ArrayList<>();
            for (Map<String, String> msg : sample.getMessages()) {
                msgsCopy.add(new HashMap<>(msg));
            }
            messagesBatch.add(msgsCopy);
            toolsBatch.add(sample.getTools());
            gtBatch.add(sample.getGt());
        }

        return new Batch(messagesBatch, toolsBatch, gtBatch);
    }

    /**
     * 重置迭代器
     */
    public void reset() {
        currentIndex = 0;
    }

    /**
     * 获取样本数量
     */
    public int getSampleCount() {
        return samples.size();
    }

    /**
     * 获取批次数量
     */
    public int getBatchCount() {
        return (int) Math.ceil((double) samples.size() / batchSize);
    }
}
