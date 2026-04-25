package io.leavesfly.tinyai.minimind.training.demo;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import org.json.JSONObject;

import java.io.*;
import java.util.*;

/**
 * MiniMind 训练演示 - 配置与工具类
 * 
 * 集中管理：
 * - 路径常量
 * - 共享状态（分词器）
 * - 模型配置
 * - JSONL 文件读写工具
 * 
 * @author TinyAI Team
 */
public class DemoConfig {

    // ========== 路径常量 ==========
    
    public static final String DATA_DIR = "./data/minimind_training";
    public static final String CHECKPOINT_DIR = "./checkpoints/minimind";

    // ========== 共享状态 ==========
    
    /** 共享分词器 - 全阶段复用 */
    private static MiniMindTokenizer sharedTokenizer;

    public static MiniMindTokenizer getSharedTokenizer() {
        return sharedTokenizer;
    }

    public static void setSharedTokenizer(MiniMindTokenizer tokenizer) {
        sharedTokenizer = tokenizer;
    }

    // ========== 模型配置 ==========

    /**
     * 创建超小型配置（用于快速演示）
     * <p>
     * 对标 Python MiniMindConfig 架构特性：
     * - RMSNorm (epsilon=1e-6)
     * - GQA (numKVHeads=2, numHeads=4)
     * - SwiGLU FFN
     * - RoPE 位置编码
     * - 权重共享 (embedding ↔ lm_head)
     */
    public static MiniMindConfig createMicroConfig(int vocabSize) {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(vocabSize);
        config.setMaxSeqLen(64);          // 序列长度
        config.setHiddenSize(128);        // 隐藏维度
        config.setNumLayers(2);           // 层数
        config.setNumHeads(4);            // Q 注意力头数
        config.setNumKVHeads(2);          // KV 头数（GQA: 每2个Q头共享1组KV）
        config.setIntermediateSize(256);  // SwiGLU 中间层维度
        config.setDropout(0.0f);          // 对标 Python dropout=0.0
        config.setEpsilon(1e-6f);         // RMSNorm eps，对标 Python rms_norm_eps=1e-6
        config.setRopeTheta(1000000.0f);  // RoPE theta，对标 Python rope_theta=1e6
        config.setUseBias(false);         // 无 bias，对标 Python
        return config;
    }

    /**
     * 创建教师模型配置（用于知识蒸馏演示）
     * <p>
     * 教师模型比学生模型更大，用于指导学生学习：
     * - 隐藏维度: 192（学生128），层数: 3（学生2）
     * - 同样使用 GQA + RMSNorm + SwiGLU + RoPE
     * - 可选使用 MoE 架构（模拟 MoE→Dense 蒸馏）
     * <p>
     * 对标 Python train_distillation.py 的 teacher/student 配置模式
     */
    public static MiniMindConfig createTeacherMicroConfig(int vocabSize, boolean useMoE) {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(vocabSize);
        config.setMaxSeqLen(64);           // 与学生保持一致
        config.setHiddenSize(192);         // 比学生(128)更大
        config.setNumLayers(3);            // 比学生(2)更深
        config.setNumHeads(6);             // Q 注意力头数
        config.setNumKVHeads(2);           // KV 头数（GQA）
        config.setIntermediateSize(384);   // SwiGLU 中间层
        config.setDropout(0.0f);
        config.setEpsilon(1e-6f);
        config.setRopeTheta(1000000.0f);
        config.setUseBias(false);

        if (useMoE) {
            config.setUseMoE(true);
            config.setNumExperts(4);
            config.setNumExpertsPerToken(1);
            config.setRouterAuxLossCoef(5e-4f);
        }

        return config;
    }

    /**
     * 创建教师模型配置（Dense 模式，默认）
     */
    public static MiniMindConfig createTeacherMicroConfig(int vocabSize) {
        return createTeacherMicroConfig(vocabSize, false);
    }

    // ========== 文件工具 ==========

    /**
     * 从 JSONL 文件读取文本行
     */
    public static List<String> readJsonlFile(String filePath) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line);
                }
            }
        }
        return lines;
    }

    /**
     * 写入 JSONL 文件
     */
    public static void writeJsonlFile(List<String> lines, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String line : lines) {
                writer.write(line);
                writer.newLine();
            }
        }
    }

    // ========== 辅助方法 ==========

    /**
     * int[] 转 List<Integer>
     */
    public static List<Integer> intArrayToList(int[] array) {
        List<Integer> list = new ArrayList<>();
        for (int value : array) {
            list.add(value);
        }
        return list;
    }
}