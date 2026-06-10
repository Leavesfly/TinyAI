package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.v3.training.DeepSeekV3MTPSpeculativeInference;
import io.leavesfly.tinyai.deepseek.v3.training.DeepSeekV3MTPSpeculativeInference.SpeculativeResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DeepSeekV3MTPSpeculativeInference 推测解码推理测试
 *
 * @author leavesfly
 */
public class DeepSeekV3MTPSpeculativeInferenceTest {

    private DeepSeekV3Model model;
    private DeepSeekV3MTPSpeculativeInference speculativeInference;

    @BeforeEach
    void setUp() {
        // 使用 micro 配置（默认 mtpDepth=1），参数量约 200K
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        assertEquals(1, config.getMtpDepth(), "micro config mtpDepth 应为 1");

        model = new DeepSeekV3Model("test_v3", config);
        speculativeInference = new DeepSeekV3MTPSpeculativeInference(model);
        speculativeInference.setSeed(42);
    }

    @Test
    void testGreedyGeneration() {
        int[] prompt = {1, 2, 3, 4, 5};
        int maxNewTokens = 5;

        SpeculativeResult result = speculativeInference.generateGreedy(prompt, maxNewTokens);

        assertNotNull(result, "结果不应为 null");
        assertTrue(result.tokens.length > prompt.length,
                "生成序列长度应大于 prompt");
        assertTrue(result.tokens.length <= prompt.length + maxNewTokens,
                "生成序列长度不应超过 prompt + maxNewTokens");

        // 验证 prompt 部分保持不变
        for (int i = 0; i < prompt.length; i++) {
            assertEquals(prompt[i], result.tokens[i], "prompt 部分应保持不变");
        }

        System.out.println("贪婪推测解码结果: " + result);
        speculativeInference.printStatistics();
    }

    @Test
    void testTemperatureGeneration() {
        int[] prompt = {10, 20, 30};
        int maxNewTokens = 8;

        SpeculativeResult result = speculativeInference.generateWithTemperature(
                prompt, maxNewTokens, 0.8f);

        assertNotNull(result);
        assertTrue(result.tokens.length > prompt.length);

        // 验证 prompt 部分保持不变
        for (int i = 0; i < prompt.length; i++) {
            assertEquals(prompt[i], result.tokens[i]);
        }

        System.out.println("Temperature 推测解码结果: " + result);
        speculativeInference.printStatistics();
    }

    @Test
    void testStatisticsTracking() {
        int[] prompt = {1, 2, 3, 4, 5};

        speculativeInference.resetStatistics();
        speculativeInference.generateGreedy(prompt, 6);

        // 验证统计信息合理性（随机权重模型的 draft 接受率可能很低但不应为负）
        double acceptanceRate = speculativeInference.getAcceptanceRate();
        assertTrue(acceptanceRate >= 0.0,
                "接受率不应为负: " + acceptanceRate);
        assertTrue(acceptanceRate <= 1.0,
                "接受率不应超过 1: " + acceptanceRate);

        double avgTokensPerRound = speculativeInference.getAvgTokensPerRound();
        assertTrue(avgTokensPerRound >= 0.0,
                "平均每轮 token 数不应为负: " + avgTokensPerRound);

        System.out.printf("接受率: %.2f%%, 平均每轮 token: %.2f%n",
                acceptanceRate * 100, avgTokensPerRound);
    }

    @Test
    void testResetStatistics() {
        int[] prompt = {1, 2, 3};
        speculativeInference.generateGreedy(prompt, 3);

        speculativeInference.resetStatistics();
        assertEquals(0.0, speculativeInference.getAcceptanceRate(), 1e-9);
        assertEquals(0.0, speculativeInference.getAvgTokensPerRound(), 1e-9);
    }

    @Test
    void testShortPrompt() {
        // 极短 prompt，验证边界情况
        int[] prompt = {1};
        SpeculativeResult result = speculativeInference.generateGreedy(prompt, 3);

        assertNotNull(result);
        assertTrue(result.tokens.length >= 1);
        assertEquals(prompt[0], result.tokens[0]);
    }

    @Test
    void testModelWithoutMTPThrowsException() {
        DeepSeekV3Config noMtpConfig = DeepSeekV3Config.createMicroConfig();
        noMtpConfig.setMtpDepth(0);
        DeepSeekV3Model noMtpModel = new DeepSeekV3Model("no_mtp", noMtpConfig);

        // 应抛出 IllegalArgumentException
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekV3MTPSpeculativeInference(noMtpModel));
    }

    @Test
    void testSpeculativeResultToString() {
        int[] prompt = {1, 2, 3};
        SpeculativeResult result = speculativeInference.generateGreedy(prompt, 4);
        String str = result.toString();

        assertNotNull(str);
        assertTrue(str.contains("tokens="), "toString 应包含 tokens 信息");
        assertTrue(str.contains("acceptanceRate="), "toString 应包含 acceptanceRate 信息");
    }
}
