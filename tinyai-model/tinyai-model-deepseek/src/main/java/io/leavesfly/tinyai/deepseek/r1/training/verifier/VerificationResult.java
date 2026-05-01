package io.leavesfly.tinyai.deepseek.r1.training.verifier;

/**
 * 验证结果类
 * 
 * 用于封装验证器的验证结果，包括：
 * - 是否正确（二值）
 * - 预测值
 * - 期望值
 * - 验证详情
 * 
 * @author leavesfly
 * @version 1.0
 */
public class VerificationResult {
    
    private final boolean correct;
    private final String predictedValue;
    private final String expectedValue;
    private final String verificationDetails;
    private final float reward;
    
    /**
     * 构造函数 - 基础版本
     * 
     * @param correct 是否正确
     */
    public VerificationResult(boolean correct) {
        this(correct, "", "", "");
    }
    
    /**
     * 构造函数 - 完整版本
     * 
     * @param correct 是否正确
     * @param predictedValue 预测值
     * @param expectedValue 期望值
     * @param verificationDetails 验证详情
     */
    public VerificationResult(boolean correct, String predictedValue, 
                            String expectedValue, String verificationDetails) {
        this.correct = correct;
        this.predictedValue = predictedValue;
        this.expectedValue = expectedValue;
        this.verificationDetails = verificationDetails;
        this.reward = correct ? 1.0f : 0.0f;
    }
    
    /**
     * 是否验证通过
     */
    public boolean isCorrect() {
        return correct;
    }
    
    /**
     * 获取二值奖励
     * 
     * @return 1.0 (正确) 或 0.0 (错误)
     */
    public float getReward() {
        return reward;
    }
    
    /**
     * 获取预测值
     */
    public String getPredictedValue() {
        return predictedValue;
    }
    
    /**
     * 获取期望值
     */
    public String getExpectedValue() {
        return expectedValue;
    }
    
    /**
     * 获取验证详情
     */
    public String getVerificationDetails() {
        return verificationDetails;
    }

    /**
     * 获取验证详情（别名方法，等价于 {@link #getVerificationDetails()}）
     *
     * <p>为测试及调用方提供更简洁的访问方式，避免命名冗余。
     */
    public String getDetails() {
        return verificationDetails;
    }
    
    @Override
    public String toString() {
        return String.format(
            "VerificationResult{\n" +
            "  正确性: %s\n" +
            "  奖励值: %.1f\n" +
            "  预测值: %s\n" +
            "  期望值: %s\n" +
            "  详情: %s\n" +
            "}",
            correct ? "✓" : "✗",
            reward,
            predictedValue,
            expectedValue,
            verificationDetails
        );
    }
}
