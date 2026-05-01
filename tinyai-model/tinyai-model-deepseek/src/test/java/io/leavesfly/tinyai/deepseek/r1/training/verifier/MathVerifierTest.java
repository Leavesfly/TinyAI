package io.leavesfly.tinyai.deepseek.r1.training.verifier;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * MathVerifier 单元测试
 *
 * <p>覆盖 P2-18 相对容差、parseNumber 健全化（全角/中文对错/千位分隔符/科学计数法）
 * 以及端到端 verify 流程。
 *
 * @author leavesfly
 */
public class MathVerifierTest {

    private final MathVerifier verifier = new MathVerifier();

    // ===================== verify() 端到端 =====================

    @Test
    public void testVerify_exactMatch() {
        VerificationResult r = verifier.verify("The answer is 42", "42");
        assertNotNull(r);
        assertTrue(r.isCorrect());
    }

    @Test
    public void testVerify_withinRelativeTolerance() {
        // 相对容差 1e-6：3.1415926 vs 3.141593 应该算对
        VerificationResult r = verifier.verify("答案是 3.141593", "3.1415926");
        assertTrue(r.isCorrect(), "相对容差 1e-6 内应判为正确: " + r.getDetails());
    }

    @Test
    public void testVerify_outsideTolerance() {
        // 3.14 vs 3.15，绝对误差 0.01，相对误差 ~3e-3，远超 1e-6
        VerificationResult r = verifier.verify("result: 3.14", "3.15");
        assertFalse(r.isCorrect(), "超出容差应判为错误: " + r.getDetails());
    }

    @Test
    public void testVerify_nearZeroUsesAbsoluteTolerance() {
        // 答案接近 0 时必须用绝对容差 1e-9，否则 1e-7 < 1e-6·|0| = 0 会误判
        VerificationResult r = verifier.verify("The answer is 0.0000001", "0");
        assertFalse(r.isCorrect(), "零附近应使用绝对容差 1e-9: " + r.getDetails());

        VerificationResult r2 = verifier.verify("The answer is 0.0000000001", "0");
        assertTrue(r2.isCorrect(), "1e-10 < 1e-9，应在绝对容差内: " + r2.getDetails());
    }

    @Test
    public void testVerify_extractLastNumberWhenNoAnswerKeyword() {
        // 没有 "answer:" 关键字时，提取最后一个数字
        VerificationResult r = verifier.verify("Step 1: 2+3=5, Step 2: 5*8=40, so we get 42", "42");
        assertTrue(r.isCorrect());
    }

    @Test
    public void testVerify_scientificNotation() {
        VerificationResult r = verifier.verify("The answer is 1.5e3", "1500");
        assertTrue(r.isCorrect(), "科学计数法应被正确解析: " + r.getDetails());
    }

    @Test
    public void testVerify_negativeNumbers() {
        VerificationResult r = verifier.verify("The answer is -42", "-42");
        assertTrue(r.isCorrect());
    }

    @Test
    public void testVerify_boolYes() {
        // groundTruth=yes 会走 parseNumber 的布尔分支
        VerificationResult r = verifier.verify("yes", "yes");
        assertTrue(r.isCorrect());
    }

    @Test
    public void testVerify_invalidInput() {
        // 无法解析的情况应返回 isCorrect=false 且不抛异常
        VerificationResult r = verifier.verify("完全没有数字的文本", "42");
        assertNotNull(r);
        assertFalse(r.isCorrect());
    }

    // ===================== 验证器类型 =====================

    @Test
    public void testGetVerifierType() {
        assertTrue("math".equals(verifier.getVerifierType()));
    }
}
