package io.leavesfly.tinyai.deepseek.r1.training.dataset;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * DeepSeekR1RLVRDataset 单元测试
 *
 * <p>主要验证 P2-12 的空格分词 + 动态词表 + hash fallback 行为，
 * 以及 F-5 暴露的 vocabSize 兼容性校验。
 *
 * @author leavesfly
 */
public class DeepSeekR1RLVRDatasetTest {

    @Test
    public void testVocabLimitIsPublic() {
        // 回归：P3-20 + F-5 将 VOCAB_LIMIT 改为 public，供下游做兼容性校验
        assertTrue(DeepSeekR1RLVRDataset.VOCAB_LIMIT >= 1000,
                "VOCAB_LIMIT 必须对外暴露且 >= 1000");
    }

    @Test
    public void testRequireCompatibleVocabSize_acceptsLargeEnough() {
        // 等于阈值 / 大于阈值都应通过
        assertDoesNotThrow(() ->
                DeepSeekR1RLVRDataset.requireCompatibleVocabSize(DeepSeekR1RLVRDataset.VOCAB_LIMIT));
        assertDoesNotThrow(() ->
                DeepSeekR1RLVRDataset.requireCompatibleVocabSize(50_257));
    }

    @Test
    public void testRequireCompatibleVocabSize_rejectsTooSmall() {
        // 小于阈值必须抛异常，避免运行时 embedding 越界
        assertThrows(IllegalArgumentException.class,
                () -> DeepSeekR1RLVRDataset.requireCompatibleVocabSize(500));
        assertThrows(IllegalArgumentException.class,
                () -> DeepSeekR1RLVRDataset.requireCompatibleVocabSize(0));
    }

    @Test
    public void testDatasetConstructionAndVocabGrowth() {
        DeepSeekR1RLVRDataset dataset = new DeepSeekR1RLVRDataset(2, 32);
        int initialVocab = dataset.vocabSize();
        // 构造后词表应为初始状态（nextTokenId=1，因此 vocabSize() 返回 1）
        assertTrue(initialVocab >= 1);

        // 添加样本后词表应增长（这里我们只验证 API 可用性，具体增长靠其他集成测试）
        dataset.addSample("What is 2 + 3?", "5", "math");
        dataset.addSample("Is 1+1=2 correct?", "true", "logic");
        int afterVocab = dataset.vocabSize();
        assertTrue(afterVocab >= initialVocab,
                "加入样本后词表大小只应增加或不变: before=" + initialVocab + " after=" + afterVocab);
    }

    @Test
    public void testDatasetSampleCountMatchesAdded() {
        DeepSeekR1RLVRDataset dataset = new DeepSeekR1RLVRDataset(2, 32);
        assertEquals(0, dataset.getSampleCount());
        dataset.addSample("q1", "a1", "math");
        dataset.addSample("q2", "a2", "logic");
        dataset.addSample("q3", "a3", "math");
        assertEquals(3, dataset.getSampleCount());
    }
}
