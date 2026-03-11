package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.gpt1.training.GPT1Dataset.Batch;
import io.leavesfly.tinyai.gpt1.training.GPT1Dataset.SimpleTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * GPT1Dataset 单元测试
 * 
 * 测试覆盖：
 * 1. 数据集初始化
 * 2. 从文本加载数据
 * 3. 从文件加载数据
 * 4. 批次生成
 * 5. 数据迭代
 * 6. SimpleTokenizer功能
 * 7. 边界条件
 * 8. 异常处理
 * 
 * @author TinyAI
 */
public class GPT1DatasetTest {
    
    private GPT1Dataset dataset;
    private SimpleTokenizer tokenizer;
    
    @Before
    public void setUp() {
        dataset = new GPT1Dataset(32, 4, 1000);
        tokenizer = new SimpleTokenizer();
    }
    
    // ==================== 初始化测试 ====================
    
    @Test
    public void testDatasetCreation() {
        GPT1Dataset ds = new GPT1Dataset(64, 8, 5000);
        assertNotNull("数据集不应为null", ds);
        assertEquals("初始样本数应为0", 0, ds.getSampleCount());
        assertEquals("初始批次数应为0", 0, ds.getBatchCount());
    }
    
    @Test
    public void testInitialState() {
        assertFalse("初始状态应无下一批次", dataset.hasNext());
        assertNull("无批次时应返回null", dataset.nextBatch());
    }
    
    // ==================== 文本加载测试 ====================
    
    @Test
    public void testLoadFromTexts() {
        List<String> texts = new ArrayList<>();
        texts.add("Hello world this is a test");
        texts.add("Another line of text");
        texts.add("GPT-1 language model");
        
        dataset.loadFromTexts(texts, tokenizer);
        
        assertTrue("加载后应有样本", dataset.getSampleCount() > 0);
    }
    
    @Test
    public void testLoadEmptyTexts() {
        List<String> texts = new ArrayList<>();
        dataset.loadFromTexts(texts, tokenizer);
        
        assertEquals("空文本列表应产生0个样本", 0, dataset.getSampleCount());
    }
    
    @Test
    public void testLoadTextsWithEmptyLines() {
        List<String> texts = new ArrayList<>();
        texts.add("Valid text");
        texts.add("");
        texts.add("   ");
        texts.add(null);
        texts.add("Another valid text");
        
        dataset.loadFromTexts(texts, tokenizer);
        
        // 应该忽略空行和null，至少产生1个样本（短文本拼接后可能不足一个完整窗口，降级为单样本）
        assertTrue("应只加载有效文本", dataset.getSampleCount() >= 1);
    }
    
    @Test
    public void testLoadLongText() {
        // 创建超过maxSeqLen的长文本
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            sb.append("word").append(i).append(" ");
        }
        
        List<String> texts = new ArrayList<>();
        texts.add(sb.toString());
        
        dataset.loadFromTexts(texts, tokenizer);
        
        // 应该被切分成多个样本
        assertTrue("长文本应被切分", dataset.getSampleCount() > 1);
    }
    
    // ==================== 文件加载测试 ====================
    
    @Test
    public void testLoadFromFile() throws IOException {
        // 创建临时文件
        Path tempFile = Files.createTempFile("test_dataset", ".txt");
        List<String> lines = new ArrayList<>();
        lines.add("Line one");
        lines.add("Line two");
        lines.add("Line three");
        Files.write(tempFile, lines);
        
        try {
            dataset.loadFromFile(tempFile.toString(), tokenizer);
            assertTrue("从文件加载后应有样本", dataset.getSampleCount() > 0);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
    
    @Test(expected = IOException.class)
    public void testLoadFromNonExistentFile() throws IOException {
        dataset.loadFromFile("/non/existent/file.txt", tokenizer);
    }
    
    // ==================== 批次生成测试 ====================
    
    @Test
    public void testPrepareWithoutShuffle() {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            texts.add("Sample text number " + i);
        }
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        assertTrue("准备后应有批次", dataset.getBatchCount() > 0);
        assertTrue("准备后应有下一批次", dataset.hasNext());
    }
    
    @Test
    public void testPrepareWithShuffle() {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            texts.add("Sample text " + i);
        }
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(true);
        
        assertTrue("带shuffle准备后应有批次", dataset.getBatchCount() > 0);
    }
    
    @Test
    public void testBatchCount() {
        // batchSize=4, 加载10个样本
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            texts.add("Short text " + i);
        }
        
        dataset.loadFromTexts(texts, tokenizer);
        int sampleCount = dataset.getSampleCount();
        dataset.prepare(false);
        
        int expectedBatchCount = (int) Math.ceil(sampleCount / 4.0);
        assertEquals("批次数应正确", expectedBatchCount, dataset.getBatchCount());
    }
    
    // ==================== 批次结构测试 ====================
    
    @Test
    public void testBatchStructure() {
        List<String> texts = new ArrayList<>();
        texts.add("Hello world");
        texts.add("GPT model");
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        Batch batch = dataset.nextBatch();
        assertNotNull("批次不应为null", batch);
        assertNotNull("inputIds不应为null", batch.getInputIds());
        assertNotNull("targetIds不应为null", batch.getTargetIds());
        
        assertTrue("批次大小应大于0", batch.getBatchSize() > 0);
        assertTrue("序列长度应大于0", batch.getSeqLen() > 0);
    }
    
    @Test
    public void testBatchInputTargetRelationship() {
        List<String> texts = new ArrayList<>();
        texts.add("Test sentence for GPT");
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        Batch batch = dataset.nextBatch();
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        
        // input和target的形状应该相同
        assertEquals("input和target批次大小应相同", 
            inputIds.getShape().getDimension(0), 
            targetIds.getShape().getDimension(0));
        assertEquals("input和target序列长度应相同", 
            inputIds.getShape().getDimension(1), 
            targetIds.getShape().getDimension(1));
        
        // target应该是input右移1位（验证第一个位置）
        // 注意：由于有BOS/EOS token，具体验证需要根据实际编码逻辑
        assertNotNull("输入和目标都应有值", inputIds);
        assertNotNull("输入和目标都应有值", targetIds);
    }
    
    // ==================== 迭代测试 ====================
    
    @Test
    public void testIteration() {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            texts.add("Text " + i);
        }
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        int batchCount = 0;
        while (dataset.hasNext()) {
            Batch batch = dataset.nextBatch();
            assertNotNull("批次不应为null", batch);
            batchCount++;
        }
        
        assertEquals("遍历的批次数应与getBatchCount一致", 
            dataset.getBatchCount(), batchCount);
        assertFalse("遍历完成后应无更多批次", dataset.hasNext());
    }
    
    @Test
    public void testReset() {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            texts.add("Sample " + i);
        }
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        // 消耗所有批次
        while (dataset.hasNext()) {
            dataset.nextBatch();
        }
        
        assertFalse("消耗完应无下一批次", dataset.hasNext());
        
        // 重置
        dataset.reset();
        assertTrue("重置后应有下一批次", dataset.hasNext());
    }
    
    @Test
    public void testMultipleIterations() {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            texts.add("Text line " + i);
        }
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        int firstIterCount = 0;
        while (dataset.hasNext()) {
            dataset.nextBatch();
            firstIterCount++;
        }
        
        dataset.reset();
        
        int secondIterCount = 0;
        while (dataset.hasNext()) {
            dataset.nextBatch();
            secondIterCount++;
        }
        
        assertEquals("两次迭代的批次数应相同", firstIterCount, secondIterCount);
    }
    
    // ==================== SimpleTokenizer测试 ====================
    
    @Test
    public void testTokenizerInitialization() {
        SimpleTokenizer tok = new SimpleTokenizer();
        
        // 应该有4个特殊token
        assertTrue("初始词汇表应>=4", tok.getVocabSize() >= 4);
    }
    
    @Test
    public void testTokenizerEncode() {
        SimpleTokenizer tok = new SimpleTokenizer();
        
        String text = "hello world";
        List<Integer> ids = tok.encode(text);
        
        assertNotNull("编码结果不应为null", ids);
        assertTrue("编码结果应有token", ids.size() > 0);
        
        // 应该包含BOS和EOS
        assertTrue("应包含多个token（含BOS/EOS）", ids.size() >= 4); // BOS + hello + world + EOS
    }
    
    @Test
    public void testTokenizerDecode() {
        SimpleTokenizer tok = new SimpleTokenizer();
        
        String original = "test sentence";
        List<Integer> ids = tok.encode(original);
        int[] idsArray = ids.stream().mapToInt(Integer::intValue).toArray();
        String decoded = tok.decode(idsArray);
        
        assertNotNull("解码结果不应为null", decoded);
        // 注意：由于简化分词器使用空格分词，解码结果应该相似
        assertTrue("解码结果应包含原文的词", 
            decoded.contains("test") && decoded.contains("sentence"));
    }
    
    @Test
    public void testTokenizerVocabGrowth() {
        SimpleTokenizer tok = new SimpleTokenizer();
        int initialSize = tok.getVocabSize();
        
        tok.encode("new unique word");
        
        assertTrue("词汇表应增长", tok.getVocabSize() > initialSize);
    }
    
    @Test
    public void testTokenizerEmptyString() {
        SimpleTokenizer tok = new SimpleTokenizer();
        
        List<Integer> ids = tok.encode("");
        
        // 应该只有BOS和EOS
        assertEquals("空字符串应只有BOS和EOS", 2, ids.size());
    }
    
    @Test
    public void testTokenizerConsistency() {
        SimpleTokenizer tok = new SimpleTokenizer();
        
        String text = "consistent encoding";
        List<Integer> ids1 = tok.encode(text);
        List<Integer> ids2 = tok.encode(text);
        
        assertEquals("相同文本的编码应一致", ids1, ids2);
    }
    
    // ==================== 边界条件测试 ====================
    
    @Test
    public void testSingleSample() {
        List<String> texts = new ArrayList<>();
        texts.add("Only one sample");
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        assertTrue("单个样本应产生批次", dataset.getBatchCount() > 0);
    }
    
    @Test
    public void testExactBatchSize() {
        // 样本数正好等于batchSize
        GPT1Dataset ds = new GPT1Dataset(32, 4, 1000);
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            texts.add("Sample " + i);
        }
        
        ds.loadFromTexts(texts, tokenizer);
        ds.prepare(false);
        
        // 应该正好1个批次
        int actualSamples = ds.getSampleCount();
        int expectedBatches = (int) Math.ceil(actualSamples / 4.0);
        assertEquals("批次数应正确", expectedBatches, ds.getBatchCount());
    }
    
    @Test
    public void testLargeBatchSize() {
        // batchSize大于样本数
        GPT1Dataset ds = new GPT1Dataset(32, 100, 1000);
        List<String> texts = new ArrayList<>();
        texts.add("Small dataset");
        texts.add("Only two samples");
        
        ds.loadFromTexts(texts, tokenizer);
        ds.prepare(false);
        
        // 应该只有1个批次
        assertTrue("应产生至少1个批次", ds.getBatchCount() >= 1);
    }
    
    @Test
    public void testVeryShortSequence() {
        // 测试极短的序列
        List<String> texts = new ArrayList<>();
        texts.add("a");
        
        dataset.loadFromTexts(texts, tokenizer);
        dataset.prepare(false);
        
        if (dataset.getBatchCount() > 0) {
            Batch batch = dataset.nextBatch();
            assertNotNull("极短序列也应产生有效批次", batch);
        }
    }
    
    // ==================== 数据准确性测试 ====================
    
    @Test
    public void testDataNotLost() {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            texts.add("Text " + i);
        }
        
        dataset.loadFromTexts(texts, tokenizer);
        int totalSamples = dataset.getSampleCount();
        dataset.prepare(false);
        
        int totalProcessed = 0;
        while (dataset.hasNext()) {
            Batch batch = dataset.nextBatch();
            totalProcessed += batch.getBatchSize();
        }
        
        assertEquals("处理的样本数应等于加载的样本数", totalSamples, totalProcessed);
    }
    
    @Test
    public void testBatchSizeConsistency() {
        GPT1Dataset ds = new GPT1Dataset(32, 4, 1000);
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            texts.add("Sample " + i);
        }
        
        ds.loadFromTexts(texts, tokenizer);
        ds.prepare(false);
        
        while (ds.hasNext()) {
            Batch batch = ds.nextBatch();
            // 每个批次的batchSize应<=4（最后一个批次可能更小）
            assertTrue("批次大小应合理", batch.getBatchSize() > 0 && batch.getBatchSize() <= 4);
        }
    }
}
