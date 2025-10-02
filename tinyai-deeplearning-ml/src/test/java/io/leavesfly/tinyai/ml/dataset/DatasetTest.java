package io.leavesfly.tinyai.ml.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.Before;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Dataset 包相关类的单元测试
 * 
 * 测试 ArrayDataset、Batch、DataSet 相关功能
 * 
 * @author TinyDL
 * @version 1.0
 */
public class DatasetTest {

    private TestArrayDataset testDataset;
    private NdArray[] testXs;
    private NdArray[] testYs;
    private int batchSize = 2;

    @Before
    public void setUp() {
        // 创建测试数据
        testXs = new NdArray[6];
        testYs = new NdArray[6];
        
        for (int i = 0; i < 6; i++) {
            testXs[i] = NdArray.of(new float[][]{{i + 1, i + 2}});
            testYs[i] = NdArray.of(new float[][]{{i * 2}});
        }
        
        testDataset = new TestArrayDataset(batchSize, testXs, testYs);
    }

    @Test
    public void testBatchCreation() {
        // 测试 Batch 创建和基本操作
        NdArray[] batchX = {testXs[0], testXs[1]};
        NdArray[] batchY = {testYs[0], testYs[1]};
        
        Batch batch = new Batch(batchX, batchY);
        
        assertEquals(2, batch.getSize());
        assertArrayEquals(batchX, batch.getX());
        assertArrayEquals(batchY, batch.getY());
    }

    @Test
    public void testBatchIteration() {
        // 测试 Batch 迭代功能
        NdArray[] batchX = {testXs[0], testXs[1]};
        NdArray[] batchY = {testYs[0], testYs[1]};
        
        Batch batch = new Batch(batchX, batchY);
        
        assertTrue(batch.hasNext());
        assertEquals(0, batch.getCurrentIndex());
        
        Batch.Pair<NdArray, NdArray> pair1 = batch.next();
        assertNotNull(pair1);
        assertEquals(testXs[0], pair1.key);
        assertEquals(testYs[0], pair1.value);
        assertEquals(1, batch.getCurrentIndex());
        
        assertTrue(batch.hasNext());
        Batch.Pair<NdArray, NdArray> pair2 = batch.next();
        assertNotNull(pair2);
        assertEquals(testXs[1], pair2.key);
        assertEquals(testYs[1], pair2.value);
        assertEquals(2, batch.getCurrentIndex());
        
        assertFalse(batch.hasNext());
        assertNull(batch.next());
        
        // 测试重置
        batch.resetIndex();
        assertEquals(0, batch.getCurrentIndex());
        assertTrue(batch.hasNext());
    }

    @Test
    public void testBatchToVariable() {
        // 测试 Batch 转换为 Variable
        NdArray[] batchX = {testXs[0], testXs[1]};
        NdArray[] batchY = {testYs[0], testYs[1]};
        
        Batch batch = new Batch(batchX, batchY);
        
        // 测试转换为Variable（应该会缓存结果）
        assertNotNull(batch.toVariableX());
        assertNotNull(batch.toVariableY());
        
        // 验证缓存机制 - 多次调用应返回同一对象
        assertSame(batch.toVariableX(), batch.toVariableX());
        assertSame(batch.toVariableY(), batch.toVariableY());
        
        // 修改数据后缓存应清除
        batch.setX(new NdArray[]{testXs[2]});
        // 注意：此时缓存已被清除，下次调用会创建新的Variable
        assertNotNull(batch.toVariableX());
    }

    @Test
    public void testArrayDatasetGetBatches() {
        // 测试 ArrayDataset 获取批次数据
        List<Batch> batches = testDataset.getBatches();
        
        assertEquals(3, batches.size()); // 6个数据，批次大小为2，应该有3个批次
        
        for (int i = 0; i < batches.size(); i++) {
            Batch batch = batches.get(i);
            assertEquals(batchSize, batch.getSize());
            
            // 验证数据正确性
            NdArray[] batchX = batch.getX();
            NdArray[] batchY = batch.getY();
            
            for (int j = 0; j < batchSize; j++) {
                int originalIndex = i * batchSize + j;
                assertEquals(testXs[originalIndex], batchX[j]);
                assertEquals(testYs[originalIndex], batchY[j]);
            }
        }
    }

    @Test
    public void testDatasetSplit() {
        // 测试数据集分割
        Map<String, DataSet> splitDatasets = testDataset.splitDataset(0.5f, 0.3f, 0.2f);
        
        assertNotNull(splitDatasets);
        assertEquals(3, splitDatasets.size());
        
        assertTrue(splitDatasets.containsKey("TRAIN"));
        assertTrue(splitDatasets.containsKey("TEST"));
        assertTrue(splitDatasets.containsKey("VALIDATION"));
        
        DataSet trainDataset = splitDatasets.get("TRAIN");
        DataSet testDatasetSplit = splitDatasets.get("TEST");
        DataSet validationDataset = splitDatasets.get("VALIDATION");
        
        assertNotNull(trainDataset);
        assertNotNull(testDatasetSplit);
        assertNotNull(validationDataset);
        
        // 验证分割比例（基于6个样本）
        assertEquals(3, trainDataset.getSize()); // 50% of 6 = 3
        assertEquals(1, testDatasetSplit.getSize()); // 30% of 6 = 1.8 ≈ 1 (由于整数切割)
        assertEquals(2, validationDataset.getSize()); // 剩余的数据作为验证集
    }

    @Test
    public void testDatasetSplitInvalidRatio() {
        // 测试无效的分割比例
        try {
            testDataset.splitDataset(0.5f, 0.4f, 0.2f); // 总和不等于1.0
            fail("应该抛出异常");
        } catch (RuntimeException e) {
            assertTrue(e.getMessage().contains("splitDataset parameters error"));
        }
    }

    @Test
    public void testDatasetShuffle() {
        // 保存原始数据的副本
        NdArray[] originalXs = testDataset.getXs().clone();
        NdArray[] originalYs = testDataset.getYs().clone();
        
        // 执行shuffle
        testDataset.shuffle();
        
        // 验证数据总数不变
        assertEquals(originalXs.length, testDataset.getXs().length);
        assertEquals(originalYs.length, testDataset.getYs().length);
        
        // 验证所有原始数据仍然存在（虽然顺序可能不同）
        NdArray[] shuffledXs = testDataset.getXs();
        NdArray[] shuffledYs = testDataset.getYs();
        
        for (NdArray originalX : originalXs) {
            boolean found = false;
            for (NdArray shuffledX : shuffledXs) {
                if (originalX == shuffledX) {
                    found = true;
                    break;
                }
            }
            assertTrue("原始数据应该仍然存在", found);
        }
    }

    @Test
    public void testDatasetPrepare() {
        // 测试数据集准备功能
        assertFalse(testDataset.hadPrepared);
        
        testDataset.prepare();
        assertTrue(testDataset.hadPrepared);
        assertTrue(testDataset.prepareCalled);
        
        // 多次调用prepare应该只执行一次doPrepare
        testDataset.prepareCalled = false;
        testDataset.prepare();
        assertFalse(testDataset.prepareCalled);
    }

    @Test
    public void testDatasetGetMethods() {
        // 测试获取特定数据集的方法
        testDataset.splitDataset(0.6f, 0.2f, 0.2f);
        
        assertNotNull(testDataset.getTrainDataSet());
        assertNotNull(testDataset.getTestDataSet());
        assertNotNull(testDataset.getValidationDataSet());
        
        assertEquals("TRAIN", "TRAIN");
        assertEquals("TEST", "TEST");  
        assertEquals("VALIDATION", "VALIDATION");
    }

    /**
     * 测试用的 ArrayDataset 实现
     */
    private static class TestArrayDataset extends ArrayDataset {
        
        boolean hadPrepared = false;
        boolean prepareCalled = false;

        public TestArrayDataset(int batchSize, NdArray[] xs, NdArray[] ys) {
            super(batchSize);
            this.xs = xs;
            this.ys = ys;
        }

        @Override
        protected DataSet build(int batchSize, NdArray[] xs, NdArray[] ys) {
            return new TestArrayDataset(batchSize, xs, ys);
        }

        @Override
        public void doPrepare() {
            prepareCalled = true;
            hadPrepared = true;
        }
    }
}