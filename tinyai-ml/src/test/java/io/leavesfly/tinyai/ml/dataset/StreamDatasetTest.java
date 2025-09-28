package io.leavesfly.tinyai.ml.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * StreamDataset 单元测试
 * 
 * 测试流式数据集的功能
 * 
 * @author TinyDL
 * @version 1.0
 */
public class StreamDatasetTest {

    private TestStreamDataset streamDataset;
    private String testData;

    @Before
    public void setUp() {
        // 创建测试数据
        testData = "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n10.0,11.0,12.0\n";
        streamDataset = new TestStreamDataset(2, testData);
    }

    @Test
    public void testStreamDatasetCreation() {
        // 测试StreamDataset基本创建
        assertNotNull(streamDataset);
        assertEquals(2, streamDataset.batchSize);
    }

    @Test
    public void testGetBatches() {
        // 测试获取批次数据
        streamDataset.prepare();
        List<Batch> batches = streamDataset.getBatches();
        
        assertNotNull(batches);
        assertFalse(batches.isEmpty());
        
        // 验证每个批次的大小
        for (Batch batch : batches) {
            assertTrue(batch.getSize() <= 2); // 批次大小不超过设定值
            assertNotNull(batch.getX());
            assertNotNull(batch.getY());
        }
    }

    @Test
    public void testPrepare() {
        // 测试数据准备功能
        assertFalse(streamDataset.isPrepared());
        
        streamDataset.prepare();
        
        assertTrue(streamDataset.isPrepared());
        assertTrue(streamDataset.prepareCalled);
    }

    @Test
    public void testMultiplePrepare() {
        // 测试多次准备不会重复执行
        streamDataset.prepare();
        streamDataset.prepareCalled = false;
        
        streamDataset.prepare();
        
        assertFalse(streamDataset.prepareCalled); // 不应该再次调用doPrepare
    }

    @Test
    public void testGetSize() {
        // 测试获取数据集大小
        streamDataset.prepare();
        int size = streamDataset.getSize();
        
        assertTrue(size >= 0);
        assertEquals(4, size); // 测试数据有4行
    }

    @Test
    public void testShuffle() {
        // 测试数据打乱功能
        streamDataset.prepare();
        
        // 获取原始数据
        List<Batch> originalBatches = streamDataset.getBatches();
        int originalSize = originalBatches.size();
        
        // 执行shuffle
        streamDataset.shuffle();
        
        // 验证数据仍然存在
        List<Batch> shuffledBatches = streamDataset.getBatches();
        assertEquals(originalSize, shuffledBatches.size());
    }

    @Test
    public void testSplitDataset() {
        // 测试数据集分割
        streamDataset.prepare();
        
        Map<String, DataSet> splitDatasets = streamDataset.splitDataset(0.5f, 0.3f, 0.2f);
        
        assertNotNull(splitDatasets);
        assertEquals(3, splitDatasets.size());
        
        assertTrue(splitDatasets.containsKey("TRAIN"));
        assertTrue(splitDatasets.containsKey("TEST"));
        assertTrue(splitDatasets.containsKey("VALIDATION"));
        
        // 验证分割后的数据集不为空
        DataSet trainDataset = splitDatasets.get("TRAIN");
        DataSet testDataset = splitDatasets.get("TEST");
        DataSet validationDataset = splitDatasets.get("VALIDATION");
        
        assertNotNull(trainDataset);
        assertNotNull(testDataset);
        assertNotNull(validationDataset);
        
        assertTrue(trainDataset.getSize() >= 0);
        assertTrue(testDataset.getSize() >= 0);
        assertTrue(validationDataset.getSize() >= 0);
    }

    @Test
    public void testSplitDatasetInvalidRatio() {
        // 测试无效的分割比例
        streamDataset.prepare();
        
        try {
            streamDataset.splitDataset(0.6f, 0.3f, 0.2f); // 总和 > 1.0
            fail("应该抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("数据集分割比例之和必须等于1.0"));
        }
    }

    @Test
    public void testEmptyStream() {
        // 测试空数据流
        TestStreamDataset emptyDataset = new TestStreamDataset(2, "");
        emptyDataset.prepare();
        
        assertEquals(0, emptyDataset.getSize());
        List<Batch> batches = emptyDataset.getBatches();
        assertNotNull(batches);
        assertTrue(batches.isEmpty());
    }

    @Test
    public void testSingleLineStream() {
        // 测试单行数据流
        TestStreamDataset singleLineDataset = new TestStreamDataset(2, "1.0,2.0,3.0\n");
        singleLineDataset.prepare();
        
        assertEquals(1, singleLineDataset.getSize());
        List<Batch> batches = singleLineDataset.getBatches();
        assertEquals(1, batches.size());
        assertEquals(1, batches.get(0).getSize());
    }

    /**
     * 测试用的 StreamDataset 实现
     */
    private static class TestStreamDataset extends StreamDataset {
        
        private String testData;
        boolean prepareCalled = false;
        private boolean prepared = false;
        private List<StreamDataset.DataItem> dataItems;

        public TestStreamDataset(int batchSize, String testData) {
            super(batchSize);
            this.testData = testData;
            this.dataItems = new ArrayList<>();
        }

        @Override
        public void doPrepare() {
            prepareCalled = true;
            prepared = true;
            
            // 解析测试数据
            String[] lines = testData.split("\\n");
            for (String line : lines) {
                if (!line.trim().isEmpty()) {
                    String[] values = line.split(",");
                    if (values.length >= 3) {
                        // 创建输入和输出数据
                        NdArray x = NdArray.of(new float[][]{{
                            Float.parseFloat(values[0]),
                            Float.parseFloat(values[1])
                        }});
                        NdArray y = NdArray.of(new float[][]{{
                            Float.parseFloat(values[2])
                        }});
                        dataItems.add(new StreamDataset.DataItem(x, y));
                    }
                }
            }
            
            // 设置数据源
            this.setDataSource(() -> dataItems.iterator());
            this.setTotalSize(dataItems.size());
        }

        @Override
        public void shuffle() {
            // 简单的shuffle实现
            if (dataItems != null && dataItems.size() > 1) {
                Collections.shuffle(dataItems);
            }
        }

        public boolean isPrepared() {
            return prepared;
        }
    }
}