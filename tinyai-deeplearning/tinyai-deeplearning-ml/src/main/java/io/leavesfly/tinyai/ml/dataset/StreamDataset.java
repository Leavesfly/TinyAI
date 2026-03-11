package io.leavesfly.tinyai.ml.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;
import java.util.function.Supplier;

/**
 * 流式数据集实现
 * 
 * 用于处理无法一次性装载到内存中的大型数据集，支持流式访问数据。
 * 该实现支持：
 * 1. 从数据源流式读取数据
 * 2. 懒加载机制，只在需要时加载数据
 * 3. 数据缓存和预取以提高性能
 * 4. 支持数据分割和打乱
 * 
 * @author TinyDL
 * @version 1.0
 */
public class StreamDataset extends DataSet {

    /**
     * 数据源提供者，用于获取数据流
     */
    private Supplier<Iterator<DataItem>> dataSourceSupplier;
    
    /**
     * 数据集总大小，-1表示未知大小
     */
    private int totalSize = -1;
    
    /**
     * 缓存队列大小
     */
    private final int cacheSize;
    
    /**
     * 是否已经打乱数据
     */
    private boolean shuffled = false;
    
    /**
     * 随机数生成器，用于数据打乱
     */
    private Random random = new Random();
    
    /**
     * 分割配置
     */
    private SplitConfig splitConfig;

    /**
     * 构造函数
     * @param batchSize 批次大小
     */
    public StreamDataset(int batchSize) {
        this(batchSize, 1000); // 默认缓存1000个数据项
    }
    
    /**
     * 构造函数
     * @param batchSize 批次大小
     * @param cacheSize 缓存大小
     */
    public StreamDataset(int batchSize, int cacheSize) {
        super(batchSize);
        this.cacheSize = cacheSize;
    }
    
    /**
     * 设置数据源
     * @param dataSourceSupplier 数据源提供者
     */
    public void setDataSource(Supplier<Iterator<DataItem>> dataSourceSupplier) {
        this.dataSourceSupplier = dataSourceSupplier;
    }
    
    /**
     * 设置数据集总大小
     * @param totalSize 总大小
     */
    public void setTotalSize(int totalSize) {
        this.totalSize = totalSize;
    }

    @Override
    public List<Batch> getBatches() {
        if (dataSourceSupplier == null) {
            throw new IllegalStateException("数据源未设置，请先调用setDataSource方法");
        }

        Iterator<DataItem> dataIterator = dataSourceSupplier.get();
        List<Batch> batches = new ArrayList<>();

        // 复用数组以减少对象创建，最后一个批次不足时会做一次复制
        NdArray[] xs = new NdArray[batchSize];
        NdArray[] ys = new NdArray[batchSize];
        int cursor = 0;

        if (shuffled) {
            // 使用有限缓存窗口做局部洗牌，兼顾随机性与内存占用
            List<DataItem> buffer = new ArrayList<>(cacheSize);
            while (dataIterator.hasNext()) {
                buffer.add(dataIterator.next());
                if (buffer.size() == cacheSize) {
                    Collections.shuffle(buffer, random);
                    for (DataItem item : buffer) {
                        xs[cursor] = item.getX();
                        ys[cursor] = item.getY();
                        cursor++;
                        if (cursor == batchSize) {
                            batches.add(new Batch(xs, ys));
                            xs = new NdArray[batchSize];
                            ys = new NdArray[batchSize];
                            cursor = 0;
                        }
                    }
                    buffer.clear();
                }
            }

            if (!buffer.isEmpty()) {
                Collections.shuffle(buffer, random);
                for (DataItem item : buffer) {
                    xs[cursor] = item.getX();
                    ys[cursor] = item.getY();
                    cursor++;
                    if (cursor == batchSize) {
                        batches.add(new Batch(xs, ys));
                        xs = new NdArray[batchSize];
                        ys = new NdArray[batchSize];
                        cursor = 0;
                    }
                }
            }
        } else {
            while (dataIterator.hasNext()) {
                DataItem item = dataIterator.next();
                xs[cursor] = item.getX();
                ys[cursor] = item.getY();
                cursor++;

                if (cursor == batchSize) {
                    batches.add(new Batch(xs, ys));
                    xs = new NdArray[batchSize];
                    ys = new NdArray[batchSize];
                    cursor = 0;
                }
            }
        }

        if (cursor > 0) {
            batches.add(createBatch(xs, ys, cursor));
        }

        return batches;
    }
    
    /**
     * 将累积的数组封装为批次
     * @param xs 输入数据
     * @param ys 输出数据
     * @param size 实际长度
     * @return 批次对象
     */
    private Batch createBatch(NdArray[] xs, NdArray[] ys, int size) {
        if (size == xs.length) {
            return new Batch(xs, ys);
        }
        // 复制有效区间，避免后续写入污染Batch数据
        return new Batch(Arrays.copyOf(xs, size), Arrays.copyOf(ys, size));
    }

    @Override
    public void doPrepare() {
        // 流式数据集的准备工作主要是验证数据源
        if (dataSourceSupplier == null) {
            throw new IllegalStateException("数据源未设置，请先调用setDataSource方法");
        }
        
        // 可以在这里进行一些预处理，比如统计数据集大小（如果未知的话）
        if (totalSize == -1) {
            try {
                totalSize = estimateSize();
            } catch (Exception e) {
                // 如果无法估算大小，保持-1
                System.out.println("警告: 无法估算数据集大小: " + e.getMessage());
            }
        }
    }
    
    /**
     * 估算数据集大小
     * @return 估算的大小
     */
    private int estimateSize() {
        // 通过遍历一次数据源来统计大小
        // 注意：这个操作可能比较耗时，实际使用时可能需要更高效的方法
        Iterator<DataItem> iterator = dataSourceSupplier.get();
        int count = 0;
        while (iterator.hasNext()) {
            iterator.next();
            count++;
        }
        return count;
    }

    @Override
    public void shuffle() {
        // 对于流式数据集，无法直接打乱所有数据
        // 这里实现一个简单的随机化策略：设置随机种子，在数据读取时进行随机采样
        this.shuffled = true;
        this.random = new Random(System.currentTimeMillis());
    }

    @Override
    public Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validationRatio) {
        if (Math.abs(trainRatio + testRatio + validationRatio - 1.0f) > 1e-6f) {
            throw new IllegalArgumentException("数据集分割比例之和必须等于1.0");
        }
        
        if (!splitDatasetMap.isEmpty()) {
            return splitDatasetMap;
        }
        
        // 保存分割配置，在数据读取时使用
        this.splitConfig = new SplitConfig(trainRatio, testRatio, validationRatio);
        
        // 创建三个子数据集
        StreamDataset trainDataset = createSubDataset(Usage.TRAIN);
        StreamDataset testDataset = createSubDataset(Usage.TEST);
        StreamDataset validationDataset = createSubDataset(Usage.VALIDATION);
        
        splitDatasetMap.put(Usage.TRAIN.name(), trainDataset);
        splitDatasetMap.put(Usage.TEST.name(), testDataset);
        splitDatasetMap.put(Usage.VALIDATION.name(), validationDataset);
        
        return splitDatasetMap;
    }
    
    /**
     * 创建子数据集
     * @param usage 数据集用途
     * @return 子数据集
     */
    private StreamDataset createSubDataset(Usage usage) {
        StreamDataset subDataset = new StreamDataset(batchSize, cacheSize);
        
        // 设置过滤后的数据源
        subDataset.setDataSource(() -> new FilteredIterator(dataSourceSupplier.get(), usage));
        
        // 估算子数据集大小
        if (totalSize > 0) {
            int subSize;
            switch (usage) {
                case TRAIN:
                    subSize = (int) (totalSize * splitConfig.trainRatio);
                    break;
                case TEST:
                    subSize = (int) (totalSize * splitConfig.testRatio);
                    break;
                case VALIDATION:
                    subSize = (int) (totalSize * splitConfig.validationRatio);
                    break;
                default:
                    subSize = 0;
                    break;
            }
            subDataset.setTotalSize(subSize);
        }
        
        return subDataset;
    }

    @Override
    public int getSize() {
        return totalSize;
    }
    
    /**
     * 数据项，包含输入和输出
     */
    public static class DataItem {
        private final NdArray x;
        private final NdArray y;
        
        public DataItem(NdArray x, NdArray y) {
            this.x = x;
            this.y = y;
        }
        
        public NdArray getX() {
            return x;
        }
        
        public NdArray getY() {
            return y;
        }
    }
    
    /**
     * 分割配置
     */
    private static class SplitConfig {
        final float trainRatio;
        final float testRatio;
        final float validationRatio;
        
        SplitConfig(float trainRatio, float testRatio, float validationRatio) {
            this.trainRatio = trainRatio;
            this.testRatio = testRatio;
            this.validationRatio = validationRatio;
        }
    }
    
    /**
     * 过滤迭代器，用于数据集分割
     */
    private class FilteredIterator implements Iterator<DataItem> {
        private final Iterator<DataItem> sourceIterator;
        private final Usage targetUsage;
        private DataItem nextItem;
        private int currentIndex = 0;
        
        FilteredIterator(Iterator<DataItem> sourceIterator, Usage targetUsage) {
            this.sourceIterator = sourceIterator;
            this.targetUsage = targetUsage;
            findNext();
        }
        
        @Override
        public boolean hasNext() {
            return nextItem != null;
        }
        
        @Override
        public DataItem next() {
            if (nextItem == null) {
                throw new NoSuchElementException();
            }
            DataItem current = nextItem;
            findNext();
            return current;
        }
        
        private void findNext() {
            nextItem = null;
            while (sourceIterator.hasNext()) {
                DataItem item = sourceIterator.next();
                if (belongsToUsage(currentIndex, targetUsage)) {
                    nextItem = item;
                    currentIndex++;
                    return;
                }
                currentIndex++;
            }
        }
        
        /**
         * 判断索引位置的数据是否属于指定用途
         */
        private boolean belongsToUsage(int index, Usage usage) {
            if (splitConfig == null || totalSize <= 0) {
                return usage == Usage.TRAIN; // 默认都是训练数据
            }
            
            float position = (float) index / totalSize;
            switch (usage) {
                case TRAIN:
                    return position < splitConfig.trainRatio;
                case TEST:
                    return position >= splitConfig.trainRatio && 
                           position < splitConfig.trainRatio + splitConfig.testRatio;
                case VALIDATION:
                    return position >= splitConfig.trainRatio + splitConfig.testRatio;
                default:
                    return false;
            }
        }
    }
}