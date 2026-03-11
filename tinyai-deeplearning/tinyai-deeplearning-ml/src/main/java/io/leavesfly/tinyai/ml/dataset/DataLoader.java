package io.leavesfly.tinyai.ml.dataset;

import java.util.*;
import java.util.concurrent.*;

/**
 * 数据加载器，PyTorch 风格的 DataLoader 实现。
 *
 * 提供灵活的数据加载能力，支持：
 * - 批次数据加载
 * - 数据打乱（shuffle）
 * - 多线程异步预取（prefetch）
 * - 自定义采样器
 *
 * 使用示例：
 *   DataSet dataset = new ArrayDataset(xs, ys);
 *   DataLoader loader = new DataLoader(dataset).batchSize(32).shuffle(true).numWorkers(4);
 *   for (Batch batch : loader) {
 *       Variable x = batch.toVariableX();
 *       Variable y = batch.toVariableY();
 *       // 训练逻辑...
 *   }
 *
 * @author TinyAI
 * @version 2.0
 */
public class DataLoader implements Iterable<Batch>, AutoCloseable {

    private final DataSet dataset;
    private int batchSize;
    private boolean shuffle;
    private int numWorkers;
    private boolean dropLast;
    private Sampler sampler;
    
    // 多线程预取相关
    private ExecutorService executorService;
    private BlockingQueue<Batch> prefetchQueue;
    private static final int DEFAULT_PREFETCH_CAPACITY = 2;
    
    /**
     * 构造函数
     *
     * @param dataset 数据集
     */
    public DataLoader(DataSet dataset) {
        this.dataset = dataset;
        this.batchSize = dataset.batchSize > 0 ? dataset.batchSize : 32;
        this.shuffle = false;
        this.numWorkers = 0; // 默认单线程
        this.dropLast = false;
        this.sampler = null;
    }

    /**
     * 设置批次大小
     *
     * @param batchSize 批次大小
     * @return 当前DataLoader实例（支持链式调用）
     */
    public DataLoader batchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        this.batchSize = batchSize;
        return this;
    }

    /**
     * 设置是否打乱数据
     *
     * @param shuffle 是否打乱
     * @return 当前DataLoader实例（支持链式调用）
     */
    public DataLoader shuffle(boolean shuffle) {
        this.shuffle = shuffle;
        return this;
    }

    /**
     * 设置工作线程数（用于异步数据加载）
     *
     * @param numWorkers 工作线程数，0表示单线程同步加载
     * @return 当前DataLoader实例（支持链式调用）
     */
    public DataLoader numWorkers(int numWorkers) {
        if (numWorkers < 0) {
            throw new IllegalArgumentException("Number of workers cannot be negative");
        }
        this.numWorkers = numWorkers;
        return this;
    }

    /**
     * 设置是否丢弃最后一个不完整的批次
     *
     * @param dropLast 是否丢弃
     * @return 当前DataLoader实例（支持链式调用）
     */
    public DataLoader dropLast(boolean dropLast) {
        this.dropLast = dropLast;
        return this;
    }

    /**
     * 设置自定义采样器
     *
     * @param sampler 采样器
     * @return 当前DataLoader实例（支持链式调用）
     */
    public DataLoader sampler(Sampler sampler) {
        this.sampler = sampler;
        return this;
    }

    /**
     * 获取迭代器
     *
     * @return 批次迭代器
     */
    @Override
    public Iterator<Batch> iterator() {
        // 如果使用多线程，返回异步迭代器
        if (numWorkers > 0) {
            return new PrefetchIterator();
        } else {
            return new SyncIterator();
        }
    }

    /**
     * 同步迭代器（单线程）
     */
    private class SyncIterator implements Iterator<Batch> {
        private final List<Batch> batches;
        private int currentIndex;

        public SyncIterator() {
            // 准备数据
            if (shuffle) {
                dataset.shuffle();
            }
            
            // 获取批次数据
            this.batches = dataset.getBatches();
            
            // 如果需要丢弃最后一个不完整批次
            if (dropLast && !batches.isEmpty()) {
                Batch lastBatch = batches.get(batches.size() - 1);
                if (lastBatch.getSize() < batchSize) {
                    batches.remove(batches.size() - 1);
                }
            }
            
            this.currentIndex = 0;
        }

        @Override
        public boolean hasNext() {
            return currentIndex < batches.size();
        }

        @Override
        public Batch next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return batches.get(currentIndex++);
        }
    }

    /**
     * 异步预取迭代器（多线程）
     */
    private class PrefetchIterator implements Iterator<Batch> {
        private final List<Batch> batches;
        private int currentIndex;
        private volatile boolean isShutdown;
        private final CountDownLatch producerLatch;

        public PrefetchIterator() {
            // 准备数据
            if (shuffle) {
                dataset.shuffle();
            }
            
            this.batches = dataset.getBatches();
            
            // 如果需要丢弃最后一个不完整批次
            if (dropLast && !batches.isEmpty()) {
                Batch lastBatch = batches.get(batches.size() - 1);
                if (lastBatch.getSize() < batchSize) {
                    batches.remove(batches.size() - 1);
                }
            }
            
            this.currentIndex = 0;
            this.isShutdown = false;
            this.producerLatch = new CountDownLatch(1);
            
            // 初始化预取队列
            prefetchQueue = new LinkedBlockingQueue<>(DEFAULT_PREFETCH_CAPACITY);
            
            // 启动预取线程
            executorService = Executors.newFixedThreadPool(numWorkers);
            startPrefetching();
        }

        private void startPrefetching() {
            executorService.submit(() -> {
                try {
                    for (Batch batch : batches) {
                        if (isShutdown) {
                            break;
                        }
                        // 阻塞式添加到队列
                        prefetchQueue.put(batch);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    producerLatch.countDown();
                }
            });
        }

        @Override
        public boolean hasNext() {
            return currentIndex < batches.size();
        }

        @Override
        public Batch next() {
            if (!hasNext()) {
                shutdown();
                throw new NoSuchElementException();
            }
            
            try {
                Batch batch = prefetchQueue.poll(10, TimeUnit.SECONDS);
                if (batch == null) {
                    throw new RuntimeException("Timeout waiting for batch data");
                }
                currentIndex++;
                
                // 如果是最后一个batch，关闭资源
                if (!hasNext()) {
                    shutdown();
                }
                
                return batch;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while waiting for batch", e);
            }
        }

        private void shutdown() {
            if (!isShutdown) {
                isShutdown = true;
                if (executorService != null) {
                    executorService.shutdownNow();
                    try {
                        if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                            System.err.println("Warning: DataLoader executor did not terminate in time");
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        }
    }

    /**
     * 关闭DataLoader，释放资源
     */
    @Override
    public void close() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdownNow();
        }
    }

    /**
     * 获取批次数量
     *
     * @return 批次数量
     */
    public int getNumBatches() {
        int datasetSize = dataset.getSize();
        if (dropLast) {
            return datasetSize / batchSize;
        } else {
            return (datasetSize + batchSize - 1) / batchSize;
        }
    }

    /**
     * 采样器接口
     */
    public interface Sampler {
        /**
         * 生成索引序列
         *
         * @param datasetSize 数据集大小
         * @return 索引列表
         */
        List<Integer> sample(int datasetSize);
    }

    /**
     * 随机采样器
     */
    public static class RandomSampler implements Sampler {
        @Override
        public List<Integer> sample(int datasetSize) {
            List<Integer> indices = new ArrayList<>(datasetSize);
            for (int i = 0; i < datasetSize; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices);
            return indices;
        }
    }

    /**
     * 顺序采样器
     */
    public static class SequentialSampler implements Sampler {
        @Override
        public List<Integer> sample(int datasetSize) {
            List<Integer> indices = new ArrayList<>(datasetSize);
            for (int i = 0; i < datasetSize; i++) {
                indices.add(i);
            }
            return indices;
        }
    }
}
