package io.leavesfly.tinyai.ml.parallel;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 梯度聚合器 - 用于多线程训练中收集和平均梯度
 * <p>
 * 支持多个线程并发提交梯度，自动进行梯度平均，确保线程安全
 */
public class GradientAggregator {

    private final Map<String, NdArray> accumulatedGradients;
    private final AtomicInteger submissionCount;
    private final int expectedSubmissions;
    private final ReentrantLock aggregationLock;
    private final Condition gradientReady;
    private volatile boolean isReady;

    /**
     * 构造梯度聚合器
     *
     * @param expectedSubmissions 期望的梯度提交次数（通常等于并行线程数）
     */
    public GradientAggregator(int expectedSubmissions) {
        this.expectedSubmissions = expectedSubmissions;
        this.accumulatedGradients = new ConcurrentHashMap<>();
        this.submissionCount = new AtomicInteger(0);
        this.aggregationLock = new ReentrantLock();
        this.gradientReady = aggregationLock.newCondition();
        this.isReady = false;
    }

    /**
     * 提交一个线程计算的梯度
     *
     * @param gradients 参数名到梯度的映射
     */
    public void submitGradients(Map<String, Parameter> gradients) {
        aggregationLock.lock();
        try {
            // 累加梯度
            for (Map.Entry<String, Parameter> entry : gradients.entrySet()) {
                String paramName = entry.getKey();
                NdArray gradient = entry.getValue().getGrad();

                if (gradient != null) {
                    accumulatedGradients.merge(paramName, gradient, NdArray::add);
                }
            }

            // 检查是否收集完所有梯度
            if (submissionCount.incrementAndGet() >= expectedSubmissions) {
                // 计算平均梯度
                for (Map.Entry<String, NdArray> entry : accumulatedGradients.entrySet()) {
                    NdArray averageGrad = entry.getValue().divNum((float) expectedSubmissions);
                    entry.setValue(averageGrad);
                }
                isReady = true;
                gradientReady.signalAll(); // 通知等待的线程
            }
        } finally {
            aggregationLock.unlock();
        }
    }

    /**
     * 等待所有梯度收集完成并返回平均梯度
     *
     * @return 平均后的梯度映射
     * @throws InterruptedException 如果等待被中断
     */
    public Map<String, NdArray> getAverageGradients() throws InterruptedException {
        aggregationLock.lock();
        try {
            while (!isReady) {
                gradientReady.await();
            }
            return new ConcurrentHashMap<>(accumulatedGradients);
        } finally {
            aggregationLock.unlock();
        }
    }

    /**
     * 重置聚合器，准备下一轮梯度收集
     */
    public void reset() {
        aggregationLock.lock();
        try {
            accumulatedGradients.clear();
            submissionCount.set(0);
            isReady = false;
        } finally {
            aggregationLock.unlock();
        }
    }

    /**
     * 检查是否已经收集完所有梯度
     *
     * @return true 如果梯度已准备就绪
     */
    public boolean isReady() {
        return isReady;
    }

    /**
     * 获取当前已提交的梯度数量
     *
     * @return 已提交的梯度数量
     */
    public int getSubmissionCount() {
        return submissionCount.get();
    }
}