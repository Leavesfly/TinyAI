package io.leavesfly.tinyai.ml.visual;

import java.util.*;

/**
 * 训练监控器，实时记录和展示训练过程中的关键指标，支持：
 * - 损失值追踪
 * - 准确率追踪
 * - 学习率追踪
 * - 自定义指标追踪
 * - 实时控制台输出
 * - 历史数据记录
 *
 * @author TinyDL
 * @version 2.0
 */
public class TrainingMonitor {

    /**
     * 指标记录类
     */
    public static class MetricRecord {
        private final int epoch;
        private final int step;
        private final String metricName;
        private final double value;
        private final long timestamp;

        public MetricRecord(int epoch, int step, String metricName, double value) {
            this.epoch = epoch;
            this.step = step;
            this.metricName = metricName;
            this.value = value;
            this.timestamp = System.currentTimeMillis();
        }

        public int getEpoch() { return epoch; }
        public int getStep() { return step; }
        public String getMetricName() { return metricName; }
        public double getValue() { return value; }
        public long getTimestamp() { return timestamp; }
    }

    /**
     * Epoch统计信息
     */
    public static class EpochStats {
        private final int epoch;
        private final Map<String, Double> metrics;
        private final long duration;

        public EpochStats(int epoch, Map<String, Double> metrics, long duration) {
            this.epoch = epoch;
            this.metrics = new HashMap<>(metrics);
            this.duration = duration;
        }

        public int getEpoch() { return epoch; }
        public Map<String, Double> getMetrics() { return metrics; }
        public long getDuration() { return duration; }
        
        public double getMetric(String name) {
            return metrics.getOrDefault(name, Double.NaN);
        }
    }

    private final String monitorName;
    private final List<MetricRecord> records;
    private final Map<String, List<Double>> metricHistory;
    private final List<EpochStats> epochHistory;
    
    private int currentEpoch = 0;
    private int currentStep = 0;
    private long epochStartTime = 0;
    private Map<String, Double> epochMetrics;
    
    private boolean enableConsoleOutput = true;
    private int printInterval = 1;  // 每多少个step打印一次
    
    /**
     * 构造函数
     *
     * @param monitorName 监控器名称
     */
    public TrainingMonitor(String monitorName) {
        this.monitorName = monitorName;
        this.records = new ArrayList<>();
        this.metricHistory = new LinkedHashMap<>();
        this.epochHistory = new ArrayList<>();
        this.epochMetrics = new HashMap<>();
    }

    /**
     * 默认构造函数
     */
    public TrainingMonitor() {
        this("TrainingMonitor");
    }

    /**
     * 开始新的epoch
     *
     * @param epoch epoch编号
     */
    public void startEpoch(int epoch) {
        this.currentEpoch = epoch;
        this.currentStep = 0;
        this.epochStartTime = System.currentTimeMillis();
        this.epochMetrics = new HashMap<>();
        
        if (enableConsoleOutput) {
            System.out.println("\n" + "=".repeat(60));
            System.out.println("Epoch " + epoch + " started");
            System.out.println("=".repeat(60));
        }
    }

    /**
     * 结束当前epoch
     */
    public void endEpoch() {
        long duration = System.currentTimeMillis() - epochStartTime;
        EpochStats stats = new EpochStats(currentEpoch, epochMetrics, duration);
        epochHistory.add(stats);
        
        if (enableConsoleOutput) {
            System.out.println("-".repeat(60));
            System.out.println("Epoch " + currentEpoch + " finished in " + formatDuration(duration));
            System.out.println("Metrics: " + formatMetrics(epochMetrics));
            System.out.println("=".repeat(60));
        }
    }

    /**
     * 记录step级别的指标
     *
     * @param step       step编号
     * @param metricName 指标名称
     * @param value      指标值
     */
    public void recordStep(int step, String metricName, double value) {
        this.currentStep = step;
        MetricRecord record = new MetricRecord(currentEpoch, step, metricName, value);
        records.add(record);
        
        // 更新历史记录
        metricHistory.computeIfAbsent(metricName, k -> new ArrayList<>()).add(value);
        
        // 更新epoch累计指标
        epochMetrics.merge(metricName, value, (old, newVal) -> (old + newVal) / 2);
        
        // 控制台输出
        if (enableConsoleOutput && step % printInterval == 0) {
            System.out.printf("[Epoch %d, Step %d] %s: %.6f%n", 
                currentEpoch, step, metricName, value);
        }
    }

    /**
     * 记录多个指标
     *
     * @param step    step编号
     * @param metrics 指标映射
     */
    public void recordStep(int step, Map<String, Double> metrics) {
        for (Map.Entry<String, Double> entry : metrics.entrySet()) {
            recordStep(step, entry.getKey(), entry.getValue());
        }
    }

    /**
     * 记录epoch级别的指标
     *
     * @param metricName 指标名称
     * @param value      指标值
     */
    public void recordEpochMetric(String metricName, double value) {
        epochMetrics.put(metricName, value);
        metricHistory.computeIfAbsent(metricName, k -> new ArrayList<>()).add(value);
        
        if (enableConsoleOutput) {
            System.out.printf("[Epoch %d] %s: %.6f%n", currentEpoch, metricName, value);
        }
    }

    /**
     * 获取指定指标的历史记录
     *
     * @param metricName 指标名称
     * @return 历史值列表
     */
    public List<Double> getMetricHistory(String metricName) {
        return metricHistory.getOrDefault(metricName, new ArrayList<>());
    }

    /**
     * 获取所有epoch统计信息
     *
     * @return epoch统计列表
     */
    public List<EpochStats> getEpochHistory() {
        return new ArrayList<>(epochHistory);
    }

    /**
     * 获取最佳指标值
     *
     * @param metricName 指标名称
     * @param minimize   是否求最小值(true=求最小,false=求最大)
     * @return 最佳值
     */
    public double getBestMetric(String metricName, boolean minimize) {
        List<Double> history = getMetricHistory(metricName);
        if (history.isEmpty()) {
            return Double.NaN;
        }
        
        return minimize ? 
            history.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN) :
            history.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
    }

    /**
     * 获取当前指标值
     *
     * @param metricName 指标名称
     * @return 当前值
     */
    public double getCurrentMetric(String metricName) {
        return epochMetrics.getOrDefault(metricName, Double.NaN);
    }

    /**
     * 清空所有记录
     */
    public void clear() {
        records.clear();
        metricHistory.clear();
        epochHistory.clear();
        epochMetrics.clear();
        currentEpoch = 0;
        currentStep = 0;
    }

    /**
     * 打印训练总结
     */
    public void printSummary() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Training Summary - " + monitorName);
        System.out.println("=".repeat(60));
        System.out.println("Total Epochs: " + epochHistory.size());
        
        if (!metricHistory.isEmpty()) {
            System.out.println("\nMetrics:");
            for (String metricName : metricHistory.keySet()) {
                List<Double> history = metricHistory.get(metricName);
                if (!history.isEmpty()) {
                    double min = history.stream().mapToDouble(Double::doubleValue).min().orElse(0);
                    double max = history.stream().mapToDouble(Double::doubleValue).max().orElse(0);
                    double avg = history.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                    double last = history.get(history.size() - 1);
                    
                    System.out.printf("  %s: Last=%.6f, Min=%.6f, Max=%.6f, Avg=%.6f%n",
                        metricName, last, min, max, avg);
                }
            }
        }
        System.out.println("=".repeat(60));
    }

    /**
     * 导出为CSV格式
     *
     * @return CSV字符串
     */
    public String toCSV() {
        StringBuilder sb = new StringBuilder();
        
        // 表头
        sb.append("Epoch,Step,MetricName,Value,Timestamp\n");
        
        // 数据行
        for (MetricRecord record : records) {
            sb.append(record.getEpoch()).append(",");
            sb.append(record.getStep()).append(",");
            sb.append(record.getMetricName()).append(",");
            sb.append(record.getValue()).append(",");
            sb.append(record.getTimestamp()).append("\n");
        }
        
        return sb.toString();
    }

    // ==================== Getter/Setter ====================

    public void setEnableConsoleOutput(boolean enable) {
        this.enableConsoleOutput = enable;
    }

    public void setPrintInterval(int interval) {
        this.printInterval = Math.max(1, interval);
    }

    public String getMonitorName() {
        return monitorName;
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public int getCurrentStep() {
        return currentStep;
    }

    // ==================== 辅助方法 ====================

    private String formatDuration(long milliseconds) {
        long seconds = milliseconds / 1000;
        if (seconds < 60) {
            return seconds + "s";
        } else if (seconds < 3600) {
            return (seconds / 60) + "m " + (seconds % 60) + "s";
        } else {
            return (seconds / 3600) + "h " + ((seconds % 3600) / 60) + "m";
        }
    }

    private String formatMetrics(Map<String, Double> metrics) {
        if (metrics.isEmpty()) {
            return "{}";
        }
        
        StringBuilder sb = new StringBuilder("{");
        int count = 0;
        for (Map.Entry<String, Double> entry : metrics.entrySet()) {
            if (count > 0) sb.append(", ");
            sb.append(entry.getKey()).append("=").append(String.format("%.6f", entry.getValue()));
            count++;
        }
        sb.append("}");
        return sb.toString();
    }
}
