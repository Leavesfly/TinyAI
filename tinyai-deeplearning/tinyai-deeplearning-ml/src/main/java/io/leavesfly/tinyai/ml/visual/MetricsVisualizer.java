package io.leavesfly.tinyai.ml.visual;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 指标可视化工具，使用 JFreeChart 绘制训练过程中的各种指标曲线，支持：
 * - 损失曲线(Loss Curve)
 * - 准确率曲线(Accuracy Curve)
 * - 学习率曲线(Learning Rate Curve)
 * - 自定义指标曲线
 * - 多曲线对比
 * - 图表导出
 *
 * @author TinyDL
 * @version 2.0
 */
public class MetricsVisualizer {

    private final Map<String, XYSeries> seriesMap;
    private final Map<String, JFreeChart> chartMap;
    
    /**
     * 默认图表配置
     */
    private int chartWidth = 800;
    private int chartHeight = 600;
    private boolean showLegend = true;
    private boolean showGrid = true;

    /**
     * 构造函数
     */
    public MetricsVisualizer() {
        this.seriesMap = new HashMap<>();
        this.chartMap = new HashMap<>();
    }

    /**
     * 添加数据点
     *
     * @param seriesName 序列名称
     * @param x          X轴值(通常是epoch或step)
     * @param y          Y轴值(指标值)
     */
    public void addDataPoint(String seriesName, double x, double y) {
        XYSeries series = seriesMap.computeIfAbsent(seriesName, XYSeries::new);
        series.add(x, y);
    }

    /**
     * 批量添加数据点
     *
     * @param seriesName 序列名称
     * @param xValues    X轴值列表
     * @param yValues    Y轴值列表
     */
    public void addDataPoints(String seriesName, List<Double> xValues, List<Double> yValues) {
        if (xValues.size() != yValues.size()) {
            throw new IllegalArgumentException("xValues and yValues must have the same size");
        }
        
        XYSeries series = seriesMap.computeIfAbsent(seriesName, XYSeries::new);
        for (int i = 0; i < xValues.size(); i++) {
            series.add(xValues.get(i), yValues.get(i));
        }
    }

    /**
     * 从TrainingMonitor导入数据
     *
     * @param monitor    训练监控器
     * @param metricName 指标名称
     */
    public void importFromMonitor(TrainingMonitor monitor, String metricName) {
        List<Double> history = monitor.getMetricHistory(metricName);
        XYSeries series = seriesMap.computeIfAbsent(metricName, XYSeries::new);
        
        for (int i = 0; i < history.size(); i++) {
            series.add(i + 1, history.get(i));  // epoch从1开始
        }
    }

    /**
     * 创建损失曲线图
     *
     * @param title       图表标题
     * @param lossSeriesNames 损失序列名称列表(如"train_loss", "val_loss")
     * @return JFreeChart对象
     */
    public JFreeChart createLossChart(String title, String... lossSeriesNames) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        for (String seriesName : lossSeriesNames) {
            if (seriesMap.containsKey(seriesName)) {
                dataset.addSeries(seriesMap.get(seriesName));
            }
        }
        
        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            "Epoch",
            "Loss",
            dataset,
            PlotOrientation.VERTICAL,
            showLegend,
            true,
            false
        );
        
        customizeChart(chart);
        chartMap.put(title, chart);
        return chart;
    }

    /**
     * 创建准确率曲线图
     *
     * @param title       图表标题
     * @param accSeriesNames 准确率序列名称列表
     * @return JFreeChart对象
     */
    public JFreeChart createAccuracyChart(String title, String... accSeriesNames) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        for (String seriesName : accSeriesNames) {
            if (seriesMap.containsKey(seriesName)) {
                dataset.addSeries(seriesMap.get(seriesName));
            }
        }
        
        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            "Epoch",
            "Accuracy",
            dataset,
            PlotOrientation.VERTICAL,
            showLegend,
            true,
            false
        );
        
        customizeChart(chart);
        chartMap.put(title, chart);
        return chart;
    }

    /**
     * 创建学习率曲线图
     *
     * @param title      图表标题
     * @param seriesName 学习率序列名称
     * @return JFreeChart对象
     */
    public JFreeChart createLearningRateChart(String title, String seriesName) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        if (seriesMap.containsKey(seriesName)) {
            dataset.addSeries(seriesMap.get(seriesName));
        }
        
        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            "Epoch",
            "Learning Rate",
            dataset,
            PlotOrientation.VERTICAL,
            showLegend,
            true,
            false
        );
        
        customizeChart(chart);
        chartMap.put(title, chart);
        return chart;
    }

    /**
     * 创建通用指标曲线图
     *
     * @param title       图表标题
     * @param xLabel      X轴标签
     * @param yLabel      Y轴标签
     * @param seriesNames 序列名称列表
     * @return JFreeChart对象
     */
    public JFreeChart createChart(String title, String xLabel, String yLabel, String... seriesNames) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        for (String seriesName : seriesNames) {
            if (seriesMap.containsKey(seriesName)) {
                dataset.addSeries(seriesMap.get(seriesName));
            }
        }
        
        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            xLabel,
            yLabel,
            dataset,
            PlotOrientation.VERTICAL,
            showLegend,
            true,
            false
        );
        
        customizeChart(chart);
        chartMap.put(title, chart);
        return chart;
    }

    /**
     * 显示图表窗口
     *
     * @param chart 图表对象
     */
    public void showChart(JFreeChart chart) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame(chart.getTitle().getText());
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(chartWidth, chartHeight));
            frame.setContentPane(chartPanel);
            
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }

    /**
     * 显示所有图表
     */
    public void showAllCharts() {
        for (JFreeChart chart : chartMap.values()) {
            showChart(chart);
        }
    }

    /**
     * 保存图表到文件
     *
     * @param chart    图表对象
     * @param filePath 文件路径(PNG格式)
     * @throws IOException 保存失败时抛出
     */
    public void saveChart(JFreeChart chart, String filePath) throws IOException {
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        ChartUtilities.saveChartAsPNG(file, chart, chartWidth, chartHeight);
        System.out.println("Chart saved to: " + filePath);
    }

    /**
     * 保存所有图表
     *
     * @param directory 保存目录
     * @throws IOException 保存失败时抛出
     */
    public void saveAllCharts(String directory) throws IOException {
        File dir = new File(directory);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        
        int index = 0;
        for (Map.Entry<String, JFreeChart> entry : chartMap.entrySet()) {
            String fileName = sanitizeFileName(entry.getKey()) + ".png";
            String filePath = new File(dir, fileName).getAbsolutePath();
            saveChart(entry.getValue(), filePath);
            index++;
        }
        
        System.out.println("Saved " + index + " charts to: " + directory);
    }

    /**
     * 清空所有数据
     */
    public void clear() {
        seriesMap.clear();
        chartMap.clear();
    }

    // ==================== 配置方法 ====================

    public MetricsVisualizer setChartSize(int width, int height) {
        this.chartWidth = width;
        this.chartHeight = height;
        return this;
    }

    public MetricsVisualizer setShowLegend(boolean show) {
        this.showLegend = show;
        return this;
    }

    public MetricsVisualizer setShowGrid(boolean show) {
        this.showGrid = show;
        return this;
    }

    // ==================== 辅助方法 ====================

    /**
     * 自定义图表样式
     */
    private void customizeChart(JFreeChart chart) {
        XYPlot plot = chart.getXYPlot();
        
        // 设置网格线
        if (showGrid) {
            plot.setDomainGridlinePaint(Color.GRAY);
            plot.setRangeGridlinePaint(Color.GRAY);
        } else {
            plot.setDomainGridlinesVisible(false);
            plot.setRangeGridlinesVisible(false);
        }
        
        // 设置背景色
        plot.setBackgroundPaint(Color.WHITE);
        chart.setBackgroundPaint(Color.WHITE);
        
        // 设置线条渲染器
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        for (int i = 0; i < plot.getDataset().getSeriesCount(); i++) {
            renderer.setSeriesShapesVisible(i, true);
            renderer.setSeriesLinesVisible(i, true);
        }
        plot.setRenderer(renderer);
    }

    /**
     * 清理文件名中的非法字符
     */
    private String sanitizeFileName(String fileName) {
        return fileName.replaceAll("[^a-zA-Z0-9_\\-]", "_");
    }
}
