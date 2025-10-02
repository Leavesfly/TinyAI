package io.leavesfly.tinyai.ml;

import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * 绘图工具类
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * Plot类提供了数据可视化功能，基于JFreeChart库实现。
 * 支持绘制散点图和折线图，用于展示训练数据和模型结果。
 */
public class Plot {

    /**
     * X轴坐标轴
     */
    private final NumberAxis xAxis = new NumberAxis("X");

    /**
     * Y轴坐标轴
     */
    private final NumberAxis yAxis = new NumberAxis("Y");

    /**
     * XY图绘制区域
     */
    private XYPlot plot;

    /**
     * 颜色索引，用于为不同的数据系列分配不同颜色
     */
    private int index = 0;

    /**
     * 颜色映射表，将索引映射到具体颜色
     */
    private Map<Integer, Color> ColorMap = new HashMap<Integer, Color>() {
        {
            put(0, Color.RED);
            put(1, Color.BLACK);
            put(2, Color.ORANGE);
            put(3, Color.PINK);
            put(4, Color.GREEN);
            put(5, Color.BLUE);
            put(6, Color.YELLOW);
        }
    };

    /**
     * 绘制散点图
     *
     * @param x X坐标数组
     * @param y Y坐标数组
     */
    public void scatter(float[] x, float[] y) {

        XYDataset dataset = createDataset(x, y, "scatter");
        XYItemRenderer scatterRenderer = new XYLineAndShapeRenderer(false, true);
        scatterRenderer.setSeriesPaint(0, ColorMap.get(index));
        if (plot == null) {
            plot = new XYPlot(dataset, xAxis, yAxis, scatterRenderer);
            index++;
        } else {
            plot.setRenderer(index, scatterRenderer);
            plot.setDataset(index, dataset);
            index++;
        }
    }


    /**
     * 绘制数据集中的散点图
     *
     * @param _dataset 数据集对象
     * @param types    要绘制的数据类型数组
     */
    public void scatter(ArrayDataset _dataset, int[] types) {
        if (plot == null) {
            plot = new XYPlot();
            plot.setDomainAxis(xAxis);
            plot.setRangeAxis(yAxis);
        }

        for (int i : types) {
            XYDataset dataset = createDataset("scatter" + i, _dataset, i);
            XYItemRenderer scatterRenderer = new XYLineAndShapeRenderer(false, true);
            scatterRenderer.setSeriesPaint(0, ColorMap.get(index));

            plot.setRenderer(index, scatterRenderer);
            plot.setDataset(index, dataset);
            index++;
        }
    }

    /**
     * 绘制折线图
     *
     * @param x    X坐标数组
     * @param y    Y坐标数组
     * @param name 数据系列名称
     */
    public void line(float[] x, float[] y, String name) {

        XYDataset dataset = createDataset(x, y, name);
        XYLineAndShapeRenderer lineRenderer = new XYLineAndShapeRenderer(true, true);
        lineRenderer.setSeriesPaint(0, ColorMap.get(index));
        if (plot == null) {
            plot = new XYPlot(dataset, xAxis, yAxis, lineRenderer);
            index++;
        } else {
            plot.setRenderer(index, lineRenderer);
            plot.setDataset(index, dataset);
            index++;
        }
    }

    /**
     * 显示图表
     */
    public void show() {
        plot.setOrientation(PlotOrientation.VERTICAL);
        // 创建图表对象
        JFreeChart chart = new JFreeChart(plot);
        Dimension dimension = new Dimension(800, 700);
        // 创建图表面板
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(dimension);

        // 创建窗口并显示图表面板
        JFrame frame = new JFrame("Scatter and LineExam Chart");
        frame.setSize(dimension);
        center(frame);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }

    /**
     * 将窗口居中显示
     *
     * @param jFrame 要居中的窗口
     */
    public static void center(JFrame jFrame) {

        // 1、获取窗体的宽和高
        int widthFrame = jFrame.getWidth();
        int heightFrame = jFrame.getHeight();
        // 2、获取屏幕的宽和高
        Toolkit defaultToolkit = Toolkit.getDefaultToolkit();
        Dimension screenSize = defaultToolkit.getScreenSize();
        double widthScreen = screenSize.getWidth();
        double heightScreen = screenSize.getHeight();
        // 3、如果窗体的尺寸超过了，则直接用屏幕的尺寸
        if (widthFrame > widthScreen) {
            widthFrame = (int) widthScreen;
        }
        if (heightFrame > heightScreen) {
            heightFrame = (int) heightScreen;
        }
        // 4、设置位置
        int positionX = (int) ((widthScreen - widthFrame) / 2);
        int positionY = (int) ((heightScreen - heightFrame) / 2);

        jFrame.setSize(new Dimension(widthFrame, heightFrame));
        jFrame.setLocation(new Point(positionX, positionY));
    }


    /**
     * 创建数据集（用于数据集对象）
     *
     * @param name    数据系列名称
     * @param dataset 数据集对象
     * @param type    数据类型
     * @return XY数据集
     */
    private static XYDataset createDataset(String name, ArrayDataset dataset, int type) {
        int size = dataset.getSize();
        XYSeries scatterSeries = new XYSeries(name);
        for (int i = 0; i < size; i++) {
            if (dataset.getYs()[i].getMatrix()[0][0] == type)
                scatterSeries.add(dataset.getXs()[i].getMatrix()[0][0], dataset.getXs()[i].getMatrix()[0][1]);
        }

        XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
        xySeriesCollection.addSeries(scatterSeries);
        return xySeriesCollection;
    }

    /**
     * 创建数据集（用于坐标数组）
     *
     * @param x    X坐标数组
     * @param y    Y坐标数组
     * @param name 数据系列名称
     * @return XY数据集
     */
    private static XYDataset createDataset(float[] x, float[] y, String name) {
        XYSeries scatterSeries = new XYSeries(name);
        for (int i = 0; i < x.length; i++) {
            scatterSeries.add(x[i], y[i]);
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(scatterSeries);
        return dataset;
    }


}