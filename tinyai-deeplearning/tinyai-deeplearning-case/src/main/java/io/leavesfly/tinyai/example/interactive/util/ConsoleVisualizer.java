package io.leavesfly.tinyai.example.interactive.util;

import java.util.List;

/**
 * 控制台可视化工具类
 *
 * <p>提供各种控制台可视化功能，用于交互式演示：
 * <ul>
 *   <li>进度条显示</li>
 *   <li>实时曲线绘制（ASCII）</li>
 *   <li>表格数据展示</li>
 *   <li>颜色高亮输出</li>
 * </ul>
 *
 * @author TinyAI Team
 */
public class ConsoleVisualizer {

    // ANSI颜色代码
    public static final String RESET = "\u001B[0m";
    public static final String RED = "\u001B[31m";
    public static final String GREEN = "\u001B[32m";
    public static final String YELLOW = "\u001B[33m";
    public static final String BLUE = "\u001B[34m";
    public static final String MAGENTA = "\u001B[35m";
    public static final String CYAN = "\u001B[36m";
    public static final String WHITE = "\u001B[37m";
    public static final String BOLD = "\u001B[1m";

    private final int terminalWidth;
    private final boolean useColor;

    public ConsoleVisualizer() {
        this(80, true);
    }

    public ConsoleVisualizer(int terminalWidth, boolean useColor) {
        this.terminalWidth = terminalWidth;
        this.useColor = useColor;
    }

    /**
     * 打印带颜色的文本
     */
    public void print(String text, String color) {
        if (useColor) {
            System.out.print(color + text + RESET);
        } else {
            System.out.print(text);
        }
    }

    /**
     * 打印带颜色的文本（带换行）
     */
    public void println(String text, String color) {
        if (useColor) {
            System.out.println(color + text + RESET);
        } else {
            System.out.println(text);
        }
    }

    /**
     * 打印标题
     */
    public void printTitle(String title) {
        String line = "=".repeat(terminalWidth);
        println(line, CYAN);
        println(centerText(title, terminalWidth), BOLD + CYAN);
        println(line, CYAN);
    }

    /**
     * 打印分隔线
     */
    public void printSeparator() {
        println("-".repeat(terminalWidth), CYAN);
    }

    /**
     * 打印进度条
     *
     * @param current 当前进度
     * @param total   总进度
     * @param width   进度条宽度
     * @param prefix  前缀文本
     */
    public void printProgressBar(int current, int total, int width, String prefix) {
        float ratio = (float) current / total;
        int filled = (int) (ratio * width);
        int empty = width - filled;

        String bar = "█".repeat(filled) + "░".repeat(empty);
        int percentage = (int) (ratio * 100);

        String output = String.format("\r%s [%s] %3d%% (%d/%d)",
            prefix, bar, percentage, current, total);

        if (useColor) {
            System.out.print(GREEN + output + RESET);
        } else {
            System.out.print(output);
        }

        if (current >= total) {
            System.out.println();
        }
    }

    /**
     * 打印训练进度（带损失和准确率）
     */
    public void printTrainingProgress(int epoch, int totalEpochs, int batch,
                                       int totalBatches, float loss, float accuracy) {
        float ratio = (float) batch / totalBatches;
        int width = 30;
        int filled = (int) (ratio * width);
        int empty = width - filled;

        String bar = "█".repeat(filled) + "░".repeat(empty);

        String output = String.format("\rEpoch %d/%d [%s] %3d%% | Loss: %.4f | Acc: %.2f%%",
            epoch, totalEpochs, bar, (int)(ratio * 100), loss, accuracy * 100);

        if (useColor) {
            System.out.print(CYAN + output + RESET);
        } else {
            System.out.print(output);
        }

        if (batch >= totalBatches) {
            System.out.println();
        }
    }

    /**
     * 打印ASCII曲线图
     *
     * @param data   数据点列表
     * @param width  图表宽度
     * @param height 图表高度
     * @param title  图表标题
     */
    public void printLineChart(List<Float> data, int width, int height, String title) {
        if (data.isEmpty()) return;

        // 找到数据范围
        float min = data.stream().min(Float::compare).orElse(0f);
        float max = data.stream().max(Float::compare).orElse(1f);
        float range = max - min;
        if (range == 0) range = 1;

        // 创建图表
        char[][] chart = new char[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                chart[i][j] = ' ';
            }
        }

        // 绘制坐标轴
        for (int i = 0; i < height; i++) {
            chart[i][0] = '│';
        }
        for (int j = 0; j < width; j++) {
            chart[height - 1][j] = '─';
        }
        chart[height - 1][0] = '└';

        // 绘制数据点
        for (int i = 0; i < data.size() && i < width - 2; i++) {
            float value = data.get(i);
            int row = height - 2 - (int) ((value - min) / range * (height - 2));
            row = Math.max(0, Math.min(height - 2, row));
            chart[row][i + 1] = '*';
        }

        // 打印标题
        if (title != null) {
            println(centerText(title, width), BOLD);
        }

        // 打印Y轴标签
        println(String.format("%.2f", max), YELLOW);

        // 打印图表
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                char c = chart[i][j];
                if (c == '*') {
                    print(String.valueOf(c), GREEN);
                } else if (c == '│' || c == '─' || c == '└') {
                    print(String.valueOf(c), CYAN);
                } else {
                    System.out.print(c);
                }
            }
            System.out.println();
        }

        // 打印X轴标签
        println(String.format("%.2f", min), YELLOW);
    }

    /**
     * 打印表格
     *
     * @param headers 表头
     * @param rows    数据行
     * @param widths  列宽
     */
    public void printTable(String[] headers, String[][] rows, int[] widths) {
        // 打印表头
        printTableRow(headers, widths, BOLD + CYAN);
        printTableSeparator(widths);

        // 打印数据行
        for (String[] row : rows) {
            printTableRow(row, widths, RESET);
        }
    }

    private void printTableRow(String[] cells, int[] widths, String color) {
        StringBuilder sb = new StringBuilder();
        sb.append("│ ");
        for (int i = 0; i < cells.length; i++) {
            String cell = cells[i];
            int width = widths[i];
            sb.append(padRight(cell, width));
            if (i < cells.length - 1) {
                sb.append(" │ ");
            }
        }
        sb.append(" │");

        if (useColor) {
            println(sb.toString(), color);
        } else {
            System.out.println(sb.toString());
        }
    }

    private void printTableSeparator(int[] widths) {
        StringBuilder sb = new StringBuilder();
        sb.append("├─");
        for (int i = 0; i < widths.length; i++) {
            sb.append("─".repeat(widths[i]));
            if (i < widths.length - 1) {
                sb.append("─┼─");
            }
        }
        sb.append("─┤");

        if (useColor) {
            println(sb.toString(), CYAN);
        } else {
            System.out.println(sb.toString());
        }
    }

    /**
     * 打印网络结构图
     */
    public void printNetworkStructure(String[] layerNames, int[] layerSizes) {
        println("网络结构:", BOLD);
        println("", RESET);

        int maxNameLength = 0;
        for (String name : layerNames) {
            maxNameLength = Math.max(maxNameLength, name.length());
        }

        for (int i = 0; i < layerNames.length; i++) {
            String name = padRight(layerNames[i], maxNameLength);
            int size = layerSizes[i];

            // 打印层
            print("  ", RESET);
            print(name + "  ", YELLOW);
            print("(" + size + " 神经元)  ", CYAN);

            // 打印神经元图标
            int iconCount = Math.min(size / 10, 10);
            for (int j = 0; j < iconCount; j++) {
                print("◎ ", GREEN);
            }
            if (size > iconCount * 10) {
                print("...", RESET);
            }
            println("", RESET);

            // 打印连接箭头
            if (i < layerNames.length - 1) {
                print("        ↓  ", CYAN);
                int connections = layerSizes[i] * layerSizes[i + 1];
                println("(" + connections + " 连接)", RESET);
            }
        }
    }

    /**
     * 打印参数统计
     */
    public void printParameterStats(int totalParams, int trainableParams) {
        println("", RESET);
        println("参数统计:", BOLD);
        println("  总参数量:     " + formatNumber(totalParams), CYAN);
        println("  可训练参数:   " + formatNumber(trainableParams), GREEN);
        println("  不可训练参数: " + formatNumber(totalParams - trainableParams), YELLOW);
    }

    /**
     * 打印菜单
     */
    public void printMenu(String title, String[] options) {
        println("", RESET);
        println(title, BOLD + CYAN);
        printSeparator();

        for (int i = 0; i < options.length; i++) {
            println("  [" + (i + 1) + "] " + options[i], RESET);
        }

        printSeparator();
    }

    /**
     * 打印信息框
     */
    public void printInfoBox(String message) {
        String[] lines = message.split("\n");
        int maxLength = 0;
        for (String line : lines) {
            maxLength = Math.max(maxLength, line.length());
        }

        String border = "┌─" + "─".repeat(maxLength) + "─┐";
        println(border, CYAN);

        for (String line : lines) {
            String padded = padRight(line, maxLength);
            println("│ " + padded + " │", CYAN);
        }

        String bottomBorder = "└─" + "─".repeat(maxLength) + "─┘";
        println(bottomBorder, CYAN);
    }

    /**
     * 打印成功消息
     */
    public void printSuccess(String message) {
        println("✓ " + message, GREEN);
    }

    /**
     * 打印警告消息
     */
    public void printWarning(String message) {
        println("⚠ " + message, YELLOW);
    }

    /**
     * 打印错误消息
     */
    public void printError(String message) {
        println("✗ " + message, RED);
    }

    // 辅助方法

    private String centerText(String text, int width) {
        int padding = (width - text.length()) / 2;
        return " ".repeat(Math.max(0, padding)) + text;
    }

    private String padRight(String text, int width) {
        if (text.length() >= width) {
            return text.substring(0, width);
        }
        return text + " ".repeat(width - text.length());
    }

    private String formatNumber(int number) {
        if (number >= 1_000_000) {
            return String.format("%.2fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.2fK", number / 1_000.0);
        }
        return String.valueOf(number);
    }

    /**
     * 清屏
     */
    public void clearScreen() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
    }

    /**
     * 等待用户按键
     */
    public void waitForKey(String message) {
        println("", RESET);
        print(message + " ", YELLOW);
        try {
            System.in.read();
        } catch (Exception e) {
            // 忽略异常
        }
    }
}
