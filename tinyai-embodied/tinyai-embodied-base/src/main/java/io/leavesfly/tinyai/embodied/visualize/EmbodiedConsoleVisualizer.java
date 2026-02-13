package io.leavesfly.tinyai.embodied.visualize;

import io.leavesfly.tinyai.embodied.model.*;

/**
 * 具身智能控制台可视化工具
 * 
 * <p>提供具身智能场景的ASCII艺术可视化，包括：
 * <ul>
 *   <li>驾驶场景可视化 - 显示道路、车辆、车道线</li>
 *   <li>传感器数据可视化 - 显示雷达点云、相机视野</li>
 *   <li>决策状态可视化 - 显示策略、Q值、置信度</li>
 *   <li>学习曲线可视化 - 显示奖励、损失曲线</li>
 * </ul>
 * 
 * <p>使用示例：
 * <pre>{@code
 * EmbodiedVisualizer visualizer = new EmbodiedVisualizer();
 * visualizer.visualizeDrivingScene(agent.getPerceptionState(), agent.getAction());
 * }</pre>
 * 
 * @author TinyAI Team
 */
public class EmbodiedConsoleVisualizer {

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
    public static final String DIM = "\u001B[2m";

    // 可视化符号
    private static final char CAR = '@';
    private static final char EGO_CAR = '@';
    private static final char LANE = '│';
    private static final char DASHED_LANE = '│';
    private static final char OBSTACLE = '#';
    private static final char ROAD = '·';
    private static final char GRASS = '░';
    private static final char TARGET = '*';
    private static final char ROBOT = 'R';
    private static final char CLEANED = '✓';
    private static final char UNCLEANED = '·';
    private static final char OBST = '█';
    private static final char LIDAR_NEAR = '#';
    private static final char LIDAR_MID = 'o';
    private static final char LIDAR_FAR = '.';

    private final int displayWidth;
    private final int displayHeight;
    private final boolean useColor;

    public EmbodiedConsoleVisualizer() {
        this(60, 20, true);
    }

    public EmbodiedConsoleVisualizer(int width, int height, boolean useColor) {
        this.displayWidth = width;
        this.displayHeight = height;
        this.useColor = useColor;
    }

    // ==================== 驾驶场景可视化 ====================

    /**
     * 可视化驾驶场景
     * 
     * <p>将当前驾驶状态渲染为ASCII艺术，包括：
     * <ul>
     *   <li>自车位置和朝向</li>
     *   <li>车道线和道路边界</li>
     *   <li>前车和障碍物</li>
     *   <li>当前速度和加速度</li>
     * </ul>
     * 
     * @param state 当前感知状态
     * @param action 当前执行的动作（可选）
     */
    public void visualizeDrivingScene(PerceptionState state, DrivingAction action) {
        char[][] grid = new char[displayHeight][displayWidth];
        initGrid(grid, GRASS);

        // 绘制道路
        drawRoad(grid, state);

        // 绘制车道线
        drawLaneMarkings(grid);

        // 绘制障碍物
        drawObstacles(grid, state);

        // 绘制自车
        drawEgoCar(grid, state);

        // 打印场景
        printScene("驾驶场景可视化");

        // 打印状态信息
        printDrivingInfo(state, action);
    }

    /**
     * 初始化网格
     */
    private void initGrid(char[][] grid, char fillChar) {
        for (int i = 0; i < displayHeight; i++) {
            for (int j = 0; j < displayWidth; j++) {
                grid[i][j] = fillChar;
            }
        }
    }

    /**
     * 绘制道路区域
     */
    private void drawRoad(char[][] grid, PerceptionState state) {
        int roadTop = displayHeight / 4;
        int roadBottom = 3 * displayHeight / 4;
        int roadLeft = displayWidth / 8;
        int roadRight = 7 * displayWidth / 8;

        for (int i = roadTop; i < roadBottom; i++) {
            for (int j = roadLeft; j < roadRight; j++) {
                if (i >= 0 && i < displayHeight && j >= 0 && j < displayWidth) {
                    grid[i][j] = ROAD;
                }
            }
        }
    }

    /**
     * 绘制车道线
     */
    private void drawLaneMarkings(char[][] grid) {
        int roadTop = displayHeight / 4;
        int roadBottom = 3 * displayHeight / 4;
        int lane1 = displayWidth / 3;
        int lane2 = 2 * displayWidth / 3;

        // 中心虚线车道线
        for (int i = roadTop; i < roadBottom; i += 2) {
            if (i >= 0 && i < displayHeight) {
                if (lane1 >= 0 && lane1 < displayWidth) {
                    grid[i][lane1] = '┆';
                }
                if (lane2 >= 0 && lane2 < displayWidth) {
                    grid[i][lane2] = '┆';
                }
            }
        }
    }

    /**
     * 绘制障碍物
     */
    private void drawObstacles(char[][] grid, PerceptionState state) {
        if (state == null || state.getObstacleMap() == null) {
            return;
        }

        for (ObstacleInfo obstacle : state.getObstacleMap()) {
            Vector3D pos = obstacle.getPosition();
            // 简单的投影映射
            int screenX = (int) (displayWidth / 2 + pos.getX() / 10);
            int screenY = displayHeight / 2 - (int) (pos.getZ() / 5);

            if (screenX >= 0 && screenX < displayWidth && screenY >= 0 && screenY < displayHeight) {
                grid[screenY][screenX] = useColor ? OBSTACLE : '#';
            }
        }
    }

    /**
     * 绘制自车
     */
    private void drawEgoCar(char[][] grid, PerceptionState state) {
        int carY = displayHeight * 3 / 4;
        int carX = displayWidth / 2;

        // 绘制车身为多个字符
        char[][] carShape = {
            {'▀', '▀', '▀'},
            {'█', '█', '█'},
            {'▄', '▄', '▄'}
        };

        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int y = carY + di;
                int x = carX + dj;
                if (y >= 0 && y < displayHeight && x >= 0 && x < displayWidth) {
                    grid[y][x] = useColor ? EGO_CAR : '@';
                }
            }
        }
    }

    /**
     * 打印驾驶场景
     */
    private void printScene(String title) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        sb.append("═".repeat(displayWidth + 4)).append("\n");
        sb.append("  ").append(title).append("\n");
        sb.append("═".repeat(displayWidth + 4)).append("\n");

        for (int i = 0; i < displayHeight; i++) {
            sb.append("  ");
            for (int j = 0; j < displayWidth; j++) {
                sb.append(' ');
                sb.append(' ');
            }
            sb.append("\n");
        }

        System.out.print(sb.toString());
    }

    /**
     * 打印驾驶状态信息
     */
    private void printDrivingInfo(PerceptionState state, DrivingAction action) {
        if (state == null) {
            return;
        }

        VehicleState vehicleState = state.getVehicleState();

        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        sb.append("┌").append("─".repeat(50)).append("┐\n");
        sb.append("│ 车辆状态信息").append(" ".repeat(35)).append("│\n");
        sb.append("├").append("─".repeat(50)).append("┤\n");

        if (vehicleState != null) {
            sb.append("│ 速度: ").append(String.format("%6.1f km/h", vehicleState.getSpeed() * 3.6))
              .append("    加速度: ").append(String.format("%6.2f m/s²", vehicleState.getAcceleration()))
              .append(" ".repeat(8)).append("│\n");

            sb.append("│ 位置: (").append(String.format("%5.1f", vehicleState.getPosition().getX()))
              .append(", ").append(String.format("%5.1f", vehicleState.getPosition().getY()))
              .append(")    航向角: ").append(String.format("%6.1f°", Math.toDegrees(vehicleState.getHeading())))
              .append(" ".repeat(6)).append("│\n");
        }

        if (state.getObstacleMap() != null) {
            sb.append("│ 障碍物数量: ").append(state.getObstacleMap().size())
              .append(" ".repeat(38)).append("│\n");
        }

        if (action != null) {
            sb.append("│ 当前动作: 驾驶控制")
              .append("  油门: ").append(String.format("%.2f", action.getThrottle()))
              .append("  刹车: ").append(String.format("%.2f", action.getBrake()))
              .append("  转向: ").append(String.format("%.2f", action.getSteering()))
              .append(" ".repeat(8)).append("│\n");
        }

        sb.append("└").append("─".repeat(50)).append("┘\n");

        System.out.print(useColor ? CYAN + sb + RESET : sb);
    }

    // ==================== 传感器数据可视化 ====================

    /**
     * 可视化激光雷达点云
     * 
     * <p>将激光雷达扫描数据可视化为ASCII点云图，
     * 展示障碍物的距离和角度分布。
     * 
     * @param lidarData 激光雷达数据
     */
    public void visualizeLidarPoints(float[] lidarData) {
        if (lidarData == null || lidarData.length == 0) {
            return;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("\n").append(BOLD).append("激光雷达点云可视化").append(RESET).append("\n");
        sb.append("─".repeat(60)).append("\n");

        int numRays = lidarData.length;
        int displayPoints = Math.min(numRays, 60);

        for (int i = 0; i < 5; i++) {
            sb.append(String.format("%3d° ", i * 45));
            for (int j = 0; j < displayPoints; j++) {
                int idx = j * numRays / displayPoints;
                float distance = lidarData[idx];
                char c;
                if (distance < 5) {
                    c = useColor ? '●' : '#';
                } else if (distance < 20) {
                    c = useColor ? '○' : 'o';
                } else if (distance < 50) {
                    c = useColor ? '·' : '.';
                } else {
                    c = ' ';
                }
                sb.append(c).append(' ');
            }
            sb.append("\n");
        }

        sb.append("─".repeat(60)).append("\n");
        sb.append("图例: ●<5m  ○<20m  ·<50m\n");

        System.out.print(sb.toString());
    }

    // ==================== 学习曲线可视化 ====================

    /**
     * 可视化训练奖励曲线
     * 
     * <p>使用ASCII字符绘制奖励随训练进程的变化曲线，
     * 帮助理解学习进度和收敛情况。
     * 
     * @param rewards 每回合的总奖励列表
     * @param windowSize 平滑窗口大小
     */
    public void visualizeLearningCurve(double[] rewards, int windowSize) {
        if (rewards == null || rewards.length == 0) {
            return;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("\n").append(BOLD).append("学习曲线可视化").append(RESET).append("\n");
        sb.append("─".repeat(70)).append("\n");

        // 计算平滑曲线
        double[] smoothed = smoothRewards(rewards, windowSize);

        int height = 15;
        int width = Math.min(rewards.length, 60);

        // 找到最大值和最小值
        double maxReward = Double.MIN_VALUE;
        double minReward = Double.MAX_VALUE;
        for (double r : rewards) {
            maxReward = Math.max(maxReward, r);
            minReward = Math.min(minReward, r);
        }

        double range = maxReward - minReward;
        if (range < 0.01) {
            range = 1;
        }

        // 绘制图表
        for (int y = height - 1; y >= 0; y--) {
            double threshold = minReward + range * y / height;
            sb.append(String.format("%8.1f │", threshold));

            for (int x = 0; x < width; x++) {
                int idx = x * rewards.length / width;
                char c;
                if (smoothed[idx] >= threshold && smoothed[idx] < threshold + range / height) {
                    c = useColor ? '●' : '*';
                } else if (rewards[idx] >= threshold && rewards[idx] < threshold + range / height) {
                    c = useColor ? '·' : '.';
                } else {
                    c = ' ';
                }
                sb.append(' ').append(c);
            }
            sb.append("\n");
        }

        sb.append(" ".repeat(9)).append("└").append("─".repeat(width + 1)).append("\n");
        sb.append(" ".repeat(9)).append("0").append(" ".repeat(width / 2))
          .append(String.format("Episode %d", rewards.length));

        sb.append("\n").append("─".repeat(70)).append("\n");
        sb.append(String.format("最新奖励: %.2f  平均奖励: %.2f  最大奖励: %.2f\n",
            rewards[rewards.length - 1],
            average(rewards),
            maxReward));

        System.out.print(useColor ? CYAN + sb + RESET : sb);
    }

    /**
     * 平滑奖励曲线
     */
    private double[] smoothRewards(double[] rewards, int windowSize) {
        double[] smoothed = new double[rewards.length];
        for (int i = 0; i < rewards.length; i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(rewards.length, i + windowSize / 2);
            double sum = 0;
            for (int j = start; j < end; j++) {
                sum += rewards[j];
            }
            smoothed[i] = sum / (end - start);
        }
        return smoothed;
    }

    /**
     * 计算平均值
     */
    private double average(double[] values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    // ==================== Q值可视化 ====================

    /**
     * 可视化Q值分布
     * 
     * <p>以热力图形式展示不同状态-动作对的Q值分布，
     * 便于理解策略的质量。
     * 
     * @param qValues Q值数组
     * @param actionNames 动作名称
     */
    public void visualizeQValues(double[] qValues, String[] actionNames) {
        if (qValues == null || qValues.length == 0) {
            return;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("\n").append(BOLD).append("Q值分布可视化").append(RESET).append("\n");
        sb.append("─".repeat(50)).append("\n");

        // 找到最大值和最小值
        double maxQ = Double.MIN_VALUE;
        double minQ = Double.MAX_VALUE;
        for (double q : qValues) {
            maxQ = Math.max(maxQ, q);
            minQ = Math.min(minQ, q);
        }

        double range = maxQ - minQ;
        if (range < 0.01) {
            range = 1;
        }

        // 绘制Q值条形图
        for (int i = 0; i < qValues.length; i++) {
            String actionName = (actionNames != null && i < actionNames.length) 
                ? actionNames[i] : "Action " + i;
            
            int barLength = (int) ((qValues[i] - minQ) / range * 30);
            
            sb.append(String.format("%-12s │", actionName));
            for (int j = 0; j < barLength; j++) {
                double ratio = (double) j / 30;
                if (ratio < 0.33) {
                    sb.append(useColor ? GREEN + "█" + RESET : "#");
                } else if (ratio < 0.66) {
                    sb.append(useColor ? YELLOW + "█" + RESET : "#");
                } else {
                    sb.append(useColor ? RED + "█" + RESET : "#");
                }
            }
            sb.append(String.format(" %.3f\n", qValues[i]));
        }

        sb.append("─".repeat(50)).append("\n");
        sb.append(String.format("最大Q值: %.3f  最小Q值: %.3f  差值: %.3f\n",
            maxQ, minQ, range));

        System.out.print(sb.toString());
    }

    // ==================== 进度条 ====================

    /**
     * 打印训练进度条
     * 
     * @param current 当前步数
     * @param total 总步数
     * @param prefix 前缀文本
     */
    public void printProgressBar(int current, int total, String prefix) {
        float ratio = (float) current / total;
        int width = 40;
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

    // ==================== 表格打印 ====================

    /**
     * 清除屏幕
     */
    public void clearScreen() {
        // 使用ANSI转义序列清除屏幕
        System.out.print("\033[H\033[2J");
        System.out.flush();
    }

    /**
     * 打印带颜色的文本（带换行）
     */
    public void println(String text, String color) {
        if (useColor && color != null && !color.isEmpty()) {
            System.out.println(color + text + RESET);
        } else {
            System.out.println(text);
        }
    }

    /**
     * 打印文本（带换行，不带颜色）
     */
    public void println(String text) {
        System.out.println(text);
    }

    /**
     * 打印带颜色的文本
     */
    public void print(String text, String color) {
        if (useColor && color != null && !color.isEmpty()) {
            System.out.print(color + text + RESET);
        } else {
            System.out.print(text);
        }
    }

    /**
     * 打印菜单
     */
    public void printMenu(String title, String[] options) {
        println("\n" + title + ":", BOLD);
        for (int i = 0; i < options.length; i++) {
            println("  " + (i + 1) + ". " + options[i], YELLOW);
        }
    }

    /**
     * 打印表格
     */
    public void printTable(String[] headers, String[][] rows, int... widths) {
        // 打印表头
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < headers.length; i++) {
            int w = widths != null && i < widths.length ? widths[i] : 10;
            sb.append(String.format("%-" + w + "s ", headers[i]));
        }
        println(sb.toString(), BOLD);

        // 打印分隔线
        sb = new StringBuilder();
        for (int i = 0; i < headers.length; i++) {
            int w = widths != null && i < widths.length ? widths[i] : 10;
            sb.append("-".repeat(w)).append(" ");
        }
        println(sb.toString(), CYAN);

        // 打印行
        for (String[] row : rows) {
            sb = new StringBuilder();
            for (int i = 0; i < row.length; i++) {
                int w = widths != null && i < widths.length ? widths[i] : 10;
                sb.append(String.format("%-" + w + "s ", row[i]));
            }
            println(sb.toString(), RESET);
        }
    }

    /**
     * 打印带颜色的信息框
     */
    public void printInfoBox(String title, String[] lines) {
        int maxLen = title.length();
        for (String line : lines) {
            maxLen = Math.max(maxLen, line.length());
        }

        StringBuilder sb = new StringBuilder();
        sb.append("\n┌").append("─".repeat(maxLen + 2)).append("┐\n");
        sb.append("│ ").append(title).append(" ".repeat(maxLen - title.length())).append(" │\n");
        sb.append("├").append("─".repeat(maxLen + 2)).append("┤\n");

        for (String line : lines) {
            sb.append("│ ").append(line).append(" ".repeat(maxLen - line.length())).append(" │\n");
        }

        sb.append("└").append("─".repeat(maxLen + 2)).append("┘\n");

        System.out.print(useColor ? CYAN + sb + RESET : sb);
    }

    /**
     * 打印带颜色的标题
     */
    public void printTitle(String title) {
        String line = "=".repeat(title.length() + 4);
        if (useColor) {
            System.out.println(CYAN + line + RESET);
            System.out.println(CYAN + "  " + title + RESET);
            System.out.println(CYAN + line + RESET);
        } else {
            System.out.println(line);
            System.out.println("  " + title);
            System.out.println(line);
        }
    }

    /**
     * 打印分隔线
     */
    public void printSeparator() {
        if (useColor) {
            System.out.println(CYAN + "─".repeat(60) + RESET);
        } else {
            System.out.println("─".repeat(60));
        }
    }
}
