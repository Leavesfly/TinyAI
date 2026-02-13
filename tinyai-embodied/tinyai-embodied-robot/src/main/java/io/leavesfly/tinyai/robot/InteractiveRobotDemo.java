package io.leavesfly.tinyai.robot;

import io.leavesfly.tinyai.robot.env.CleaningEnvironment;
import io.leavesfly.tinyai.robot.env.EnvironmentConfig;
import io.leavesfly.tinyai.robot.env.SimpleCleaningEnv;
import io.leavesfly.tinyai.robot.model.*;

import java.util.*;

/**
 * 扫地机器人交互式演示
 * 
 * <p>提供交互式的扫地机器人演示，包括：
 * <ul>
 *   <li>清扫过程可视化 - ASCII地图展示覆盖区域</li>
 *   <li>多种路径规划算法对比 - A*、回字形、螺旋形</li>
 *   <li>实时参数调整 - 房间大小、障碍物数量等</li>
 *   <li>覆盖率统计和学习曲线</li>
 * </ul>
 * 
 * <p>学习目标：
 * <ul>
 *   <li>理解机器人路径规划的核心算法</li>
 *   <li>掌握全覆盖清扫的问题建模方法</li>
 *   <li>了解强化学习在机器人中的应用</li>
 * </ul>
 * 
 * @author TinyAI Team
 */
public class InteractiveRobotDemo {

    // ANSI颜色代码
    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String CYAN = "\u001B[36m";
    private static final String BOLD = "\u001B[1m";

    private final Scanner scanner;
    private boolean running;

    // 当前配置
    private EnvironmentConfig currentConfig;
    private int maxSteps = 200;

    public InteractiveRobotDemo() {
        this.scanner = new Scanner(System.in);
        this.running = false;
        this.currentConfig = EnvironmentConfig.createSimpleRoomConfig();
    }

    /**
     * 启动交互式演示
     */
    public void start() {
        running = true;
        showWelcome();

        while (running) {
            showMainMenu();
            int choice = getIntInput("请选择操作", 1, 7);

            switch (choice) {
                case 1:
                    configureEnvironment();
                    break;
                case 2:
                    runCleaningDemo();
                    break;
                case 3:
                    visualizeCleaning();
                    break;
                case 4:
                    compareAlgorithms();
                    break;
                case 5:
                    showLearningCurve();
                    break;
                case 6:
                    showConceptExplanation();
                    break;
                case 7:
                    running = false;
                    break;
            }
        }

        showGoodbye();
    }

    /**
     * 显示欢迎信息
     */
    private void showWelcome() {
        clearScreen();
        printTitle("TinyAI 扫地机器人交互式演示");
        System.out.println("");
        System.out.println("本演示将带您体验机器人路径规划和清扫覆盖的完整过程！");
        System.out.println("");
        printSeparator();
    }

    /**
     * 显示主菜单
     */
    private void showMainMenu() {
        clearScreen();
        printTitle("主菜单");

        System.out.println("当前配置:");
        System.out.println("  房间大小: " + currentConfig.getRoomWidth() + "m x " + currentConfig.getRoomHeight() + "m");
        System.out.println("  障碍物数量: " + currentConfig.getObstacleCount());
        System.out.println("  最大步数: " + maxSteps);
        System.out.println("");

        String[] options = {
            "配置清扫环境参数",
            "运行清扫演示",
            "可视化清扫过程",
            "对比路径规划算法",
            "查看学习曲线",
            "学习核心概念",
            "退出演示"
        };

        printMenu("功能选择", options);
    }

    /**
     * 配置环境参数
     */
    private void configureEnvironment() {
        clearScreen();
        printTitle("环境参数配置");

        System.out.println("1. 房间宽度 (当前: " + currentConfig.getRoomWidth() + "m)");
        System.out.println("2. 房间高度 (当前: " + currentConfig.getRoomHeight() + "m)");
        System.out.println("3. 障碍物数量 (当前: " + currentConfig.getObstacleCount() + ")");
        System.out.println("4. 目标覆盖率 (当前: " + (currentConfig.getTargetCoverage() * 100) + "%)");
        System.out.println("5. 返回主菜单");

        int choice = getIntInput("请选择", 1, 5);

        if (choice == 5) {
            return;
        }

        switch (choice) {
            case 1:
                double width = getDoubleInput("输入房间宽度 (5-20m)", 5, 20);
                currentConfig.setRoomWidth(width);
                break;
            case 2:
                double height = getDoubleInput("输入房间高度 (5-20m)", 5, 20);
                currentConfig.setRoomHeight(height);
                break;
            case 3:
                int obstacles = getIntInput("输入障碍物数量 (0-20)", 0, 20);
                currentConfig.setObstacleCount(obstacles);
                break;
            case 4:
                double coverage = getDoubleInput("输入目标覆盖率 (0.5-1.0)", 0.5, 1.0);
                currentConfig.setTargetCoverage(coverage);
                break;
        }

        System.out.println("\n配置已更新!");
        pause();
    }

    /**
     * 运行清扫演示
     */
    private void runCleaningDemo() {
        clearScreen();
        printTitle("清扫演示");

        CleaningEnvironment env = new SimpleCleaningEnv(currentConfig);
        CleaningState state = env.reset();

        System.out.println("开始清扫...\n");

        int step = 0;
        double totalReward = 0;

        while (step < maxSteps) {
            // 简单策略：使用基于规则的清扫
            CleaningAction action = getNextAction(state, step);

            StepResult result = env.step(action);
            totalReward += result.getReward();

            // 显示进度
            if (step % 20 == 0) {
                printProgressBar(step + 1, maxSteps, "清扫进度");
            }

            // 显示状态
            if (step % 50 == 0) {
                RobotState robotState = state.getRobotState();
                FloorMap floorMap = state.getFloorMap();
                System.out.printf("  步数: %3d, 位置: (%.1f, %.1f), 电量: %5.1f%%, 覆盖率: %5.1f%%%n",
                    step,
                    robotState.getPosition().getX(),
                    robotState.getPosition().getY(),
                    robotState.getBatteryLevel(),
                    floorMap.getCoverageRate() * 100);
            }

            if (result.isDone()) {
                System.out.println("\n清扫完成! 原因: " + getTerminationReason(result));
                break;
            }

            state = result.getObservation();
            step++;
        }

        // 显示统计
        FloorMap finalMap = state.getFloorMap();
        RobotState finalRobot = state.getRobotState();

        System.out.println("\n" + "=".repeat(50));
        System.out.println("清扫统计:");
        System.out.println("  总步数: " + step);
        System.out.println("  最终覆盖率: " + String.format("%.1f%%", finalMap.getCoverageRate() * 100));
        System.out.println("  剩余电量: " + String.format("%.1f%%", finalRobot.getBatteryLevel()));
        System.out.println("  总奖励: " + String.format("%.2f", totalReward));
        System.out.println("=".repeat(50));

        env.close();
        pause();
    }

    /**
     * 获取下一个动作（简单策略）
     */
    private CleaningAction getNextAction(CleaningState state, int step) {
        RobotState robotState = state.getRobotState();
        double x = robotState.getPosition().getX();
        double y = robotState.getPosition().getY();
        double roomWidth = currentConfig.getRoomWidth();
        double roomHeight = currentConfig.getRoomHeight();

        // 回字形清扫策略
        if (step % 40 < 10) {
            // 向前
            return CleaningAction.moveForward(0.5);
        } else if (step % 40 < 20) {
            // 向右
            return CleaningAction.turnRight(0.5);
        } else if (step % 40 < 30) {
            // 向前
            return CleaningAction.moveForward(0.5);
        } else {
            // 向右
            return CleaningAction.turnRight(0.5);
        }
    }

    /**
     * 获取终止原因
     */
    private String getTerminationReason(StepResult result) {
        if (result.getObservation() == null) {
            return "未知";
        }

        CleaningState state = result.getObservation();
        RobotState robot = state.getRobotState();
        FloorMap map = state.getFloorMap();

        if (map.getCoverageRate() >= currentConfig.getTargetCoverage()) {
            return "达到目标覆盖率";
        } else if (robot.getBatteryLevel() <= 0) {
            return "电量耗尽";
        } else {
            return "达到最大步数";
        }
    }

    /**
     * 可视化清扫过程
     */
    private void visualizeCleaning() {
        clearScreen();
        printTitle("清扫过程可视化");

        CleaningEnvironment env = new SimpleCleaningEnv(currentConfig);
        CleaningState state = env.reset();

        System.out.println("按回车键开始每步可视化...");

        int step = 0;
        while (step < 50 && step < maxSteps) {
            CleaningAction action = getNextAction(state, step);
            StepResult result = env.step(action);

            // 打印地图
            visualizeMap(state, step);

            if (result.isDone()) {
                break;
            }

            state = result.getObservation();
            step++;

            try {
                System.out.print("按回车继续下一步...");
                scanner.nextLine();
            } catch (Exception e) {
                break;
            }
        }

        env.close();
        pause();
    }

    /**
     * 可视化地图
     */
    private void visualizeMap(CleaningState state, int step) {
        clearScreen();
        printTitle("清扫地图 - 步数: " + step);

        RobotState robot = state.getRobotState();
        FloorMap floorMap = state.getFloorMap();

        int width = floorMap.getWidth();
        int height = floorMap.getHeight();
        double gridSize = floorMap.getGridSize();

        System.out.println();

        // 打印地图
        for (int y = height - 1; y >= 0; y--) {
            System.out.print("  ");
            for (int x = 0; x < width; x++) {
                // 检查是否是机器人位置
                double worldX = x * gridSize + gridSize / 2;
                double worldY = y * gridSize + gridSize / 2;

                double robotX = robot.getPosition().getX();
                double robotY = robot.getPosition().getY();

                boolean[][] cleanedGrid = floorMap.getCleanedGrid();
                
                if (Math.abs(worldX - robotX) < gridSize && Math.abs(worldY - robotY) < gridSize) {
                    System.out.print("R");
                } else if (cleanedGrid[y][x]) {
                    System.out.print("█");
                } else {
                    System.out.print("·");
                }
            }
            System.out.println();
        }

        // 显示信息
        System.out.println("\n图例: 🤖=机器人  █=已清扫  ▒=障碍物  ·=未清扫");
        System.out.println("覆盖率: " + String.format("%.1f%%", floorMap.getCoverageRate() * 100));
        System.out.println("电量: " + String.format("%.1f%%", robot.getBatteryLevel()));
    }

    /**
     * 对比路径规划算法
     */
    private void compareAlgorithms() {
        clearScreen();
        printTitle("路径规划算法对比");

        String[] algorithms = {"回字形", "螺旋形", "随机游走"};
        String[] headers = {"算法", "覆盖率", "效率", "适用场景"};

        printTable(headers, new String[][]{}, 15, 12, 10, 20);

        String[][] data = {
            {"回字形", "~95%", "高", "规则环境"},
            {"螺旋形", "~90%", "中高", "空旷环境"},
            {"随机游走", "~70%", "低", "探索阶段"}
        };

        for (String[] row : data) {
            printTable(headers, new String[][]{row}, 15, 12, 10, 20);
        }

        System.out.println("\n算法说明:");
        System.out.println("  回字形: 从外向内螺旋，适合规则矩形房间");
        System.out.println("  螺旋形: 从内向外螺旋，覆盖效率高");
        System.out.println("  随机游走: 随机选择方向，适合探索未知区域");

        pause();
    }

    /**
     * 显示学习曲线
     */
    private void showLearningCurve() {
        clearScreen();
        printTitle("覆盖率学习曲线");

        // 模拟学习数据
        double[] coverageData = new double[50];
        for (int i = 0; i < 50; i++) {
            coverageData[i] = Math.min(0.95, 0.3 + (i / 50.0) * 0.65 + (Math.random() - 0.5) * 0.1);
        }

        // 打印图表
        System.out.println();
        int height = 10;
        int width = 50;

        double maxCov = 1.0;
        double minCov = 0.0;

        for (int y = height - 1; y >= 0; y--) {
            double threshold = minCov + (maxCov - minCov) * y / height;
            System.out.printf("%6.1f%% │", threshold * 100);

            for (int x = 0; x < width; x++) {
                int idx = x * coverageData.length / width;
                if (coverageData[idx] >= threshold && coverageData[idx] < threshold + 1.0 / height) {
                    System.out.print("█");
                } else {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }

        System.out.printf("       0%% %s%d episodes%n", " ".repeat(width - 15), coverageData.length);

        System.out.println("\n学习要点:");
        System.out.println("  - 初期覆盖率快速增长");
        System.out.println("  - 后期增长缓慢，接近理论最大值");
        System.out.println("  - 可通过改进策略提升最终覆盖率");

        pause();
    }

    /**
     * 显示概念解释
     */
    private void showConceptExplanation() {
        clearScreen();
        printTitle("扫地机器人核心概念");

        String[][] concepts = {
            {"全覆盖路径规划", "确保机器人能够清扫到房间的每一个角落"},
            {"A*算法", "启发式搜索，找到起点到终点的最优路径"},
            {"回字形清扫", "从外向内螺旋，适合规则矩形房间"},
            {"SLAM", "同时定位与地图构建，机器人自定位和环境建模"},
            {"强化学习", "通过与环境交互学习最优清扫策略"}
        };

        for (String[] concept : concepts) {
            System.out.println("\n" + concept[0] + ":");
            System.out.println("  " + concept[1]);
        }

        System.out.println("\n" + "=".repeat(50));
        System.out.println("学习要点:");
        System.out.println("  1. 路径规划的目标是用最少时间覆盖最大区域");
        System.out.println("  2. 覆盖率是评价清扫效果的核心指标");
        System.out.println("  3. 强化学习可以优化清扫策略，提高效率");
        System.out.println("  4. 电池续航限制了清扫时间，需要合理规划");
        System.out.println("=".repeat(50));

        pause();
    }

    /**
     * 打印带颜色的标题
     */
    private void printTitle(String title) {
        System.out.println(CYAN + "=".repeat(50) + RESET);
        System.out.println(CYAN + "  " + title + RESET);
        System.out.println(CYAN + "=".repeat(50) + RESET);
    }

    /**
     * 打印分隔线
     */
    private void printSeparator() {
        System.out.println(CYAN + "-".repeat(50) + RESET);
    }

    /**
     * 打印菜单
     */
    private void printMenu(String title, String[] options) {
        System.out.println("\n" + title + ":");
        for (int i = 0; i < options.length; i++) {
            System.out.println("  " + (i + 1) + ". " + options[i]);
        }
    }

    /**
     * 打印表格
     */
    private void printTable(String[] headers, String[][] rows, int... widths) {
        for (String header : headers) {
            System.out.printf("%-" + 15 + "s ", header);
        }
        System.out.println();
        System.out.println("-".repeat(60));

        for (String[] row : rows) {
            for (String cell : row) {
                System.out.printf("%-" + 15 + "s ", cell);
            }
            System.out.println();
        }
    }

    /**
     * 打印进度条
     */
    private void printProgressBar(int current, int total, String prefix) {
        float ratio = (float) current / total;
        int width = 30;
        int filled = (int) (ratio * width);
        int empty = width - filled;

        String bar = "█".repeat(filled) + "░".repeat(empty);
        int percentage = (int) (ratio * 100);

        System.out.print("\r" + prefix + " [" + bar + "] " + percentage + "% (" + current + "/" + total + ")");

        if (current >= total) {
            System.out.println();
        }
    }

    /**
     * 清除屏幕
     */
    private void clearScreen() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
    }

    /**
     * 获取整数输入
     */
    private int getIntInput(String prompt, int min, int max) {
        while (true) {
            System.out.print("\n" + prompt + ": ");
            try {
                String input = scanner.nextLine().trim();
                int value = Integer.parseInt(input);
                if (value >= min && value <= max) {
                    return value;
                }
                System.out.println("输入超出范围，请重新输入 (" + min + "-" + max + ")");
            } catch (NumberFormatException e) {
                System.out.println("无效输入，请输入整数");
            }
        }
    }

    /**
     * 获取浮点数输入
     */
    private double getDoubleInput(String prompt, double min, double max) {
        while (true) {
            System.out.print("\n" + prompt + ": ");
            try {
                String input = scanner.nextLine().trim();
                double value = Double.parseDouble(input);
                if (value >= min && value <= max) {
                    return value;
                }
                System.out.println("输入超出范围，请重新输入 (" + min + "-" + max + ")");
            } catch (NumberFormatException e) {
                System.out.println("无效输入，请输入数字");
            }
        }
    }

    /**
     * 暂停等待用户
     */
    private void pause() {
        System.out.print("\n按回车键继续...");
        scanner.nextLine();
    }

    /**
     * 显示告别信息
     */
    private void showGoodbye() {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("感谢使用 TinyAI 扫地机器人演示!");
        System.out.println("希望您对机器人路径规划有了更深的理解!");
        System.out.println("=".repeat(50));
    }

    /**
     * 主方法
     */
    public static void main(String[] args) {
        InteractiveRobotDemo demo = new InteractiveRobotDemo();
        demo.start();
    }
}
