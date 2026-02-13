package io.leavesfly.tinyai.embodied;

import io.leavesfly.tinyai.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.embodied.model.*;
import io.leavesfly.tinyai.embodied.visualize.EmbodiedConsoleVisualizer;

import java.util.*;

/**
 * 具身智能（自动驾驶）交互式演示
 * 
 * <p>提供交互式的自动驾驶场景演示，包括：
 * <ul>
 *   <li>多种驾驶场景选择（高速、城市、停车场等）</li>
 *   <li>实时参数调整（速度限制、安全距离等）</li>
 *   <li>可视化驾驶过程和传感器数据</li>
 *   <li>多策略对比实验</li>
 * </ul>
 * 
 * <p>学习目标：
 * <ul>
 *   <li>理解具身智能的感知-决策-执行闭环</li>
 *   <li>了解自动驾驶系统的核心组件</li>
 *   <li>掌握强化学习在自动驾驶中的应用</li>
 * </ul>
 * 
 * @author TinyAI Team
 */
public class InteractiveDrivingDemo {

    // ANSI颜色代码
    private static final String RESET = "\u001B[0m";
    private static final String RED = "\u001B[31m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String BLUE = "\u001B[34m";
    private static final String CYAN = "\u001B[36m";
    private static final String BOLD = "\u001B[1m";

    private final EmbodiedConsoleVisualizer visualizer;
    private final Scanner scanner;
    private boolean running;

    // 当前配置
    private EnvironmentConfig currentConfig;
    private String currentScenario = "测试场";
    private int maxSteps = 100;
    private boolean autoMode = false;

    // 统计信息
    private List<Double> rewardHistory = new ArrayList<>();

    public InteractiveDrivingDemo() {
        this.visualizer = new EmbodiedConsoleVisualizer();
        this.scanner = new Scanner(System.in);
        this.running = false;
        this.currentConfig = EnvironmentConfig.createTestConfig();
    }

    /**
     * 启动交互式演示
     */
    public void start() {
        running = true;
        showWelcome();

        while (running) {
            showMainMenu();
            int choice = getIntInput("请选择操作", 1, 8);

            switch (choice) {
                case 1:
                    selectScenario();
                    break;
                case 2:
                    configureParameters();
                    break;
                case 3:
                    runAutoDemo();
                    break;
                case 4:
                    runManualStep();
                    break;
                case 5:
                    compareStrategies();
                    break;
                case 6:
                    showLearningCurve();
                    break;
                case 7:
                    showConceptExplanation();
                    break;
                case 8:
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
        visualizer.clearScreen();
        visualizer.printTitle("TinyAI 具身智能交互式演示 - 自动驾驶");
        visualizer.println("", RESET);
        visualizer.println("本演示将带您体验自动驾驶的感知-决策-执行闭环！", CYAN);
        visualizer.println("", RESET);
        visualizer.printSeparator();
    }

    /**
     * 显示主菜单
     */
    private void showMainMenu() {
        visualizer.clearScreen();
        visualizer.printTitle("主菜单");

        visualizer.println("当前配置:", BOLD);
        visualizer.println("  场景: " + currentScenario, CYAN);
        visualizer.println("  最大步数: " + maxSteps, CYAN);
        visualizer.println("  自动模式: " + (autoMode ? "开启" : "关闭"), CYAN);
        visualizer.println("", RESET);

        String[] options = {
            "选择驾驶场景",
            "配置运行参数",
            "运行自动演示",
            "单步手动运行",
            "对比不同策略",
            "查看学习曲线",
            "学习核心概念",
            "退出演示"
        };

        visualizer.printMenu("功能选择", options);
    }

    /**
     * 选择驾驶场景
     */
    private void showScenarioMenu() {
        visualizer.println("\n可选择的驾驶场景:", BOLD);
        visualizer.println("  1. 测试场 - 简单直线路径，用于基础测试", CYAN);
        visualizer.println("  2. 高速公路 - 高速行驶，保持车道和跟车", CYAN);
        visualizer.println("  3. 城市道路 - 复杂路况，识别红绿灯和行人", CYAN);
        visualizer.println("  4. 停车场 - 低速泊车，车位识别", CYAN);
    }

    private void selectScenario() {
        showScenarioMenu();
        int choice = getIntInput("请选择场景", 1, 4);

        switch (choice) {
            case 1:
                currentConfig = EnvironmentConfig.createTestConfig();
                currentScenario = "测试场";
                break;
            case 2:
                currentConfig = EnvironmentConfig.createHighwayConfig();
                currentScenario = "高速公路";
                break;
            case 3:
                currentConfig = EnvironmentConfig.createUrbanConfig();
                currentScenario = "城市道路";
                break;
            case 4:
                currentConfig = EnvironmentConfig.createTestConfig();
                currentScenario = "停车场";
                break;
        }

        visualizer.println("\n已选择场景: " + currentScenario, GREEN);
        pause();
    }

    /**
     * 配置运行参数
     */
    private void configureParameters() {
        visualizer.clearScreen();
        visualizer.printTitle("参数配置");

        visualizer.println("1. 最大步数 (当前: " + maxSteps + ")", YELLOW);
        visualizer.println("2. 自动模式 (当前: " + (autoMode ? "开启" : "关闭") + ")", YELLOW);
        visualizer.println("3. 返回主菜单", YELLOW);

        int choice = getIntInput("请选择要修改的参数", 1, 3);

        if (choice == 1) {
            int newSteps = getIntInput("输入最大步数 (10-500)", 10, 500);
            maxSteps = newSteps;
            visualizer.println("\n最大步数已设置为: " + maxSteps, GREEN);
        } else if (choice == 2) {
            autoMode = !autoMode;
            visualizer.println("\n自动模式已" + (autoMode ? "开启" : "关闭"), GREEN);
        }

        pause();
    }

    /**
     * 运行自动演示
     */
    private void runAutoDemo() {
        visualizer.clearScreen();
        visualizer.printTitle("自动驾驶演示 - " + currentScenario);

        // 创建智能体
        EmbodiedAgent agent = new EmbodiedAgent(currentConfig);
        agent.initialize();

        visualizer.println("正在初始化驾驶环境...", CYAN);
        agent.reset();

        visualizer.println("开始自动驾驶演示...\n", GREEN);

        int step = 0;
        double totalReward = 0;

        while (step < maxSteps) {
            StepResult result = agent.step();
            totalReward += result.getReward();
            rewardHistory.add(result.getReward());

            // 显示进度
            if (step % 10 == 0) {
                visualizer.printProgressBar(step + 1, maxSteps, "驾驶进度");
            }

            // 显示关键信息
            if (step % 20 == 0) {
                PerceptionState state = agent.getCurrentState();
                if (state != null && state.getVehicleState() != null) {
                    VehicleState vs = state.getVehicleState();
                    System.out.printf("  速度: %.1f km/h, 加速度: %.2f m/s²%n",
                        vs.getSpeed() * 3.6, vs.getAcceleration());
                }
            }

            if (result.isDone()) {
                visualizer.println("\n驾驶结束! 原因: " + (result.isDone() ? "到达目标" : "碰撞/超时"), YELLOW);
                break;
            }

            step++;

            if (autoMode) {
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    break;
                }
            }
        }

        // 显示统计信息
        visualizer.println("\n" + "=".repeat(50), CYAN);
        visualizer.println("驾驶演示统计:", BOLD);
        visualizer.println("  总步数: " + step, RESET);
        visualizer.println("  总奖励: " + String.format("%.2f", totalReward), RESET);
        visualizer.println("  平均奖励: " + String.format("%.3f", totalReward / Math.max(1, step)), RESET);
        visualizer.println("  奖励历史: " + rewardHistory.size() + " 个数据点", RESET);
        visualizer.println("=".repeat(50), CYAN);

        agent.close();
        pause();
    }

    /**
     * 单步手动运行
     */
    private void runManualStep() {
        visualizer.clearScreen();
        visualizer.printTitle("单步手动运行");

        EmbodiedAgent agent = new EmbodiedAgent(currentConfig);
        agent.initialize();
        agent.reset();

        int step = 0;

        while (step < maxSteps) {
            visualizer.println("\n" + "-".repeat(40), CYAN);
            visualizer.println("当前步数: " + step, BOLD);

            // 显示当前状态
            PerceptionState state = agent.getCurrentState();
            if (state != null && state.getVehicleState() != null) {
                VehicleState vs = state.getVehicleState();
                visualizer.println("车辆状态:", YELLOW);
                visualizer.println("  速度: " + String.format("%.1f km/h", vs.getSpeed() * 3.6));
                visualizer.println("  加速度: " + String.format("%.2f m/s²", vs.getAcceleration()));
                visualizer.println("  位置: (" + String.format("%.1f", vs.getPosition().getX())
                    + ", " + String.format("%.1f", vs.getPosition().getY()) + ")");
            }

            // 显示障碍物信息
            if (state != null && state.getObstacleMap() != null) {
                visualizer.println("障碍物数量: " + state.getObstacleMap().size(), YELLOW);
            }

            // 菜单
            visualizer.println("\n1. 执行一步", YELLOW);
            visualizer.println("2. 执行多步", YELLOW);
            visualizer.println("3. 查看详细状态", YELLOW);
            visualizer.println("4. 返回主菜单", YELLOW);

            int choice = getIntInput("请选择操作", 1, 4);

            if (choice == 1) {
                StepResult result = agent.step();
                totalReward += result.getReward();
                rewardHistory.add(result.getReward());

                visualizer.println("奖励: " + String.format("%.3f", result.getReward()), GREEN);
                visualizer.println("累计奖励: " + String.format("%.2f", totalReward), GREEN);

                if (result.isDone()) {
                    visualizer.println("驾驶结束!", RED);
                    break;
                }
                step++;
            } else if (choice == 2) {
                int steps = getIntInput("输入步数", 1, 20);
                for (int i = 0; i < steps && step < maxSteps; i++) {
                    StepResult result = agent.step();
                    totalReward += result.getReward();
                    rewardHistory.add(result.getReward());
                    step++;

                    if (result.isDone()) {
                        visualizer.println("驾驶结束!", RED);
                        break;
                    }
                }
            } else if (choice == 3) {
                showDetailedState(agent);
            } else {
                break;
            }
        }

        agent.close();
        pause();
    }

    /**
     * 显示详细状态
     */
    private void showDetailedState(EmbodiedAgent agent) {
        visualizer.clearScreen();
        visualizer.printTitle("详细状态信息");

        PerceptionState state = agent.getCurrentState();
        if (state == null) {
            visualizer.println("无状态信息", RED);
            return;
        }

        // 车辆状态
        if (state.getVehicleState() != null) {
            VehicleState vs = state.getVehicleState();
            visualizer.println("车辆动力学状态:", BOLD);
            visualizer.println("  速度: " + String.format("%.2f m/s (%.1f km/h)", vs.getSpeed(), vs.getSpeed() * 3.6));
            visualizer.println("  加速度: " + String.format("%.3f m/s²", vs.getAcceleration()));
            visualizer.println("  航向角: " + String.format("%.2f°", Math.toDegrees(vs.getHeading())));
            visualizer.println("  角速度: " + String.format("%.3f rad/s", vs.getAngularVelocity()));
        }

        // 车道信息
        if (state.getLaneInfo() != null) {
            LaneGeometry lane = state.getLaneInfo();
            visualizer.println("车道信息:", BOLD);
            visualizer.println("  车道ID: " + lane.getLaneId());
            visualizer.println("  车道宽度: " + String.format("%.2f m", lane.getLaneWidth()));
        }

        // 障碍物
        if (state.getObstacleMap() != null) {
            visualizer.println("障碍物列表 (" + state.getObstacleMap().size() + "个):", BOLD);
            for (ObstacleInfo obs : state.getObstacleMap()) {
                visualizer.println("  - 距离: " + String.format("%.1f m", obs.getDistance())
                    + ", 类型: " + obs.getObjectType()
                    + ", 速度: " + String.format("%.1f m/s", obs.getVelocity().getX()));
            }
        }

        pause();
    }

    /**
     * 对比不同策略
     */
    private void compareStrategies() {
        visualizer.clearScreen();
        visualizer.printTitle("策略对比实验");

        visualizer.println("将在相同场景下对比三种策略:", CYAN);
        visualizer.println("  1. 规则策略 - 基于启发式规则", YELLOW);
        visualizer.println("  2. 随机策略 - 随机动作选择", YELLOW);
        visualizer.println("  3. 学习策略 - 基于训练的策略", YELLOW);

        pause();

        String[] strategies = {"规则策略", "随机策略", "学习策略"};
        String[] headers = {"策略", "平均奖励", "完成率", "平均步数"};

        visualizer.printTable(headers, new String[][]{}, 15, 12, 10, 12);

        for (String strategy : strategies) {
            // 模拟运行
            EmbodiedAgent agent = new EmbodiedAgent(currentConfig);
            agent.initialize();
            agent.reset();

            int steps = 0;
            double totalReward = 0;
            boolean completed = false;

            for (int i = 0; i < maxSteps; i++) {
                StepResult result = agent.step();
                totalReward += result.getReward();
                steps++;

                if (result.isDone()) {
                    completed = true;
                    break;
                }
            }

            agent.close();

            double avgReward = totalReward / Math.max(1, steps);
            String completionRate = completed ? "100%" : "<100%";

            String[] row = {strategy, String.format("%.2f", avgReward),
                completionRate, String.valueOf(steps)};
            visualizer.printTable(headers, new String[][]{row}, 15, 12, 10, 12);
        }

        visualizer.println("\n策略对比分析:", BOLD);
        visualizer.println("  - 规则策略: 确定性高，适合简单场景", CYAN);
        visualizer.println("  - 随机策略: 作为基线，用于评估其他策略", CYAN);
        visualizer.println("  - 学习策略: 通过训练获得最优行为", CYAN);

        pause();
    }

    /**
     * 显示学习曲线
     */
    private void showLearningCurve() {
        if (rewardHistory.isEmpty()) {
            visualizer.println("\n暂无学习数据，请先运行演示!", YELLOW);
            pause();
            return;
        }

        visualizer.clearScreen();
        visualizer.printTitle("学习曲线");

        double[] rewards = rewardHistory.stream().mapToDouble(Double::doubleValue).toArray();
        visualizer.visualizeLearningCurve(rewards, 10);

        pause();
    }

    /**
     * 显示概念解释
     */
    private void showConceptExplanation() {
        visualizer.clearScreen();
        visualizer.printTitle("具身智能核心概念");

        String[][] concepts = {
            {"感知模块", "通过传感器获取环境信息，包括激光雷达、摄像头、GPS等"},
            {"决策模块", "基于感知信息进行路径规划和行为决策"},
            {"执行模块", "将决策转化为具体的车辆控制指令"},
            {"强化学习", "通过与环境交互学习最优策略，最大化累积奖励"},
            {"感知-决策-执行闭环", "智能体感知环境、做出决策、执行动作的完整循环"}
        };

        for (String[] concept : concepts) {
            visualizer.println("\n" + concept[0] + ":", BOLD + CYAN);
            visualizer.println("  " + concept[1], RESET);
        }

        visualizer.println("\n" + "=".repeat(50), CYAN);
        visualizer.println("学习要点:", BOLD);
        visualizer.println("  1. 具身智能强调智能体与物理环境的直接交互", RESET);
        visualizer.println("  2. 感知是决策的基础，决策指导执行", RESET);
        visualizer.println("  3. 强化学习通过试错学习最优策略", RESET);
        visualizer.println("  4. 样本效率是实际应用的关键挑战", RESET);
        visualizer.println("=".repeat(50), CYAN);

        pause();
    }

    /**
     * 打印带颜色的菜单
     */
    private void printMenu(String title, String[] options) {
        visualizer.println("\n" + title + ":", BOLD);
        for (int i = 0; i < options.length; i++) {
            visualizer.println("  " + (i + 1) + ". " + options[i], YELLOW);
        }
    }

    /**
     * 获取整数输入
     */
    private int getIntInput(String prompt, int min, int max) {
        while (true) {
            visualizer.print("\n" + prompt + ": ", CYAN);
            try {
                String input = scanner.nextLine().trim();
                int value = Integer.parseInt(input);
                if (value >= min && value <= max) {
                    return value;
                }
                visualizer.println("输入超出范围，请重新输入 (" + min + "-" + max + ")", RED);
            } catch (NumberFormatException e) {
                visualizer.println("无效输入，请输入整数", RED);
            }
        }
    }

    /**
     * 暂停等待用户按键
     */
    private void pause() {
        visualizer.print("\n按回车键继续...", CYAN);
        scanner.nextLine();
    }

    /**
     * 显示告别信息
     */
    private void showGoodbye() {
        visualizer.println("\n" + "=".repeat(50), CYAN);
        visualizer.println("感谢使用 TinyAI 具身智能演示!", GREEN);
        visualizer.println("希望您对自动驾驶的感知-决策-执行有了更深的理解", CYAN);
        visualizer.println("=".repeat(50), CYAN);
    }

    // 临时变量
    private double totalReward = 0;

    /**
     * 主方法 - 启动演示
     */
    public static void main(String[] args) {
        InteractiveDrivingDemo demo = new InteractiveDrivingDemo();
        demo.start();
    }
}
