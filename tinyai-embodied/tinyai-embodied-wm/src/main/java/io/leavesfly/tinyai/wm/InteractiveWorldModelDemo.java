package io.leavesfly.tinyai.wm;

import io.leavesfly.tinyai.wm.core.WorldModel;
import io.leavesfly.tinyai.wm.env.Environment;
import io.leavesfly.tinyai.wm.env.SimpleDrivingEnvironment;
import io.leavesfly.tinyai.wm.model.Episode;

import java.util.*;

/**
 * 世界模型（World Model）交互式演示
 * 
 * <p>提供交互式的世界模型演示：
 * <ul>
 *   <li>VAE编码可视化 - 观察压缩为潜在表示</li>
 *   <li>MDN-RNN预测展示 - 时序动态预测</li>
 *   <li>想象训练演示 - 在内部模型中学习</li>
 *   <li>样本效率对比 - 真实vs想象训练</li>
 * </ul>
 * 
 * <p>学习目标：
 * <ul>
 *   <li>理解世界模型的核心思想</li>
 *   <li>掌握VAE和MDN-RNN的工作原理</li>
 *   <li>了解想象训练如何提升样本效率</li>
 * </ul>
 * 
 * @author TinyAI Team
 */
public class InteractiveWorldModelDemo {

    // ANSI颜色代码
    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String CYAN = "\u001B[36m";
    private static final String RED = "\u001B[31m";

    private final Scanner scanner;
    private boolean running;

    // 配置参数
    private int observationSize = 8;
    private int latentSize = 32;
    private int hiddenSize = 256;
    private int actionSize = 3;

    public InteractiveWorldModelDemo() {
        this.scanner = new Scanner(System.in);
        this.running = false;
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
                    configureParameters();
                    break;
                case 2:
                    demonstrateVAE();
                    break;
                case 3:
                    demonstrateMDNRNN();
                    break;
                case 4:
                    demonstrateDreamTraining();
                    break;
                case 5:
                    compareEfficiency();
                    break;
                case 6:
                    showArchitecture();
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
        printTitle("TinyAI 世界模型 交互式演示");
        System.out.println("");
        System.out.println("本演示将带您理解世界模型的核心概念！");
        System.out.println("世界模型让智能体能够'在想象中学习'，大幅提升样本效率。");
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
        System.out.println("  观察维度: " + observationSize);
        System.out.println("  潜在空间维度: " + latentSize);
        System.out.println("  RNN隐藏维度: " + hiddenSize);
        System.out.println("  动作维度: " + actionSize);
        System.out.println("");

        String[] options = {
            "配置模型参数",
            "VAE编码演示",
            "MDN-RNN预测演示",
            "想象训练演示",
            "样本效率对比",
            "架构说明",
            "退出演示"
        };

        printMenu("功能选择", options);
    }

    /**
     * 配置参数
     */
    private void configureParameters() {
        clearScreen();
        printTitle("模型参数配置");

        System.out.println("1. 观察维度 (当前: " + observationSize + ")");
        System.out.println("2. 潜在空间维度 (当前: " + latentSize + ")");
        System.out.println("3. RNN隐藏维度 (当前: " + hiddenSize + ")");
        System.out.println("4. 动作维度 (当前: " + actionSize + ")");
        System.out.println("5. 返回主菜单");

        int choice = getIntInput("请选择", 1, 5);

        if (choice == 5) {
            return;
        }

        switch (choice) {
            case 1:
                observationSize = getIntInput("输入观察维度 (4/8/16)", 4, 16);
                break;
            case 2:
                latentSize = getIntInput("输入潜在维度 (16/32/64)", 16, 64);
                break;
            case 3:
                hiddenSize = getIntInput("输入隐藏维度 (128/256/512)", 128, 512);
                break;
            case 4:
                actionSize = getIntInput("输入动作维度 (2/3/4)", 2, 4);
                break;
        }

        System.out.println("\n配置已更新!");
        pause();
    }

    /**
     * VAE编码演示
     */
    private void demonstrateVAE() {
        clearScreen();
        printTitle("VAE 编码演示");

        System.out.println("VAE (Variational Autoencoder) 将高维观察压缩为低维潜在表示：\n");

        // 模拟观察数据
        double[] observation = new double[observationSize];
        Random random = new Random(42);
        for (int i = 0; i < observationSize; i++) {
            observation[i] = random.nextDouble() * 2 - 1;
        }

        System.out.println("原始观察 (维度: " + observationSize + "):");
        System.out.print("  [");
        for (int i = 0; i < Math.min(observationSize, 8); i++) {
            System.out.printf("%.2f ", observation[i]);
        }
        if (observationSize > 8) {
            System.out.print("...");
        }
        System.out.println("]");

        System.out.println();

        // 模拟编码
        System.out.println("编码过程:");
        System.out.println("  1. 输入层: " + observationSize + "维观察向量");
        System.out.println("  2. 隐藏层: 压缩到中间表示");
        System.out.println("  3. 潜在层: 输出均值和方差");

        System.out.println();

        // 模拟潜在表示
        double[] latent = new double[latentSize];
        for (int i = 0; i < latentSize; i++) {
            latent[i] = (random.nextDouble() - 0.5) * 2;
        }

        System.out.println("压缩后的潜在表示 (维度: " + latentSize + "):");
        System.out.print("  [");
        for (int i = 0; i < Math.min(latentSize, 10); i++) {
            System.out.printf("%.2f ", latent[i]);
        }
        if (latentSize > 10) {
            System.out.print("...");
        }
        System.out.println("]");

        System.out.println();

        // 压缩比
        double ratio = (double) observationSize / latentSize;
        System.out.println(GREEN + "压缩比: " + String.format("%.1f", ratio) + "x" + RESET);
        System.out.println("  - 原始维度: " + observationSize);
        System.out.println("  - 潜在维度: " + latentSize);
        System.out.println("  - 压缩效率提升 " + String.format("%.0f", (ratio - 1) * 100) + "%");

        pause();
    }

    /**
     * MDN-RNN预测演示
     */
    private void demonstrateMDNRNN() {
        clearScreen();
        printTitle("MDN-RNN 预测演示");

        System.out.println("MDN-RNN (Mixture Density Network RNN) 预测环境动态：\n");

        System.out.println("工作原理:");
        System.out.println("  - 输入: 当前潜在状态 + 动作");
        System.out.println("  - 输出: 下一时刻潜在状态的概率分布");
        System.out.println("  - 核心: 混合高斯分布预测不确定性");

        System.out.println();

        // 模拟预测
        System.out.println("预测示例:");
        System.out.println("  时间步 t:");
        System.out.println("    潜在状态: [0.1, 0.3, -0.2, ...]");
        System.out.println("    执行动作: [0.5, 0.0, -0.1]");

        System.out.println();
        System.out.println("  时间步 t+1 (预测):");
        System.out.println("    预测分布: 5个高斯混合");
        System.out.println("    均值: [0.15, 0.35, -0.15, ...]");
        System.out.println("    方差: [0.01, 0.02, 0.01, ...]");
        System.out.println("    混合权重: [0.3, 0.25, 0.2, 0.15, 0.1]");

        System.out.println();

        // 可视化
        System.out.println("预测可视化:");
        System.out.println("  实际轨迹: ●●●●●●●●●●");
        System.out.println("  预测轨迹: ○○○○○○○○○○");
        System.out.println("  不确定区域: ░░░░░░░░░");

        pause();
    }

    /**
     * 想象训练演示
     */
    private void demonstrateDreamTraining() {
        clearScreen();
        printTitle("想象训练演示");

        System.out.println("想象训练 (Dream Training) 让智能体在内部模型中学习：\n");

        System.out.println("训练流程:");
        System.out.println("  1. 在真实环境中收集少量经验");
        System.out.println("  2. 训练VAE学习观察表示");
        System.out.println("  3. 训练MDN-RNN学习动态");
        System.out.println("  4. 在世界模型中大量想象训练");
        System.out.println("  5. 将学到的策略迁移到真实环境");

        System.out.println();

        // 模拟数据
        System.out.println("样本效率对比:");
        System.out.println();
        System.out.println("  真实环境训练:");
        System.out.println("    - 收集10万步真实交互");
        System.out.println("    - 耗时: 数小时到数天");
        System.out.println("    - 成本: 高");

        System.out.println();
        System.out.println("  想象训练 (世界模型):");
        System.out.println("    - 收集1千步真实交互");
        System.out.println("    - 在模型中想象100万步");
        System.out.println("    - 耗时: 分钟级别");
        System.out.println("    " + GREEN + "效率提升: 100倍+" + RESET);

        System.out.println();

        // 模拟训练过程
        System.out.println("想象训练过程:");
        System.out.println("  [初始化] ──→ [编码观察] ──→ [预测动态]");
        System.out.println("     ↑                                        ↓");
        System.out.println("     └────────── [更新策略] ←──────────────┘");

        System.out.println();
        System.out.println("优势:");
        System.out.println("  ✓ 样本效率极高");
        System.out.println("  ✓ 可以做危险/昂贵的实验");
        System.out.println("  ✓ 训练速度快");
        System.out.println("  ✗ 受限于模型精度");

        pause();
    }

    /**
     * 样本效率对比
     */
    private void compareEfficiency() {
        clearScreen();
        printTitle("样本效率对比");

        System.out.println("真实训练 vs 想象训练 对比实验\n");

        // 模拟数据
        String[] methods = {"DQN (真实)", "世界模型 (想象)"};
        String[] headers = {"方法", "真实步数", "想象步数", "训练时间", "最终性能"};

        printTable(headers, new String[][]{}, 18, 14, 14, 14, 12);

        String[][] data = {
            {"DQN (真实)", "100,000", "0", "10小时", "95分"},
            {"世界模型", "1,000", "99,000", "30分钟", "92分"}
        };

        for (String[] row : data) {
            printTable(headers, new String[][]{row}, 18, 14, 14, 14, 12);
        }

        System.out.println();
        System.out.println("关键发现:");
        System.out.println("  1. 真实步数减少 99%");
        System.out.println("  2. 训练时间减少 95%");
        System.out.println("  3. 最终性能仅下降 3%");

        System.out.println();
        System.out.println("适用场景:");
        System.out.println("  ✓ 机器人控制 (真实训练昂贵)");
        System.out.println("  ✓ 自动驾驶 (危险场景)");
        System.out.println("  ✓ 医疗模拟 (高风险干预)");
        System.out.println("  ✗ 简单任务 (直接训练更高效)");

        pause();
    }

    /**
     * 架构说明
     */
    private void showArchitecture() {
        clearScreen();
        printTitle("世界模型架构");

        System.out.println("V-M-C 架构 (Vision-Memory-Controller)");
        System.out.println("=".repeat(50));
        System.out.println();

        System.out.println("┌─────────────────────────────────────────────────────────┐");
        System.out.println("│                    真实环境                              │");
        System.out.println("│  观察(o_t) ──────→ 动作(a_t) ──────→ 奖励(r_t)         │");
        System.out.println("└───────────────────────────┬─────────────────────────────┘");
        System.out.println("                            ↓");
        System.out.println("┌─────────────────────────────────────────────────────────┐");
        System.out.println("│                      V (Vision)                        │");
        System.out.println("│                  VAE 编码器                            │");
        System.out.println("│  高维观察 ──→ 压缩 ──→ 潜在表示 z_t                   │");
        System.out.println("└───────────────────────────┬─────────────────────────────┘");
        System.out.println("                            ↓");
        System.out.println("┌─────────────────────────────────────────────────────────┐");
        System.out.println("│                    M (Memory)                          │");
        System.out.println("│                  MDN-RNN                                │");
        System.out.println("│  z_t + a_t ──→ 预测 ──→ z_{t+1} 分布                  │");
        System.out.println("└───────────────────────────┬─────────────────────────────┘");
        System.out.println("                            ↓");
        System.out.println("┌─────────────────────────────────────────────────────────┐");
        System.out.println("│                   C (Controller)                       │");
        System.out.println("│                    控制器                              │");
        System.out.println("│  z_t ──→ 动作 a_t                                     │");
        System.out.println("└─────────────────────────────────────────────────────────┘");

        System.out.println();

        System.out.println("训练流程:");
        System.out.println("  阶段1: 收集数据训练 VAE");
        System.out.println("  阶段2: 收集序列训练 MDN-RNN");
        System.out.println("  阶段3: 在想象环境中训练 Controller");

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
            System.out.printf("%-" + 18 + "s ", header);
        }
        System.out.println();
        System.out.println("-".repeat(80));

        for (String[] row : rows) {
            for (String cell : row) {
                System.out.printf("%-" + 18 + "s ", cell);
            }
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
        System.out.println("感谢使用 TinyAI 世界模型交互式演示!");
        System.out.println("希望您对世界模型和想象训练有了更深的理解!");
        System.out.println("=".repeat(50));
    }

    /**
     * 主方法
     */
    public static void main(String[] args) {
        InteractiveWorldModelDemo demo = new InteractiveWorldModelDemo();
        demo.start();
    }
}
