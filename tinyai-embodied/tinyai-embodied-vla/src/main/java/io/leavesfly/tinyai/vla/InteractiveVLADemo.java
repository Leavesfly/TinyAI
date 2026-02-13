package io.leavesfly.tinyai.vla;

import io.leavesfly.tinyai.vla.env.RobotEnvironment;
import io.leavesfly.tinyai.vla.env.SimpleRobotEnv;
import io.leavesfly.tinyai.vla.env.TaskScenario;
import io.leavesfly.tinyai.vla.model.TaskConfig;
import io.leavesfly.tinyai.vla.model.VLAAction;
import io.leavesfly.tinyai.vla.model.VLAState;

import java.util.*;

/**
 * VLA（视觉-语言-动作）智能体交互式演示
 * 
 * <p>提供交互式的VLA系统：
 * <ul>
 *   <li>多模演示，包括态输入可视化 - 视觉、语言、本体感知</li>
 *   <li>跨模态融合展示 - 理解不同模态如何结合</li>
 *   <li>任务场景选择 - 拾取放置、堆叠、抽屉等</li>
 *   <li>零样本泛化演示 - 语言组合新任务</li>
 * </ul>
 * 
 * <p>学习目标：
 * <ul>
 *   <li>理解VLA架构的设计理念</li>
 *   <li>掌握多模态融合的核心技术</li>
 *   <li>了解零样本泛化的实现原理</li>
 * </ul>
 * 
 * @author TinyAI Team
 */
public class InteractiveVLADemo {

    // ANSI颜色代码
    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String CYAN = "\u001B[36m";
    private static final String MAGENTA = "\u001B[35m";
    private static final String BOLD = "\u001B[1m";

    private final Scanner scanner;
    private boolean running;

    // 当前配置
    private int hiddenDim = 768;
    private int numHeads = 8;
    private int numLayers = 6;
    private TaskScenario currentTask = TaskScenario.PICK_AND_PLACE;

    public InteractiveVLADemo() {
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
                    configureModel();
                    break;
                case 2:
                    selectTask();
                    break;
                case 3:
                    runDemoEpisode();
                    break;
                case 4:
                    demonstrateZeroShot();
                    break;
                case 5:
                    visualizeMultimodal();
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
        printTitle("TinyAI VLA 视觉-语言-动作 交互式演示");
        System.out.println("");
        System.out.println("本演示将带您体验多模态具身智能的魅力！");
        System.out.println("VLA架构将视觉感知、自然语言理解和动作生成统一建模。");
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
        System.out.println("  隐藏层维度: " + hiddenDim);
        System.out.println("  注意力头数: " + numHeads);
        System.out.println("  Transformer层数: " + numLayers);
        System.out.println("  当前任务: " + currentTask.getName() + " (" + currentTask.getDescription() + ")");
        System.out.println("");

        String[] options = {
            "配置模型参数",
            "选择任务场景",
            "运行演示回合",
            "零样本泛化演示",
            "多模态输入可视化",
            "查看架构说明",
            "退出演示"
        };

        printMenu("功能选择", options);
    }

    /**
     * 配置模型参数
     */
    private void configureModel() {
        clearScreen();
        printTitle("模型参数配置");

        System.out.println("1. 隐藏层维度 (当前: " + hiddenDim + ")");
        System.out.println("2. 注意力头数 (当前: " + numHeads + ")");
        System.out.println("3. Transformer层数 (当前: " + numLayers + ")");
        System.out.println("4. 返回主菜单");

        int choice = getIntInput("请选择", 1, 4);

        if (choice == 4) {
            return;
        }

        switch (choice) {
            case 1:
                hiddenDim = getIntInput("输入隐藏层维度 (256/512/768)", 256, 768);
                break;
            case 2:
                numHeads = getIntInput("输入注意力头数 (4/8/16)", 4, 16);
                break;
            case 3:
                numLayers = getIntInput("输入Transformer层数 (2/4/6/8)", 2, 8);
                break;
        }

        System.out.println("\n配置已更新!");
        pause();
    }

    /**
     * 选择任务场景
     */
    private void selectTask() {
        clearScreen();
        printTitle("任务场景选择");

        System.out.println("可选择的任务场景:");
        TaskScenario[] scenarios = TaskScenario.values();
        for (int i = 0; i < scenarios.length; i++) {
            TaskScenario task = scenarios[i];
            System.out.println("  " + (i + 1) + ". " + task.getName() + " " + task.getDifficultyStars());
            System.out.println("     描述: " + task.getDescription());
        }

        int choice = getIntInput("请选择任务", 1, scenarios.length);
        currentTask = scenarios[choice - 1];

        System.out.println("\n已选择任务: " + currentTask.getName());
        pause();
    }

    /**
     * 运行演示回合
     */
    private void runDemoEpisode() {
        clearScreen();
        printTitle("VLA 演示回合 - " + currentTask.getName());

        // 创建VLA智能体
        VLAAgent agent = new VLAAgent(hiddenDim, numHeads, numLayers, 7);
        System.out.println();

        // 创建任务环境
        TaskConfig taskConfig = new TaskConfig();
        taskConfig.setTaskName(currentTask.getName());
        taskConfig.setTaskDescription(currentTask.getDescription());
        taskConfig.setMaxSteps(20);
        taskConfig.setSuccessReward(100.0);
        taskConfig.setRender(false);

        RobotEnvironment env = new SimpleRobotEnv(taskConfig);
        VLAState state = env.reset();

        System.out.println("任务: " + currentTask.getName());
        System.out.println("难度: " + currentTask.getDifficultyStars());
        System.out.println("描述: " + currentTask.getDescription());
        System.out.println();

        // 显示初始状态
        System.out.println("初始状态:");
        visualizeState(state);

        System.out.println("\n开始执行任务...\n");

        int step = 0;
        double totalReward = 0;

        while (step < 20) {
            // 预测动作
            VLAAction action = agent.predict(state);

            // 显示动作信息
            System.out.printf("步骤 %2d | 动作类型: %-20s | 置信度: %.2f%n",
                step, action.getActionType().getDescription(), action.getConfidence());

            // 执行动作
            RobotEnvironment.EnvironmentStep envStep = env.step(action);
            totalReward += envStep.getReward();

            if (envStep.isDone()) {
                System.out.println("\n任务完成! 奖励: " + String.format("%.2f", totalReward));
                break;
            }

            state = envStep.getNextState();
            step++;
        }

        if (step >= 20) {
            System.out.println("\n任务超时! 奖励: " + String.format("%.2f", totalReward));
        }

        env.close();
        pause();
    }

    /**
     * 可视化状态
     */
    private void visualizeState(VLAState state) {
        if (state.getVisionInput() != null) {
            System.out.println("  视觉输入: 已接收");
        }
        if (state.getLanguageInput() != null) {
            System.out.println("  语言指令: 已接收");
        }
        if (state.getProprioceptionInput() != null) {
            System.out.println("  本体感知: 已接收");
        }
        if (state.getFusedFeatures() != null) {
            System.out.println("  融合特征: 维度 " + state.getFusedFeatures().getShape());
        }
    }

    /**
     * 零样本泛化演示
     */
    private void demonstrateZeroShot() {
        clearScreen();
        printTitle("零样本泛化演示");

        System.out.println("VLA的核心能力之一是零样本泛化：");
        System.out.println("通过组合语言指令，模型可以执行未见过的新任务。");
        System.out.println();

        // 创建智能体
        VLAAgent agent = new VLAAgent(hiddenDim, numHeads, numLayers, 7);

        // 组合任务示例
        String[][] compositions = {
            {"Pick up the red cube and place it on the left", "拾取红方块放到左边"},
            {"Push the blue block gently to the right", "轻轻向右推蓝方块"},
            {"Stack two small blocks on the large one", "把两个小方块堆到大方块上"}
        };

        System.out.println("语言指令组合示例:");
        for (int i = 0; i < compositions.length; i++) {
            System.out.println("  " + (i + 1) + ". " + compositions[i][0]);
            System.out.println("     中文: " + compositions[i][1]);
        }

        System.out.println();

        // 模拟执行
        System.out.println("模拟执行第一个指令...");
        TaskConfig config = new TaskConfig();
        config.setTaskName("ZeroShot");
        config.setMaxSteps(10);

        RobotEnvironment env = new SimpleRobotEnv(config);
        VLAState state = env.reset();

        // 模拟语言输入变化
        System.out.println("  - 视觉编码: 处理图像特征");
        System.out.println("  - 语言编码: 解析指令语义");
        System.out.println("  - 跨模态融合: 结合视觉和语言");
        System.out.println("  - 动作解码: 生成控制指令");
        System.out.println("  - 输出动作: 连续动作 + 离散动作");

        System.out.println("\n零样本泛化原理:");
        System.out.println("  1. 语言编码器理解指令意图");
        System.out.println("  2. 跨模态注意力将语言与视觉对齐");
        System.out.println("  3. 动作解码器根据语义生成动作");
        System.out.println("  4. 未见过的指令也能正确执行");

        env.close();
        pause();
    }

    /**
     * 多模态输入可视化
     */
    private void visualizeMultimodal() {
        clearScreen();
        printTitle("多模态输入可视化");

        System.out.println("VLA系统接收三种模态的输入：\n");

        // 视觉输入
        System.out.println("1. 视觉输入 (Vision):");
        System.out.println("   ┌─────────────────────────┐");
        System.out.println("   │     📷 相机图像         │");
        System.out.println("   │  [物体检测框]          │");
        System.out.println("   │  [深度图]              │");
        System.out.println("   └─────────────────────────┘");
        System.out.println("   → CNN特征提取 → 空间编码");
        System.out.println();

        // 语言输入
        System.out.println("2. 语言输入 (Language):");
        System.out.println("   ┌─────────────────────────┐");
        System.out.println("   │  \"Pick up the red cube\" │");
        System.out.println("   └─────────────────────────┘");
        System.out.println("   → Tokenize → Transformer编码");
        System.out.println();

        // 本体感知
        System.out.println("3. 本体感知 (Proprioception):");
        System.out.println("   ┌─────────────────────────┐");
        System.out.println("   │  关节角度: [0.5,0.3..] │");
        System.out.println("   │  关节速度: [0.1,0.0..] │");
        System.out.println("   │  夹爪状态: 0.8          │");
        System.out.println("   └─────────────────────────┘");
        System.out.println("   → MLP编码 → 状态表示");
        System.out.println();

        System.out.println("融合过程:");
        System.out.println("   ┌──────────────────────────────────────────┐");
        System.out.println("   │        跨模态注意力机制                  │");
        System.out.println("   │  Vision ↔ Language ↔ Proprioception     │");
        System.out.println("   └──────────────────────────────────────────┘");

        pause();
    }

    /**
     * 显示架构说明
     */
    private void showArchitecture() {
        clearScreen();
        printTitle("VLA 架构说明");

        System.out.println("VLA (Vision-Language-Action) 架构");
        System.out.println("=".repeat(50));
        System.out.println();

        System.out.println("┌─────────────────────────────────────────────────┐");
        System.out.println("│                 输入层                           │");
        System.out.println("│  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │");
        System.out.println("│  │  Vision │  │ Language │  │Proprioception│ │");
        System.out.println("│  └────┬────┘  └────┬────┘  └──────┬──────┘ │");
        System.out.println("└────────┼───────────┼───────────────┼──────────┘");
        System.out.println("         │           │               │");
        System.out.println("┌────────▼───────────▼───────────────▼──────────┐");
        System.out.println("│              编码器层                          │");
        System.out.println("│  ┌──────────┐ ┌──────────┐ ┌───────────┐     │");
        System.out.println("│  │  CNN     │ │ Transformer│ │   MLP     │     │");
        System.out.println("│  │  Encoder │ │  Encoder  │ │  Encoder  │     │");
        System.out.println("│  └────┬─────┘ └────┬─────┘ └─────┬─────┘     │");
        System.out.println("└────────┼───────────┼───────────────┼──────────┘");
        System.out.println("         │           │               │");
        System.out.println("┌────────▼───────────▼───────────────▼──────────┐");
        System.out.println("│           跨模态融合层                        │");
        System.out.println("│         Cross-Modal Attention                │");
        System.out.println("└────────────────────┬──────────────────────────┘");
        System.out.println("                     │");
        System.out.println("┌────────────────────▼──────────────────────────┐");
        System.out.println("│              动作解码器                       │");
        System.out.println("│    ┌─────────────┐  ┌─────────────────┐     │");
        System.out.println("│    │ 连续动作    │  │  离散动作        │     │");
        System.out.println("│    │ Continuous  │  │  Discrete       │     │");
        System.out.println("│    └─────────────┘  └─────────────────┘     │");
        System.out.println("└─────────────────────────────────────────────┘");
        System.out.println();

        System.out.println("核心特性:");
        System.out.println("  ✓ 统一建模: 视觉、语言、动作在一个模型中处理");
        System.out.println("  ✓ 跨模态注意力: 不同模态之间相互对齐");
        System.out.println("  ✓ 零样本泛化: 通过语言组合完成新任务");
        System.out.println("  ✓ 端到端: 从感知到动作的完整可微分化");

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
        System.out.println("感谢使用 TinyAI VLA 交互式演示!");
        System.out.println("希望您对多模态具身智能有了更深的理解!");
        System.out.println("=".repeat(50));
    }

    /**
     * 主方法
     */
    public static void main(String[] args) {
        InteractiveVLADemo demo = new InteractiveVLADemo();
        demo.start();
    }
}
