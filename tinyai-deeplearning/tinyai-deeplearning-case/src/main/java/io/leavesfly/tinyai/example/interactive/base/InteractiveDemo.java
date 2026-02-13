package io.leavesfly.tinyai.example.interactive.base;

import io.leavesfly.tinyai.example.interactive.util.ConsoleVisualizer;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;

/**
 * 交互式演示基类
 *
 * <p>为所有交互式演示提供统一的基础设施：
 * <ul>
 *   <li>菜单系统</li>
 *   <li>参数输入验证</li>
 *   <li>进度显示</li>
 *   <li>结果保存</li>
 * </ul>
 *
 * <p>使用示例：</
> * <pre>{@code
 * public class MyDemo extends InteractiveDemo {
 *     public MyDemo() {
 *         super("我的演示");
 *     }
 *
 *     @Override
 *     protected void run() {
 *         showMenu("主菜单", new String[]{"选项1", "选项2"});
 *         int choice = getIntInput("请选择", 1, 2);
 *         // ... 处理选择
 *     }
 * }
 * }</pre>
 *
 * @author TinyAI Team
 */
public abstract class InteractiveDemo {

    protected final String demoName;
    protected final ConsoleVisualizer visualizer;
    protected final Scanner scanner;
    protected final BufferedReader reader;

    private boolean running = false;

    public InteractiveDemo(String demoName) {
        this(demoName, true);
    }

    public InteractiveDemo(String demoName, boolean useColor) {
        this.demoName = demoName;
        this.visualizer = new ConsoleVisualizer(80, useColor);
        this.scanner = new Scanner(System.in);
        this.reader = new BufferedReader(new InputStreamReader(System.in));
    }

    /**
     * 启动演示
     */
    public void start() {
        running = true;
        showWelcome();

        try {
            run();
        } catch (Exception e) {
            visualizer.printError("演示运行时出错: " + e.getMessage());
            e.printStackTrace();
        }

        showGoodbye();
    }

    /**
     * 子类实现此方法定义演示逻辑
     */
    protected abstract void run();

    /**
     * 停止演示
     */
    protected void stop() {
        running = false;
    }

    /**
     * 检查是否仍在运行
     */
    protected boolean isRunning() {
        return running;
    }

    // ==================== 显示方法 ====================

    protected void showWelcome() {
        visualizer.clearScreen();
        visualizer.printTitle("交互式演示: " + demoName);
        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("本演示支持交互式参数调整和实时可视化。", ConsoleVisualizer.CYAN);
        visualizer.println("", ConsoleVisualizer.RESET);
    }

    protected void showGoodbye() {
        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.printSeparator();
        visualizer.println("感谢使用！", ConsoleVisualizer.GREEN);
        visualizer.printSeparator();
    }

    protected void showMenu(String title, String[] options) {
        visualizer.printMenu(title, options);
    }

    protected void showSection(String title) {
        visualizer.println("", ConsoleVisualizer.RESET);
        visualizer.println("[" + title + "]", ConsoleVisualizer.BOLD + ConsoleVisualizer.CYAN);
        visualizer.printSeparator();
    }

    protected void showInfo(String message) {
        visualizer.printInfoBox(message);
    }

    protected void showSuccess(String message) {
        visualizer.printSuccess(message);
    }

    protected void showWarning(String message) {
        visualizer.printWarning(message);
    }

    protected void showError(String message) {
        visualizer.printError(message);
    }

    // ==================== 输入方法 ====================

    /**
     * 获取整数输入
     *
     * @param prompt 提示文本
     * @param min    最小值
     * @param max    最大值
     * @return 用户输入的整数
     */
    protected int getIntInput(String prompt, int min, int max) {
        while (true) {
            visualizer.print(prompt + " (" + min + "-" + max + "): ", ConsoleVisualizer.YELLOW);
            try {
                String input = scanner.nextLine().trim();
                if (input.isEmpty()) {
                    showWarning("请输入一个值");
                    continue;
                }
                int value = Integer.parseInt(input);
                if (value < min || value > max) {
                    showWarning("请输入 " + min + " 到 " + max + " 之间的值");
                    continue;
                }
                return value;
            } catch (NumberFormatException e) {
                showWarning("请输入有效的整数");
            }
        }
    }

    /**
     * 获取整数输入（带默认值）
     */
    protected int getIntInput(String prompt, int min, int max, int defaultValue) {
        while (true) {
            visualizer.print(prompt + " (" + min + "-" + max + ", 默认" + defaultValue + "): ", ConsoleVisualizer.YELLOW);
            try {
                String input = scanner.nextLine().trim();
                if (input.isEmpty()) {
                    return defaultValue;
                }
                int value = Integer.parseInt(input);
                if (value < min || value > max) {
                    showWarning("请输入 " + min + " 到 " + max + " 之间的值");
                    continue;
                }
                return value;
            } catch (NumberFormatException e) {
                showWarning("请输入有效的整数");
            }
        }
    }

    /**
     * 获取浮点数输入
     */
    protected float getFloatInput(String prompt, float min, float max) {
        while (true) {
            visualizer.print(prompt + " (" + min + "-" + max + "): ", ConsoleVisualizer.YELLOW);
            try {
                String input = scanner.nextLine().trim();
                if (input.isEmpty()) {
                    showWarning("请输入一个值");
                    continue;
                }
                float value = Float.parseFloat(input);
                if (value < min || value > max) {
                    showWarning("请输入 " + min + " 到 " + max + " 之间的值");
                    continue;
                }
                return value;
            } catch (NumberFormatException e) {
                showWarning("请输入有效的数字");
            }
        }
    }

    /**
     * 获取浮点数输入（带默认值）
     */
    protected float getFloatInput(String prompt, float min, float max, float defaultValue) {
        while (true) {
            visualizer.print(prompt + " (" + min + "-" + max + ", 默认" + defaultValue + "): ", ConsoleVisualizer.YELLOW);
            try {
                String input = scanner.nextLine().trim();
                if (input.isEmpty()) {
                    return defaultValue;
                }
                float value = Float.parseFloat(input);
                if (value < min || value > max) {
                    showWarning("请输入 " + min + " 到 " + max + " 之间的值");
                    continue;
                }
                return value;
            } catch (NumberFormatException e) {
                showWarning("请输入有效的数字");
            }
        }
    }

    /**
     * 获取字符串输入
     */
    protected String getStringInput(String prompt) {
        visualizer.print(prompt + ": ", ConsoleVisualizer.YELLOW);
        return scanner.nextLine().trim();
    }

    /**
     * 获取字符串输入（带默认值）
     */
    protected String getStringInput(String prompt, String defaultValue) {
        visualizer.print(prompt + " (默认: " + defaultValue + "): ", ConsoleVisualizer.YELLOW);
        String input = scanner.nextLine().trim();
        return input.isEmpty() ? defaultValue : input;
    }

    /**
     * 获取确认输入
     */
    protected boolean getYesNoInput(String prompt) {
        while (true) {
            visualizer.print(prompt + " (y/n): ", ConsoleVisualizer.YELLOW);
            String input = scanner.nextLine().trim().toLowerCase();
            if (input.equals("y") || input.equals("yes")) {
                return true;
            } else if (input.equals("n") || input.equals("no")) {
                return false;
            }
            showWarning("请输入 y 或 n");
        }
    }

    /**
     * 获取选项输入
     */
    protected int getChoiceInput(String prompt, int min, int max) {
        return getIntInput(prompt, min, max);
    }

    /**
     * 等待用户按键继续
     */
    protected void pause() {
        pause("按回车键继续...");
    }

    /**
     * 等待用户按键继续（自定义消息）
     */
    protected void pause(String message) {
        visualizer.waitForKey(message);
    }

    // ==================== 进度显示方法 ====================

    /**
     * 显示进度条
     */
    protected void showProgress(int current, int total, String prefix) {
        visualizer.printProgressBar(current, total, 30, prefix);
    }

    /**
     * 显示训练进度
     */
    protected void showTrainingProgress(int epoch, int totalEpochs, int batch,
                                         int totalBatches, float loss, float accuracy) {
        visualizer.printTrainingProgress(epoch, totalEpochs, batch, totalBatches, loss, accuracy);
    }

    /**
     * 清屏
     */
    protected void clearScreen() {
        visualizer.clearScreen();
    }

    // ==================== 工具方法 ====================

    /**
     * 格式化数字
     */
    protected String formatNumber(int number) {
        if (number >= 1_000_000) {
            return String.format("%.2fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.2fK", number / 1_000.0);
        }
        return String.valueOf(number);
    }

    /**
     * 格式化时间（毫秒转为可读格式）
     */
    protected String formatTime(long milliseconds) {
        if (milliseconds < 1000) {
            return milliseconds + "ms";
        } else if (milliseconds < 60000) {
            return String.format("%.2fs", milliseconds / 1000.0);
        } else {
            long minutes = milliseconds / 60000;
            long seconds = (milliseconds % 60000) / 1000;
            return minutes + "m " + seconds + "s";
        }
    }

    /**
     * 睡眠指定毫秒
     */
    protected void sleep(long milliseconds) {
        try {
            Thread.sleep(milliseconds);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
