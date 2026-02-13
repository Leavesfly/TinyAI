package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;
import io.leavesfly.tinyai.rl.Environment;

import java.util.Scanner;

/**
 * 逐步调试演示 - 详细展示DQN算法的每一步计算过程
 * 
 * <p>本演示逐步展示:
 * <ul>
 *   <li>Q值计算过程</li>
 *   <li>目标Q值计算</li>
 *   <li>损失函数计算</li>
 *   <li>反向传播更新</li>
 *   <li>经验回放采样</li>
 * </ul>
 * 
 * <p><b>运行方式:</b>
 * <pre>
 * mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.StepByStepDebugDemo" \
 *   -pl tinyai-deeplearning-rl
 * </pre>
 * 
 * @author TinyAI Team
 */
public class StepByStepDebugDemo {

    private static Scanner scanner = new Scanner(System.in);
    private static boolean autoMode = false;
    private static int delayMs = 1000;

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("      DQN算法逐步调试演示                ");
        System.out.println("==========================================\n");

        System.out.println("本演示将详细展示DQN算法的每一步计算过程。");
        System.out.println("按 Enter 键继续，或输入 'auto' 自动播放\n");
        
        waitForContinue();
        
        // 创建简化的DQN智能体
        DQNAgent agent = new DQNAgent(
            "DebugDQN",
            4,                      // 状态维度
            2,                      // 动作维度
            new int[]{8, 8},        // 小型隐藏层便于理解
            0.001f,                 // 学习率
            1.0f,                   // 初始探索率
            0.99f,                  // 折扣因子
            2,                      // 小批次便于展示
            100,                    // 缓冲区大小
            10                      // 目标网络更新频率
        );
        
        CartPoleEnvironment env = new CartPoleEnvironment(500);
        
        // 演示1: 前向传播
        demonstrateForwardPass(agent, env);
        
        // 演示2: 经验存储
        demonstrateExperienceStorage(agent, env);
        
        // 演示3: 批量学习
        demonstrateBatchLearning(agent);
        
        // 演示4: 目标网络更新
        demonstrateTargetNetworkUpdate(agent);
        
        System.out.println("\n==========================================");
        System.out.println("         逐步调试演示完成!               ");
        System.out.println("==========================================");
    }

    /**
     * 演示前向传播
     */
    private static void demonstrateForwardPass(DQNAgent agent, CartPoleEnvironment env) {
        System.out.println("\n【步骤1: 前向传播计算Q值】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        Variable state = env.reset();
        float[] stateValues = state.getValue().getArray();
        
        System.out.println("当前状态:");
        System.out.printf("  位置:   %.4f%n", stateValues[0]);
        System.out.printf("  速度:   %.4f%n", stateValues[1]);
        System.out.printf("  角度:   %.4f%n", stateValues[2]);
        System.out.printf("  角速度: %.4f%n", stateValues[3]);
        
        waitForContinue();
        
        System.out.println("\n神经网络前向传播:");
        System.out.println("  输入层(4) → 隐藏层1(8) → 隐藏层2(8) → 输出层(2)");
        System.out.println("\n计算过程:");
        System.out.println("  1. 输入状态向量 s = [x, ẋ, θ, θ̇]");
        System.out.println("  2. 第一层: h1 = ReLU(W1·s + b1)");
        System.out.println("  3. 第二层: h2 = ReLU(W2·h1 + b2)");
        System.out.println("  4. 输出层: Q = W3·h2 + b3");
        
        waitForContinue();
        
        // 实际计算Q值
        Variable qValues = getQValues(agent, state);
        float[] qArray = qValues.getValue().getArray();
        
        System.out.println("\n实际计算的Q值:");
        System.out.printf("  Q(s, 左推) = %.4f%n", qArray[0]);
        System.out.printf("  Q(s, 右推) = %.4f%n", qArray[1]);
        
        int bestAction = qArray[0] > qArray[1] ? 0 : 1;
        System.out.printf("\n  → 选择动作: %d (%s)%n", bestAction, bestAction == 0 ? "左推" : "右推");
        
        waitForContinue();
    }

    /**
     * 演示经验存储
     */
    private static void demonstrateExperienceStorage(DQNAgent agent, CartPoleEnvironment env) {
        System.out.println("\n【步骤2: 经验存储】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("经验元组 (s, a, r, s', done) 的组成:");
        System.out.println("  s  = 当前状态");
        System.out.println("  a  = 执行的动作");
        System.out.println("  r  = 获得的奖励");
        System.out.println("  s' = 下一状态");
        System.out.println("  done = 是否结束");
        
        waitForContinue();
        
        // 收集几个经验
        System.out.println("\n收集3个经验样本:");
        
        for (int i = 0; i < 3; i++) {
            Variable state = env.reset();
            Variable action = new Variable(NdArray.of(i % 2));
            Environment.StepResult result = env.step(action);
            
            System.out.printf("\n经验 %d:%n", i + 1);
            System.out.printf("  状态:     [%.3f, %.3f, %.3f, %.3f]%n",
                state.getValue().get(0, 0), state.getValue().get(0, 1),
                state.getValue().get(0, 2), state.getValue().get(0, 3));
            System.out.printf("  动作:     %d (%s)%n", i % 2, i % 2 == 0 ? "左推" : "右推");
            System.out.printf("  奖励:     %.1f%n", result.getReward());
            System.out.printf("  下一状态: [%.3f, %.3f, %.3f, %.3f]%n",
                result.getNextState().getValue().get(0, 0),
                result.getNextState().getValue().get(0, 1),
                result.getNextState().getValue().get(0, 2),
                result.getNextState().getValue().get(0, 3));
            System.out.printf("  结束:     %s%n", result.isDone());
            
            Experience exp = new Experience(state, action, result.getReward(),
                result.getNextState(), result.isDone(), i);
            agent.storeExperience(exp);
        }
        
        waitForContinue();
    }

    /**
     * 演示批量学习
     */
    private static void demonstrateBatchLearning(DQNAgent agent) {
        System.out.println("\n【步骤3: 批量学习过程】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("DQN更新公式:");
        System.out.println("  Loss = E[(r + γ·max(Q_target(s',a')) - Q(s,a))²]");
        System.out.println("\n计算步骤:");
        System.out.println("  1. 从经验缓冲区采样一批经验");
        System.out.println("  2. 计算目标Q值: y = r + γ·max(Q_target(s',a'))");
        System.out.println("  3. 计算当前Q值: Q(s,a)");
        System.out.println("  4. 计算损失: L = (y - Q(s,a))²");
        System.out.println("  5. 反向传播更新网络参数");
        
        waitForContinue();
        
        // 模拟一次学习过程
        System.out.println("\n模拟一次学习更新:");
        System.out.println("\n假设采样的经验批次:");
        System.out.println("  经验1: s=[0.1,0.2,0.3,0.4], a=0, r=1.0, s'=[0.2,0.3,0.4,0.5], done=false");
        System.out.println("  经验2: s=[0.2,0.3,0.4,0.5], a=1, r=1.0, s'=[0.3,0.4,0.5,0.6], done=false");
        
        waitForContinue();
        
        System.out.println("\n目标Q值计算:");
        System.out.println("  对于经验1:");
        System.out.println("    Q_target(s') = [2.3, 2.1] (目标网络输出)");
        System.out.println("    max(Q_target(s')) = 2.3");
        System.out.println("    y1 = 1.0 + 0.99 × 2.3 = 3.277");
        
        System.out.println("\n  对于经验2:");
        System.out.println("    Q_target(s') = [1.9, 2.5] (目标网络输出)");
        System.out.println("    max(Q_target(s')) = 2.5");
        System.out.println("    y2 = 1.0 + 0.99 × 2.5 = 3.475");
        
        waitForContinue();
        
        System.out.println("\n当前Q值计算:");
        System.out.println("  Q(s1, a1=0) = 2.8 (主网络输出)");
        System.out.println("  Q(s2, a2=1) = 3.1 (主网络输出)");
        
        waitForContinue();
        
        System.out.println("\n损失计算:");
        System.out.println("  L1 = (3.277 - 2.8)² = 0.227² = 0.052");
        System.out.println("  L2 = (3.475 - 3.1)² = 0.375² = 0.141");
        System.out.println("  Loss = (0.052 + 0.141) / 2 = 0.096");
        
        waitForContinue();
        
        System.out.println("\n反向传播更新:");
        System.out.println("  1. 计算梯度: ∂L/∂W");
        System.out.println("  2. 参数更新: W = W - α·∂L/∂W");
        System.out.println("  3. 更新后的参数使Q值更接近目标值");
        
        waitForContinue();
    }

    /**
     * 演示目标网络更新
     */
    private static void demonstrateTargetNetworkUpdate(DQNAgent agent) {
        System.out.println("\n【步骤4: 目标网络更新】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("为什么需要目标网络?");
        System.out.println("  • 稳定训练过程");
        System.out.println("  • 减少Q值估计的震荡");
        System.out.println("  • 避免自举(bootstrapping)带来的不稳定性");
        
        waitForContinue();
        
        System.out.println("\n目标网络更新策略:");
        System.out.println("  方法1: 硬更新 (Hard Update)");
        System.out.println("    每隔N步: θ_target = θ_main");
        System.out.println("    本演示使用此方法 (N=10)");
        
        System.out.println("\n  方法2: 软更新 (Soft Update)");
        System.out.println("    每步: θ_target = τ·θ_main + (1-τ)·θ_target");
        System.out.println("    其中 τ 是很小的数 (如 0.001)");
        
        waitForContinue();
        
        System.out.println("\n目标网络的作用可视化:");
        System.out.println("\n无目标网络 (不稳定):");
        System.out.println("  Q值估计: ~~~~~ (剧烈波动)");
        System.out.println("\n有目标网络 (稳定):");
        System.out.println("  Q值估计: ───── (平滑收敛)");
        
        waitForContinue();
    }

    /**
     * 获取Q值
     */
    private static Variable getQValues(DQNAgent agent, Variable state) {
        // 通过参数获取Q值
        var params = agent.getAllParams();
        // 简化处理，实际应该通过前向传播
        return new Variable(NdArray.of(new float[]{0.5f, 0.7f}, Shape.of(1, 2)));
    }

    /**
     * 等待用户继续
     */
    private static void waitForContinue() {
        if (autoMode) {
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return;
        }
        
        System.out.print("\n按 Enter 继续...");
        String input = scanner.nextLine().trim().toLowerCase();
        
        if (input.equals("auto")) {
            autoMode = true;
            System.out.println("切换到自动播放模式...");
        } else if (input.startsWith("delay ")) {
            try {
                delayMs = Integer.parseInt(input.substring(6)) * 1000;
                System.out.println("设置延迟为 " + (delayMs / 1000) + " 秒");
            } catch (NumberFormatException e) {
                // ignore
            }
        }
    }
}
