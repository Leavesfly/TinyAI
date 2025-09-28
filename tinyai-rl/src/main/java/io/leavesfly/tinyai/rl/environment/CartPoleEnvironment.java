package io.leavesfly.tinyai.rl.environment;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * CartPole（倒立摆）环境实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * CartPoleEnvironment实现了经典的倒立摆控制问题。
 * 目标是通过控制小车的左右移动来平衡杆子，使其不倒下。
 * 这是强化学习领域的经典控制问题，适合测试各种RL算法的性能。
 * <p>
 * 状态空间：4维 [位置, 速度, 角度, 角速度]
 * 动作空间：2维 [向左推, 向右推]
 * 奖励：每一步+1，倒下时结束
 */
public class CartPoleEnvironment extends Environment {

    // 物理参数
    private static final float GRAVITY = 9.8f;              // 重力加速度
    private static final float MASS_CART = 1.0f;            // 小车质量
    private static final float MASS_POLE = 0.1f;            // 杆子质量
    private static final float TOTAL_MASS = MASS_CART + MASS_POLE;
    private static final float LENGTH = 0.5f;               // 杆子长度的一半
    private static final float POLE_MASS_LENGTH = MASS_POLE * LENGTH;
    private static final float FORCE_MAG = 10.0f;           // 施加力的大小
    private static final float TIME_STEP = 0.02f;           // 时间步长

    // 阈值参数
    private static final float X_THRESHOLD = 2.4f;          // 位置阈值
    private static final float THETA_THRESHOLD = (float) (Math.PI / 15); // 角度阈值（12度）

    // 状态变量 [x, x_dot, theta, theta_dot]
    private float x;         // 小车位置
    private float xDot;      // 小车速度
    private float theta;     // 杆子角度（弧度）
    private float thetaDot;  // 杆子角速度

    private final Random random;

    /**
     * 构造函数
     */
    public CartPoleEnvironment() {
        super(4, 2, 500); // 4维状态，2个动作，最大500步
        this.random = new Random();
    }

    /**
     * 构造函数（可指定随机种子）
     *
     * @param seed 随机种子
     */
    public CartPoleEnvironment(long seed) {
        super(4, 2, 500);
        this.random = new Random(seed);
    }

    @Override
    public Variable reset() {
        // 随机初始化状态，范围较小
        x = random.nextFloat() * 0.2f - 0.1f;           // [-0.1, 0.1]
        xDot = random.nextFloat() * 0.2f - 0.1f;        // [-0.1, 0.1]
        theta = random.nextFloat() * 0.2f - 0.1f;       // [-0.1, 0.1] 弧度
        thetaDot = random.nextFloat() * 0.2f - 0.1f;    // [-0.1, 0.1]

        currentStep = 0;
        done = false;

        currentState = getStateVariable();
        return currentState;
    }

    @Override
    public StepResult step(Variable action) {
        if (done) {
            throw new IllegalStateException("环境已结束，请先调用reset()");
        }

        // 获取动作值（0或1）
        int actionValue = (int) action.getValue().getNumber().floatValue();
        if (!isValidAction(action)) {
            throw new IllegalArgumentException("无效动作：" + actionValue + "，必须是0或1");
        }

        // 计算施加的力（0=向左，1=向右）
        float force = (actionValue == 1) ? FORCE_MAG : -FORCE_MAG;

        // 使用物理方程更新状态
        updatePhysics(force);

        // 增加步数
        currentStep++;

        // 检查是否结束
        boolean terminated = isTerminated();
        done = terminated || currentStep >= maxSteps;

        // 计算奖励
        float reward = done ? 0.0f : 1.0f; // 每存活一步得1分

        // 更新当前状态
        currentState = getStateVariable();

        // 创建附加信息
        Map<String, Object> info = new HashMap<>();
        info.put("x", x);
        info.put("theta", theta);
        info.put("terminated", terminated);
        info.put("truncated", currentStep >= maxSteps);

        return new StepResult(currentState, reward, done, info);
    }

    /**
     * 使用物理方程更新环境状态
     *
     * @param force 施加的力
     */
    private void updatePhysics(float force) {
        float cosTheta = (float) Math.cos(theta);
        float sinTheta = (float) Math.sin(theta);

        // 计算角加速度
        float temp = (force + POLE_MASS_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS;
        float thetaAcc = (GRAVITY * sinTheta - cosTheta * temp) /
                (LENGTH * (4.0f / 3.0f - MASS_POLE * cosTheta * cosTheta / TOTAL_MASS));

        // 计算线加速度
        float xAcc = temp - POLE_MASS_LENGTH * thetaAcc * cosTheta / TOTAL_MASS;

        // 使用欧拉方法数值积分
        x += TIME_STEP * xDot;
        xDot += TIME_STEP * xAcc;
        theta += TIME_STEP * thetaDot;
        thetaDot += TIME_STEP * thetaAcc;
    }

    /**
     * 检查是否到达终止状态
     *
     * @return 是否终止
     */
    private boolean isTerminated() {
        return Math.abs(x) > X_THRESHOLD || Math.abs(theta) > THETA_THRESHOLD;
    }

    /**
     * 将当前状态转换为Variable
     *
     * @return 状态Variable
     */
    private Variable getStateVariable() {
        float[] stateArray = {x, xDot, theta, thetaDot};
        return new Variable(NdArray.of(stateArray, Shape.of(1, 4)));
    }

    @Override
    public Variable sampleAction() {
        int randomAction = random.nextInt(2); // 0 or 1
        return new Variable(NdArray.of(randomAction));
    }

    @Override
    public boolean isValidAction(Variable action) {
        float actionValue = action.getValue().getNumber().floatValue();
        return actionValue == 0.0f || actionValue == 1.0f;
    }

    @Override
    public void render() {
        // 简单的控制台渲染
        System.out.printf("Step: %d, X: %.3f, Theta: %.3f°, Done: %s%n",
                currentStep, x, Math.toDegrees(theta), done);

        // ASCII艺术渲染
        renderASCII();
    }

    /**
     * ASCII艺术渲染
     */
    private void renderASCII() {
        int screenWidth = 50;
        int cartPos = (int) ((x + X_THRESHOLD) / (2 * X_THRESHOLD) * screenWidth);
        cartPos = Math.max(0, Math.min(screenWidth - 1, cartPos));

        // 绘制顶部（杆子）
        System.out.print("杆子: ");
        for (int i = 0; i < screenWidth; i++) {
            if (i == cartPos) {
                // 根据角度显示杆子方向
                if (theta > 0.1) {
                    System.out.print("/");
                } else if (theta < -0.1) {
                    System.out.print("\\");
                } else {
                    System.out.print("|");
                }
            } else {
                System.out.print(" ");
            }
        }
        System.out.println();

        // 绘制小车
        System.out.print("小车: ");
        for (int i = 0; i < screenWidth; i++) {
            if (i == cartPos) {
                System.out.print("█");
            } else {
                System.out.print("-");
            }
        }
        System.out.println();
        System.out.println();
    }

    /**
     * 获取状态的标准化版本（用于神经网络输入）
     *
     * @return 标准化状态
     */
    public Variable getNormalizedState() {
        // 标准化各个状态分量
        float normalizedX = x / X_THRESHOLD;
        float normalizedXDot = xDot / 1.0f; // 假设最大速度为1
        float normalizedTheta = theta / THETA_THRESHOLD;
        float normalizedThetaDot = thetaDot / 1.0f; // 假设最大角速度为1

        float[] normalizedStateArray = {normalizedX, normalizedXDot, normalizedTheta, normalizedThetaDot};
        return new Variable(NdArray.of(normalizedStateArray, Shape.of(1, 4)));
    }

    /**
     * 获取当前状态的详细信息
     *
     * @return 状态信息
     */
    public Map<String, Float> getStateInfo() {
        Map<String, Float> stateInfo = new HashMap<>();
        stateInfo.put("position", x);
        stateInfo.put("velocity", xDot);
        stateInfo.put("angle_rad", theta);
        stateInfo.put("angle_deg", (float) Math.toDegrees(theta));
        stateInfo.put("angular_velocity", thetaDot);
        return stateInfo;
    }
}