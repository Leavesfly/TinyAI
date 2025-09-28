package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * UCB（Upper Confidence Bound）多臂老虎机智能体
 * <p>
 * UCB算法基于"乐观面对不确定性"的原则：
 * - 选择具有最高上置信界的臂
 * - 上置信界 = 估计奖励 + 置信区间
 * - 置信区间反映了对该臂估计的不确定性
 * <p>
 * UCB1公式：UCB(i) = Q(i) + c * sqrt(ln(t) / N(i))
 * 其中：
 * - Q(i): 臂i的平均奖励估计
 * - c: 置信度参数（通常取√2）
 * - t: 总选择次数
 * - N(i): 臂i被选择的次数
 * <p>
 * 算法特点：
 * - 理论上有较好的悔恨上界保证
 * - 自动平衡探索和利用
 * - 不需要手动调参（除了置信度参数c）
 *
 * @author leavesfly
 */
public class UCBBanditAgent extends BanditAgent {

    /**
     * 置信度参数，控制探索的程度
     * 典型值：√2 ≈ 1.414
     */
    private float confidenceParam;

    /**
     * 构造函数，使用默认置信度参数
     *
     * @param name    智能体名称
     * @param numArms 臂的数量
     */
    public UCBBanditAgent(String name, int numArms) {
        this(name, numArms, (float) Math.sqrt(2.0));
    }

    /**
     * 完整构造函数
     *
     * @param name            智能体名称
     * @param numArms         臂的数量
     * @param confidenceParam 置信度参数
     */
    public UCBBanditAgent(String name, int numArms, float confidenceParam) {
        super(name, numArms);
        this.confidenceParam = confidenceParam;
    }

    @Override
    public Variable selectAction(Variable state) {
        int armIndex = selectArm();
        return new Variable(NdArray.of(new float[]{armIndex}, Shape.of(1)));
    }

    @Override
    public int selectArm() {
        // 如果还有未被选择过的臂，优先选择它们
        for (int i = 0; i < actionDim; i++) {
            if (actionCounts[i] == 0) {
                return i;
            }
        }

        // 所有臂都被选择过至少一次，使用UCB公式选择
        return selectArmByUCB();
    }

    /**
     * 使用UCB公式选择臂
     *
     * @return 选择的臂索引
     */
    private int selectArmByUCB() {
        int bestArm = 0;
        float bestUCB = calculateUCB(0);

        for (int i = 1; i < actionDim; i++) {
            float ucb = calculateUCB(i);
            if (ucb > bestUCB) {
                bestUCB = ucb;
                bestArm = i;
            }
        }

        return bestArm;
    }

    /**
     * 计算指定臂的UCB值
     * <p>
     * UCB(i) = Q(i) + c * sqrt(ln(t) / N(i))
     *
     * @param armIndex 臂索引
     * @return UCB值
     */
    private float calculateUCB(int armIndex) {
        if (actionCounts[armIndex] == 0) {
            return Float.POSITIVE_INFINITY; // 未选择过的臂具有无限大的UCB
        }

        float averageReward = estimatedRewards[armIndex];
        float confidence = confidenceParam * (float) Math.sqrt(Math.log(totalActions) / actionCounts[armIndex]);

        return averageReward + confidence;
    }

    /**
     * 获取所有臂的UCB值
     *
     * @return UCB值数组
     */
    public float[] getAllUCBValues() {
        float[] ucbValues = new float[actionDim];
        for (int i = 0; i < actionDim; i++) {
            ucbValues[i] = calculateUCB(i);
        }
        return ucbValues;
    }

    /**
     * 获取指定臂的UCB值
     *
     * @param armIndex 臂索引
     * @return UCB值
     */
    public float getUCBValue(int armIndex) {
        return calculateUCB(armIndex);
    }

    /**
     * 获取指定臂的置信区间
     *
     * @param armIndex 臂索引
     * @return 置信区间大小
     */
    public float getConfidenceInterval(int armIndex) {
        if (actionCounts[armIndex] == 0 || totalActions == 0) {
            return Float.POSITIVE_INFINITY;
        }

        return confidenceParam * (float) Math.sqrt(Math.log(totalActions) / actionCounts[armIndex]);
    }

    /**
     * 获取所有臂的置信区间
     *
     * @return 置信区间数组
     */
    public float[] getAllConfidenceIntervals() {
        float[] intervals = new float[actionDim];
        for (int i = 0; i < actionDim; i++) {
            intervals[i] = getConfidenceInterval(i);
        }
        return intervals;
    }

    /**
     * 获取置信度参数
     *
     * @return 置信度参数
     */
    public float getConfidenceParam() {
        return confidenceParam;
    }

    /**
     * 设置置信度参数
     *
     * @param confidenceParam 新的置信度参数
     */
    public void setConfidenceParam(float confidenceParam) {
        this.confidenceParam = Math.max(0.0f, confidenceParam);
    }

    @Override
    public void printStatus() {
        super.printStatus();
        System.out.println("置信度参数: " + String.format("%.4f", confidenceParam));

        System.out.println("各臂UCB值:");
        float[] ucbValues = getAllUCBValues();
        for (int i = 0; i < actionDim; i++) {
            System.out.println("  臂 " + i + ": UCB=" +
                    (Float.isInfinite(ucbValues[i]) ? "∞" : String.format("%.4f", ucbValues[i])) +
                    ", 置信区间=" +
                    (Float.isInfinite(getConfidenceInterval(i)) ? "∞" : String.format("%.4f", getConfidenceInterval(i))));
        }
    }

    /**
     * 获取算法描述信息
     *
     * @return 算法描述
     */
    public String getAlgorithmDescription() {
        return String.format("UCB算法 (置信度参数=%.4f)", confidenceParam);
    }

    /**
     * 获取当前选择策略的详细信息
     *
     * @return 策略信息
     */
    public String getSelectionInfo() {
        StringBuilder info = new StringBuilder();
        info.append("UCB选择策略:\n");

        float[] ucbValues = getAllUCBValues();
        int bestArm = 0;
        float bestUCB = ucbValues[0];

        for (int i = 1; i < actionDim; i++) {
            if (ucbValues[i] > bestUCB) {
                bestUCB = ucbValues[i];
                bestArm = i;
            }
        }

        info.append(String.format("  推荐选择臂 %d (UCB=%.4f)\n", bestArm, bestUCB));
        info.append("  各臂详情:\n");

        for (int i = 0; i < actionDim; i++) {
            info.append(String.format("    臂 %d: 奖励=%.4f, UCB=%s, 选择次数=%d\n",
                    i,
                    estimatedRewards[i],
                    Float.isInfinite(ucbValues[i]) ? "∞" : String.format("%.4f", ucbValues[i]),
                    actionCounts[i]));
        }

        return info.toString();
    }
}