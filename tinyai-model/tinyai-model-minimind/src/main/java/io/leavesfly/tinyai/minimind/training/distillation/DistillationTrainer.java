package io.leavesfly.tinyai.minimind.training.distillation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 知识蒸馏训练器
 * <p>
 * 对标 Python minimind3 train_distillation.py，将大模型（教师）的知识迁移到小模型（学生）。
 * <p>
 * 核心原理：
 * 1. 温度缩放：通过温度参数 T 软化教师模型的概率分布，暴露更多"暗知识"
 * 2. 混合损失：总损失 = alpha * CE_loss + (1-alpha) * KL_loss
 * 3. KL 散度损失乘以 T²，保持梯度尺度与 CE 损失一致
 * <p>
 * 温度的作用：
 * - T > 1：软化概率分布，暴露更多暗知识（推荐 1.5-2.0）
 * - T = 1：标准 softmax，无软化效果
 * - T < 1：锐化概率分布，强调高概率类别
 * <p>
 * 训练流程（对标 Python train_epoch）：
 * 1. 学生模型前向传播 → 计算学生 logits
 * 2. 教师模型前向传播（eval + no_grad）→ 计算教师 logits
 * 3. CE 损失：学生 logits vs 真实标签
 * 4. 蒸馏损失：KL(student || teacher) * T²
 * 5. 混合损失 = alpha * CE + (1-alpha) * KL
 * <p>
 * 支持 MoE 教师蒸馏到 Dense 学生：
 * 当教师和学生词表大小不同时，自动截断教师 logits
 *
 * @author TinyAI Team
 * @since 2025
 */
public class DistillationTrainer extends BaseTrainer {

    /**
     * 标签忽略位标记（对齐 SoftmaxCE 的 ignore_index 约定）
     */
    private static final int IGNORE_INDEX = -100;

    private final MiniMindModel teacherModel;      // 教师模型（冻结）
    private final MiniMindModel studentModel;       // 学生模型（训练）
    private final SFTDataset dataset;
    private final DistillationConfig distillConfig;
    private final SoftmaxCrossEntropy ceLossFunction;
    private final Adam optimizer;

    // 训练状态
    private float currentLearningRate;
    private int accumulationCounter = 0;

    // 蒸馏损失追踪
    private final List<Float> ceLossHistory;
    private final List<Float> klLossHistory;

    /**
     * 构造函数
     *
     * @param studentModel  学生模型（将被训练）
     * @param teacherModel  教师模型（冻结，不参与梯度更新）
     * @param dataset       SFT 数据集
     * @param distillConfig 蒸馏配置
     */
    public DistillationTrainer(MiniMindModel studentModel, MiniMindModel teacherModel,
                                SFTDataset dataset, DistillationConfig distillConfig) {
        super(studentModel);
        this.studentModel = studentModel;
        this.teacherModel = teacherModel;
        this.dataset = dataset;
        this.distillConfig = distillConfig;
        this.ceLossFunction = new SoftmaxCrossEntropy();

        // 验证配置
        distillConfig.validate();

        // 冻结教师模型
        freezeTeacherModel();

        // 训练参数
        this.maxEpochs = distillConfig.getMaxEpochs();
        this.maxGradNorm = distillConfig.getGradClip();
        this.logInterval = distillConfig.getLogInterval();
        this.saveInterval = distillConfig.getSaveInterval();
        this.checkpointDir = "./checkpoints/minimind/distillation";
        this.currentLearningRate = distillConfig.getLearningRate();

        // 优化器
        this.optimizer = new Adam(studentModel, distillConfig.getLearningRate(),
                0.9f, 0.999f, 1e-8f);

        // 损失追踪
        this.ceLossHistory = new ArrayList<>();
        this.klLossHistory = new ArrayList<>();
    }

    /**
     * 冻结教师模型
     * <p>
     * 对标 Python:
     * teacher_model.eval()
     * teacher_model.requires_grad_(False)
     */
    private void freezeTeacherModel() {
        teacherModel.setTraining(false);
        // 显式关闭教师参数的梯度：反向传播到教师叶子即终止，
        // 不会被优化器更新，也不会白耗内存累积梯度
        for (Parameter param : teacherModel.getAllParams().values()) {
            param.setRequiresGrad(false);
        }
    }

    // ==================== 核心训练逻辑 ====================

    @Override
    protected float trainStep(Object batch) {
        // 更新学习率（余弦退火 with 10% floor，对标 Python get_lr）
        updateLearningRate();

        SFTDataset.Batch sftBatch = (SFTDataset.Batch) batch;

        NdArray inputArray = sftBatch.getInput();
        // answer-only：prompt / padding 位置置 -100，CE 与 KL 均仅由回复部分贡献
        NdArray labelArray = applyIgnoreIndex(
                sftBatch.getLabels(), sftBatch.getLossMask(), IGNORE_INDEX);

        int batchSize = sftBatch.getBatchSize();
        int seqLen = sftBatch.getSeqLen();
        int totalTokens = batchSize * seqLen;

        // ========== 1. 学生模型前向传播 ==========
        studentModel.setTraining(true);
        Variable input = new Variable(inputArray);
        MiniMindBlock.MoEOutput studentOutput = forwardWithAux(input);
        Variable studentLogits = studentOutput.getOutput();
        int vocabSize = studentLogits.getValue().getShape().getDimension(2);

        Variable studentFlat = studentLogits.reshape(Shape.of(totalTokens, vocabSize));

        // ========== 2. 教师模型前向传播（无梯度） ==========
        teacherModel.setTraining(false);
        Variable teacherLogitsVar = teacherModel.predict(new Variable(inputArray.copy())).detach();

        // 如果词表大小不同，截断教师 logits（对标 Python: teacher_logits[..., :vocab_size_student]）
        NdArray teacherLogitsNd = teacherLogitsVar.getValue();
        int teacherVocabSize = teacherLogitsNd.getShape().getDimension(2);
        if (teacherVocabSize > vocabSize) {
            teacherLogitsNd = truncateLastDim(teacherLogitsNd,
                    teacherLogitsNd.getShape().getShapeDims(), vocabSize);
        }
        // 教师已 detach，释放其计算图以节省内存
        teacherLogitsVar.unChainBackward();

        Variable teacherFlat = new Variable(teacherLogitsNd).reshape(Shape.of(totalTokens, vocabSize));
        teacherFlat.setRequireGrad(false);

        // ========== 3. 计算 CE 损失（Ground-Truth） ==========
        Variable labels = new Variable(labelArray);
        labels.setRequireGrad(false);
        Variable labelsReshaped = labels.reshape(Shape.of(totalTokens, 1));

        Variable ceLoss = ceLossFunction.loss(labelsReshaped, studentFlat);

        // ========== 4. 计算蒸馏损失（可微 KL 散度） ==========
        float[] maskData = sftBatch.getLossMask().getArray();
        float maskSum = 0.0f;
        for (float m : maskData) {
            if (m > 0.5f) {
                maskSum += 1.0f;
            }
        }
        Variable tokenMask = constant(maskData).reshape(Shape.of(totalTokens, 1));

        Variable klLoss = computeKLDivergenceLossVar(
                studentFlat, teacherFlat, distillConfig.getTemperature(), tokenMask, maskSum);

        // ========== 5. 计算混合损失 ==========
        // 总损失 = alpha * CE + (1-alpha) * KL（对标 Python）
        // 两项均为可微 Variable，梯度方向同时来自真实标签与教师分布
        float alpha = distillConfig.getAlpha();
        Variable loss = toScalar(
                ceLoss.mul(constant(alpha)).add(klLoss.mul(constant(1.0f - alpha))));
        loss = withMoeAuxLoss(loss, studentOutput);

        float ceLossValue = ceLoss.getValue().getNumber().floatValue();
        float klLossValue = klLoss.getValue().getNumber().floatValue();
        float totalLossValue = loss.getValue().getNumber().floatValue();

        // 损失异常时跳过本 batch：不反向、不推进累积计数，并释放计算图
        if (Float.isNaN(totalLossValue) || Float.isInfinite(totalLossValue)) {
            System.err.println("警告: 蒸馏损失异常 (" + totalLossValue + "), 跳过此batch");
            loss.unChainBackward();
            model.clearGrads();
            accumulationCounter = 0;
            return Float.NaN;
        }

        // 梯度累积
        int accumSteps = distillConfig.getAccumulationSteps();
        if (accumSteps > 1) {
            loss = loss.mul(constant(1.0f / accumSteps));
        }

        // ========== 6. 反向传播 ==========
        loss.backward();

        accumulationCounter++;

        if (accumulationCounter % accumSteps == 0) {
            clipGradients();
            optimizer.update();
            model.clearGrads();
            accumulationCounter = 0;
        }

        // 断开计算图
        loss.unChainBackward();

        // 记录损失
        ceLossHistory.add(ceLossValue);
        klLossHistory.add(klLossValue);

        return totalLossValue;
    }

    /**
     * 计算可微的 KL 散度蒸馏损失
     * <p>
     * 对标 Python distillation_loss():
     * <pre>
     * teacher_probs = softmax(teacher_logits / T)
     * student_log_probs = log_softmax(student_logits / T)
     * kl = KL_div(student_log_probs, teacher_probs, reduction='batchmean')
     * return T² * kl
     * </pre>
     * <p>
     * 实现要点：
     * 1. 学生侧全程使用 Variable 算子，保证 KL 对 logits 可微 —— 这是蒸馏真正生效的前提。
     *    （旧实现用纯 float 循环算 KL，再用 totalLoss/ceLoss 的比例去缩放 CE 梯度，
     *    结果梯度方向 100% 来自 CE，教师分布的信息完全没有传给学生）
     * 2. 教师侧已 detach，softMax/logSoftmax 均不产生梯度
     * 3. 用 sumTo([N,1]) 先在 vocab 维归约得到逐 token KL，再乘 token 掩码求和，
     *    避开构造 [N,V] 的掩码广播张量（省 vocabSize 倍内存）
     * 4. 归约采用"有效 token 均值"，与 CE 的 ignore_index 均值口径保持一致
     *
     * @param studentFlat 学生 logits [N, V]（保留计算图）
     * @param teacherFlat 教师 logits [N, V]（已 detach）
     * @param temperature 蒸馏温度
     * @param tokenMask   token 级掩码 [N, 1]，1 表示参与蒸馏
     * @param maskSum     掩码中 1 的个数
     * @return KL 散度损失标量（已乘以 T²）
     */
    private Variable computeKLDivergenceLossVar(Variable studentFlat, Variable teacherFlat,
                                                float temperature, Variable tokenMask,
                                                float maskSum) {
        Variable temperatureVar = constant(temperature);

        // 温度缩放
        Variable studentScaled = studentFlat.div(temperatureVar);
        Variable teacherScaled = teacherFlat.div(temperatureVar);

        // 学生 log 概率（可微）与教师概率/对数概率（无梯度）
        Variable studentLogProbs = studentScaled.logSoftmax(-1);
        Variable teacherLogProbs = teacherScaled.logSoftmax(-1);
        Variable teacherProbs = teacherScaled.softMax();

        // KL(teacher || student) = Σ_v p_teacher * (log p_teacher - log p_student)
        Variable diff = teacherLogProbs.sub(studentLogProbs);
        Variable weighted = teacherProbs.mul(diff);                        // [N, V]
        Variable perTokenKl = weighted.sumTo(Shape.of(tokenMask.getValue()
                .getShape().getDimension(0), 1));                          // [N, 1]

        Variable maskedKl = perTokenKl.mul(tokenMask);                     // [N, 1]
        Variable klSum = maskedKl.sum();                                   // 标量

        float divisor = maskSum > 0 ? maskSum : 1.0f;
        Variable klMean = klSum.div(constant(divisor));

        // 温度平方缩放（对标 Python: return temperature ** 2 * kl）
        // toScalar 把框架的 [1,1] 标量归一到 [1]，与 CE 项形状一致
        return toScalar(klMean.mul(constant(temperature * temperature)));
    }

    /**
     * 截断 NdArray 的最后一个维度
     */
    private NdArray truncateLastDim(NdArray nd, int[] shape, int newLastDim) {
        int batchSize = shape[0];
        int seqLen = shape[1];
        int oldLastDim = shape[2];

        float[] oldData = nd.getArray();
        float[] newData = new float[batchSize * seqLen * newLastDim];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int oldOffset = (b * seqLen + s) * oldLastDim;
                int newOffset = (b * seqLen + s) * newLastDim;
                System.arraycopy(oldData, oldOffset, newData, newOffset, newLastDim);
            }
        }

        return NdArray.of(newData, Shape.of(batchSize, seqLen, newLastDim));
    }

    /**
     * 更新学习率（余弦退火 with 10% floor，对标 Python get_lr）
     */
    private void updateLearningRate() {
        int totalSteps = maxEpochs * dataset.getBatchCount();
        currentLearningRate = computeScheduledLearningRate(
                distillConfig.getLearningRate(), currentStep, totalSteps, 0);
        optimizer.setLearningRate(currentLearningRate);
    }

    // ==================== 训练入口 ====================

    @Override
    public void train() {
        printTrainingInfo();
        createCheckpointDir();

        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }

        System.out.println("\n知识蒸馏训练完成!");
        printDistillationStats();
    }

    /**
     * 打印蒸馏训练统计
     */
    private void printDistillationStats() {
        System.out.println("=".repeat(70));
        System.out.println("蒸馏训练统计");
        System.out.println("=".repeat(70));

        if (!lossHistory.isEmpty()) {
            double avgTotalLoss = lossHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            double finalTotalLoss = lossHistory.get(lossHistory.size() - 1);
            System.out.printf("  平均总损失: %.4f%n", avgTotalLoss);
            System.out.printf("  最终总损失: %.4f%n", finalTotalLoss);
        }

        if (!ceLossHistory.isEmpty()) {
            double avgCe = ceLossHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            double avgKl = klLossHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            System.out.printf("  平均 CE 损失: %.4f%n", avgCe);
            System.out.printf("  平均 KL 损失: %.4f%n", avgKl);
        }

        System.out.printf("  总训练步数: %d%n", currentStep);
        System.out.printf("  蒸馏配置: %s%n", distillConfig);
        System.out.println("=".repeat(70));
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected String getTrainerName() {
        return "知识蒸馏";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(70));
        System.out.println("开始知识蒸馏训练");
        System.out.println("=".repeat(70));

        // 教师模型信息
        MiniMindConfig teacherConfig = teacherModel.getConfig();
        long teacherParams = teacherConfig.estimateParameters();
        System.out.printf("  教师模型: %s (%.3f M 参数)%n",
                teacherConfig.getModelSize(), teacherParams / 1_000_000.0);
        System.out.printf("    - 隐藏维度: %d, 层数: %d, MoE: %b%n",
                teacherConfig.getHiddenSize(), teacherConfig.getNumLayers(), teacherConfig.isUseMoE());

        // 学生模型信息
        MiniMindConfig studentConfig = studentModel.getConfig();
        long studentParams = studentConfig.estimateParameters();
        System.out.printf("  学生模型: %s (%.3f M 参数)%n",
                studentConfig.getModelSize(), studentParams / 1_000_000.0);
        System.out.printf("    - 隐藏维度: %d, 层数: %d, MoE: %b%n",
                studentConfig.getHiddenSize(), studentConfig.getNumLayers(), studentConfig.isUseMoE());

        // 压缩比
        if (teacherParams > 0) {
            System.out.printf("  压缩比: %.1fx%n", (double) teacherParams / studentParams);
        }

        // 训练配置
        System.out.println("  蒸馏配置: " + distillConfig);
        System.out.printf("  训练样本数: %d%n", dataset.getSampleCount());
        System.out.printf("  批次数量: %d%n", dataset.getBatchCount());
        System.out.printf("  最大轮次: %d%n", maxEpochs);
        System.out.printf("  混合损失: alpha=%.2f, temperature=%.2f%n",
                distillConfig.getAlpha(), distillConfig.getTemperature());
        System.out.println("=".repeat(70));
    }

    @Override
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
                .skip(Math.max(0, lossHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        double avgCe = ceLossHistory.stream()
                .skip(Math.max(0, ceLossHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        double avgKl = klLossHistory.stream()
                .skip(Math.max(0, klLossHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | CE: %.4f | KL: %.4f | LR: %.6f%n",
                currentEpoch + 1, maxEpochs, currentStep, avgLoss, avgCe, avgKl, currentLearningRate);
    }

    @Override
    protected void prepareDataset() {
        dataset.prepare(true);
    }

    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNextBatch();
    }

    @Override
    protected Object getNextBatch() {
        return dataset.getNextBatch();
    }

    @Override
    protected void resetDataset() {
        dataset.reset();
    }

    @Override
    protected String getCheckpointPrefix() {
        return "distillation";
    }

    // ==================== Getter ====================

    public List<Float> getCeLossHistory() {
        return new ArrayList<>(ceLossHistory);
    }

    public List<Float> getKlLossHistory() {
        return new ArrayList<>(klLossHistory);
    }

    public MiniMindModel getStudentModel() {
        return studentModel;
    }

    public MiniMindModel getTeacherModel() {
        return teacherModel;
    }
}
