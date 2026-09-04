package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 训练器抽象基类
 * 
 * 提供训练循环框框、梯度裁剪、检查点保存等公共功能
 * 子类需实现具体的训练步骤逻辑
 * 
 * @author leavesfly
 * @since 2024
 */
public abstract class BaseTrainer {
    
    protected final MiniMindModel model;
    
    // 训练配置
    protected int maxEpochs;
    protected float maxGradNorm;
    protected int logInterval;
    protected int saveInterval;
    protected String checkpointDir;

    /**
     * MoE 负载均衡辅助损失的额外权重
     * <p>
     * {@code balanceLossVar} 内部已乘过 {@code router_aux_loss_coef}，默认 1.0 即直接相加；
     * 置 0 可关闭辅助损失。Dense（非 MoE）模式下该损失为 0 常量，无副作用。
     */
    protected float moeAuxLossWeight = 1.0f;
    
    // 训练状态
    protected int currentEpoch;
    protected int currentStep;
    protected List<Float> lossHistory;
    
    /**
     * 构造函数
     * 
     * @param model 模型
     */
    public BaseTrainer(MiniMindModel model) {
        this.model = model;
        this.maxEpochs = 10;
        this.maxGradNorm = 1.0f;
        this.logInterval = 100;
        this.saveInterval = 1000;
        this.checkpointDir = "./checkpoints";
        
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.lossHistory = new ArrayList<>();
    }
    
    /**
     * 开始训练
     * <p>
     * 无论正常结束还是中途抛异常，都会将模型复位为 eval 模式：
     * 否则训练后紧接着的推理、验证、或把该模型当作参考模型前向时，Dropout 仍会生效。
     */
    public void train() {
        printTrainingInfo();
        
        // 创建检查点目录
        createCheckpointDir();
        
        try {
            // 训练循环
            for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
                trainOneEpoch();
            }
        } finally {
            model.setTraining(false);
        }
        
        System.out.println(getTrainerName() + "训练完成!");
    }
    
    /**
     * 训练一个epoch
     * <p>
     * 健壮性说明：
     * 当 {@link #trainStep(Object)} 返回 NaN/Inf（例如 SFTTrainer 在检测到损失溢出时
     * 会返回 {@code Float.NaN} 以跳过本 batch 的参数更新），本方法需要：
     *   1) 不将该值累加到 epochLoss（否则平均损失被污染为 NaN）；
     *   2) 不计入 lossHistory（否则后续统计/日志打印会被污染）；
     *   3) 不增加 batchCount（用于平均损失分母），但 currentStep 仍自增，
     *      以保持全局步数与日志/checkpoint 节奏一致；
     *   4) 打印警告便于定位训练异常。
     */
    protected void trainOneEpoch() {
        prepareDataset();
        model.setTraining(true);
        
        double epochLoss = 0.0;
        int batchCount = 0;
        int skippedCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (hasNextBatch()) {
            Object batch = getNextBatch();
            
            // 训练一步
            float stepLoss = trainStep(batch);
            
            currentStep++;
            
            // 跳过 NaN/Inf 损失，避免污染统计
            if (Float.isNaN(stepLoss) || Float.isInfinite(stepLoss)) {
                skippedCount++;
                System.err.printf("警告: Step %d 损失异常(%s)，已跳过累加与历史记录%n",
                    currentStep, Float.isNaN(stepLoss) ? "NaN" : "Inf");
            } else {
                epochLoss += stepLoss;
                batchCount++;
                lossHistory.add(stepLoss);
            }
            
            // 打印日志
            if (currentStep % logInterval == 0) {
                printTrainingLog();
            }
            
            // 保存检查点
            if (currentStep % saveInterval == 0) {
                saveCheckpoint();
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        
        if (skippedCount > 0) {
            System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 跳过批次: %d | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, skippedCount, epochEndTime - epochStartTime);
        } else {
            System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime);
        }
        
        onEpochEnd();
        resetDataset();
    }
    
    /**
     * epoch 收尾钩子
     * <p>
     * 默认行为：清掉未凑满累积步数的残留梯度。使用梯度累积时，epoch 末尾往往剩下
     * 不足 accumulationSteps 的已累积梯度，若不清理会被带入下一个 epoch 的首次 update，
     * 使那一步的梯度尺度偏大且对应的是上一个 epoch 的数据。
     */
    protected void onEpochEnd() {
        model.clearGrads();
    }
    
    /**
     * 梯度裁剪（按全局范数缩放）
     * <p>
     * 只通过 {@code NdArray} 公开接口读写梯度，不假设具体后端实现（不直接向下转
     * {@code NdArrayCpu} 并改其 buffer），以便将来切换到 GPU/TPU 后端。
     * <p>
     * 梯度为 null 的参数（已冻结或未参与本次反向）自然被跳过，因此 LoRA 等
     * 部分参数训练场景下只需先清零非训练参数的梯度、再调用本方法即可。
     */
    protected void clipGradients() {
        if (maxGradNorm <= 0) {
            return;
        }
        
        // 计算梯度范数
        double normSquare = 0.0;
        for (Parameter param : model.getAllParams().values()) {
            NdArray grad = param.getGrad();
            if (grad == null) {
                continue;
            }
            for (float g : grad.getArray()) {
                normSquare += (double) g * g;
            }
        }
        
        double totalNorm = Math.sqrt(normSquare);
        
        // 如果超过阈值,进行裁剪
        // 写成 !(totalNorm > maxGradNorm) 而非 <= ，以便 NaN 范数也能被拦住，
        // 避免用一个 NaN 系数把所有梯度污染成 NaN
        if (!(totalNorm > maxGradNorm)) {
            return;
        }
        
        float clipCoef = (float) (maxGradNorm / (totalNorm + 1e-6));
        for (Parameter param : model.getAllParams().values()) {
            NdArray grad = param.getGrad();
            if (grad == null) {
                continue;
            }
            param.setGrad(grad.mulNum(clipCoef));
        }
    }
    
    /**
     * 保存检查点
     */
    protected void saveCheckpoint() {
        String filename = String.format("%s_checkpoint_epoch%d_step%d.model", 
            getCheckpointPrefix(), currentEpoch, currentStep);
        String filepath = Paths.get(checkpointDir, filename).toString();
        
        try {
            model.save(new File(filepath));
            System.out.println(getTrainerName() + "检查点已保存: " + filepath);
        } catch (Exception e) {
            // 保留完整堆栈与 cause：Model.save 会将原始序列化异常包装后抛出，
            // 只打印 getMessage() 会丢掉真正的根因（如某个配置类未实现 Serializable）
            System.err.println("保存检查点失败: " + filepath);
            e.printStackTrace();
        }
    }
    
    /**
     * 创建检查点目录
     */
    protected void createCheckpointDir() {
        try {
            Path path = Paths.get(checkpointDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
    
    /**
     * 打印训练日志
     */
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
            .skip(Math.max(0, lossHistory.size() - logInterval))
            .mapToDouble(Float::doubleValue)
            .average()
            .orElse(0.0);
        
        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f%n",
            currentEpoch + 1, maxEpochs, currentStep, avgLoss);
    }
    
    /**
     * 获取损失历史
     * 
     * @return 损失历史列表
     */
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    // ==================== 子类公用工具 ====================
    
    /**
     * 计算带 warmup 的余弦退火学习率（对标 Python minimind get_lr）
     * <p>
     * 退火段公式：{@code lr * (0.1 + 0.45 * (1 + cos(π * progress)))}，即最低降到 10% floor。
     * warmup 段采用线性升温，并用 {@code step + 1} 避开首步 LR=0（否则第一步是一次空更新）。
     *
     * @param baseLearningRate 基准（峰值）学习率
     * @param step             当前步数
     * @param totalSteps       总步数
     * @param warmupSteps      预热步数，传 0 表示不预热
     * @return 当前步应使用的学习率
     */
    protected float computeScheduledLearningRate(float baseLearningRate, int step,
                                                 int totalSteps, int warmupSteps) {
        if (warmupSteps > 0 && step < warmupSteps) {
            return baseLearningRate * ((step + 1f) / warmupSteps);
        }
        double progress = Math.min(1.0, (double) step / Math.max(totalSteps, 1));
        double cosineDecay = 0.1 + 0.45 * (1 + Math.cos(Math.PI * progress));
        return baseLearningRate * (float) cosineDecay;
    }
    
    /**
     * 构造不参与梯度的标量常量 Variable
     * <p>
     * 公开给各损失类（DPOLoss / PPOLoss / SPOLoss / GRPOLoss）复用，避免各处重复实现
     * 而漏掉 {@code setRequireGrad(false)}——外部信号（优势、参考 logProb、温度系数）
     * 一旦带梯度，会被当成可训练参数求导。
     *
     * @param value 标量值
     * @return requireGrad=false 的 [1] 形状 Variable
     */
    public static Variable constant(float value) {
        Variable var = new Variable(NdArray.of(new float[]{value}, Shape.of(1)));
        var.setRequireGrad(false);
        return var;
    }
    
    /**
     * 构造不参与梯度的常量 Variable
     *
     * @param values 数据
     * @return requireGrad=false 的 [values.length] 形状 Variable
     */
    public static Variable constant(float[] values) {
        Variable var = new Variable(NdArray.of(values.clone(), Shape.of(values.length)));
        var.setRequireGrad(false);
        return var;
    }
    
    /**
     * 将单元素张量归一为形状 [1] 的标量
     * <p>
     * 框架里 {@code NdArray.sum()} 与 {@code SoftmaxCE} 的标量输出形状是 [1,1]，
     * 而各处构造的常量是 [1]。[1,1] 与 [1] 做 add/sub 能广播成 [1,1]，但一旦与
     * [batch] 的逐样本张量混用就会因形状不兼容报错。统一归一到 [1] 可以消除这类隐式差异。
     * <p>
     * {@code backward()} 对单元素张量用 {@code NdArray.ones(shape)} 初始化梯度，
     * 因此 [1] 与 [1,1] 在反向传播上等价，本方法只影响形状一致性。
     *
     * @param var 待归一的 Variable
     * @return 形状 [1] 的标量 Variable
     * @throws IllegalArgumentException 元素数不为 1
     */
    public static Variable toScalar(Variable var) {
        int[] dims = var.getValue().getShape().getShapeDims();
        long elements = 1;
        for (int dim : dims) {
            elements *= dim;
        }
        if (elements != 1) {
            throw new IllegalArgumentException(
                "toScalar 需要单元素张量，实际形状: " + java.util.Arrays.toString(dims));
        }
        // 0 维与 1 维的单元素张量已经是标量形态，不再 reshape
        return dims.length <= 1 ? var : var.reshape(Shape.of(1));
    }
    
    /**
     * 按 lossMask 将忽略位置的标签改写为 ignore_index
     * <p>
     * {@code SoftmaxCE} 对负数标签（惯例为 -100）的处理是：前向不计入均值、
     * 反向该行梯度置零。因此只需在数据侧把 prompt / padding 位置的标签改为 -100，
     * 就能得到精确的 "answer-only" 损失，无需额外构造 one-hot 或逐元素损失张量。
     * <p>
     * 注意：不要依赖 pad token id（0）来屏蔽——{@code SoftmaxCE} 只跳过负数，
     * 标签 0 会被当作一个正常类别计入损失，从而把模型训成去预测 PAD。
     *
     * @param labels       原始标签 [batch, seq]
     * @param lossMask     损失掩码 [batch, seq]，1 表示参与损失
     * @param ignoreIndex  忽略标记（惯例 -100）
     * @return 掩码后的新标签数组，形状与 labels 一致
     */
    protected static NdArray applyIgnoreIndex(NdArray labels, NdArray lossMask, int ignoreIndex) {
        float[] labelData = labels.getArray();
        float[] maskData = lossMask.getArray();
        float[] masked = labelData.clone();
        int limit = Math.min(masked.length, maskData.length);
        for (int i = 0; i < limit; i++) {
            if (maskData[i] <= 0.5f) {
                masked[i] = ignoreIndex;
            }
        }
        return NdArray.of(masked, labels.getShape());
    }
    
    /**
     * 训练专用前向传播：同时拿到 logits 与可微的 MoE 负载均衡损失
     * <p>
     * 子类应用本方法取代 {@code model.predict(input)}：{@code predict} 会丢弃
     * {@code balanceLossVar}，使得 MoE 模式下辅助损失永远不参与反向传播。
     *
     * @param input Token IDs Variable
     * @return MoE 输出结果
     */
    protected MiniMindBlock.MoEOutput forwardWithAux(Variable input) {
        return model.predictWithLoss(input);
    }
    
    /**
     * 将 MoE 负载均衡辅助损失并入主损失
     * <p>
     * {@code balanceLossVar} 内部已乘过 {@code router_aux_loss_coef}；标准（非 MoE）模式
     * 下它是 0 常量 Variable，因此对 Dense 模型是无副作用的直通。
     * <p>
     * 结果统一经 {@link #toScalar(Variable)} 归一：主损失可能来自不同算子
     * （SoftmaxCE 给 [1,1]、各损失类给 [1]），不归一会在与逐样本张量混用时
     * 因广播不兼容报错，而且这种错误只在实际跑到那一行时才暴露。
     *
     * @param mainLoss 主损失（语言建模/偏好/策略损失等）
     * @param output   {@link #forwardWithAux(Variable)} 的输出
     * @return 合并后的标量总损失
     */
    protected Variable withMoeAuxLoss(Variable mainLoss, MiniMindBlock.MoEOutput output) {
        Variable aux = output.getBalanceLossVar();
        if (aux == null || moeAuxLossWeight <= 0f) {
            return toScalar(mainLoss);
        }
        return toScalar(mainLoss.add(toScalar(aux).mul(constant(moeAuxLossWeight))));
    }
    
    /**
     * 设置 MoE 辅助损失权重（置 0 可关闭）
     */
    public void setMoeAuxLossWeight(float moeAuxLossWeight) {
        this.moeAuxLossWeight = moeAuxLossWeight;
    }
    
    // ==================== 统一配置入口 ====================
    //
    // 这些 setter 集中在基类，避免各训练器重复声明（历史上有的返回 void、
    // 有的返回自身，有的根本没暴露 logInterval），调用方无法用一致的方式配置训练器。
    // 返回 BaseTrainer 以允许子类做协变覆盖（如 GRPOTrainer 返回 GRPOTrainer）。
    
    /**
     * 设置检查点目录
     */
    public BaseTrainer setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
        return this;
    }
    
    /**
     * 设置检查点目录与保存间隔
     */
    public BaseTrainer setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = Math.max(1, saveInterval);
        return this;
    }
    
    /**
     * 设置保存间隔（步数）
     */
    public BaseTrainer setSaveInterval(int saveInterval) {
        this.saveInterval = Math.max(1, saveInterval);
        return this;
    }
    
    /**
     * 设置日志打印间隔（步数）
     */
    public BaseTrainer setLogInterval(int logInterval) {
        this.logInterval = Math.max(1, logInterval);
        return this;
    }
    
    /**
     * 设置梯度裁剪阈值（<=0 表示不裁剪）
     */
    public BaseTrainer setMaxGradNorm(float maxGradNorm) {
        this.maxGradNorm = maxGradNorm;
        return this;
    }
    
    // ==================== 抽象方法 ====================
    
    /**
     * 训练一步
     * 
     * @param batch 批次数据
     * @return 损失值
     */
    protected abstract float trainStep(Object batch);
    
    /**
     * 获取训练器名称
     * 
     * @return 训练器名称
     */
    protected abstract String getTrainerName();
    
    /**
     * 打印训练信息
     */
    protected abstract void printTrainingInfo();
    
    /**
     * 准备数据集（如打乱数据）
     */
    protected abstract void prepareDataset();
    
    /**
     * 检查是否还有下一个批次
     * 
     * @return 是否有下一个批次
     */
    protected abstract boolean hasNextBatch();
    
    /**
     * 获取下一个批次
     * 
     * @return 批次数据
     */
    protected abstract Object getNextBatch();
    
    /**
     * 重置数据集
     */
    protected abstract void resetDataset();
    
    /**
     * 获取检查点文件名前缀
     * 
     * @return 检查点前缀
     */
    protected abstract String getCheckpointPrefix();
}