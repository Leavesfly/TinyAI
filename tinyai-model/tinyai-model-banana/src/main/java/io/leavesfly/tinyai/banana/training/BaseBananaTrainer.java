package io.leavesfly.tinyai.banana.training;

import io.leavesfly.tinyai.banana.block.BananaBlock;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Banana训练器基类
 * 
 * 提供训练过程中常用的公共方法，包括梯度裁剪、参数计算和检查点目录创建
 * 
 * @author TinyAI
 * @since 2024
 */
public class BaseBananaTrainer {
    
    /**
     * 梯度裁剪
     * 
     * 防止梯度爆炸，将梯度范数裁剪到指定阈值内
     * 
     * @param model 模型
     * @param maxGradNorm 最大梯度范数阈值
     */
    protected static void clipGradients(BananaModel model, float maxGradNorm) {
        // 计算梯度范数
        double totalNorm = 0.0;
        
        for (var param : model.getAllParams().values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradData = grad.getArray();
                
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 如果超过阈值,进行裁剪
        if (totalNorm > maxGradNorm) {
            float clipCoef = maxGradNorm / (float) totalNorm;
            
            for (var param : model.getAllParams().values()) {
                if (param.getGrad() != null) {
                    NdArray grad = param.getGrad();
                    float[] gradData = grad.getArray();
                    
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= clipCoef;
                    }
                }
            }
        }
    }
    
    /**
     * 计算模型总参数量（元素个数之和）。
     *
     * @param model 模型，不能为 {@code null}
     * @return 所有参数的元素总数
     */
    public static long calculateTotalParams(BananaModel model) {
        long totalParams = 0;
        for (var param : model.getAllParams().values()) {
            int[] dims = param.getValue().getShape().getShapeDims();
            long size = 1;
            for (int d : dims) size *= d;
            totalParams += size;
        }
        return totalParams;
    }
    
    /**
     * 创建检查点目录；若已存在则直接返回，IO 异常仅打印到标准错误，不抛出。
     *
     * @param checkpointDir 检查点目录路径，不能为 {@code null}
     */
    public static void createCheckpointDir(String checkpointDir) {
        try {
            Path path = Paths.get(checkpointDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }

    // ==================== 多模态对比损失 (CLIP 风格) ====================

    /** CLIP 默认温度系数的倒数：1 / 0.07。*/
    public static final float DEFAULT_LOGIT_SCALE = 1.0f / 0.07f;

    /**
     * L2 归一化（在最后一维上），保持反向传播可导。
     *
     * <p>公式：{@code x / sqrt(mean(x^2, axis=-1, keepdims=True) + eps)}。
     * 使用 {@code mean} 而非 {@code sum} 只差一个常数因子，对方向无影响。</p>
     *
     * @param x 输入特征，形状 {@code [..., H]}
     * @return L2 归一化后的特征，形状不变
     */
    public static Variable l2Normalize(Variable x) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int lastAxis = shape.length - 1;

        Variable squared = x.mul(x);
        Variable meanSq = squared.mean(lastAxis, true);
        Variable norm = meanSq.add(new Variable(1e-8f)).sqrt();
        Variable normBroadcast = norm.broadcastTo(Shape.of(shape));
        return x.div(normBroadcast);
    }

    /**
     * CLIP 风格对称 InfoNCE 对比损失。
     *
     * <p>流程：</p>
     * <ol>
     *   <li>相似度矩阵 {@code S = textEmb · imageEmb^T * logitScale}，形状 {@code [B, B]}；</li>
     *   <li>行方向 softmax 得到 text→image 概率，对角为正样本概率；</li>
     *   <li>列方向 softmax 得到 image→text 概率；</li>
     *   <li>两方向 -log(p_diag) 的均值对称平均。</li>
     * </ol>
     *
     * @param textEmb    L2 归一化后的文本池化特征 {@code [B, H]}
     * @param imageEmb   L2 归一化后的图像池化特征 {@code [B, H]}
     * @param logitScale 温度倒数，CLIP 默认 {@link #DEFAULT_LOGIT_SCALE}
     * @return 标量损失 Variable
     */
    public static Variable computeContrastiveLoss(Variable textEmb, Variable imageEmb, float logitScale) {
        int batchSize = textEmb.getValue().getShape().getShapeDims()[0];

        Variable logits = textEmb.matMul(imageEmb.transpose())
                .mul(new Variable(logitScale));

        Variable rowProb = logits.softMax();                  // text→image
        Variable colProb = logits.transpose().softMax();      // image→text

        Variable lossT2I = diagonalNegLogMean(rowProb, batchSize);
        Variable lossI2T = diagonalNegLogMean(colProb, batchSize);

        return lossT2I.add(lossI2T).mul(new Variable(0.5f));
    }

    /**
     * 取 B×B 概率矩阵对角元素的 -log 均值（Variable 版，参与反向传播）。
     *
     * <p>由于当前 Variable 层没有 gather 算子，这里逐行用 {@link Variable#sliceRange(int, int, int)}
     * 取出 {@code prob[i:i+1, i:i+1]}，先 clip 防 log(0)，再累加 -log。</p>
     */
    private static Variable diagonalNegLogMean(Variable prob, int batchSize) {
        Variable accumulator = null;
        Variable negOne = new Variable(-1.0f);
        for (int i = 0; i < batchSize; i++) {
            Variable row = prob.sliceRange(0, i, i + 1);      // [1, B]
            Variable cell = row.sliceRange(1, i, i + 1);      // [1, 1]
            Variable safe = cell.clip(1e-8f, 1.0f);
            Variable negLog = safe.log().mul(negOne);
            accumulator = (accumulator == null) ? negLog : accumulator.add(negLog);
        }
        return accumulator.mul(new Variable(1.0f / Math.max(1, batchSize)));
    }

    // ==================== 多模态编码 + 池化 公共入口 ====================

    /**
     * 多模态"编码 → 可选双向融合 → 均值池化 → L2 归一化"的标准流水线。
     *
     * <p>{@link PretrainTrainer} 和 {@link FinetuneTrainer} 的 {@code trainStep} / {@code evaluate}
     * 都会重复这段逻辑，抽到这里集中维护，保证训练 / 验证两端口径完全一致。</p>
     *
     * <p>处理步骤：</p>
     * <ol>
     *   <li>{@code model.encodeText(textVar)} → {@code [B, text_len, H]}；</li>
     *   <li>{@code model.encodeImage(imageVar)} → {@code [B, num_patches, H]}；</li>
     *   <li>若 {@code bananaBlock.getFusionLayer() != null}，调用
     *       {@code forwardBoth} 让双向 cross-attention 参数接入计算图；</li>
     *   <li>序列维度均值池化 {@code mean(1, false)} → {@code [B, H]}；</li>
     *   <li>{@link #l2Normalize(Variable)} 最后一维归一化。</li>
     * </ol>
     *
     * @param model       Banana 模型，提供 encodeText / encodeImage 入口
     * @param bananaBlock 对应的 {@link BananaBlock}，用于访问 fusion 层
     * @param textVar     文本 token ids 变量 {@code [B, text_len]}
     * @param imageVar    图像像素变量 {@code [B, C, H, W]}
     * @return 长度为 2 的数组：{@code [textEmbedding, imageEmbedding]}，均为 {@code [B, H]}
     */
    public static Variable[] encodeAndPoolPair(BananaModel model,
                                                BananaBlock bananaBlock,
                                                Variable textVar,
                                                Variable imageVar) {
        // 1. 编码
        Variable textFeatures = model.encodeText(textVar);      // [B, text_len, H]
        Variable imageFeatures = model.encodeImage(imageVar);   // [B, num_patches, H]

        // 2. 可选双向融合，让 fusion 层参数也接入计算图
        Variable textForLoss;
        Variable imageForLoss;
        if (bananaBlock != null && bananaBlock.getFusionLayer() != null) {
            Variable[] fused = bananaBlock.getFusionLayer()
                    .forwardBoth(textFeatures, imageFeatures);
            textForLoss = fused[0];
            imageForLoss = fused[1];
        } else {
            textForLoss = textFeatures;
            imageForLoss = imageFeatures;
        }

        // 3. 序列维度均值池化
        Variable textPooled = textForLoss.mean(1, false);     // [B, H]
        Variable imagePooled = imageForLoss.mean(1, false);   // [B, H]

        // 4. L2 归一化
        Variable textEmb = l2Normalize(textPooled);
        Variable imageEmb = l2Normalize(imagePooled);

        return new Variable[]{textEmb, imageEmb};
    }
}
