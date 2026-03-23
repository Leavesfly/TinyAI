package io.leavesfly.tinyai.banana.decoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * 上采样块 (Upsample Block)
 * 
 * 将低分辨率特征图上采样到高分辨率:
 * 1. 最近邻插值上采样 - 2x分辨率提升
 * 2. 卷积投影 - 通道数调整和特征精炼
 * 3. LayerNorm + ReLU - 归一化和激活
 * 
 * 架构流程:
 * 输入 [batch, in_channels, in_h, in_w]
 *   ↓ Nearest Neighbor Upsample (2x)
 * [batch, in_channels, in_h*2, in_w*2]
 *   ↓ Spatial Conv (模拟卷积)
 * [batch, out_channels, out_h, out_w]
 *   ↓ LayerNorm + ReLU
 * 输出 [batch, out_channels, out_h, out_w]
 * 
 * 注意: 由于TinyAI V2暂无转置卷积,使用最近邻插值+Linear投影模拟
 * 
 * @author leavesfly
 * @version 1.0
 */
public class UpsampleBlock extends Module {
    
    private final int inChannels;
    private final int outChannels;
    private final int inSize;
    private final int outSize;
    private final String activationType;
    
    // 通道投影层
    private final Linear channelProjection;
    
    // 归一化层
    private final LayerNorm layerNorm;
    
    /**
     * 构造函数（默认使用 ReLU 激活函数）
     * 
     * @param name 模块名称
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     * @param inSize 输入空间尺寸（假设正方形）
     * @param outSize 输出空间尺寸（假设正方形）
     */
    public UpsampleBlock(String name, int inChannels, int outChannels, 
                         int inSize, int outSize) {
        this(name, inChannels, outChannels, inSize, outSize, "relu");
    }
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     * @param inSize 输入空间尺寸（假设正方形）
     * @param outSize 输出空间尺寸（假设正方形）
     * @param activationType 激活函数类型: "relu", "gelu", "tanh", "none"
     */
    public UpsampleBlock(String name, int inChannels, int outChannels, 
                         int inSize, int outSize, String activationType) {
        super(name);
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.inSize = inSize;
        this.outSize = outSize;
        this.activationType = activationType;
        
        // 通道投影：in_channels -> out_channels
        this.channelProjection = new Linear(
            name + "_ch_proj",
            inChannels,
            outChannels,
            true
        );
        registerModule("ch_proj", channelProjection);
        
        // LayerNorm（应用在通道维度）
        this.layerNorm = new LayerNorm(
            name + "_ln",
            outChannels,
            1e-5f
        );
        registerModule("ln", layerNorm);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入特征图 [batch, in_channels, in_h, in_w]
     * @return 上采样后的特征图 [batch, out_channels, out_h, out_w]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("UpsampleBlock需要输入特征图");
        }
        
        Variable x = inputs[0];
        
        // 1. 最近邻插值上采样
        Variable upsampled = nearestNeighborUpsample(x, outSize, outSize);
        
        // 2. 手动重排维度：[batch, in_channels, out_h, out_w] -> [batch, out_h, out_w, in_channels]
        upsampled = permuteNCHWToNHWC(upsampled);
        
        // 3. 通道投影
        // [batch, out_h, out_w, in_channels] -> [batch, out_h, out_w, out_channels]
        Variable projected = channelProjection.forward(upsampled);
        
        // 4. LayerNorm
        projected = layerNorm.forward(projected);
        
        // 5. 激活函数
        projected = applyActivation(projected);
        
        // 6. 恢复维度顺序: [batch, out_h, out_w, out_channels] -> [batch, out_channels, out_h, out_w]
        Variable output = permuteNHWCToNCHW(projected);
        
        return output;
    }
    
    /**
     * 应用激活函数
     * 
     * @param x 输入张量
     * @return 激活后的张量
     */
    private Variable applyActivation(Variable x) {
        switch (activationType.toLowerCase()) {
            case "relu":
                return x.relu();
            case "gelu":
                return x.gelu();
            case "tanh":
                return x.tanh();
            case "none":
                return x;
            default:
                throw new IllegalArgumentException("不支持的激活函数类型: " + activationType);
        }
    }
    
    /**
     * 最近邻上采样
     * 
     * 使用 Variable 算子实现以保持梯度计算图连通。
     * 通过 reshape 和 expand 操作实现像素复制。
     * [B, C, H, W] -> [B, C, H, scale, W, scale] -> [B, C, outH, outW]
     * 
     * @param x 输入 [batch, channels, in_h, in_w]
     * @param outH 目标高度
     * @param outW 目标宽度
     * @return 上采样结果 [batch, channels, out_h, out_w]
     */
    private Variable nearestNeighborUpsample(Variable x, int outH, int outW) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int channels = shape[1];
        int inH = shape[2];
        int inW = shape[3];
            
        int scaleH = outH / inH;
        int scaleW = outW / inW;
            
        if (scaleH * inH != outH || scaleW * inW != outW) {
            throw new IllegalArgumentException(
                String.format("上采样比例必须为整数: inH=%d->outH=%d, inW=%d->outW=%d",
                    inH, outH, inW, outW)
            );
        }
            
        // [B, C, H, W] -> [B, C, H, 1, W, 1] -> expand [B, C, H, scaleH, W, scaleW] -> [B, C, outH, outW]
        Variable reshaped = x.reshape(Shape.of(batchSize, channels, inH, 1, inW, 1));
        Variable expanded = reshaped.expand(Shape.of(batchSize, channels, inH, scaleH, inW, scaleW));
        return expanded.reshape(Shape.of(batchSize, channels, outH, outW));
    }
        
    /**
     * 维度置换: NCHW -> NHWC
     * [batch, channels, height, width] -> [batch, height, width, channels]
     * 使用 Permute Function 保持梯度计算图连通。
     */
    private Variable permuteNCHWToNHWC(Variable x) {
        return new Permute(0, 2, 3, 1).call(x);
    }
        
    /**
     * 维度置换: NHWC -> NCHW
     * [batch, height, width, channels] -> [batch, channels, height, width]
     * 使用 Permute Function 保持梯度计算图连通。
     */
    private Variable permuteNHWCToNCHW(Variable x) {
        return new Permute(0, 3, 1, 2).call(x);
    }
    
    // ==================== Getter方法 ====================
    
    public int getInChannels() {
        return inChannels;
    }
    
    public int getOutChannels() {
        return outChannels;
    }
    
    public int getInSize() {
        return inSize;
    }
    
    public int getOutSize() {
        return outSize;
    }
    
    @Override
    public String toString() {
        return String.format(
            "UpsampleBlock{inChannels=%d, outChannels=%d, %dx%d->%dx%d}",
            inChannels, outChannels, inSize, inSize, outSize, outSize
        );
    }
}
