package io.leavesfly.tinyai.banana.decoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * 像素投影层 (Pixel Projection)
 * 
 * 将高分辨率特征图投影为RGB像素值:
 * 1. 通道投影 - 将特征通道投影为RGB 3通道
 * 2. 逐像素线性变换
 * 
 * 架构流程:
 * 输入 [batch, in_channels, height, width]
 *   ↓ Permute
 * [batch, height, width, in_channels]
 *   ↓ Linear Projection
 * [batch, height, width, 3]
 *   ↓ Permute
 * 输出 [batch, 3, height, width]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class PixelProjection extends Module {
    
    private final int inChannels;
    private final int outChannels; // RGB: 3
    
    // 像素级线性投影
    private final Linear pixelLinear;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数（通常是3，对应RGB）
     */
    public PixelProjection(String name, int inChannels, int outChannels) {
        super(name);
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        
        // 像素级投影: in_channels -> out_channels (3)
        this.pixelLinear = new Linear(
            name + "_pixel_linear",
            inChannels,
            outChannels,
            true  // 使用偏置
        );
        registerModule("pixel_linear", pixelLinear);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入特征图 [batch, in_channels, height, width]
     * @return RGB图像 [batch, 3, height, width]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("PixelProjection需要输入特征图");
        }
        
        Variable x = inputs[0];
        
        int[] shape = x.getValue().getShape().getShapeDims();
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                "输入必须是4维 [batch, channels, height, width], 当前: " + 
                java.util.Arrays.toString(shape)
            );
        }
        
        // 1. 重排维度: [batch, channels, height, width] -> [batch, height, width, channels]
        // 使用 Permute Function 保持梯度计算图连通
        Variable permuted = permuteNCHWToNHWC(x);
        
        // 2. 应用像素级线性投影
        // [batch, height, width, in_channels] -> [batch, height, width, out_channels]
        Variable projected = pixelLinear.forward(permuted);
        
        // 3. 恢复维度顺序: [batch, height, width, out_channels] -> [batch, out_channels, height, width]
        // 使用 Permute Function 保持梯度计算图连通
        Variable output = permuteNHWCToNCHW(projected);
        
        return output;
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
    
    @Override
    public String toString() {
        return String.format(
            "PixelProjection{inChannels=%d, outChannels=%d}",
            inChannels, outChannels
        );
    }
}
