package io.leavesfly.tinyai.minimind.model.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.activation.SiLU;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * SwiGLU 前馈网络（对标 Python MiniMind FeedForward）
 * <p>
 * SwiGLU 结构：
 * output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
 * <p>
 * 包含三个线性投影层：
 * - gate_proj: hidden_size -> intermediate_size（门控分支）
 * - up_proj:   hidden_size -> intermediate_size（上投影分支）
 * - down_proj:  intermediate_size -> hidden_size（下投影）
 * <p>
 * 相比标准 FFN (Linear -> ReLU -> Linear) 的改进：
 * - 门控机制过滤信息，训练更稳定
 * - SiLU 激活函数比 GELU 计算简单且效果相当
 *
 * @author TinyAI Team
 * @version 1.0
 */
public class SwiGLUFeedForward extends Module {

    private final Linear gateProj;   // 门控投影: hiddenSize -> intermediateSize
    private final SiLU silu;         // SiLU 激活函数
    private final Linear upProj;     // 上投影: hiddenSize -> intermediateSize
    private final Linear downProj;   // 下投影: intermediateSize -> hiddenSize

    /**
     * 构造 SwiGLU 前馈网络
     *
     * @param name             层名称
     * @param hiddenSize       隐藏层维度
     * @param intermediateSize 中间层维度
     */
    public SwiGLUFeedForward(String name, int hiddenSize, int intermediateSize) {
        super(name);

        // 门控分支: hiddenSize -> intermediateSize（无 bias，对标 Python）
        gateProj = new Linear(name + "_gate_proj", hiddenSize, intermediateSize, false);
        registerModule("gate_proj", gateProj);

        // SiLU 激活
        silu = new SiLU(name + "_silu");
        registerModule("silu", silu);

        // 上分支: hiddenSize -> intermediateSize（无 bias）
        upProj = new Linear(name + "_up_proj", hiddenSize, intermediateSize, false);
        registerModule("up_proj", upProj);

        // 下投影: intermediateSize -> hiddenSize（无 bias）
        downProj = new Linear(name + "_down_proj", intermediateSize, hiddenSize, false);
        registerModule("down_proj", downProj);
    }

    /**
     * 前向传播
     * <p>
     * SwiGLU: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // SwiGLU: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
        Variable gate = silu.forward(gateProj.forward(x));
        Variable up = upProj.forward(x);
        Variable hidden = gate.mul(up);
        return downProj.forward(hidden);
    }

    @Override
    public String extraRepr() {
        return String.format("gate=%s, up=%s, down=%s", gateProj, upProj, downProj);
    }
}
