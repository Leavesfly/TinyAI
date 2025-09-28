package io.leavesfly.tinyai.modality.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.transformer.GPT2Block;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2OutputHead;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2TokenEmbedding;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2 小规模语言模型实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * GPT2Model类实现了GPT-2语言模型，基于Transformer解码器的自回归语言模型。
 * 特点：
 * 1. 仅使用解码器架构
 * 2. 使用掩码多头自注意力防止未来信息泄露
 * 3. Pre-LayerNorm结构
 * 4. 残差连接
 * <p>
 * 模型结构：
 * Token Embedding + Position Embedding
 * → N × GPT2Block
 * → Final LayerNorm
 * → Output Head
 * <p>
 * <p>
 * todo
 */
public class GPT2Model extends Block {


    public GPT2Model(String _name, Shape _inputShape) {
        super(_name, _inputShape);
    }

    @Override
    public void init() {

    }
}