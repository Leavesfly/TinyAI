package io.leavesfly.tinyai.example.v2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.transformer.MultiHeadAttention;
import io.leavesfly.tinyai.nnet.layer.transformer.PositionalEncoding;
import io.leavesfly.tinyai.nnet.layer.transformer.TransformerEncoderLayer;
import io.leavesfly.tinyai.nnet.layer.transformer.TransformerDecoderLayer;

/**
 * Transformer模型示例
 * <p>
 * 本示例展示如何:
 * 1. 使用多头注意力机制
 * 2. 使用位置编码
 * 3. 构建Transformer编码器和解码器
 */
public class TransformerExample {

    /**
     * 简单的Transformer编码器
     */
    static class TransformerEncoder extends Module {
        private final PositionalEncoding posEnc;
        private final TransformerEncoderLayer encoderLayer;

        public TransformerEncoder(String name, int dModel, int nHead, int dimFeedforward,
                                  int maxSeqLen, float dropout) {
            super(name);

            posEnc = new PositionalEncoding("pos_enc", dModel, maxSeqLen, dropout);
            encoderLayer = new TransformerEncoderLayer("encoder", dModel, nHead,
                    dimFeedforward, dropout, false);

            registerModule("pos_enc", posEnc);
            registerModule("encoder_layer", encoderLayer);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0]; // (batch_size, seq_len, d_model)

            // 添加位置编码
            x = posEnc.forward(x);

            // Transformer编码器层
            x = encoderLayer.forward(x);

            return x;
        }
    }

    /**
     * 简单的Transformer解码器
     */
    static class TransformerDecoder extends Module {
        private final PositionalEncoding posEnc;
        private final TransformerDecoderLayer decoderLayer;

        public TransformerDecoder(String name, int dModel, int nHead, int dimFeedforward,
                                  int maxSeqLen, float dropout) {
            super(name);

            posEnc = new PositionalEncoding("pos_dec", dModel, maxSeqLen, dropout);
            decoderLayer = new TransformerDecoderLayer("decoder", dModel, nHead,
                    dimFeedforward, dropout, false);

            registerModule("pos_dec", posEnc);
            registerModule("decoder_layer", decoderLayer);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable tgt = inputs[0];    // (batch_size, tgt_len, d_model)
            Variable memory = inputs[1]; // (batch_size, src_len, d_model)

            // 添加位置编码
            tgt = posEnc.forward(tgt);

            // Transformer解码器层
            tgt = decoderLayer.forward(tgt, memory);

            return tgt;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Transformer模型示例 ===\n");

        int batchSize = 2;
        int seqLen = 10;
        int dModel = 64;      // 模型维度
        int nHead = 4;        // 注意力头数
        int dimFeedforward = 256; // 前馈网络维度
        int maxSeqLen = 100;  // 最大序列长度
        float dropout = 0.1f;

        // 示例1: 多头注意力
        System.out.println("示例1: 多头注意力（Multi-Head Attention）");
        System.out.println("----------------------------------------");

        MultiHeadAttention mha = new MultiHeadAttention("mha", dModel, nHead, dropout);
        mha.eval();

        System.out.println("配置:");
        System.out.println("  模型维度 (d_model): " + dModel);
        System.out.println("  注意力头数 (n_head): " + nHead);
        System.out.println("  每个头的维度: " + (dModel / nHead));
        System.out.println();

        // 创建输入 (batch_size, seq_len, d_model)
        float[] qkvData = new float[batchSize * seqLen * dModel];
        for (int i = 0; i < qkvData.length; i++) {
            qkvData[i] = (float) (Math.random() * 0.1 - 0.05);
        }
        Variable q = new Variable(NdArray.of(qkvData, Shape.of(batchSize, seqLen, dModel)));
        Variable k = new Variable(NdArray.of(qkvData, Shape.of(batchSize, seqLen, dModel)));
        Variable v = new Variable(NdArray.of(qkvData, Shape.of(batchSize, seqLen, dModel)));

        System.out.println("输入:");
        System.out.println("  Query (Q): " + shapeToString(q.getValue().getShape()));
        System.out.println("  Key (K):   " + shapeToString(k.getValue().getShape()));
        System.out.println("  Value (V): " + shapeToString(v.getValue().getShape()));
        System.out.println();

        Variable mhaOutput = mha.forward(q, k, v);
        System.out.println("输出:");
        System.out.println("  形状: " + shapeToString(mhaOutput.getValue().getShape()));
        System.out.println("  说明: 多头注意力聚合了来自不同表示子空间的信息");
        System.out.println();

        // 示例2: 位置编码
        System.out.println("\n示例2: 位置编码（Positional Encoding）");
        System.out.println("----------------------------------------");

        PositionalEncoding posEnc = new PositionalEncoding("pos_enc", dModel, maxSeqLen, dropout);
        posEnc.eval();

        System.out.println("配置:");
        System.out.println("  模型维度: " + dModel);
        System.out.println("  最大序列长度: " + maxSeqLen);
        System.out.println();

        Variable posInput = new Variable(NdArray.of(qkvData, Shape.of(batchSize, seqLen, dModel)));
        System.out.println("输入:");
        System.out.println("  形状: " + shapeToString(posInput.getValue().getShape()));
        System.out.println();

        Variable posOutput = posEnc.forward(posInput);
        System.out.println("输出:");
        System.out.println("  形状: " + shapeToString(posOutput.getValue().getShape()));
        System.out.println("  说明: 位置编码为序列添加了位置信息");
        System.out.println();

        // 示例3: Transformer编码器
        System.out.println("\n示例3: Transformer编码器");
        System.out.println("----------------------------------------");

        TransformerEncoder encoder = new TransformerEncoder("encoder", dModel, nHead,
                dimFeedforward, maxSeqLen, dropout);
        encoder.eval();

        System.out.println("配置:");
        System.out.println("  模型维度: " + dModel);
        System.out.println("  注意力头数: " + nHead);
        System.out.println("  前馈网络维度: " + dimFeedforward);
        System.out.println();

        Variable srcInput = new Variable(NdArray.of(qkvData, Shape.of(batchSize, seqLen, dModel)));
        System.out.println("输入:");
        System.out.println("  源序列: " + shapeToString(srcInput.getValue().getShape()));
        System.out.println();

        Variable encoderOutput = encoder.forward(srcInput);
        System.out.println("输出:");
        System.out.println("  编码表示: " + shapeToString(encoderOutput.getValue().getShape()));
        System.out.println("  说明: 编码器输出包含了输入序列的上下文信息");
        System.out.println();

        // 示例4: Transformer解码器
        System.out.println("\n示例4: Transformer解码器");
        System.out.println("----------------------------------------");

        TransformerDecoder decoder = new TransformerDecoder("decoder", dModel, nHead,
                dimFeedforward, maxSeqLen, dropout);
        decoder.eval();

        System.out.println("配置:");
        System.out.println("  模型维度: " + dModel);
        System.out.println("  注意力头数: " + nHead);
        System.out.println("  前馈网络维度: " + dimFeedforward);
        System.out.println();

        // 创建目标序列（通常比源序列短）
        int tgtLen = 8;
        float[] tgtData = new float[batchSize * tgtLen * dModel];
        for (int i = 0; i < tgtData.length; i++) {
            tgtData[i] = (float) (Math.random() * 0.1 - 0.05);
        }
        Variable tgtInput = new Variable(NdArray.of(tgtData, Shape.of(batchSize, tgtLen, dModel)));

        System.out.println("输入:");
        System.out.println("  目标序列: " + shapeToString(tgtInput.getValue().getShape()));
        System.out.println("  记忆（编码器输出）: " + shapeToString(encoderOutput.getValue().getShape()));
        System.out.println();

        Variable decoderOutput = decoder.forward(tgtInput, encoderOutput);
        System.out.println("输出:");
        System.out.println("  解码表示: " + shapeToString(decoderOutput.getValue().getShape()));
        System.out.println("  说明: 解码器结合了目标序列和源序列的信息");
        System.out.println();

        // Transformer应用场景
        System.out.println("\nTransformer应用场景:");
        System.out.println("----------------------------------------");
        System.out.println("1. 机器翻译: 将一种语言翻译成另一种语言");
        System.out.println("2. 文本摘要: 生成文本的简短摘要");
        System.out.println("3. 问答系统: 根据上下文回答问题");
        System.out.println("4. 语言建模: GPT系列使用仅解码器架构");
        System.out.println("5. 文本分类: BERT使用仅编码器架构");
        System.out.println("6. 图像处理: Vision Transformer (ViT)");
        System.out.println();

        // 架构优势
        System.out.println("Transformer架构优势:");
        System.out.println("----------------------------------------");
        System.out.println("1. 并行计算: 不像RNN需要顺序处理，可以并行计算");
        System.out.println("2. 长距离依赖: 自注意力机制可以捕获任意距离的依赖");
        System.out.println("3. 可解释性: 注意力权重可以可视化，增强可解释性");
        System.out.println("4. 可扩展性: 可以堆叠多层，构建深层网络");
        System.out.println();

        System.out.println("=== 示例完成 ===");
    }

    private static String shapeToString(Shape shape) {
        int[] dims = shape.getShapeDims();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(dims[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}

