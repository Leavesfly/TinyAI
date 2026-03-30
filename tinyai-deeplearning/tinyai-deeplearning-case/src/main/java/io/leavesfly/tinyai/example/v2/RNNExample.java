package io.leavesfly.tinyai.example.v2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.rnn.LSTM;
import io.leavesfly.tinyai.nnet.v2.layer.rnn.GRU;
import io.leavesfly.tinyai.nnet.v2.layer.rnn.SimpleRNN;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.util.PlantUML;
import io.leavesfly.tinyai.util.UmlPrinter;

/**
 * RNN序列建模示例
 * <p>
 * 本示例展示如何:
 * 1. 使用LSTM、GRU、SimpleRNN处理序列数据
 * 2. 管理RNN的隐藏状态
 * 3. 构建序列分类模型
 */
public class RNNExample {

    /**
     * 使用LSTM的序列分类器
     * 用于情感分析等任务
     */
    static class LSTMClassifier extends Module {
        private final LSTM lstm;
        private final Linear fc;

        public LSTMClassifier(String name, int inputSize, int hiddenSize, int numClasses) {
            super(name);

            lstm = new LSTM("lstm", inputSize, hiddenSize, true);
            fc = new Linear("fc", hiddenSize, numClasses, true);

            registerModule("lstm", lstm);
            registerModule("fc", fc);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0]; // (seq_len, batch_size, input_size)

            // 重置隐藏状态（对于每个新序列）
            lstm.resetState();

            // 处理序列
            int[] shape = x.getValue().getShape().getShapeDims();
            int seqLen = shape[0];
            int batchSize = shape[1];
            int inputSize = shape[2];

            Variable lastHidden = null;
            for (int t = 0; t < seqLen; t++) {
                // 提取第t个时间步的输入
                Variable xt = extractTimeStep(x, t, batchSize, inputSize);

                // LSTM前向传播
                lastHidden = lstm.forward(xt);
            }

            // 使用最后一个隐藏状态进行分类
            Variable output = fc.forward(lastHidden);

            return output;
        }

        private Variable extractTimeStep(Variable x, int t, int batchSize, int inputSize) {
            float[] data = x.getValue().getArray();
            float[] stepData = new float[batchSize * inputSize];

            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < inputSize; i++) {
                    int srcIdx = t * (batchSize * inputSize) + b * inputSize + i;
                    int dstIdx = b * inputSize + i;
                    stepData[dstIdx] = data[srcIdx];
                }
            }

            return new Variable(NdArray.of(stepData, Shape.of(batchSize, inputSize)));
        }
    }

    /**
     * 使用GRU的序列分类器
     */
    static class GRUClassifier extends Module {
        private final GRU gru;
        private final Linear fc;

        public GRUClassifier(String name, int inputSize, int hiddenSize, int numClasses) {
            super(name);

            gru = new GRU("gru", inputSize, hiddenSize, true);
            fc = new Linear("fc", hiddenSize, numClasses, true);

            registerModule("gru", gru);
            registerModule("fc", fc);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];

            gru.resetState();

            int[] shape = x.getValue().getShape().getShapeDims();
            int seqLen = shape[0];
            int batchSize = shape[1];
            int inputSize = shape[2];

            Variable lastHidden = null;
            for (int t = 0; t < seqLen; t++) {
                Variable xt = extractTimeStep(x, t, batchSize, inputSize);
                lastHidden = gru.forward(xt);
            }

            Variable output = fc.forward(lastHidden);
            return output;
        }

        private Variable extractTimeStep(Variable x, int t, int batchSize, int inputSize) {
            float[] data = x.getValue().getArray();
            float[] stepData = new float[batchSize * inputSize];

            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < inputSize; i++) {
                    int srcIdx = t * (batchSize * inputSize) + b * inputSize + i;
                    int dstIdx = b * inputSize + i;
                    stepData[dstIdx] = data[srcIdx];
                }
            }

            return new Variable(NdArray.of(stepData, Shape.of(batchSize, inputSize)));
        }
    }

    /**
     * 使用SimpleRNN的序列分类器
     */
    static class SimpleRNNClassifier extends Module {
        private final SimpleRNN rnn;
        private final Linear fc;

        public SimpleRNNClassifier(String name, int inputSize, int hiddenSize, int numClasses) {
            super(name);

            rnn = new SimpleRNN("rnn", inputSize, hiddenSize, true, "tanh");
            fc = new Linear("fc", hiddenSize, numClasses, true);

            registerModule("rnn", rnn);
            registerModule("fc", fc);
        }

        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];

            rnn.resetState();

            int[] shape = x.getValue().getShape().getShapeDims();
            int seqLen = shape[0];
            int batchSize = shape[1];
            int inputSize = shape[2];

            Variable lastHidden = null;
            for (int t = 0; t < seqLen; t++) {
                Variable xt = extractTimeStep(x, t, batchSize, inputSize);
                lastHidden = rnn.forward(xt);
            }

            Variable output = fc.forward(lastHidden);
            return output;
        }

        private Variable extractTimeStep(Variable x, int t, int batchSize, int inputSize) {
            float[] data = x.getValue().getArray();
            float[] stepData = new float[batchSize * inputSize];

            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < inputSize; i++) {
                    int srcIdx = t * (batchSize * inputSize) + b * inputSize + i;
                    int dstIdx = b * inputSize + i;
                    stepData[dstIdx] = data[srcIdx];
                }
            }

            return new Variable(NdArray.of(stepData, Shape.of(batchSize, inputSize)));
        }
    }

    public static void main(String[] args) {
        System.out.println("=== RNN序列建模示例 ===\n");

        int seqLen = 10;      // 序列长度
        int batchSize = 4;    // 批次大小
        int inputSize = 50;   // 输入维度（如词向量维度）
        int hiddenSize = 128; // 隐藏层维度
        int numClasses = 3;   // 分类数量（如正面/中性/负面）

        // 创建模拟序列数据 (seq_len, batch_size, input_size)
        float[] seqData = new float[seqLen * batchSize * inputSize];
        for (int i = 0; i < seqData.length; i++) {
            seqData[i] = (float) (Math.random() * 0.1 - 0.05);
        }
        NdArray seqArray = NdArray.of(seqData, Shape.of(seqLen, batchSize, inputSize));
        Variable input = new Variable(seqArray,"input");

        // 示例1: LSTM分类器
        System.out.println("示例1: LSTM序列分类器");
        System.out.println("----------------------------------------");

        LSTMClassifier lstmModel = new LSTMClassifier("lstm_classifier", inputSize, hiddenSize, numClasses);
        lstmModel.eval();

        System.out.println("1. 前向传播:");
        System.out.println("   输入形状: [" + seqLen + ", " + batchSize + ", " + inputSize + "]");
        Variable lstmOutput = lstmModel.forward(input);

//        System.out.println(PlantUML.generateSimpleGraph(lstmOutput));

        System.out.println("   输出形状: " + shapeToString(lstmOutput.getValue().getShape()));

        float[] lstmOutputData = lstmOutput.getValue().getArray();
        System.out.println("   第一个样本的预测:");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("     类别%d: %.4f%n", i, lstmOutputData[i]);
        }
        System.out.println();

        // 示例2: GRU分类器
        System.out.println("\n示例2: GRU序列分类器");
        System.out.println("----------------------------------------");

        GRUClassifier gruModel = new GRUClassifier("gru_classifier", inputSize, hiddenSize, numClasses);
        gruModel.eval();

        System.out.println("1. 前向传播:");
        Variable gruOutput = gruModel.forward(input);
        System.out.println("   输出形状: " + shapeToString(gruOutput.getValue().getShape()));

        float[] gruOutputData = gruOutput.getValue().getArray();
        System.out.println("   第一个样本的预测:");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("     类别%d: %.4f%n", i, gruOutputData[i]);
        }
        System.out.println();

        // 示例3: SimpleRNN分类器
        System.out.println("\n示例3: SimpleRNN序列分类器");
        System.out.println("----------------------------------------");

        SimpleRNNClassifier rnnModel = new SimpleRNNClassifier("rnn_classifier", inputSize, hiddenSize, numClasses);
        rnnModel.eval();

        System.out.println("1. 前向传播:");
        Variable rnnOutput = rnnModel.forward(input);

        System.out.println(PlantUML.generateSimpleGraph(rnnOutput));

        System.out.println("   输出形状: " + shapeToString(rnnOutput.getValue().getShape()));

        float[] rnnOutputData = rnnOutput.getValue().getArray();
        System.out.println("   第一个样本的预测:");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("     类别%d: %.4f%n", i, rnnOutputData[i]);
        }
        System.out.println();




        // RNN变体对比
        System.out.println("\nRNN变体对比:");
        System.out.println("----------------------------------------");
        System.out.println("LSTM: 3个门（输入门、遗忘门、输出门）+ 细胞状态，参数量最多");
        System.out.println("GRU: 2个门（重置门、更新门），参数量适中");
        System.out.println("SimpleRNN: 无门控机制，参数量最少");
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

