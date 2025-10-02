package io.leavesfly.tinyai.gpt2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2 输出头实现
 * 
 * 负责将Transformer的隐藏状态映射到词汇表上的概率分布
 * 实现线性变换：hidden_states -> logits
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT2OutputHead extends Layer {
    
    /** 输出权重矩阵 (nEmbd, vocabSize) */
    private Parameter outputWeight;
    
    /** 是否使用偏置 */
    private boolean useBias;
    
    /** 偏置参数 (1, vocabSize) */
    private Parameter outputBias;
    
    /** 词汇表大小 */
    private int vocabSize;
    
    /** 模型维度 */
    private int nEmbd;
    
    /**
     * 构造GPT-2输出头
     * 
     * @param name 层名称
     * @param nEmbd 模型维度
     * @param vocabSize 词汇表大小
     * @param useBias 是否使用偏置
     */
    public GPT2OutputHead(String name, int nEmbd, int vocabSize, boolean useBias) {
        super(name, 
              Shape.of(-1, -1, nEmbd), 
              Shape.of(-1, -1, vocabSize));
        
        this.nEmbd = nEmbd;
        this.vocabSize = vocabSize;
        this.useBias = useBias;
        
        init();
    }
    
    /**
     * 使用GPT2Config的构造函数
     */
    public GPT2OutputHead(String name, GPT2Config config) {
        this(name, config.getNEmbd(), config.getVocabSize(), false);  // GPT-2通常不使用偏置
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化输出权重矩阵
            // 使用标准正态分布初始化，然后缩放
            double initStd = 0.02;  // GPT-2标准初始化
            outputWeight = new Parameter(
                NdArray.likeRandomN(Shape.of(nEmbd, vocabSize))
                       .mulNum((float) initStd)
            );
            outputWeight.setName(name + "_weight");
            addParam(outputWeight.getName(), outputWeight);
            
            // 如果使用偏置，初始化偏置参数
            if (useBias) {
                outputBias = new Parameter(NdArray.zeros(Shape.of(1, vocabSize)));
                outputBias.setName(name + "_bias");
                addParam(outputBias.getName(), outputBias);
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable hiddenStates = inputs[0];  // shape: (batch_size, seq_len, n_embd)
        NdArray inputData = hiddenStates.getValue();
        
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 验证输入维度
        if (inputData.getShape().getDimension(2) != nEmbd) {
            throw new IllegalArgumentException(
                String.format("输入的最后一维(%d)必须等于模型维度(%d)",
                            inputData.getShape().getDimension(2), nEmbd)
            );
        }
        
        // 1. 将三维输入重塑为二维矩阵：(batch_size * seq_len, n_embd)
        NdArray input2D = reshapeTo2D(inputData, batchSize * seqLen, nEmbd);
        
        // 2. 执行矩阵乘法：(batch_size * seq_len, n_embd) × (n_embd, vocab_size)
        Variable logits2D = matmul2D(new Variable(input2D), outputWeight);
        
        // 3. 如果使用偏置，添加偏置
        if (useBias) {
            logits2D = addBias(logits2D, outputBias);
        }
        
        // 4. 重塑回三维：(batch_size, seq_len, vocab_size)
        NdArray logits3D = reshapeFrom2D(logits2D.getValue(), batchSize, seqLen, vocabSize);
        
        return new Variable(logits3D);
    }
    
    /**
     * 将三维张量重塑为二维矩阵
     * 
     * @param input 输入张量
     * @param rows 输出行数
     * @param cols 输出列数
     * @return 重塑后的二维矩阵
     */
    private NdArray reshapeTo2D(NdArray input, int rows, int cols) {
        return input.reshape(Shape.of(rows, cols));
    }
    
    /**
     * 将二维矩阵重塑为三维张量
     * 
     * @param input 输入矩阵
     * @param batchSize 批次大小
     * @param seqLen 序列长度
     * @param vocabSize 词汇表大小
     * @return 重塑后的三维张量
     */
    private NdArray reshapeFrom2D(NdArray input, int batchSize, int seqLen, int vocabSize) {
        return input.reshape(Shape.of(batchSize, seqLen, vocabSize));
    }
    
    /**
     * 执行二维矩阵乘法
     * 
     * @param input 输入变量 (m, k)
     * @param weight 权重参数 (k, n)
     * @return 矩阵乘法结果 (m, n)
     */
    private Variable matmul2D(Variable input, Parameter weight) {
        // 使用现有的线性变换功能
        return input.linear(weight, null);
    }
    
    /**
     * 添加偏置
     * 
     * @param input 输入变量
     * @param bias 偏置参数
     * @return 添加偏置后的变量
     */
    private Variable addBias(Variable input, Parameter bias) {
        NdArray inputData = input.getValue();
        NdArray biasData = bias.getValue();
        
        // 广播加法
        NdArray result = inputData.add(biasData);
        return new Variable(result);
    }
    
    /**
     * 预测下一个token的概率分布
     * 
     * @param hiddenState 最后一个位置的隐藏状态 (batch_size, n_embd)
     * @return 词汇表上的概率分布 (batch_size, vocab_size)
     */
    public Variable predictNextToken(Variable hiddenState) {
        NdArray inputData = hiddenState.getValue();
        
        // 如果输入是二维的，直接处理
        if (inputData.getShape().getDimNum() == 2) {
            int batchSize = inputData.getShape().getDimension(0);
            if (inputData.getShape().getDimension(1) != nEmbd) {
                throw new IllegalArgumentException("隐藏状态维度不匹配");
            }
            
            Variable logits = matmul2D(hiddenState, outputWeight);
            if (useBias) {
                logits = addBias(logits, outputBias);
            }
            return logits;
        }
        
        // 如果输入是三维的，取最后一个时间步
        else if (inputData.getShape().getDimNum() == 3) {
            int batchSize = inputData.getShape().getDimension(0);
            int seqLen = inputData.getShape().getDimension(1);
            
            // 提取最后一个时间步的隐藏状态
            NdArray lastHidden = NdArray.of(Shape.of(batchSize, nEmbd));
            for (int b = 0; b < batchSize; b++) {
                for (int d = 0; d < nEmbd; d++) {
                    float value = inputData.get(b, seqLen - 1, d);
                    lastHidden.set(value, b, d);
                }
            }
            
            return predictNextToken(new Variable(lastHidden));
        }
        
        else {
            throw new IllegalArgumentException("不支持的输入维度：" + inputData.getShape().getDimNum());
        }
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 简化的反向传播实现
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取输出权重参数
     * 
     * @return 输出权重参数
     */
    public Parameter getOutputWeight() {
        return outputWeight;
    }
    
    /**
     * 获取偏置参数（如果使用）
     * 
     * @return 偏置参数，如果不使用偏置则返回null
     */
    public Parameter getOutputBias() {
        return outputBias;
    }
    
    /**
     * 是否使用偏置
     * 
     * @return 是否使用偏置
     */
    public boolean isUseBias() {
        return useBias;
    }
    
    /**
     * 获取词汇表大小
     * 
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }
    
    /**
     * 获取模型维度
     * 
     * @return 模型维度
     */
    public int getNEmbd() {
        return nEmbd;
    }
}