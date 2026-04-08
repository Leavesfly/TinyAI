package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/**
 * Expert Router - 专家路由网络
 * 
 * 根据输入计算每个专家的权重,并选择Top-K个专家激活
 * 
 * 核心功能:
 * 1. 计算门控权重: gate_logits = W_g · x
 * 2. Top-K选择: 只保留权重最大的K个专家
 * 3. Softmax归一化: 确保权重和为1
 * 4. Noisy Top-K: 添加噪声避免专家过载
 * 
 * 路由公式:
 * w_i = Softmax(W_g · x + noise)
 * Top-K: 选择权重最大的K个专家
 * 
 * @author leavesfly
 * @since 2024
 */
public class ExpertRouter extends Module {
    
    private final int inputDim;
    private final int numExperts;
    private final int topK;
    private final float noiseFactor;
    
    private final Linear gateLinear;  // 门控线性层: input_dim -> num_experts
    
    /**
     * 构造函数
     * 
     * @param inputDim 输入维度
     * @param numExperts 专家数量
     * @param topK Top-K选择数量
     * @param noiseFactor 噪声因子(用于负载均衡)
     */
    public ExpertRouter(int inputDim, int numExperts, int topK, float noiseFactor) {
        super("expert_router");
        
        if (topK > numExperts) {
            throw new IllegalArgumentException("topK must be <= numExperts");
        }
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be >= 1");
        }
        
        this.inputDim = inputDim;
        this.numExperts = numExperts;
        this.topK = topK;
        this.noiseFactor = noiseFactor;
        
        // 创建门控层
        this.gateLinear = new Linear("gate", inputDim, numExperts, true);
        
        // 注册子模块
        registerModule("gate", gateLinear);
    }
    
    /**
     * 前向传播(Variable版本)
     * 
     * 委托到forwardRouter，将topKWeights打包为NdArray返回。
     * 如需完整路由信息(indices等)，请直接调用forwardRouter。
     */
    @Override
    public Variable forward(Variable... inputs) {
        RouterOutput routerOutput = forwardRouter(inputs[0]);
        float[][] topKWeights = routerOutput.getTopKWeights();
        int batchSize = routerOutput.getBatchSize();
        int topKSize = routerOutput.getTopK();
        float[] flatWeights = new float[batchSize * topKSize];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(topKWeights[b], 0, flatWeights, b * topKSize, topKSize);
        }
        NdArray weightsArray = NdArray.of(flatWeights, Shape.of(batchSize, topKSize));
        return new Variable(weightsArray);
    }
    
    /**
     * 前向传播(NdArray版本)
     * 
     * 委托到forward(Variable...)，将输入包装为Variable后调用。
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        Variable input = new Variable(inputs[0]);
        return forward(input).getValue();
    }
    
    /**
     * 前向传播(返回RouterOutput)
     * 
     * @param input 输入 [batch_size, input_dim]
     * @return 路由结果 RouterOutput
     */
    public RouterOutput forwardRouter(Variable input) {
        // 支持3D输入 [batch_size, seq_len, input_dim] -> reshape为 [batch_size * seq_len, input_dim]
        Variable flatInput = input;
        if (input.getShape().getDimNum() == 3) {
            int batchSize = input.getShape().getDimension(0);
            int seqLen = input.getShape().getDimension(1);
            flatInput = input.reshape(Shape.of(batchSize * seqLen, inputDim));
        }
        int flatBatchSize = flatInput.getShape().getDimension(0);
        
        // 1. 计算门控logits: [flatBatchSize, num_experts]
        Variable gateLogits = gateLinear.forward(flatInput);
        
        // 2. 添加噪声(训练时)
        if (isTraining() && noiseFactor > 0) {
            gateLogits = addNoise(gateLogits);
        }
        
        // 3. Top-K选择和Softmax
        RouterOutput output = topKGating(gateLogits, flatBatchSize);
        
        return output;
    }
    
    /**
     * Top-K门控计算
     * 
     * @param gateLogits 门控logits [batch_size, num_experts]
     * @param batchSize 批次大小
     * @return 路由结果
     */
    private RouterOutput topKGating(Variable gateLogits, int batchSize) {
        // 准备输出数组
        int[][] topKIndices = new int[batchSize][topK];     // Top-K专家索引
        float[][] topKWeights = new float[batchSize][topK]; // Top-K专家权重
        float[][] allWeights = new float[batchSize][numExperts]; // 所有专家权重(用于负载均衡)
        
        // 对每个样本进行Top-K选择
        for (int b = 0; b < batchSize; b++) {
            // 提取当前样本的logits: 使用 indexSelect
            Variable batchIndexVar = new Variable(NdArray.of(new float[]{b}));
            batchIndexVar.setRequireGrad(false);
            Variable sampleLogits = gateLogits.indexSelect(0, batchIndexVar);  // [1, num_experts]
            
            // 计算Softmax (使用 Variable 算子)
            Variable softmaxVar = sampleLogits.softMax();  // [1, num_experts]
            NdArray softmaxData = softmaxVar.getValue();
            float[] softmaxWeights = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) softmaxData).buffer;
            
            // 保存所有权重
            System.arraycopy(softmaxWeights, 0, allWeights[b], 0, numExperts);
            
            // Top-K选择 (离散操作，不参与梯度计算)
            Integer[] indices = new Integer[numExperts];
            for (int i = 0; i < numExperts; i++) {
                indices[i] = i;
            }
            
            // 按权重排序
            Arrays.sort(indices, Comparator.comparingDouble(i -> -softmaxWeights[i]));
            
            // 提取Top-K
            float topKSum = 0.0f;
            for (int k = 0; k < topK; k++) {
                topKIndices[b][k] = indices[k];
                topKWeights[b][k] = softmaxWeights[indices[k]];
                topKSum += topKWeights[b][k];
            }
            
            // 重新归一化Top-K权重
            if (topKSum > 0) {
                for (int k = 0; k < topK; k++) {
                    topKWeights[b][k] /= topKSum;
                }
            }
        }
        
        return new RouterOutput(topKIndices, topKWeights, allWeights);
    }
    
    /**
     * 添加噪声(Noisy Top-K Gating)
     * 
     * 使用标准高斯噪声,噪声幅度由 noiseFactor 控制:
     * noisy_logits = logits + StandardNormal() * noiseFactor
     * 
     * 参考: Shazeer et al. 2017 "Outrageously Large Neural Networks"
     * 
     * 修复说明: 通过 Variable 运算添加噪声,确保噪声参与梯度计算
     */
    private Variable addNoise(Variable gateLogits) {
        NdArray logitsData = gateLogits.getValue();
        int[] shape = logitsData.getShape().getShapeDims();
        int totalElements = 1;
        for (int dim : shape) {
            totalElements *= dim;
        }
        
        // 创建噪声数组
        float[] noiseBuffer = new float[totalElements];
        Random random = new Random();
        for (int i = 0; i < totalElements; i++) {
            noiseBuffer[i] = (float) random.nextGaussian() * noiseFactor;
        }
        
        // 将噪声包装为 Variable(不需要梯度)
        NdArray noiseNdArray = NdArray.of(noiseBuffer, logitsData.getShape());
        Variable noiseVar = new Variable(noiseNdArray);
        noiseVar.setRequireGrad(false);
        
        // 通过 Variable.add() 运算添加噪声,确保噪声通过计算图传播
        return gateLogits.add(noiseVar);
    }
    
    // 已删除 softmax 方法，改用 Variable.softMax() 算子
    
    /**
     * 获取输入维度
     */
    public int getInputDim() {
        return inputDim;
    }
    
    /**
     * 获取专家数量
     */
    public int getNumExperts() {
        return numExperts;
    }
    
    /**
     * 获取Top-K数量
     */
    public int getTopK() {
        return topK;
    }
    
    @Override
    public String toString() {
        return String.format("ExpertRouter(input=%d, experts=%d, topK=%d, noise=%.3f)",
            inputDim, numExperts, topK, noiseFactor);
    }
    
    /**
     * 路由输出结果
     */
    public static class RouterOutput {
        private final int[][] topKIndices;    // [batch_size, topK]
        private final float[][] topKWeights;  // [batch_size, topK]
        private final float[][] allWeights;   // [batch_size, num_experts]
        
        public RouterOutput(int[][] topKIndices, float[][] topKWeights, float[][] allWeights) {
            this.topKIndices = topKIndices;
            this.topKWeights = topKWeights;
            this.allWeights = allWeights;
        }
        
        public int[][] getTopKIndices() {
            return topKIndices;
        }
        
        public float[][] getTopKWeights() {
            return topKWeights;
        }
        
        public float[][] getAllWeights() {
            return allWeights;
        }
        
        public int getBatchSize() {
            return topKIndices.length;
        }
        
        public int getTopK() {
            return topKIndices[0].length;
        }
    }
}