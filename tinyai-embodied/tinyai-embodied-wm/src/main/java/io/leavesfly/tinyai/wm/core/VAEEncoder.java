package io.leavesfly.tinyai.wm.core;

import io.leavesfly.tinyai.wm.model.LatentState;
import io.leavesfly.tinyai.wm.model.Observation;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * VAE编码器（Variational Autoencoder Encoder）
 * 将高维观察（如图像）编码为低维潜在表示
 * 
 * 架构：
 * - 编码器：Observation -> μ, log(σ²)
 * - 重参数化：z = μ + σ * ε, ε ~ N(0,1)
 * - 解码器：z -> Reconstructed Observation
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class VAEEncoder {
    
    private final int latentSize;
    private final int observationSize;
    
    // 编码器网络（观察 -> 隐藏层）
    private final Sequential encoder;
    
    // 均值和方差层
    private final Linear muLayer;
    private final Linear logVarLayer;
    
    // 解码器网络（z -> 观察）
    private final Sequential decoder;
    
    /**
     * 构造函数
     *
     * @param observationSize 观察空间维度
     * @param latentSize 潜在空间维度
     * @param hiddenSize 隐藏层维度
     */
    public VAEEncoder(int observationSize, int latentSize, int hiddenSize) {
        this.observationSize = observationSize;
        this.latentSize = latentSize;
        
        // 构建编码器：Observation -> Hidden
        this.encoder = new Sequential("encoder")
            .add(new Linear("fc1", observationSize, hiddenSize))
            .add(new ReLU())
            .add(new Linear("fc2", hiddenSize, hiddenSize))
            .add(new ReLU());
        
        // 均值和方差映射层
        this.muLayer = new Linear("mu", hiddenSize, latentSize);
        this.logVarLayer = new Linear("logvar", hiddenSize, latentSize);
        
        // 构建解码器：z -> Observation
        this.decoder = new Sequential("decoder")
            .add(new Linear("fc1", latentSize, hiddenSize))
            .add(new ReLU())
            .add(new Linear("fc2", hiddenSize, hiddenSize))
            .add(new ReLU())
            .add(new Linear("fc3", hiddenSize, observationSize));
    }
    
    /**
     * 编码观察为潜在状态
     *
     * @param observation 观察
     * @return 潜在状态（包含z, μ, log(σ²)）
     */
    public LatentState encode(Observation observation) {
        // 1. 获取观察向量
        NdArray obsVector = observation.getStateVector();
        
        // 2. 确保是二维数组 (1, features)
        if (obsVector.getShape().getDimNum() == 1) {
            int size = obsVector.getShape().size();
            obsVector = NdArray.of(obsVector.getArray(), Shape.of(1, size));
        }
        
        // 3. 通过编码器
        Variable hidden = encoder.forward(new Variable(obsVector));
        
        // 4. 计算均值和对数方差
        Variable mu = muLayer.forward(hidden);
        Variable logVar = logVarLayer.forward(hidden);
        
        // 5. 重参数化技巧：z = μ + σ * ε
        NdArray muData = mu.getValue();
        NdArray logVarData = logVar.getValue();
        
        // 如果是二维，取第一行
        if (muData.getShape().getDimNum() == 2) {
            float[] muArray = new float[latentSize];
            float[] logVarArray = new float[latentSize];
            for (int i = 0; i < latentSize; i++) {
                muArray[i] = muData.get(0, i);
                logVarArray[i] = logVarData.get(0, i);
            }
            muData = NdArray.of(muArray, Shape.of(latentSize));
            logVarData = NdArray.of(logVarArray, Shape.of(latentSize));
        }
        
        NdArray sigma = logVarData.mulNum(0.5f).exp(); // σ = exp(0.5 * log(σ²))
        NdArray epsilon = NdArray.randn(Shape.of(latentSize)); // ε ~ N(0,1)
        NdArray z = muData.add(sigma.mul(epsilon)); // z = μ + σ * ε
        
        return new LatentState(z, muData, logVarData);
    }
    
    /**
     * 解码潜在状态为观察
     *
     * @param latentState 潜在状态
     * @return 重建的观察向量
     */
    public NdArray decode(LatentState latentState) {
        NdArray z = latentState.getZ();
        
        // 确保是二维数组 (1, features)
        if (z.getShape().getDimNum() == 1) {
            int size = z.getShape().size();
            z = NdArray.of(z.getArray(), Shape.of(1, size));
        }
        
        Variable zVar = new Variable(z);
        Variable reconstructed = decoder.forward(zVar);
        
        // 返回一维数组
        NdArray result = reconstructed.getValue();
        if (result.getShape().getDimNum() == 2) {
            int size = result.getShape().getDimension(1);
            float[] data = new float[size];
            for (int i = 0; i < size; i++) {
                data[i] = result.get(0, i);
            }
            result = NdArray.of(data, Shape.of(size));
        }
        
        return result;
    }
    
    /**
     * 完整的前向传播（编码 + 解码）
     *
     * @param observation 观察
     * @return 重建的观察和潜在状态
     */
    public EncoderOutput forward(Observation observation) {
        LatentState latent = encode(observation);
        NdArray reconstructed = decode(latent);
        return new EncoderOutput(latent, reconstructed);
    }
    
    /**
     * 计算VAE损失
     * Loss = Reconstruction Loss + KL Divergence
     *
     * @param observation 原始观察
     * @param output 编码器输出
     * @return 总损失
     */
    public double calculateLoss(Observation observation, EncoderOutput output) {
        // 1. 重建损失（MSE）
        NdArray original = observation.getStateVector();
        NdArray reconstructed = output.getReconstructed();
        NdArray diff = original.sub(reconstructed);
        double reconLoss = diff.mul(diff).sum().getNumber().doubleValue() / original.getShape().size();
        
        // 2. KL散度损失
        double klLoss = output.getLatentState().calculateKLDivergence();
        
        // 3. 总损失
        return reconLoss + klLoss;
    }
    

    
    // Getters
    public int getLatentSize() {
        return latentSize;
    }
    
    public int getObservationSize() {
        return observationSize;
    }
    
    /**
     * 编码器输出包装类
     */
    public static class EncoderOutput {
        private final LatentState latentState;
        private final NdArray reconstructed;
        
        public EncoderOutput(LatentState latentState, NdArray reconstructed) {
            this.latentState = latentState;
            this.reconstructed = reconstructed;
        }
        
        public LatentState getLatentState() {
            return latentState;
        }
        
        public NdArray getReconstructed() {
            return reconstructed;
        }
    }
}
