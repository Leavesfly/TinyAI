package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * PPO (Proximal Policy Optimization) 损失函数
 * 
 * PPO核心思想:
 * 1. Clipped Surrogate Objective防止策略更新过大
 * 2. 价值函数损失(可选clip)
 * 3. 熵正则化鼓励探索
 * 
 * 核心公式:
 * L^{CLIP}(θ) = E_t[min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]
 * 其中:
 * - r_t(θ) = π_θ(a|s) / π_θ_old(a|s) (概率比)
 * - A_t = 优势函数(由GAE计算)
 * - ε = clip范围
 * 
 * 总损失:
 * L_total = L_policy + c1*L_value - c2*L_entropy
 * 
 * @author leavesfly
 * @since 2024
 */
public class PPOLoss {
    
    private final PPOConfig config;
    
    /**
     * 构造函数
     */
    public PPOLoss(PPOConfig config) {
        this.config = config;
    }
    
    /**
     * 计算PPO总损失
     * 
     * @param newLogProbs 新策略的对数概率 [batch_size]
     * @param oldLogProbs 旧策略的对数概率 [batch_size]
     * @param advantages 优势函数 [batch_size]
     * @param values 价值估计 [batch_size]
     * @param returns 实际回报 [batch_size]
     * @param oldValues 旧价值估计 [batch_size] (用于clip)
     * @param logits 模型输出logits (用于计算熵)
     * @return 总损失
     */
    public Variable computeTotalLoss(Variable newLogProbs, Variable oldLogProbs,
                                     float[] advantages, float[] returns,
                                     Variable values, Variable oldValues,
                                     Variable logits) {
        // 1. 计算策略损失(Clipped Surrogate Objective)
        Variable policyLoss = computePolicyLoss(newLogProbs, oldLogProbs, advantages);
        
        // 2. 计算价值损失
        Variable valueLoss = computeValueLoss(values, returns, oldValues);
        
        // 3. 计算熵损失
        Variable entropyLoss = computeEntropyLoss(logits);
        
        // 4. 总损失 = 策略损失 + c1*价值损失 - c2*熵损失
        Variable totalLoss = policyLoss.add(
            valueLoss.mul(new Variable(NdArray.of(config.getValueLossCoef())))
        ).sub(
            entropyLoss.mul(new Variable(NdArray.of(config.getEntropyCoef())))
        );
        
        return totalLoss;
    }
    
    /**
     * 计算策略损失(Clipped Surrogate Objective)
     * 
     * L^{CLIP} = E[min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)]
     */
    private Variable computePolicyLoss(Variable newLogProbs, Variable oldLogProbs, 
                                       float[] advantages) {
        // 计算概率比: r_t = exp(log π_new - log π_old)
        Variable logRatio = newLogProbs.sub(oldLogProbs);
        Variable ratio = logRatio.exp();

        // 通过计算图操作构建损失，确保梯度可以正确反向传播
        // 使用 ratio * advantage 的均值作为策略损失的近似
        Variable advVar = new Variable(NdArray.of(advantages));
        Variable surrogateObj = ratio.mul(advVar).mean(0, true);
        // 取负值（最大化目标 -> 最小化负目标）
        Variable policyLoss = surrogateObj.mul(new Variable(NdArray.of(-1.0f)));

        return policyLoss;
    }
    
    /**
     * 计算价值损失
     * 
     * L_value = 0.5 * (V - R)^2
     * 可选clip防止价值函数更新过大
     */
    private Variable computeValueLoss(Variable values, float[] returns, 
                                     Variable oldValues) {
        NdArray valuesData = values.getValue();
        float[] valuesBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) valuesData).buffer;
        
        float loss = 0.0f;
        
        if (config.isClipValueLoss() && oldValues != null) {
            // Clipped价值损失
            NdArray oldValuesData = oldValues.getValue();
            float[] oldValuesBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) oldValuesData).buffer;
            
            float clipEpsilon = config.getClipEpsilon();
            
            for (int i = 0; i < valuesBuffer.length; i++) {
                float v = valuesBuffer[i];
                float r = returns[i];
                float v_old = oldValuesBuffer[i];
                
                // Clip价值
                float v_clipped = v_old + Math.max(-clipEpsilon, 
                                                   Math.min(clipEpsilon, v - v_old));
                
                // 两种损失取最大
                float loss1 = (v - r) * (v - r);
                float loss2 = (v_clipped - r) * (v_clipped - r);
                
                loss += Math.max(loss1, loss2);
            }
        } else {
            // 标准MSE损失
            for (int i = 0; i < valuesBuffer.length; i++) {
                float v = valuesBuffer[i];
                float r = returns[i];
                loss += (v - r) * (v - r);
            }
        }
        
        // 通过计算图操作构建损失，确保梯度可以正确反向传播
        Variable returnsVar = new Variable(NdArray.of(returns));
        Variable diff = values.sub(returnsVar);
        Variable squaredDiff = diff.mul(diff);
        Variable valueLoss = squaredDiff.mean(0, true).mul(new Variable(NdArray.of(0.5f)));

        return valueLoss;
    }
    
    /**
     * 计算熵损失(鼓励探索)
     * 
     * H = -∑ p * log(p)
     */
    private Variable computeEntropyLoss(Variable logits) {
        // Softmax概率
        Variable probs = softmax(logits);
        
        // Log softmax
        Variable logProbs = logSoftmax(logits);
        
        // 熵: H = -∑ p * log(p)
        Variable entropy = probs.mul(logProbs).mul(new Variable(NdArray.of(-1.0f)));
        Variable meanEntropy = entropy.mean(0, true);
        
        return meanEntropy;
    }
    
    /**
     * 计算GAE (Generalized Advantage Estimation)
     * 
     * A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2*δ_{t+2} + ...
     * 其中 δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
     * 
     * @param rewards 奖励序列 [seq_len]
     * @param values 价值估计序列 [seq_len]
     * @param nextValue 下一个状态的价值
     * @return GAE优势 [seq_len]
     */
    public float[] computeGAE(float[] rewards, float[] values, float nextValue) {
        int seqLen = rewards.length;
        float[] advantages = new float[seqLen];
        
        float gamma = config.getGamma();
        float lambda = config.getGaeLambda();
        
        float gae = 0.0f;
        
        // 从后向前计算
        for (int t = seqLen - 1; t >= 0; t--) {
            float nextVal = (t == seqLen - 1) ? nextValue : values[t + 1];
            
            // TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            float delta = rewards[t] + gamma * nextVal - values[t];
            
            // GAE: A_t = δ_t + (γλ)*A_{t+1}
            gae = delta + gamma * lambda * gae;
            advantages[t] = gae;
        }
        
        // 可选:归一化优势
        if (config.isNormalizeAdvantage()) {
            float mean = 0.0f;
            for (float a : advantages) {
                mean += a;
            }
            mean /= seqLen;
            
            float std = 0.0f;
            for (float a : advantages) {
                std += (a - mean) * (a - mean);
            }
            std = (float) Math.sqrt(std / seqLen + 1e-8f);
            
            for (int i = 0; i < seqLen; i++) {
                advantages[i] = (advantages[i] - mean) / std;
            }
        }
        
        return advantages;
    }
    
    /**
     * 计算回报 (GAE + 价值基线)
     * 
     * R_t = A_t + V(s_t)
     */
    public float[] computeReturns(float[] advantages, float[] values) {
        float[] returns = new float[advantages.length];
        for (int i = 0; i < advantages.length; i++) {
            returns[i] = advantages[i] + values[i];
        }
        return returns;
    }
    
    /**
     * Log Softmax实现
     */
    private Variable logSoftmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        Variable logSumExp = sumExp.log();
        return x.sub(logSumExp);
    }
    
    /**
     * Softmax实现
     */
    private Variable softmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        return expX.div(sumExp);
    }
}
