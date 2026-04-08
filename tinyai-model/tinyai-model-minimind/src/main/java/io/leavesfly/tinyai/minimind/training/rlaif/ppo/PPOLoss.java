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
        Variable valueLossCoef = new Variable(NdArray.of(config.getValueLossCoef()));
        valueLossCoef.setRequireGrad(false);
        Variable entropyCoef = new Variable(NdArray.of(config.getEntropyCoef()));
        entropyCoef.setRequireGrad(false);
        
        Variable totalLoss = policyLoss
            .add(valueLoss.mul(valueLossCoef))
            .sub(entropyLoss.mul(entropyCoef));
        
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

        // advantage是外部信号常量,不参与反向传播
        Variable advVar = new Variable(NdArray.of(advantages));
        advVar.setRequireGrad(false);
        
        Variable surrogateObj = ratio.mul(advVar).mean(0, true);
        
        // 取负值（最大化目标 → 最小化负目标）
        return surrogateObj.neg();
    }
    
    /**
     * 计算价值损失
     * 
     * L_value = 0.5 * (V - R)^2
     * 可选clip防止价值函数更新过大
     */
    private Variable computeValueLoss(Variable values, float[] returns, 
                                     Variable oldValues) {
        // L_value = 0.5 * mean((V - R)^2), returns是外部信号不需要梯度
        Variable returnsVar = new Variable(NdArray.of(returns));
        returnsVar.setRequireGrad(false);
        
        Variable diff = values.sub(returnsVar);
        Variable squaredDiff = diff.squ();
        
        Variable half = new Variable(NdArray.of(0.5f));
        half.setRequireGrad(false);
        
        return squaredDiff.mean(0, true).mul(half);
    }
    
    /**
     * 计算熵损失(鼓励探索)
     * 
     * H = -∑ p * log(p)
     */
    public Variable computeEntropyLoss(Variable logits) {
        // 使用框架内置的logSoftmax,在vocab维度(axis=-1)上计算
        Variable logProbs = logits.logSoftmax();
        
        // softMax结果detach,避免梯度通过两条路径重复回传
        Variable probs = logits.softMax().detach();
        
        // 熵: H = -∑ p * log(p), p已detach,梯度只通过logProbs流回
        Variable entropy = probs.mul(logProbs).neg();
        return entropy.mean(-1, true);
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
    

}
