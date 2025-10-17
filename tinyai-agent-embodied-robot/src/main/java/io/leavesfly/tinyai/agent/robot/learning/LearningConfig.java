package io.leavesfly.tinyai.agent.robot.learning;

import io.leavesfly.tinyai.agent.robot.model.LearningStrategy;

/**
 * 学习配置类
 * 
 * <p>定义学习算法的配置参数。</p>
 * 
 * @author TinyAI Team
 */
public class LearningConfig {
    private LearningStrategy strategy;
    private double learningRate;
    private double gamma;
    private double epsilon;
    private double epsilonDecay;
    private double epsilonMin;
    private int batchSize;
    private int bufferSize;
    private int updateFreq;
    
    public LearningConfig() {
        this.strategy = LearningStrategy.DQN;
        this.learningRate = 0.001;
        this.gamma = 0.99;
        this.epsilon = 1.0;
        this.epsilonDecay = 0.995;
        this.epsilonMin = 0.1;
        this.batchSize = 32;
        this.bufferSize = 10000;
        this.updateFreq = 100;
    }
    
    // Getters and Setters
    public LearningStrategy getStrategy() { return strategy; }
    public void setStrategy(LearningStrategy strategy) { this.strategy = strategy; }
    
    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }
    
    public double getGamma() { return gamma; }
    public void setGamma(double gamma) { this.gamma = gamma; }
    
    public double getEpsilon() { return epsilon; }
    public void setEpsilon(double epsilon) { this.epsilon = epsilon; }
    
    public double getEpsilonDecay() { return epsilonDecay; }
    public void setEpsilonDecay(double epsilonDecay) { this.epsilonDecay = epsilonDecay; }
    
    public double getEpsilonMin() { return epsilonMin; }
    public void setEpsilonMin(double epsilonMin) { this.epsilonMin = epsilonMin; }
    
    public int getBatchSize() { return batchSize; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
    
    public int getBufferSize() { return bufferSize; }
    public void setBufferSize(int bufferSize) { this.bufferSize = bufferSize; }
    
    public int getUpdateFreq() { return updateFreq; }
    public void setUpdateFreq(int updateFreq) { this.updateFreq = updateFreq; }
}
