package io.leavesfly.tinyai.agent.vla.decoder;

import io.leavesfly.tinyai.agent.vla.model.ActionType;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 语言反馈生成器
 * 根据动作和状态生成自然语言反馈
 * 
 * @author TinyAI
 */
public class LanguageFeedbackGenerator {
    
    /**
     * 生成动作执行反馈
     * 
     * @param actionType 动作类型
     * @param confidence 置信度
     * @return 自然语言反馈
     */
    public String generateFeedback(ActionType actionType, double confidence) {
        String template = getFeedbackTemplate(actionType);
        String confidenceLevel = getConfidenceLevel(confidence);
        
        return String.format(template, confidenceLevel);
    }
    
    /**
     * 生成详细反馈
     */
    public String generateDetailedFeedback(ActionType actionType, NdArray continuousAction, 
                                          double confidence, boolean success) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("Action: ").append(actionType.getDescription()).append("\n");
        sb.append("Confidence: ").append(String.format("%.2f", confidence)).append("\n");
        sb.append("Status: ").append(success ? "Success" : "In Progress").append("\n");
        
        if (continuousAction != null) {
            sb.append("Parameters: ");
            float[] params = continuousAction.getArray();
            for (int i = 0; i < Math.min(3, params.length); i++) {
                sb.append(String.format("%.3f", params[i])).append(" ");
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * 获取反馈模板
     */
    private String getFeedbackTemplate(ActionType actionType) {
        switch (actionType) {
            case MOVE_END_EFFECTOR:
                return "Moving end effector with %s confidence";
            case GRASP_OBJECT:
                return "Attempting to grasp object with %s confidence";
            case RELEASE_OBJECT:
                return "Releasing object with %s confidence";
            case NAVIGATE_TO_TARGET:
                return "Navigating to target with %s confidence";
            case ROTATE_JOINTS:
                return "Rotating joints with %s confidence";
            case WAIT:
                return "Waiting with %s confidence";
            case SPEAK:
                return "Speaking with %s confidence";
            default:
                return "Executing action with %s confidence";
        }
    }
    
    /**
     * 获取置信度级别描述
     */
    private String getConfidenceLevel(double confidence) {
        if (confidence > 0.9) return "high";
        if (confidence > 0.7) return "medium";
        if (confidence > 0.5) return "low";
        return "very low";
    }
}
