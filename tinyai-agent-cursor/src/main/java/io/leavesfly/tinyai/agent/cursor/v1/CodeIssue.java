package io.leavesfly.tinyai.agent.cursor.v1;

import java.util.Objects;

/**
 * 代码问题或建议
 * 表示在代码分析过程中发现的问题
 * 
 * @author 山泽
 */
public class CodeIssue {
    
    private final String issueType;     // 问题类型
    private final String severity;      // 严重程度: "low", "medium", "high", "critical"
    private final String message;       // 问题描述
    private final int lineNumber;       // 行号
    private final String suggestion;    // 修复建议
    
    /**
     * 构造函数
     * @param issueType 问题类型
     * @param severity 严重程度
     * @param message 问题描述
     * @param lineNumber 行号
     * @param suggestion 修复建议
     */
    public CodeIssue(String issueType, String severity, String message, int lineNumber, String suggestion) {
        this.issueType = issueType;
        this.severity = severity;
        this.message = message;
        this.lineNumber = lineNumber;
        this.suggestion = suggestion;
    }
    
    // Getter 方法
    public String getIssueType() {
        return issueType;
    }
    
    public String getSeverity() {
        return severity;
    }
    
    public String getMessage() {
        return message;
    }
    
    public int getLineNumber() {
        return lineNumber;
    }
    
    public String getSuggestion() {
        return suggestion;
    }
    
    /**
     * 获取严重程度的数值
     * @return 数值越高越严重
     */
    public int getSeverityLevel() {
        switch (severity.toLowerCase()) {
            case "critical": return 4;
            case "high": return 3;
            case "medium": return 2;
            case "low": return 1;
            default: return 0;
        }
    }
    
    @Override
    public String toString() {
        return String.format("[%s] %s (行%d): %s - %s", 
                           severity.toUpperCase(), 
                           issueType, 
                           lineNumber, 
                           message, 
                           suggestion);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        CodeIssue that = (CodeIssue) obj;
        return lineNumber == that.lineNumber &&
               issueType.equals(that.issueType) &&
               severity.equals(that.severity) &&
               message.equals(that.message);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(issueType, severity, message, lineNumber);
    }
}