package io.leavesfly.tinyai.agent.cursor.v2.tool;

import io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;

import java.util.Map;

/**
 * 工具接口
 * 所有工具都需要实现此接口
 * 
 * 工具用于扩展AI助手的能力，如代码分析、文件读写、网络请求等
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public interface Tool {
    
    /**
     * 获取工具名称
     * 
     * @return 工具名称（唯一标识）
     */
    String getName();
    
    /**
     * 获取工具描述
     * 
     * @return 工具描述信息
     */
    String getDescription();
    
    /**
     * 获取工具定义（用于LLM理解工具功能和参数）
     * 
     * @return 工具定义对象
     */
    ToolDefinition getDefinition();
    
    /**
     * 执行工具
     * 
     * @param parameters 工具参数（JSON对象）
     * @return 执行结果
     */
    ToolResult execute(Map<String, Object> parameters);
    
    /**
     * 验证参数
     * 
     * @param parameters 工具参数
     * @return 参数是否有效
     */
    default boolean validateParameters(Map<String, Object> parameters) {
        return parameters != null;
    }
    
    /**
     * 工具是否需要在沙箱中执行
     * 
     * @return true表示需要沙箱隔离
     */
    default boolean requiresSandbox() {
        return false;
    }
    
    /**
     * 获取工具类别
     * 
     * @return 工具类别
     */
    default ToolCategory getCategory() {
        return ToolCategory.GENERAL;
    }
    
    /**
     * 工具类别枚举
     */
    enum ToolCategory {
        /** 代码分析 */
        CODE_ANALYSIS,
        /** 代码生成 */
        CODE_GENERATION,
        /** 文件操作 */
        FILE_OPERATION,
        /** 检索增强 */
        RETRIEVAL,
        /** 网络请求 */
        NETWORK,
        /** 通用工具 */
        GENERAL
    }
}
