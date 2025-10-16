package io.leavesfly.tinyai.agent.research.v2.component.tool;

import java.util.Map;

/**
 * 工具接口
 * 所有工具必须实现此接口
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public interface Tool {
    
    /**
     * 获取工具名称
     */
    String getName();
    
    /**
     * 获取工具描述
     */
    String getDescription();
    
    /**
     * 获取工具分类
     */
    String getCategory();
    
    /**
     * 获取工具参数定义
     */
    ToolParameters getParameters();
    
    /**
     * 验证参数
     */
    boolean validate(Map<String, Object> parameters);
    
    /**
     * 执行工具
     */
    ToolResult execute(Map<String, Object> parameters);
}
