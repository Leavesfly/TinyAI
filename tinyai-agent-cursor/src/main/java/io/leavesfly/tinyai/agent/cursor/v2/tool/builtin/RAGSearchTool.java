package io.leavesfly.tinyai.agent.cursor.v2.tool.builtin;

import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.model.Context.CodeSnippet;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;
import io.leavesfly.tinyai.agent.cursor.v2.tool.Tool;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * RAG检索工具
 * 从代码库中检索相关代码片段
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class RAGSearchTool implements Tool {
    
    private static final String NAME = "rag_search";
    private static final String DESCRIPTION = "Search codebase for relevant code snippets using RAG";
    
    private final RAGEngine ragEngine;
    
    public RAGSearchTool(RAGEngine ragEngine) {
        this.ragEngine = ragEngine;
    }
    
    @Override
    public String getName() {
        return NAME;
    }
    
    @Override
    public String getDescription() {
        return DESCRIPTION;
    }
    
    @Override
    public ToolDefinition getDefinition() {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("type", "object");
        
        Map<String, Object> properties = new HashMap<>();
        
        // query参数
        Map<String, Object> queryParam = new HashMap<>();
        queryParam.put("type", "string");
        queryParam.put("description", "Search query text");
        properties.put("query", queryParam);
        
        // topK参数
        Map<String, Object> topKParam = new HashMap<>();
        topKParam.put("type", "integer");
        topKParam.put("description", "Number of results to return (default: 5)");
        properties.put("topK", topKParam);
        
        // projectId参数
        Map<String, Object> projectIdParam = new HashMap<>();
        projectIdParam.put("type", "string");
        projectIdParam.put("description", "Project ID to search within (optional)");
        properties.put("projectId", projectIdParam);
        
        parameters.put("properties", properties);
        parameters.put("required", new String[]{"query"});
        
        return ToolDefinition.create(NAME, DESCRIPTION, parameters);
    }
    
    @Override
    public ToolResult execute(Map<String, Object> parameters) {
        long startTime = System.currentTimeMillis();
        
        try {
            if (!validateParameters(parameters)) {
                ToolResult result = ToolResult.failure(NAME, "Invalid parameters: query is required");
                result.setExecutionTime(System.currentTimeMillis() - startTime);
                return result;
            }
            
            String query = (String) parameters.get("query");
            int topK = parameters.containsKey("topK") ? 
                      ((Number) parameters.get("topK")).intValue() : 5;
            String projectId = (String) parameters.get("projectId");
            
            // 执行检索
            List<CodeSnippet> snippets = ragEngine.semanticSearch(query, topK, projectId);
            
            // 格式化结果
            String resultText = formatSearchResults(snippets);
            
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.success(NAME, resultText);
            result.setExecutionTime(executionTime);
            result.putMetadata("snippetCount", snippets.size());
            
            return result;
            
        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.failure(NAME, "Search failed: " + e.getMessage());
            result.setExecutionTime(executionTime);
            return result;
        }
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null || !parameters.containsKey("query")) {
            return false;
        }
        
        Object query = parameters.get("query");
        return query instanceof String && !((String) query).isEmpty();
    }
    
    @Override
    public ToolCategory getCategory() {
        return ToolCategory.RETRIEVAL;
    }
    
    /**
     * 格式化检索结果
     */
    private String formatSearchResults(List<CodeSnippet> snippets) {
        if (snippets.isEmpty()) {
            return "No relevant code snippets found.";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("Found ").append(snippets.size()).append(" relevant code snippet(s):\n\n");
        
        for (int i = 0; i < snippets.size(); i++) {
            CodeSnippet snippet = snippets.get(i);
            sb.append("## Snippet ").append(i + 1)
              .append(" (Score: ").append(String.format("%.2f", snippet.getScore())).append(")\n");
            sb.append("File: ").append(snippet.getFilePath()).append("\n");
            sb.append("```\n").append(snippet.getContent()).append("\n```\n\n");
        }
        
        return sb.toString();
    }
}
