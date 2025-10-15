package io.leavesfly.tinyai.agent.cursor.v2.service;

import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.analyzer.CodeAnalyzerV2;
import io.leavesfly.tinyai.agent.cursor.v2.component.debug.DebugAgentV2;
import io.leavesfly.tinyai.agent.cursor.v2.component.generator.CodeGeneratorV2;
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.refactor.RefactorAgentV2;
import io.leavesfly.tinyai.agent.cursor.v2.tool.ToolOrchestrator;

/**
 * 代码智能服务统一层
 * 整合所有智能服务组件，提供统一的对外接口
 * 
 * 核心职责：
 * 1. 组件初始化和依赖注入
 * 2. 服务编排和协调
 * 3. 统一的错误处理
 * 4. 性能监控和日志
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class CodeIntelligenceService {
    
    // 核心组件
    private final LLMGateway llmGateway;
    private final MemoryManager memoryManager;
    private final RAGEngine ragEngine;
    private final ContextEngine contextEngine;
    private final ToolOrchestrator toolOrchestrator;
    
    // 智能服务
    private final CodeAnalyzerV2 codeAnalyzer;
    private final CodeGeneratorV2 codeGenerator;
    private final RefactorAgentV2 refactorAgent;
    private final DebugAgentV2 debugAgent;
    
    /**
     * 构造函数（完整依赖注入）
     */
    public CodeIntelligenceService(
            LLMGateway llmGateway,
            MemoryManager memoryManager,
            RAGEngine ragEngine,
            ContextEngine contextEngine,
            ToolOrchestrator toolOrchestrator) {
        
        this.llmGateway = llmGateway;
        this.memoryManager = memoryManager;
        this.ragEngine = ragEngine;
        this.contextEngine = contextEngine;
        this.toolOrchestrator = toolOrchestrator;
        
        // 初始化智能服务
        this.codeAnalyzer = new CodeAnalyzerV2(llmGateway, contextEngine, toolOrchestrator);
        this.codeGenerator = new CodeGeneratorV2(llmGateway, contextEngine);
        this.refactorAgent = new RefactorAgentV2(llmGateway, contextEngine, codeAnalyzer);
        this.debugAgent = new DebugAgentV2(llmGateway, contextEngine);
    }
    
    /**
     * 简化构造函数（自动创建组件）
     */
    public static CodeIntelligenceService create(LLMGateway llmGateway, 
                                                 ToolOrchestrator toolOrchestrator) {
        MemoryManager memoryManager = new MemoryManager();
        RAGEngine ragEngine = new RAGEngine();
        ContextEngine contextEngine = new ContextEngine(memoryManager, ragEngine);
        
        return new CodeIntelligenceService(
            llmGateway,
            memoryManager,
            ragEngine,
            contextEngine,
            toolOrchestrator
        );
    }
    
    // ========== 对外服务接口 ==========
    
    /**
     * 获取代码分析器
     */
    public CodeAnalyzerV2 getCodeAnalyzer() {
        return codeAnalyzer;
    }
    
    /**
     * 获取代码生成器
     */
    public CodeGeneratorV2 getCodeGenerator() {
        return codeGenerator;
    }
    
    /**
     * 获取重构助手
     */
    public RefactorAgentV2 getRefactorAgent() {
        return refactorAgent;
    }
    
    /**
     * 获取调试助手
     */
    public DebugAgentV2 getDebugAgent() {
        return debugAgent;
    }
    
    /**
     * 获取上下文引擎
     */
    public ContextEngine getContextEngine() {
        return contextEngine;
    }
    
    /**
     * 获取记忆管理器
     */
    public MemoryManager getMemoryManager() {
        return memoryManager;
    }
    
    /**
     * 获取RAG引擎
     */
    public RAGEngine getRAGEngine() {
        return ragEngine;
    }
    
    /**
     * 获取工具编排器
     */
    public ToolOrchestrator getToolOrchestrator() {
        return toolOrchestrator;
    }
    
    /**
     * 获取LLM网关
     */
    public LLMGateway getLLMGateway() {
        return llmGateway;
    }
    
    /**
     * 获取服务统计信息
     */
    public ServiceStats getStats() {
        ServiceStats stats = new ServiceStats();
        stats.memoryStats = memoryManager.getStats();
        stats.ragStats = ragEngine.getStats();
        stats.contextStats = contextEngine.getStats();
        stats.toolStats = toolOrchestrator.getStats();
        return stats;
    }
    
    /**
     * 清理资源
     */
    public void cleanup() {
        // 清理过期记忆
        memoryManager.cleanupExpiredMemories();
        
        // 清理工具编排器
        toolOrchestrator.shutdown();
    }
    
    /**
     * 服务统计信息
     */
    public static class ServiceStats {
        public MemoryManager.MemoryStats memoryStats;
        public RAGEngine.RAGStats ragStats;
        public ContextEngine.ContextStats contextStats;
        public ToolOrchestrator.OrchestratorStats toolStats;
        
        @Override
        public String toString() {
            return "ServiceStats{" +
                    "memory=" + memoryStats +
                    ", rag=" + ragStats +
                    ", context=" + contextStats +
                    ", tools=" + toolStats +
                    '}';
        }
    }
}
