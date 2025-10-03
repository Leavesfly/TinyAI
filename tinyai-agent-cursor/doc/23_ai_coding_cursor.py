#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Coding Agent Cursor - 智能编程助手系统

模拟实现类似 AI Coding Cursor 的智能编程助手系统。
专门针对编程任务进行优化，具备代码理解、生成、分析、重构和调试等核心能力。

作者: 山泽
日期: 2025-10-03
"""

import ast
import re
import json
import time
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class CodeLanguage(Enum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    TYPESCRIPT = "typescript"


class TaskType(Enum):
    """编程任务类型"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_REFACTOR = "code_refactor"
    BUG_FIX = "bug_fix"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"


@dataclass
class CodeContext:
    """代码上下文信息"""
    language: CodeLanguage
    file_path: str = ""
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    line_count: int = 0
    complexity_score: float = 0.0


@dataclass
class CodeIssue:
    """代码问题或建议"""
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    line_number: int = 0
    suggestion: str = ""


@dataclass
class RefactorSuggestion:
    """重构建议"""
    suggestion_type: str
    description: str
    original_code: str
    refactored_code: str
    benefits: List[str]
    estimated_impact: str


class CodeAnalyzer:
    """代码分析器 - 分析代码结构、质量和潜在问题"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_python_code(self, code: str) -> Dict[str, Any]:
        """分析Python代码"""
        try:
            tree = ast.parse(code)
            analysis = {
                "syntax_valid": True,
                "imports": self._extract_imports(tree),
                "classes": self._extract_classes(tree),
                "functions": self._extract_functions(tree),
                "variables": self._extract_variables(tree),
                "complexity": self._calculate_complexity(tree),
                "issues": self._find_code_issues(tree, code),
                "metrics": self._calculate_metrics(code, tree)
            }
            return analysis
        except SyntaxError as e:
            return {
                "syntax_valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset
            }
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """提取导入语句"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """提取类定义"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    "docstring": ast.get_docstring(node)
                }
                classes.append(class_info)
        return classes
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """提取函数定义"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "returns": bool(node.returns),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "docstring": ast.get_docstring(node),
                    "decorators": [decorator.id if isinstance(decorator, ast.Name) 
                                 else str(decorator) for decorator in node.decorator_list]
                }
                functions.append(func_info)
        return functions
    
    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """提取变量定义"""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            "name": target.id,
                            "line": node.lineno,
                            "type": "assignment"
                        })
        return variables
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """计算圈复杂度"""
        complexity = 1  # 基础复杂度
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                               ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _find_code_issues(self, tree: ast.AST, code: str) -> List[CodeIssue]:
        """查找代码问题"""
        issues = []
        lines = code.split('\n')
        
        # 检查长函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = len([line for line in lines[node.lineno-1:] 
                                if line.strip() and not line.strip().startswith('#')])
                if func_lines > 50:
                    issues.append(CodeIssue(
                        issue_type="long_function",
                        severity="medium",
                        message=f"函数 '{node.name}' 过长 ({func_lines} 行)，建议分解",
                        line_number=node.lineno,
                        suggestion="考虑将大函数分解为多个小函数"
                    ))
        
        # 检查缺少文档字符串
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(CodeIssue(
                        issue_type="missing_docstring",
                        severity="low",
                        message=f"{node.__class__.__name__.lower()} '{node.name}' 缺少文档字符串",
                        line_number=node.lineno,
                        suggestion="添加描述性的文档字符串"
                    ))
        
        return issues
    
    def _calculate_metrics(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """计算代码度量"""
        lines = code.split('\n')
        return {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "function_count": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
            "class_count": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        }


class CodeGenerator:
    """代码生成器 - 基于上下文和需求生成代码"""
    
    def __init__(self):
        self.templates = {
            "python_function": '''def {name}({params}):
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    """
    {body}
    return {return_value}''',
            
            "python_class": '''class {name}({inheritance}):
    """
    {description}
    """
    
    def __init__(self{init_params}):
        """初始化{name}实例"""
        {init_body}
    
    {methods}''',
            
            "python_test": '''def test_{name}():
    """测试{name}函数"""
    # 准备测试数据
    {test_setup}
    
    # 执行测试
    result = {name}({test_params})
    
    # 断言结果
    assert {assertion}, "{assertion_message}"'''
        }
    
    def generate_function(self, name: str, description: str, params: List[str], 
                         return_type: str = "Any", body_hint: str = "") -> str:
        """生成函数代码"""
        # 构建参数文档
        args_doc = "\n        ".join([f"{param}: 参数描述" for param in params])
        
        # 生成函数体
        if body_hint:
            body = f"    # {body_hint}\n    pass"
        else:
            body = "    pass"
        
        return self.templates["python_function"].format(
            name=name,
            params=", ".join(params),
            description=description,
            args_doc=args_doc,
            return_doc=return_type,
            body=body,
            return_value="None"
        )
    
    def generate_class(self, name: str, description: str, 
                      inheritance: str = "", methods: Optional[List[str]] = None) -> str:
        """生成类代码"""
        methods = methods or []
        inheritance = inheritance or "object"
        
        # 生成方法代码
        methods_code = "\n    ".join([
            f"def {method}(self):\n        \"\"\"实现{method}方法\"\"\"\n        pass"
            for method in methods
        ])
        
        return self.templates["python_class"].format(
            name=name,
            inheritance=inheritance,
            description=description,
            init_params="",
            init_body="pass",
            methods=methods_code
        )
    
    def generate_test(self, function_name: str, test_params: str = "") -> str:
        """生成测试代码"""
        return self.templates["python_test"].format(
            name=function_name,
            test_setup="# 设置测试数据",
            test_params=test_params or "test_data",
            assertion="result is not None",
            assertion_message="函数应该返回有效结果"
        )


class RefactorAgent:
    """重构代理 - 智能分析和建议代码重构"""
    
    def __init__(self, analyzer: CodeAnalyzer):
        self.analyzer = analyzer
        self.refactor_patterns = self._load_refactor_patterns()
    
    def _load_refactor_patterns(self) -> Dict[str, Any]:
        """加载重构模式"""
        return {
            "long_function": {
                "description": "分解长函数",
                "threshold": 30,
                "strategy": "extract_method"
            },
            "duplicate_code": {
                "description": "消除重复代码",
                "threshold": 3,
                "strategy": "extract_common"
            },
            "large_class": {
                "description": "分解大类",
                "threshold": 500,
                "strategy": "split_class"
            }
        }
    
    def analyze_refactor_opportunities(self, code: str) -> List[RefactorSuggestion]:
        """分析重构机会"""
        analysis = self.analyzer.analyze_python_code(code)
        suggestions = []
        
        if analysis.get("syntax_valid", False):
            # 检查长函数
            for func in analysis.get("functions", []):
                if self._is_long_function(code, func):
                    suggestions.append(self._suggest_function_extraction(func))
            
            # 检查复杂度
            if analysis.get("complexity", 0) > 10:
                suggestions.append(self._suggest_complexity_reduction(analysis))
        
        return suggestions
    
    def _is_long_function(self, code: str, func_info: Dict[str, Any]) -> bool:
        """检查是否为长函数"""
        lines = code.split('\n')
        func_start = func_info["line"] - 1
        
        # 简单估算函数长度
        func_lines = 0
        indent_level = None
        
        for i in range(func_start, len(lines)):
            line = lines[i]
            if line.strip():
                if indent_level is None:
                    indent_level = len(line) - len(line.lstrip())
                elif len(line) - len(line.lstrip()) <= indent_level and line.strip() != "":
                    break
                func_lines += 1
        
        return func_lines > self.refactor_patterns["long_function"]["threshold"]
    
    def _suggest_function_extraction(self, func_info: Dict[str, Any]) -> RefactorSuggestion:
        """建议函数提取"""
        return RefactorSuggestion(
            suggestion_type="extract_method",
            description=f"函数 '{func_info['name']}' 过长，建议分解为多个小函数",
            original_code=f"def {func_info['name']}(...):",
            refactored_code="# 分解为多个小函数\ndef helper_function_1():\n    pass\n\ndef helper_function_2():\n    pass",
            benefits=["提高代码可读性", "便于单元测试", "降低维护成本"],
            estimated_impact="中等"
        )
    
    def _suggest_complexity_reduction(self, analysis: Dict[str, Any]) -> RefactorSuggestion:
        """建议复杂度降低"""
        return RefactorSuggestion(
            suggestion_type="reduce_complexity",
            description=f"代码复杂度过高 ({analysis['complexity']})，建议简化逻辑",
            original_code="# 复杂的条件判断",
            refactored_code="# 使用策略模式或提取方法简化逻辑",
            benefits=["降低圈复杂度", "提高代码可理解性", "减少bug风险"],
            estimated_impact="高"
        )


class DebugAgent:
    """调试代理 - 智能错误诊断和修复建议"""
    
    def __init__(self, analyzer: CodeAnalyzer):
        self.analyzer = analyzer
        self.error_patterns = self._load_error_patterns()
    
    def _load_error_patterns(self) -> Dict[str, Dict[str, str]]:
        """加载错误模式"""
        return {
            "SyntaxError": {
                "description": "语法错误",
                "common_causes": "缺少冒号、括号不匹配、缩进错误",
                "fix_strategy": "检查语法结构和缩进"
            },
            "NameError": {
                "description": "名称错误",
                "common_causes": "变量未定义、函数名拼写错误",
                "fix_strategy": "检查变量定义和导入语句"
            },
            "TypeError": {
                "description": "类型错误",
                "common_causes": "类型不匹配、方法调用错误",
                "fix_strategy": "检查数据类型和方法调用"
            }
        }
    
    def diagnose_error(self, code: str, error_message: str = "") -> Dict[str, Any]:
        """诊断代码错误"""
        diagnosis = {
            "error_found": False,
            "error_type": "",
            "error_line": 0,
            "diagnosis": "",
            "suggestions": [],
            "fixed_code": ""
        }
        
        # 分析代码语法
        analysis = self.analyzer.analyze_python_code(code)
        
        if not analysis.get("syntax_valid", True):
            diagnosis["error_found"] = True
            diagnosis["error_type"] = "SyntaxError"
            diagnosis["error_line"] = analysis.get("line", 0)
            diagnosis["diagnosis"] = analysis.get("error", "")
            diagnosis["suggestions"] = self._get_syntax_fix_suggestions(analysis)
            diagnosis["fixed_code"] = self._attempt_syntax_fix(code, analysis)
        
        return diagnosis
    
    def _get_syntax_fix_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """获取语法修复建议"""
        suggestions = []
        error_msg = analysis.get("error", "").lower()
        
        if "invalid syntax" in error_msg:
            suggestions.extend([
                "检查是否缺少冒号 (:)",
                "检查括号是否匹配",
                "检查字符串引号是否闭合"
            ])
        elif "indentation" in error_msg:
            suggestions.extend([
                "检查缩进是否一致",
                "避免混用空格和Tab",
                "使用4个空格作为标准缩进"
            ])
        
        return suggestions
    
    def _attempt_syntax_fix(self, code: str, analysis: Dict[str, Any]) -> str:
        """尝试自动修复语法错误"""
        lines = code.split('\n')
        error_line = analysis.get("line", 1) - 1
        
        if error_line < len(lines):
            line = lines[error_line]
            
            # 尝试修复常见语法错误
            if line.strip().endswith(("if", "else", "elif", "for", "while", "def", "class")):
                lines[error_line] = line + ":"
            elif "print " in line and not line.strip().startswith("#"):
                # Python 2 -> Python 3
                fixed_line = re.sub(r'print\s+([^(].*)', r'print(\1)', line)
                lines[error_line] = fixed_line
        
        return '\n'.join(lines)


class AICodingCursor:
    """AI Coding Cursor 主系统 - 智能编程助手"""
    
    def __init__(self, name: str = "AI Coding Cursor"):
        self.name = name
        self.analyzer = CodeAnalyzer()
        self.generator = CodeGenerator()
        self.refactor_agent = RefactorAgent(self.analyzer)
        self.debug_agent = DebugAgent(self.analyzer)
        
        # 系统状态
        self.current_context = None
        self.session_history = deque(maxlen=100)
        self.preferences = {
            "language": CodeLanguage.PYTHON,
            "style": "pep8",
            "auto_refactor": True,
            "debug_level": "detailed"
        }
        
        print(f"🚀 {self.name} 初始化完成!")
        print("💡 支持的功能：代码分析、生成、重构、调试、优化建议")
    
    def analyze_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> Dict[str, Any]:
        """分析代码"""
        print(f"🔍 正在分析代码...")
        
        if language == CodeLanguage.PYTHON:
            analysis = self.analyzer.analyze_python_code(code)
        else:
            analysis = {"error": f"暂不支持 {language.value} 语言分析"}
        
        # 记录到会话历史
        self.session_history.append({
            "action": "analyze",
            "timestamp": time.time(),
            "language": language.value,
            "result": analysis
        })
        
        return analysis
    
    def generate_code(self, request: str, context: Optional[CodeContext] = None) -> str:
        """根据需求生成代码"""
        print(f"🤖 正在生成代码: {request}")
        
        # 解析生成请求
        if "function" in request.lower():
            # 提取函数信息
            name_match = re.search(r'function\s+(\w+)', request)
            name = name_match.group(1) if name_match else "new_function"
            
            return self.generator.generate_function(
                name=name,
                description=f"根据请求生成的函数: {request}",
                params=["param1", "param2"],
                body_hint="实现具体逻辑"
            )
        
        elif "class" in request.lower():
            # 提取类信息
            name_match = re.search(r'class\s+(\w+)', request)
            name = name_match.group(1) if name_match else "NewClass"
            
            return self.generator.generate_class(
                name=name,
                description=f"根据请求生成的类: {request}",
                methods=["method1", "method2"]
            )
        
        else:
            return f"# 根据请求生成的代码: {request}\n# TODO: 实现具体功能\npass"
    
    def suggest_refactor(self, code: str) -> List[RefactorSuggestion]:
        """建议代码重构"""
        print("🔧 正在分析重构机会...")
        return self.refactor_agent.analyze_refactor_opportunities(code)
    
    def debug_code(self, code: str, error_message: str = "") -> Dict[str, Any]:
        """调试代码"""
        print("🐛 正在进行智能调试...")
        return self.debug_agent.diagnose_error(code, error_message)
    
    def code_review(self, code: str) -> Dict[str, Any]:
        """代码审查"""
        print("📋 正在进行代码审查...")
        
        # 综合分析
        analysis = self.analyze_code(code)
        refactor_suggestions = self.suggest_refactor(code)
        debug_info = self.debug_code(code)
        
        # 生成审查报告
        review = {
            "overall_score": self._calculate_code_score(analysis),
            "analysis": analysis,
            "refactor_suggestions": refactor_suggestions,
            "potential_issues": debug_info,
            "recommendations": self._generate_recommendations(analysis, refactor_suggestions)
        }
        
        return review
    
    def _calculate_code_score(self, analysis: Dict[str, Any]) -> float:
        """计算代码质量评分"""
        if not analysis.get("syntax_valid", False):
            return 0.0
        
        score = 100.0
        
        # 根据问题扣分
        issues = analysis.get("issues", [])
        for issue in issues:
            if issue.severity == "critical":
                score -= 20
            elif issue.severity == "high":
                score -= 10
            elif issue.severity == "medium":
                score -= 5
            elif issue.severity == "low":
                score -= 2
        
        # 根据复杂度扣分
        complexity = analysis.get("complexity", 0)
        if complexity > 15:
            score -= 15
        elif complexity > 10:
            score -= 10
        elif complexity > 5:
            score -= 5
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, analysis: Dict[str, Any], 
                                refactor_suggestions: List[RefactorSuggestion]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if refactor_suggestions:
            recommendations.append("考虑进行代码重构以提高可维护性")
        
        issues = analysis.get("issues", [])
        if issues:
            recommendations.append("修复已识别的代码问题")
        
        metrics = analysis.get("metrics", {})
        if metrics.get("comment_lines", 0) == 0:
            recommendations.append("增加代码注释以提高可读性")
        
        complexity = analysis.get("complexity", 0)
        if complexity > 10:
            recommendations.append("简化复杂的逻辑结构")
        
        return recommendations
    
    def chat_mode(self):
        """进入交互式聊天模式"""
        print(f"\n🤖 {self.name} 交互式模式")
        print("输入 'help' 查看可用命令，输入 'quit' 退出")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！感谢使用 AI Coding Cursor!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                
                elif user_input.startswith('analyze:'):
                    code = user_input[8:].strip()
                    result = self.analyze_code(code)
                    self._print_analysis_result(result)
                
                elif user_input.startswith('generate:'):
                    request = user_input[9:].strip()
                    result = self.generate_code(request)
                    print(f"\n🤖 生成的代码:\n{result}")
                
                elif user_input.startswith('refactor:'):
                    code = user_input[9:].strip()
                    suggestions = self.suggest_refactor(code)
                    self._print_refactor_suggestions(suggestions)
                
                elif user_input.startswith('debug:'):
                    code = user_input[6:].strip()
                    result = self.debug_code(code)
                    self._print_debug_result(result)
                
                elif user_input.startswith('review:'):
                    code = user_input[7:].strip()
                    result = self.code_review(code)
                    self._print_review_result(result)
                
                else:
                    print("🤖 请使用指定格式输入命令，输入 'help' 查看帮助")
            
            except KeyboardInterrupt:
                print("\n👋 再见！感谢使用 AI Coding Cursor!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n📚 可用命令:")
        print("analyze:<code>    - 分析代码")
        print("generate:<request> - 生成代码")
        print("refactor:<code>   - 重构建议")
        print("debug:<code>     - 调试代码")
        print("review:<code>    - 代码审查")
        print("help             - 显示帮助")
        print("quit/exit        - 退出程序")
    
    def _print_analysis_result(self, result: Dict[str, Any]):
        """打印分析结果"""
        print(f"\n🔍 分析结果:")
        if result.get("syntax_valid", False):
            print(f"✅ 语法有效")
            print(f"📊 函数数量: {len(result.get('functions', []))}")
            print(f"📊 类数量: {len(result.get('classes', []))}")
            print(f"📊 复杂度: {result.get('complexity', 0)}")
            
            issues = result.get("issues", [])
            if issues:
                print(f"⚠️  发现 {len(issues)} 个问题:")
                for i, issue in enumerate(issues[:3], 1):
                    print(f"  {i}. {issue.message}")
        else:
            print(f"❌ 语法错误: {result.get('error', '未知错误')}")
    
    def _print_refactor_suggestions(self, suggestions: List[RefactorSuggestion]):
        """打印重构建议"""
        print(f"\n🔧 重构建议:")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion.description}")
                print(f"     影响: {suggestion.estimated_impact}")
        else:
            print("✅ 代码结构良好，无需重构")
    
    def _print_debug_result(self, result: Dict[str, Any]):
        """打印调试结果"""
        print(f"\n🐛 调试结果:")
        if result.get("error_found", False):
            print(f"❌ 发现错误: {result.get('error_type', '未知错误')}")
            print(f"📍 行号: {result.get('error_line', 0)}")
            print(f"💡 诊断: {result.get('diagnosis', '')}")
            
            suggestions = result.get("suggestions", [])
            if suggestions:
                print("🔧 修复建议:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")
        else:
            print("✅ 未发现明显错误")
    
    def _print_review_result(self, result: Dict[str, Any]):
        """打印审查结果"""
        print(f"\n📋 代码审查报告:")
        print(f"🏆 总体评分: {result.get('overall_score', 0):.1f}/100")
        
        recommendations = result.get("recommendations", [])
        if recommendations:
            print("📝 改进建议:")
            for rec in recommendations:
                print(f"  • {rec}")


def demo_basic_features():
    """基础功能演示"""
    print("🚀 AI Coding Cursor 基础功能演示")
    print("=" * 50)
    
    # 创建 AI Coding Cursor 实例
    cursor = AICodingCursor()
    
    # 测试代码
    test_code = '''
def calculate_statistics(data):
    if not data:
        return None
    
    total = 0
    count = 0
    
    for item in data:
        if isinstance(item, (int, float)):
            total += item
            count += 1
        else:
            print(f"Skipping invalid item: {item}")
    
    if count == 0:
        return None
        
    mean = total / count
    
    # Calculate variance
    variance = 0
    for item in data:
        if isinstance(item, (int, float)):
            variance += (item - mean) ** 2
    
    variance = variance / count
    std_dev = variance ** 0.5
    
    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "count": count,
        "total": total
    }

class DataAnalyzer:
    def __init__(self):
        self.results = []
    
    def analyze(self, dataset):
        result = calculate_statistics(dataset)
        if result:
            self.results.append(result)
        return result
'''
    
    print("\n1. 🔍 代码分析演示")
    analysis = cursor.analyze_code(test_code)
    if analysis.get("syntax_valid", False):
        print(f"✅ 语法正确")
        print(f"📊 函数数量: {len(analysis['functions'])}")
        print(f"📊 类数量: {len(analysis['classes'])}")
        print(f"📊 代码行数: {analysis['metrics']['total_lines']}")
        print(f"📊 复杂度: {analysis['complexity']}")
        
        if analysis['issues']:
            print(f"⚠️  发现 {len(analysis['issues'])} 个问题:")
            for issue in analysis['issues'][:3]:
                print(f"  • {issue.message}")
    
    print("\n2. 🤖 代码生成演示")
    
    # 生成函数
    print("\n生成函数:")
    function_code = cursor.generate_code("function validate_data")
    print(function_code[:200] + "..." if len(function_code) > 200 else function_code)
    
    # 生成类
    print("\n生成类:")
    class_code = cursor.generate_code("class EmailValidator")
    print(class_code[:200] + "..." if len(class_code) > 200 else class_code)
    
    print("\n3. 🔧 重构建议演示")
    refactor_suggestions = cursor.suggest_refactor(test_code)
    if refactor_suggestions:
        for i, suggestion in enumerate(refactor_suggestions, 1):
            print(f"{i}. {suggestion.description}")
            print(f"   影响级别: {suggestion.estimated_impact}")
            print(f"   优势: {', '.join(suggestion.benefits)}")
    else:
        print("✅ 代码结构良好，无需重构")
    
    print("\n4. 🐛 调试功能演示")
    
    # 测试有语法错误的代码
    buggy_code = '''
def broken_function(x, y)
    if x > y
        return x
    else
        return y
'''
    
    debug_result = cursor.debug_code(buggy_code)
    if debug_result.get("error_found", False):
        print(f"❌ 发现错误: {debug_result['error_type']}")
        print(f"📍 行号: {debug_result['error_line']}")
        print(f"💡 建议:")
        for suggestion in debug_result.get("suggestions", []):
            print(f"  • {suggestion}")
        
        # 显示修复后的代码
        fixed_code = debug_result.get("fixed_code", "")
        if fixed_code and fixed_code != buggy_code:
            print(f"\n🔧 尝试自动修复:")
            print(fixed_code)
    
    print("\n5. 📋 代码审查演示")
    review_result = cursor.code_review(test_code)
    print(f"🏆 总体评分: {review_result['overall_score']:.1f}/100")
    
    recommendations = review_result.get("recommendations", [])
    if recommendations:
        print("📝 改进建议:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print("\n🎉 基础功能演示完成!")


def demo_advanced_features():
    """高级功能演示"""
    print("\n🚀 AI Coding Cursor 高级功能演示")
    print("=" * 50)
    
    cursor = AICodingCursor()
    
    # 复杂代码示例
    complex_code = '''
import json
import sqlite3
from typing import List, Dict, Optional

class UserManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self._create_tables()
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise
    
    def _create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                bio TEXT,
                avatar_url TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        self.connection.commit()
    
    def create_user(self, username: str, email: str, password_hash: str) -> Optional[int]:
        if not username or not email or not password_hash:
            raise ValueError("All fields are required")
        
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        
        if "@" not in email:
            raise ValueError("Invalid email format")
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            self.connection.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise ValueError("Username already exists")
            elif "email" in str(e):
                raise ValueError("Email already exists")
            else:
                raise ValueError("User creation failed")
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT u.*, p.first_name, p.last_name, p.bio, p.avatar_url
            FROM users u
            LEFT JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = ? AND u.is_active = 1
        """, (user_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def update_user_profile(self, user_id: int, profile_data: Dict) -> bool:
        allowed_fields = ['first_name', 'last_name', 'bio', 'avatar_url']
        filtered_data = {k: v for k, v in profile_data.items() if k in allowed_fields}
        
        if not filtered_data:
            return False
        
        cursor = self.connection.cursor()
        
        # Check if profile exists
        cursor.execute("SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,))
        exists = cursor.fetchone()
        
        if exists:
            # Update existing profile
            set_clause = ", ".join([f"{field} = ?" for field in filtered_data.keys()])
            values = list(filtered_data.values()) + [user_id]
            cursor.execute(f"UPDATE user_profiles SET {set_clause} WHERE user_id = ?", values)
        else:
            # Create new profile
            fields = "user_id, " + ", ".join(filtered_data.keys())
            placeholders = "?, " + ", ".join(["?" for _ in filtered_data])
            values = [user_id] + list(filtered_data.values())
            cursor.execute(f"INSERT INTO user_profiles ({fields}) VALUES ({placeholders})", values)
        
        self.connection.commit()
        return True
    
    def delete_user(self, user_id: int) -> bool:
        cursor = self.connection.cursor()
        cursor.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
        self.connection.commit()
        return cursor.rowcount > 0
    
    def search_users(self, query: str, limit: int = 10) -> List[Dict]:
        cursor = self.connection.cursor()
        search_pattern = f"%{query}%"
        cursor.execute("""
            SELECT u.id, u.username, u.email, p.first_name, p.last_name
            FROM users u
            LEFT JOIN user_profiles p ON u.id = p.user_id
            WHERE (u.username LIKE ? OR u.email LIKE ? OR 
                   p.first_name LIKE ? OR p.last_name LIKE ?)
                  AND u.is_active = 1
            ORDER BY u.username
            LIMIT ?
        """, (search_pattern, search_pattern, search_pattern, search_pattern, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_user_stats(self) -> Dict:
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) as total_users FROM users WHERE is_active = 1")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) as users_with_profiles 
            FROM users u 
            JOIN user_profiles p ON u.id = p.user_id 
            WHERE u.is_active = 1
        """)
        users_with_profiles = cursor.fetchone()[0]
        
        return {
            "total_users": total_users,
            "users_with_profiles": users_with_profiles,
            "completion_rate": (users_with_profiles / total_users * 100) if total_users > 0 else 0
        }
    
    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
'''
    
    print("\n1. 🔍 复杂代码分析")
    analysis = cursor.analyze_code(complex_code)
    
    print(f"📊 代码统计:")
    print(f"  总行数: {analysis['metrics']['total_lines']}")
    print(f"  代码行数: {analysis['metrics']['code_lines']}")
    print(f"  注释行数: {analysis['metrics']['comment_lines']}")
    print(f"  函数数量: {len(analysis['functions'])}")
    print(f"  类数量: {len(analysis['classes'])}")
    print(f"  复杂度: {analysis['complexity']}")
    
    print(f"\n📦 导入分析:")
    for imp in analysis['imports'][:5]:  # 显示前5个导入
        print(f"  • {imp}")
    
    print(f"\n🏠 类结构分析:")
    for cls in analysis['classes']:
        print(f"  • {cls['name']} (行 {cls['line']}, {len(cls['methods'])} 个方法)")
        for method in cls['methods'][:3]:  # 显示前3个方法
            print(f"    - {method}()")
        if len(cls['methods']) > 3:
            print(f"    - ... 还有 {len(cls['methods']) - 3} 个方法")
    
    print("\n2. 🔧 高级重构分析")
    refactor_suggestions = cursor.suggest_refactor(complex_code)
    if refactor_suggestions:
        for i, suggestion in enumerate(refactor_suggestions, 1):
            print(f"{i}. 🔧 {suggestion.description}")
            print(f"   类型: {suggestion.suggestion_type}")
            print(f"   影响: {suggestion.estimated_impact}")
            print(f"   优势: {', '.join(suggestion.benefits)}")
    else:
        print("✅ 代码结构良好")
    
    print("\n3. 📋 综合代码审查")
    review_result = cursor.code_review(complex_code)
    print(f"🏆 代码质量评分: {review_result['overall_score']:.1f}/100")
    
    recommendations = review_result.get("recommendations", [])
    if recommendations:
        print("📝 专业建议:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print("\n4. 🤖 智能代码生成")
    
    # 生成辅助方法
    print("生成辅助工具方法:")
    utility_code = cursor.generate_code("function hash_password")
    print(utility_code[:150] + "...")
    
    # 生成测试用例
    print("\n生成测试代码:")
    test_code = cursor.generate_code("test UserManager")
    print(test_code[:150] + "...")
    
    print("\n🎉 高级功能演示完成!")


def main():
    """主程序入口"""
    print("🚀 欢迎使用 AI Coding Cursor - 智能编程助手!")
    print("💻 模拟实现类似 Cursor 的 AI 编程助手系统")
    print("=" * 60)
    
    while True:
        print("\n📚 请选择操作:")
        print("1. 🔍 基础功能演示")
        print("2. 🚀 高级功能演示")
        print("3. 💬 交互式聊天模式")
        print("4. 📝 使用指南")
        print("5. 🚪 退出")
        
        try:
            choice = input("\n请输入选项 (1-5): ").strip()
            
            if choice == '1':
                demo_basic_features()
            elif choice == '2':
                demo_advanced_features()
            elif choice == '3':
                cursor = AICodingCursor()
                cursor.chat_mode()
            elif choice == '4':
                show_usage_guide()
            elif choice == '5':
                print("👋 感谢使用 AI Coding Cursor！再见!")
                break
            else:
                print("❌ 无效选项，请输入 1-5")
        
        except KeyboardInterrupt:
            print("\n👋 感谢使用 AI Coding Cursor！再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def show_usage_guide():
    """显示使用指南"""
    print("\n📝 AI Coding Cursor 使用指南")
    print("=" * 40)
    
    print("\n🎁 主要特性:")
    print("• 🔍 智能代码分析 - 深度理解代码结构和质量")
    print("• 🤖 代码生成 - 基于上下文的智能代码生成")
    print("• 🔧 智能重构 - 自动识别并建议代码改进")
    print("• 🐛 错误诊断 - 实时调试和错误修复建议")
    print("• 📋 代码审查 - 综合质量评估和改进建议")
    
    print("\n💬 交互式命令:")
    print("• analyze:<code>    - 分析代码结构和质量")
    print("• generate:<request> - 根据描述生成代码")
    print("• refactor:<code>   - 分析重构机会")
    print("• debug:<code>     - 诊断代码错误")
    print("• review:<code>    - 综合代码审查")
    print("• help             - 显示帮助信息")
    print("• quit/exit        - 退出程序")
    
    print("\n🌱 使用示例:")
    print("analyze:def hello(): print('Hello World')")
    print("generate:function calculate_fibonacci")
    print("refactor:def long_function_with_many_lines()...")
    print("debug:def broken_syntax(")
    
    print("\n💡 小贴士:")
    print("• 使用多行代码时，可以使用三引号 ''' 包裹")
    print("• 代码生成支持函数、类、测试等多种类型")
    print("• 重构建议基于代码复杂度和最佳实践")
    print("• 调试功能可以自动修复常见语法错误")


if __name__ == "__main__":
    main()
    
    def chat_mode(self):
        """进入交互式聊天模式"""
        print(f"\n🤖 {self.name} 交互式模式")
        print("输入 'help' 查看可用命令，输入 'quit' 退出")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！感谢使用 AI Coding Cursor!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                
                elif user_input.startswith('analyze:'):
                    code = user_input[8:].strip()
                    result = self.analyze_code(code)
                    self._print_analysis_result(result)
                
                elif user_input.startswith('generate:'):
                    request = user_input[9:].strip()
                    result = self.generate_code(request)
                    print(f"\n🤖 生成的代码:\n{result}")
                
                elif user_input.startswith('refactor:'):
                    code = user_input[9:].strip()
                    suggestions = self.suggest_refactor(code)
                    self._print_refactor_suggestions(suggestions)
                
                elif user_input.startswith('debug:'):
                    code = user_input[6:].strip()
                    result = self.debug_code(code)
                    self._print_debug_result(result)
                
                elif user_input.startswith('review:'):
                    code = user_input[7:].strip()
                    result = self.code_review(code)
                    self._print_review_result(result)
                
                else:
                    print("🤖 请使用指定格式输入命令，输入 'help' 查看帮助")
            
            except KeyboardInterrupt:
                print("\n👋 再见！感谢使用 AI Coding Cursor!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n📚 可用命令:")
        print("analyze:<code>    - 分析代码")
        print("generate:<request> - 生成代码")
        print("refactor:<code>   - 重构建议")
        print("debug:<code>     - 调试代码")
        print("review:<code>    - 代码审查")
        print("help             - 显示帮助")
        print("quit/exit        - 退出程序")
    
    def _print_analysis_result(self, result: Dict[str, Any]):
        """打印分析结果"""
        print(f"\n🔍 分析结果:")
        if result.get("syntax_valid", False):
            print(f"✅ 语法有效")
            print(f"📊 函数数量: {len(result.get('functions', []))}")
            print(f"📊 类数量: {len(result.get('classes', []))}")
            print(f"📊 复杂度: {result.get('complexity', 0)}")
            
            issues = result.get("issues", [])
            if issues:
                print(f"⚠️  发现 {len(issues)} 个问题:")
                for i, issue in enumerate(issues[:3], 1):
                    print(f"  {i}. {issue.message}")
        else:
            print(f"❌ 语法错误: {result.get('error', '未知错误')}")
    
    def _print_refactor_suggestions(self, suggestions: List[RefactorSuggestion]):
        """打印重构建议"""
        print(f"\n🔧 重构建议:")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion.description}")
                print(f"     影响: {suggestion.estimated_impact}")
        else:
            print("✅ 代码结构良好，无需重构")
    
    def _print_debug_result(self, result: Dict[str, Any]):
        """打印调试结果"""
        print(f"\n🐛 调试结果:")
        if result.get("error_found", False):
            print(f"❌ 发现错误: {result.get('error_type', '未知错误')}")
            print(f"📍 行号: {result.get('error_line', 0)}")
            print(f"💡 诊断: {result.get('diagnosis', '')}")
            
            suggestions = result.get("suggestions", [])
            if suggestions:
                print("🔧 修复建议:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")
        else:
            print("✅ 未发现明显错误")
    
    def _print_review_result(self, result: Dict[str, Any]):
        """打印审查结果"""
        print(f"\n📋 代码审查报告:")
        print(f"🏆 总体评分: {result.get('overall_score', 0):.1f}/100")
        
        recommendations = result.get("recommendations", [])
        if recommendations:
            print("📝 改进建议:")
            for rec in recommendations:
                print(f"  • {rec}")