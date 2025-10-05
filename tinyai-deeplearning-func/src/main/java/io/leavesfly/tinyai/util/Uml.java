package io.leavesfly.tinyai.util;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;

import java.util.Objects;

/**
 * UML绘图工具类，用于绘制计算图
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * Uml类提供了将计算图转换为PlantUML格式的功能，
 * 可以可视化展示变量和函数之间的依赖关系。
 */
public class Uml {

    /**
     * 在线验证：http://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000
     */

    /**
     * 获取变量节点的DOT图表示
     *
     * @param variableNode 变量节点
     * @return DOT格式的图表示字符串
     */
    public static String getDotGraph(Variable variableNode) {
        StringBuilder dotGraph = new StringBuilder();
        getDotNode(variableNode, dotGraph);
        return "@startuml\ndigraph g {\n" + "rankdir=LR;\n" + dotGraph + "}\n@enduml";
    }

    /**
     * 递归获取变量节点的DOT表示
     *
     * @param variableNode 变量节点
     * @param dotGraph     DOT图字符串构建器
     */
    private static void getDotNode(Variable variableNode, StringBuilder dotGraph) {

        if (!Objects.isNull(variableNode)) {

            String dotNode = getDotVar(variableNode);
            dotGraph.append(dotNode);

            Function functionNode = variableNode.getCreator();

            if (!Objects.isNull(functionNode)) {

                dotGraph.append(getDotFunc(functionNode));

                Variable[] inputs = functionNode.getInputs();
                if (!Objects.isNull(inputs)) {
                    for (Variable input : inputs) {
                        getDotNode(input, dotGraph);
                    }
                }
            }
        }
    }

    /**
     * 获取变量节点的DOT表示
     *
     * @param node 变量节点
     * @return DOT格式的节点表示字符串
     */
    private static String getDotVar(Variable node) {
        String dotVar = "%s [label=\"%s\", color=orange, style=filled]\n";

        StringBuilder label = new StringBuilder();
        if (node.getName() != null) {
            label.append(node.getName()).append(" ");
        }
//        label.append(" value:").append(node.getValue().getNumber().floatValue()).append(" ");
//        if (node.getGrad() != null) {
//            label.append(" grad:").append(node.getGrad().getNumber().floatValue()).append(" ");
//        }

        return String.format(dotVar, node.hashCode(), label);
    }

    /**
     * 获取函数节点的DOT表示
     *
     * @param node 函数节点
     * @return DOT格式的函数节点表示字符串
     */
    private static String getDotFunc(Function node) {
        StringBuilder dotFunc = new StringBuilder("%s [label=\"%s\", color=lightblue, style=filled, shape=box]\n");
        String docEdge = " %s -> %s\n";
        dotFunc = new StringBuilder(String.format(dotFunc.toString(), node.hashCode(), node.getClass().getSimpleName()));
        for (Variable input : node.getInputs()) {
            dotFunc.append(String.format(docEdge, input.hashCode(), node.hashCode()));
        }
        dotFunc.append(String.format(docEdge, node.hashCode(), node.getOutput().hashCode()));
        return dotFunc.toString();
    }
}