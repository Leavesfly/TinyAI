package io.leavesfly.tinyai.minimind.cli;

import java.util.*;

/**
 * MiniMind命令行工具主入口
 * 
 * 支持以下命令:
 * - train-pretrain: 预训练模型
 * - train-sft: 监督微调(SFT)
 * - train-lora: LoRA微调
 * - generate: 文本生成
 * - chat: 交互式对话
 * - evaluate: 模型评估
 * 
 * 使用示例:
 * ```bash
 * java -jar minimind-cli.jar train-pretrain --config config.yaml
 * java -jar minimind-cli.jar generate --prompt "今天天气" --max-length 100
 * java -jar minimind-cli.jar chat --model-path model.pt
 * ```
 * 
 * @author leavesfly
 * @since 2024
 */
public class MiniMindCLI {
    
    private static final String VERSION = "1.0.0";
    
    /**
     * 命令注册表
     */
    private static final Map<String, Command> COMMANDS = new LinkedHashMap<>();
    
    static {
        registerCommand("train-pretrain", new TrainPretrainCommand());
        registerCommand("train-sft", new TrainSFTCommand());
        registerCommand("train-lora", new TrainLoRACommand());
        registerCommand("generate", new GenerateCommand());
        registerCommand("chat", new ChatCommand());
        registerCommand("evaluate", new EvaluateCommand());
        registerCommand("help", new HelpCommand());
        registerCommand("version", new VersionCommand());
    }
    
    /**
     * 注册命令
     */
    private static void registerCommand(String name, Command command) {
        COMMANDS.put(name, command);
    }
    
    /**
     * 主入口
     */
    public static void main(String[] args) {
        try {
            // 参数检查
            if (args.length == 0) {
                printUsage();
                System.exit(0);
            }
            
            // 解析命令
            String commandName = args[0];
            String[] commandArgs = Arrays.copyOfRange(args, 1, args.length);
            
            // 查找并执行命令
            Command command = COMMANDS.get(commandName);
            if (command == null) {
                System.err.println("错误: 未知命令 '" + commandName + "'");
                System.err.println("使用 'help' 查看可用命令");
                System.exit(1);
            }
            
            // 执行命令
            command.execute(commandArgs);
            
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    /**
     * 打印使用说明
     */
    private static void printUsage() {
        System.out.println("MiniMind CLI v" + VERSION);
        System.out.println();
        System.out.println("用法: minimind <command> [options]");
        System.out.println();
        System.out.println("可用命令:");
        System.out.println("  train-pretrain    预训练模型");
        System.out.println("  train-sft         监督微调(SFT)");
        System.out.println("  train-lora        LoRA微调");
        System.out.println("  generate          文本生成");
        System.out.println("  chat              交互式对话");
        System.out.println("  evaluate          模型评估");
        System.out.println("  help              显示帮助信息");
        System.out.println("  version           显示版本信息");
        System.out.println();
        System.out.println("使用 'minimind <command> --help' 查看命令详细帮助");
    }
    
    /**
     * 命令接口
     */
    public interface Command {
        void execute(String[] args) throws Exception;
        String getDescription();
        void printHelp();
    }
    
    /**
     * 帮助命令
     */
    static class HelpCommand implements Command {
        @Override
        public void execute(String[] args) {
            if (args.length > 0) {
                String commandName = args[0];
                Command command = COMMANDS.get(commandName);
                if (command != null) {
                    command.printHelp();
                } else {
                    System.err.println("未知命令: " + commandName);
                }
            } else {
                printUsage();
            }
        }
        
        @Override
        public String getDescription() {
            return "显示帮助信息";
        }
        
        @Override
        public void printHelp() {
            printUsage();
        }
    }
    
    /**
     * 版本命令
     */
    static class VersionCommand implements Command {
        @Override
        public void execute(String[] args) {
            System.out.println("MiniMind CLI v" + VERSION);
            System.out.println("基于TinyAI深度学习框架");
        }
        
        @Override
        public String getDescription() {
            return "显示版本信息";
        }
        
        @Override
        public void printHelp() {
            System.out.println("用法: minimind version");
            System.out.println();
            System.out.println("显示MiniMind CLI版本信息");
        }
    }
    
    /**
     * 参数解析器
     */
    public static class ArgParser {
        private final Map<String, String> options = new HashMap<>();
        private final List<String> positional = new ArrayList<>();
        
        public ArgParser(String[] args) {
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (arg.startsWith("--")) {
                    String key = arg.substring(2);
                    if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                        options.put(key, args[++i]);
                    } else {
                        options.put(key, "true");
                    }
                } else if (arg.startsWith("-")) {
                    String key = arg.substring(1);
                    if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
                        options.put(key, args[++i]);
                    } else {
                        options.put(key, "true");
                    }
                } else {
                    positional.add(arg);
                }
            }
        }
        
        public String get(String key) {
            return options.get(key);
        }
        
        public String get(String key, String defaultValue) {
            return options.getOrDefault(key, defaultValue);
        }
        
        public int getInt(String key, int defaultValue) {
            String value = options.get(key);
            if (value == null) {
                return defaultValue;
            }
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                System.err.println("警告: 参数 '" + key + "' 的值 '" + value + "' 不是有效的整数，使用默认值 " + defaultValue);
                return defaultValue;
            }
        }
        
        public float getFloat(String key, float defaultValue) {
            String value = options.get(key);
            if (value == null) {
                return defaultValue;
            }
            try {
                return Float.parseFloat(value);
            } catch (NumberFormatException e) {
                System.err.println("警告: 参数 '" + key + "' 的值 '" + value + "' 不是有效的浮点数，使用默认值 " + defaultValue);
                return defaultValue;
            }
        }
        
        public boolean getBoolean(String key) {
            return "true".equals(options.get(key));
        }
        
        public boolean has(String key) {
            return options.containsKey(key);
        }
        
        public List<String> getPositional() {
            return positional;
        }
    }
}