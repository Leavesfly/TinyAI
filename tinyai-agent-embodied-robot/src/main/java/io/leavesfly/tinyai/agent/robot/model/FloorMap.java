package io.leavesfly.tinyai.agent.robot.model;

/**
 * 地面清扫地图类
 * 
 * <p>表示清扫区域的网格地图，记录已清扫区域和灰尘分布。</p>
 * 
 * @author TinyAI Team
 */
public class FloorMap {
    /**
     * 地图宽度（网格数）
     */
    private int width;
    
    /**
     * 地图高度（网格数）
     */
    private int height;
    
    /**
     * 网格大小（米）
     */
    private double gridSize;
    
    /**
     * 已清扫标记
     */
    private boolean[][] cleanedGrid;
    
    /**
     * 灰尘密度分布（0-1）
     */
    private double[][] dustDensity;
    
    /**
     * 地面类型分布
     */
    private FloorType[][] floorType;
    
    /**
     * 构造函数
     * 
     * @param width 地图宽度（网格数）
     * @param height 地图高度（网格数）
     * @param gridSize 网格大小（米）
     */
    public FloorMap(int width, int height, double gridSize) {
        this.width = width;
        this.height = height;
        this.gridSize = gridSize;
        this.cleanedGrid = new boolean[height][width];
        this.dustDensity = new double[height][width];
        this.floorType = new FloorType[height][width];
        
        // 初始化
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                cleanedGrid[i][j] = false;
                dustDensity[i][j] = 0.5; // 默认中等灰尘密度
                floorType[i][j] = FloorType.TILE; // 默认瓷砖
            }
        }
    }
    
    /**
     * 位置转换为网格坐标
     * 
     * @param position 实际位置
     * @return 网格坐标 [行, 列]
     */
    public int[] positionToGrid(Vector2D position) {
        int col = (int) (position.getX() / gridSize);
        int row = (int) (position.getY() / gridSize);
        
        // 边界检查
        col = Math.max(0, Math.min(width - 1, col));
        row = Math.max(0, Math.min(height - 1, row));
        
        return new int[]{row, col};
    }
    
    /**
     * 网格坐标转换为位置
     * 
     * @param row 行
     * @param col 列
     * @return 实际位置（网格中心）
     */
    public Vector2D gridToPosition(int row, int col) {
        double x = (col + 0.5) * gridSize;
        double y = (row + 0.5) * gridSize;
        return new Vector2D(x, y);
    }
    
    /**
     * 标记位置已清扫
     * 
     * @param position 位置
     * @param cleaningEfficiency 清扫效率（0-1）
     */
    public void markCleaned(Vector2D position, double cleaningEfficiency) {
        int[] grid = positionToGrid(position);
        int row = grid[0];
        int col = grid[1];
        
        cleanedGrid[row][col] = true;
        // 减少灰尘密度
        dustDensity[row][col] = Math.max(0, dustDensity[row][col] - cleaningEfficiency * 0.1);
    }
    
    /**
     * 获取某位置的灰尘量
     * 
     * @param position 位置
     * @return 灰尘密度（0-1）
     */
    public double getDustAt(Vector2D position) {
        int[] grid = positionToGrid(position);
        return dustDensity[grid[0]][grid[1]];
    }
    
    /**
     * 计算覆盖率
     * 
     * @return 覆盖率（0-1）
     */
    public double getCoverageRate() {
        int cleanedCount = 0;
        int totalCount = width * height;
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (cleanedGrid[i][j]) {
                    cleanedCount++;
                }
            }
        }
        
        return (double) cleanedCount / totalCount;
    }
    
    /**
     * 计算剩余灰尘总量
     * 
     * @return 灰尘总量
     */
    public double getTotalDust() {
        double total = 0.0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                total += dustDensity[i][j];
            }
        }
        return total;
    }
    
    /**
     * 设置灰尘分布
     * 
     * @param pattern 分布模式：uniform（均匀）、concentrated（集中）、random（随机）
     */
    public void setDustDistribution(String pattern) {
        switch (pattern.toLowerCase()) {
            case "uniform":
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        dustDensity[i][j] = 0.5;
                    }
                }
                break;
            case "concentrated":
                // 中心区域灰尘多
                int centerRow = height / 2;
                int centerCol = width / 2;
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        double distToCenter = Math.sqrt(
                            Math.pow(i - centerRow, 2) + Math.pow(j - centerCol, 2)
                        );
                        dustDensity[i][j] = Math.max(0.1, 1.0 - distToCenter / (width / 2.0));
                    }
                }
                break;
            case "random":
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        dustDensity[i][j] = Math.random();
                    }
                }
                break;
            default:
                // 默认均匀分布
                setDustDistribution("uniform");
        }
    }
    
    /**
     * 重置地图
     */
    public void reset() {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                cleanedGrid[i][j] = false;
                dustDensity[i][j] = 0.5;
            }
        }
    }
    
    // Getters
    public int getWidth() {
        return width;
    }
    
    public int getHeight() {
        return height;
    }
    
    public double getGridSize() {
        return gridSize;
    }
    
    public boolean[][] getCleanedGrid() {
        return cleanedGrid;
    }
    
    public double[][] getDustDensity() {
        return dustDensity;
    }
    
    public FloorType[][] getFloorType() {
        return floorType;
    }
    
    @Override
    public String toString() {
        return String.format("FloorMap(size=%dx%d, gridSize=%.2f, coverage=%.1f%%)",
                             width, height, gridSize, getCoverageRate() * 100);
    }
}
