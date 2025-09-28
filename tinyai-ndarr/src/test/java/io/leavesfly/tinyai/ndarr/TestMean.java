package io.leavesfly.tinyai.ndarr;


public class TestMean {
    public static void main(String[] args) {
        // 创建一个3D数组
        float[][][] data3d = {
                {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                {{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}}
        };

        NdArray array3d = NdArray.of(data3d);
        System.out.println("3D数组: " + array3d);
        System.out.println("形状: " + array3d.getShape());

        // 计算最后一个维度的均值
        int lastAxis = array3d.getShape().getDimNum() - 1;
        System.out.println("最后一个轴: " + lastAxis);

        try {
            NdArray meanArray = array3d.mean(lastAxis);
            System.out.println("均值数组: " + meanArray);
            System.out.println("均值数组形状: " + meanArray.getShape());
        } catch (Exception e) {
            System.out.println("计算均值时出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
}