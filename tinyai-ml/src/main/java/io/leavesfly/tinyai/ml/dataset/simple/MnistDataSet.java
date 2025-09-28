package io.leavesfly.tinyai.ml.dataset.simple;

import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ndarr.NdArray;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

/**
 * 手写数字的图像集合
 */
public class MnistDataSet extends ArrayDataset {

    private String UserHome = System.getProperty("user.home");

    private String MnistDir = UserHome + "/mnist/";

    private String trainImages = MnistDir + "train-images-idx3-ubyte";
    private String trainLabels = MnistDir + "train-labels-idx1-ubyte";


    private String testImages = MnistDir + "t10k-images-idx3-ubyte";
    private String testLabels = MnistDir + "t10k-labels-idx1-ubyte";


    public MnistDataSet(int batchSize) {
        super(batchSize);
    }

    @Override
    public void doPrepare() {

        try {
            downloadMnist();
            NdArray[] trainX = readMnistImageFile(trainImages);
            NdArray[] trainY = readMnistLabelFile(trainLabels);
            DataSet trainDataset = build(batchSize, trainX, trainY);
            splitDatasetMap.put(Usage.TRAIN.name(), trainDataset);

            NdArray[] testX = readMnistImageFile(testImages);
            NdArray[] testY = readMnistLabelFile(testLabels);
            DataSet testDataset = build(batchSize, testX, testY);
            splitDatasetMap.put(Usage.TEST.name(), testDataset);

        } catch (Exception e) {
            throw new RuntimeException("MnistDataSet prepare() error!");
        }
    }


    @Override
    protected DataSet build(int batchSize, NdArray[] _xs, NdArray[] _ys) {
        MnistDataSet dataSet = new MnistDataSet(batchSize);
        dataSet.xs = _xs;
        dataSet.ys = _ys;
        return dataSet;
    }


    public NdArray[] readMnistImageFile(String imagesFile) throws IOException {
        NdArray[] arrays = null;
        try (DataInputStream images = new DataInputStream(Files.newInputStream(Paths.get(imagesFile)));) {
            int magicNumber = images.readInt();
            int numImages = images.readInt();
            arrays = new NdArray[numImages];

            int numRows = images.readInt();
            int numColumns = images.readInt();
            for (int i = 0; i < numImages; i++) {
                byte[] image = new byte[numRows * numColumns];
                images.readFully(image);
                float[] values = new float[numRows * numColumns];
                for (int j = 0; j < image.length; j++) {
                    values[j] = (float) (image[j] & 0xff) / 255;
                }
                arrays[i] = NdArray.of(values);
            }
        }
        return arrays;
    }

    public NdArray[] readMnistLabelFile(String labelsFile) throws IOException {
        NdArray[] arrays = null;
        try (DataInputStream labels = new DataInputStream(Files.newInputStream(Paths.get(labelsFile)))) {
            // 读取标签
            int magicNumber = labels.readInt();
            int numLabels = labels.readInt();
            arrays = new NdArray[numLabels];
            for (int i = 0; i < numLabels; i++) {
                int label = labels.readByte();
                arrays[i] = NdArray.of(label);
            }
        }
        return arrays;
    }


    public void downloadMnist() throws IOException {
        File outputDir = new File(MnistDir);
        String mnistUrl = "http://yann.lecun.com/exdb/mnist/";
        String[] files = {"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"};

        for (String file : files) {
            URL url = new URL(mnistUrl + file);
            if (!outputDir.exists()) {
                outputDir.mkdirs();
            }
            File outputFile = new File(outputDir, file.replace(".gz", ""));
            if (!outputFile.exists()) {
                System.out.println("Downloading " + url + "...");
                try (GZIPInputStream gzis = new GZIPInputStream(new BufferedInputStream(url.openStream()));
                     BufferedOutputStream bos = new BufferedOutputStream(Files.newOutputStream(outputFile.toPath()))) {
                    byte[] buffer = new byte[1024];
                    int len;
                    while ((len = gzis.read(buffer)) != -1) {
                        bos.write(buffer, 0, len);
                    }
                }
                System.out.println("Done.");
            } else {
                System.out.println(outputFile + " already exists.");
            }
        }
    }

    private void drawImage(float[] pixelValues, String fileName) throws IOException {
        // 将第一张图片转换为二维数组
        float[][] image = new float[28][28];
        for (int i = 0; i < 28; i++) {
            System.arraycopy(pixelValues, i * 28, image[i], 0, 28);
        }
        // 使用Swing生成图片并保存到本地
        BufferedImage bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = bufferedImage.createGraphics();
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int gray = (int) (image[x][y] * 255);
                Color color = new Color(gray, gray, gray);
                graphics.setColor(color);
                graphics.fillRect(x, y, 1, 1);
            }
        }
        ImageIO.write(bufferedImage, "png", new File(fileName));
    }

    public String getTrainImages() {
        return trainImages;
    }

    public String getTrainLabels() {
        return trainLabels;
    }

    public String getMnistDir() {
        return MnistDir;
    }

    public static void main(String[] args) throws IOException {

        MnistDataSet mnist = new MnistDataSet(100);

        mnist.downloadMnist();

        NdArray[] trainX = mnist.readMnistImageFile(mnist.getTrainImages());
        NdArray[] trainY = mnist.readMnistLabelFile(mnist.getTrainLabels());

        mnist.drawImage(trainX[1].getMatrix()[0], mnist.getMnistDir() + "1.png");
        System.out.println(trainY[1].getNumber().intValue());

//        mnist.prepare();

    }
}
