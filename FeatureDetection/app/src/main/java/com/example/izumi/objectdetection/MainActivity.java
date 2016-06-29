package com.example.izumi.objectdetection;

import android.app.Activity;
import android.os.Bundle;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    // カメラビューのインスタンス
    // CameraBridgeViewBase は JavaCameraView/NativeCameraView のスーパークラス
    private CameraBridgeViewBase mCameraView;
    private Mat mOutputFrame;
    private Mat saveImage;
    private Mat descriptorSaved;
    private MatOfKeyPoint keyPointSaved;
    private int switch1 = 1;
    private FeatureDetector orbDetector;
    private DescriptorExtractor orbExtractor;
    private DescriptorMatcher matcher;
    // ライブラリ初期化完了後に呼ばれるコールバック (onManagerConnected)

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                // 読み込みが成功したらカメラプレビューを開始
                case LoaderCallbackInterface.SUCCESS:
                    mCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // カメラビューのインスタンスを変数にバインド
        mCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        // リスナーの設定 (後述)
        mCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 非同期でライブラリの読み込み/初期化を行う
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_4, this, mLoaderCallback);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mCameraView != null) {
            mCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // カメラプレビュー開始時に呼ばれる
        // Mat(int rows, int cols, int type)
        // rows(行): height, cols(列): width
        mOutputFrame = new Mat(height, width, CvType.CV_8UC1);
        orbExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        orbDetector = FeatureDetector.create(FeatureDetector.ORB);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
    }

    @Override
    public void onCameraViewStopped() {
        // カメラプレビュー終了時に呼ばれる
        mOutputFrame.release();
    }

    // CvCameraViewListener の場合
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // フレームをキャプチャする毎(30fpsなら毎秒30回)に呼ばれる
        mOutputFrame = inputFrame.rgba();
        Mat grayImage = inputFrame.gray();
        if (switch1 == 1) {
            keyPointSaved = new MatOfKeyPoint();
            orbDetector.detect(grayImage, keyPointSaved);

            descriptorSaved = new Mat(mOutputFrame.rows(), mOutputFrame.cols(), mOutputFrame.type());
            orbExtractor.compute(grayImage, keyPointSaved, descriptorSaved);
            switch1 = 0;
        }

        if (switch1==0) {
            MatOfKeyPoint keyPoint = new MatOfKeyPoint();
            orbDetector.detect(grayImage, keyPoint);

            Mat descriptor = new Mat(mOutputFrame.rows(), mOutputFrame.cols(), mOutputFrame.type());
            orbExtractor.compute(grayImage, keyPoint, descriptor);

            MatOfDMatch matches = new MatOfDMatch();
            matcher.match(descriptorSaved, descriptor, matches);

            //
            DMatch[] tmp01 = matches.toArray();
            KeyPoint[] kp = keyPoint.toArray();
            int idx;
            int i;
            for (i=0;i<3;i++) {
                idx = tmp01[i].trainIdx;
                Point point = kp[idx].pt;
                Core.circle(mOutputFrame, point, 10, new Scalar(255, 0, 0), 2);
            }

        }



        return mOutputFrame;
    }
}
