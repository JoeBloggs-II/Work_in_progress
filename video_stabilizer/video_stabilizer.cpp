#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Smooth transformation matrix to avoid jitter
Mat smoothTransform(const vector<Mat>& transforms, int idx, int radius = 15) {
    Mat smoothed = Mat::zeros(2, 3, CV_64F);
    int count = 0;
    for (int i = max(0, idx - radius); i <= min((int)transforms.size() - 1, idx + radius); i++) {
        smoothed += transforms[i];
        count++;
    }
    smoothed /= count;
    return smoothed;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input_video> <output_video>\n";
        return -1;
    }

    string inputPath = argv[1];
    string outputPath = argv[2];

    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file.\n";
        return -1;
    }

    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    int totalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);

    VideoWriter writer(outputPath, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    // Prepare variables for stabilization
    Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) return -1;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    vector<Mat> transforms;
    cout << "Calculating frame-to-frame transforms...\n";

    for (int i = 1; i < totalFrames; i++) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect optical flow
        vector<Point2f> prevPts, nextPts;
        goodFeaturesToTrack(prevGray, prevPts, 200, 0.01, 30);
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prevGray, gray, prevPts, nextPts, status, err);

        // Keep only good points
        vector<Point2f> goodPrev, goodNext;
        for (size_t j = 0; j < status.size(); j++) {
            if (status[j]) {
                goodPrev.push_back(prevPts[j]);
                goodNext.push_back(nextPts[j]);
            }
        }

        Mat transform = estimateAffinePartial2D(goodPrev, goodNext);
        if (transform.empty())
            transform = Mat::eye(2, 3, CV_64F);
        transforms.push_back(transform);

        prevGray = gray.clone();
    }

    // Reset and write stabilized video
    cap.set(CAP_PROP_POS_FRAMES, 0);
    cout << "Stabilizing and processing frames...\n";

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    int idx = 0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Apply smoothing transform
        Mat smoothed = smoothTransform(transforms, idx);
        Mat stabilized;
        warpAffine(frame, stabilized, smoothed, frame.size());

        // Convert to LAB and apply CLAHE to L channel
        Mat lab;
        cvtColor(stabilized, lab, COLOR_BGR2Lab);
        vector<Mat> labChannels;
        split(lab, labChannels);
        clahe->apply(labChannels[0], labChannels[0]);
        merge(labChannels, lab);
        cvtColor(lab, stabilized, COLOR_Lab2BGR);

        writer.write(stabilized);
        idx++;
    }

    cap.release();
    writer.release();
    cout << "Processing complete! Saved to: " << outputPath << endl;

    return 0;
}
