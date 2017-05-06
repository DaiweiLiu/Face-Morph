#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace dlib;
using namespace std;

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float> Triplet;

/* ============= This part of the code is modified based on codes from LearnOpenCv.com ============= */

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri) {
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = getAffineTransform( srcTri, dstTri );
    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morphTriangle(cv::Mat &img1, cv::Mat &img2, cv::Mat &img, std::vector<Point2f> &t1, std::vector<Point2f> &t2, std::vector<Point2f> &t, float alpha) {
    // Find bounding rectangle for each triangle
    Rect r = boundingRect(t);
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> t1Rect, t2Rect, tRect;
    std::vector<Point> tRectInt;
    for(int i = 0; i < 3; i++)
    {
        tRect.push_back( Point2f( t[i].x - r.x, t[i].y -  r.y) );
        tRectInt.push_back( Point(t[i].x - r.x, t[i].y - r.y) ); // for fillConvexPoly
        
        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
    }
    
    // Get mask by filling triangle
    cv::Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
    fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    
    // Apply warpImage to small rectangular patches
    cv::Mat img1Rect, img2Rect;
    img1(r1).copyTo(img1Rect);
    img2(r2).copyTo(img2Rect);
    
    cv::Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
    cv::Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    
    applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
    applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    
    // Alpha blend rectangular patches
    cv::Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

    // Copy triangular region of the rectangular patch to the output image
    cv::multiply(imgRect, mask, imgRect);
    cv::multiply(img(r), Scalar(1.0,1.0,1.0) - mask, img(r));
    img(r) = img(r) + imgRect;
}

void warpTriangle(cv::Mat &src, cv::Mat &dst, std::vector<Point2f> &ts, std::vector<Point2f> &td) {
    Rect rs = boundingRect(ts);
    Rect rd = boundingRect(td);
    std::vector<Point2f> srcRect, dstRect;
    std::vector<Point> dstRectInt;
    for(int i = 0; i < 3; i++) {
        dstRect.push_back( Point2f( td[i].x - rd.x, td[i].y - rd.y) );
        dstRectInt.push_back( Point(td[i].x - rd.x, td[i].y - rd.y) );
        srcRect.push_back( Point2f( ts[i].x - rs.x, ts[i].y - rs.y) );
    }

    cv::Mat mask = Mat::zeros(rd.height, rd.width, CV_32FC3);
    fillConvexPoly(mask, dstRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    cv::Mat srcImgRect;
    src(rs).copyTo(srcImgRect);

    cv::Mat warpImg = Mat::zeros(rd.height, rd.width, srcImgRect.type());
    applyAffineTransform(warpImg, srcImgRect, srcRect, dstRect);

    cv::multiply(warpImg, mask, warpImg);
    cv::multiply(dst(rd), Scalar(1.0,1.0,1.0) - mask, dst(rd));
    dst(rd) = dst(rd) + warpImg;
}
/* ================================================================================== */

/* ===>>> Poisson Blending Support Code <<<=== */
Mat getLaplacian() {
    Mat laplacian = Mat::zeros(3, 3, CV_64FC1);
    laplacian.at<double>(0, 1) = 1.0;
    laplacian.at<double>(1, 0) = 1.0;
    laplacian.at<double>(1, 2) = 1.0;
    laplacian.at<double>(2, 1) = 1.0;
    laplacian.at<double>(1, 1) = -4.0; 
    return laplacian;
}

SpMat getA(Mat &mask, Rect ROI, int N, std::vector<double> &idxMap) {
    // populate coefficients
    std::vector<Triplet> coefficients;
    coefficients.reserve(5 * N);
    // diagnoal entreis are -4
    for (int i = 0; i < N; ++i)
        coefficients.push_back(Triplet(i, i, -4.0));
    for (int row = ROI.y; row < ROI.y + ROI.height; ++row) {
        for (int col = ROI.x; col < ROI.x + ROI.width; ++col) {
            if (mask.at<uchar>(row, col) == 1) {
                int i = idxMap[(row - ROI.y) * ROI.width + col - ROI.x];
                if (mask.at<uchar>(row - 1, col) == 1) {
                    int j = idxMap[(row - ROI.y - 1) * ROI.width + col - ROI.x];
                    coefficients.push_back(Triplet(i, j, 1.0));
                }
                if (mask.at<uchar>(row + 1, col) == 1) {
                    int j = idxMap[(row - ROI.y + 1) * ROI.width + col - ROI.x];
                    coefficients.push_back(Triplet(i, j, 1.0));
                }
                if (mask.at<uchar>(row, col - 1) == 1) {
                    int j = idxMap[(row - ROI.y) * ROI.width + col - ROI.x - 1];
                    coefficients.push_back(Triplet(i, j, 1.0));
                }
                if (mask.at<uchar>(row, col + 1) == 1) {
                    int j = idxMap[(row - ROI.y) * ROI.width + col - ROI.x + 1];
                    coefficients.push_back(Triplet(i, j, 1.0));
                }
            }
        }
    }
    SpMat A(N, N);
    A.setFromTriplets(coefficients.begin(), coefficients.end());
    return A;
}

Mat getB(Mat &src, Mat &dst, Mat &mask, Rect ROI, int N, std::vector<double> &idxMap) {
    Mat grad;
    filter2D(src, grad, -1, getLaplacian());
    Mat B = Mat::zeros(N, 1, CV_32FC1);
    for (int row = ROI.y; row < ROI.y + ROI.height; ++row) {
        for (int col = ROI.x; col < ROI.x + ROI.width; ++col) {
            if (mask.at<uchar>(row, col) == 1) {
                float s = grad.at<float>(row, col);
                if (mask.at<uchar>(row - 1, col) == 0)
                    s -= dst.at<float>(row - 1, col);
                if (mask.at<uchar>(row + 1, col) == 0)
                    s -= dst.at<float>(row + 1, col);
                if (mask.at<uchar>(row, col - 1) == 0)
                    s -= dst.at<float>(row, col - 1);
                if (mask.at<uchar>(row, col + 1) == 0)
                    s -= dst.at<float>(row, col + 1);
                int i = idxMap[(row - ROI.y) * ROI.width + col - ROI.x];
                B.at<float>(i, 0) = s;
            }
        }
    }
    return B;
}

Mat getResult(SpMat &A, Mat &B, Rect &ROI) {
    Mat result;
    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
    cg.compute(A);
    Eigen::VectorXf eb(B.rows);
    cv2eigen(B, eb);

    Eigen::VectorXf x(B.rows);
    x = cg.solve(eb);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;
    eigen2cv(x, result);

    return result;
}

Mat poisson_blending(Mat &src, Mat &dst, Mat &mask, Rect ROI) {
    std::vector<double> idxMap(ROI.height * ROI.width);
    int N = 0;
    for (int row = ROI.y; row < ROI.y + ROI.height; ++row) {
        for (int col = ROI.x; col < ROI.x + ROI.width; ++col) {
            if (mask.at<uchar>(row, col) == 1) {
                idxMap[(row - ROI.y) * ROI.width + col - ROI.x] = N;
                ++N;
            }
        }
    }
    cout << "Starting to build A..." << endl;
    SpMat A = getA(mask, ROI, N, idxMap);
    cout << ">>> finished." << endl;
    std::vector<Mat> bgrSrc;
    split(src, bgrSrc);
    std::vector<Mat> bgrDst;
    split(dst, bgrDst);

    std::vector<Mat> result;
    Mat merged, res, Bb, Bg, Br;

    Bb = getB(bgrSrc[0], bgrDst[0], mask, ROI, N, idxMap);
    res = getResult(A, Bb, ROI);
    result.push_back(res);
    cout<<"B channel finished..."<<endl;
    
    Bg = getB(bgrSrc[1], bgrDst[1], mask, ROI, N, idxMap);
    res = getResult(A, Bg, ROI);
    result.push_back(res);
    cout<<"G channel finished..."<<endl;
    
    Br = getB(bgrSrc[2], bgrDst[2], mask, ROI, N, idxMap);
    res = getResult(A, Br, ROI);
    result.push_back(res);
    cout<<"R channel finished..."<<endl;

    merge(result, merged);
    cout << "Finished merging" << endl;

    Mat output;
    dst.copyTo(output);
    int curr_index = 0;
    for (int row = ROI.y; row < ROI.y + ROI.height; ++row) {
        for (int col = ROI.x; col < ROI.x + ROI.width; ++col) {
            if (mask.at<uchar>(row, col) == 1) {
                output.at<Vec3f>(row, col) = merged.at<Vec3f>(curr_index, 0);
                ++curr_index;
            }
        }
    }
    return output;
}

/* ===>>> End Poisson Blending Support Code <<<=== */

static std::vector<Point2f> getControlPoints(dlib::full_object_detection shape, Size size) {
    std::vector<Point2f> landmarks;
    // detect face and get the 68 landmarks
    for (int j = 0; j < shape.num_parts(); ++j) {
        dlib::point p = shape.part(j);
        landmarks.push_back(cv::Point2f(p.x(), p.y()));
    }
    // add landmarks on the edges of the image
    landmarks.push_back(Point2f(0, 0));
    landmarks.push_back(Point2f(0, 0.5 * size.height));
    landmarks.push_back(Point2f(0, size.height - 1));
    landmarks.push_back(Point2f(0.5 * size.width, 0));
    landmarks.push_back(Point2f(0.5 * size.width, size.height - 1));
    landmarks.push_back(Point2f(size.width - 1, 0));
    landmarks.push_back(Point2f(size.width - 1, 0.5 * size.height));
    landmarks.push_back(Point2f(size.width - 1, size.height - 1));
    return landmarks;
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            cout << "Call this program like this:" << endl;
            cout << "./face_morph <face_A.jpg> <face_B.jpg> <alpha value 0.0 - 1.0> <path_to_save>" << endl;
            cout << "<face_A.jpg>: image that the face from <face_B.jpg> will be morphed into" << endl;
            cout << "<face_B.jpg>: image whose face will be extracted and morph into <face_A.jpg>" << endl;
            cout << "<alpha>: controls how much the resulting composite looks like A or B";
            return 0;
        }
        float alpha = std::strtof(argv[3], NULL);

        dlib::frontal_face_detector detector = get_frontal_face_detector();
        dlib::shape_predictor sp;
        deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

        // Get control points on image 1
        cv::Mat img1 = imread(argv[1]);
        img1.convertTo(img1, CV_32F);
        Size imgSize = img1.size();
        dlib::array2d<rgb_pixel> img1_dlib;
        dlib::load_image(img1_dlib, argv[1]);
        dlib::rectangle bbox1 = detector(img1_dlib)[0];
        dlib::full_object_detection shape1 = sp(img1_dlib, bbox1);
        std::vector<cv::Point2f> ctrlPts1 = getControlPoints(shape1, imgSize);
        // Perform Delaunay triangulation on image 1
        cv::Rect rect(0, 0, imgSize.width, imgSize.height);
        cv::Subdiv2D subdiv(rect);
        for(std::vector<Point2f>::iterator it = ctrlPts1.begin(); it != ctrlPts1.end(); it++) {
            subdiv.insert(*it);
        }
        std::vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        // Store the triangles in terms of point indices... O(n^2)
        std::vector<Point2f> pts(3);
        std::vector<std::vector<int>> triangle_list_in_index;
        for (int i = 0; i < triangleList.size(); ++i) {
            Vec6f t = triangleList[i];
            std::vector<int> indices(3);
            pts[0] = Point2f(t[0], t[1]);
		    pts[1] = Point2f(t[2], t[3]);
		    pts[2] = Point2f(t[4], t[5]);
            if (rect.contains(pts[0]) && rect.contains(pts[1]) && rect.contains(pts[2])) {
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < ctrlPts1.size(); ++k)
                        if (abs(pts[j].x - ctrlPts1[k].x) < 1 && abs(pts[j].y - ctrlPts1[k].y) < 1)
                            indices[j] = k;
                triangle_list_in_index.push_back(indices);
            }
        }

        // Get control points on image 2
        cv::Mat img2 = imread(argv[2]);
        img2.convertTo(img2, CV_32F);
        dlib::array2d<rgb_pixel> img2_dlib;
        dlib::load_image(img2_dlib, argv[2]);
        dlib::rectangle bbox2 = detector(img2_dlib)[0];
        dlib::full_object_detection shape2 = sp(img2_dlib, bbox2);
        std::vector<cv::Point2f> ctrlPts2 = getControlPoints(shape2, imgSize);

        // Set up result image
        cv::Mat alphaBlend = cv::Mat::zeros(imgSize, CV_32FC3);
        cv::Mat warpedDst = cv::Mat::zeros(imgSize, CV_32FC3);
        std::vector<Point2f> resPts;

        // compute linearly interpolated control points in result image
        for (int i = 0; i < ctrlPts1.size(); ++i) {
            float x = (1 - alpha) * ctrlPts1[i].x + alpha * ctrlPts2[i].x;
            float y = (1 - alpha) * ctrlPts1[i].y + alpha * ctrlPts2[i].y;
            resPts.push_back(Point2f(x, y));
        }

        // get the morphed face
        for (std::vector<int> indices : triangle_list_in_index) {
            std::vector<Point2f> t1, t2, tRes;

            t1.push_back(ctrlPts1[indices[0]]);
            t1.push_back(ctrlPts1[indices[1]]);
            t1.push_back(ctrlPts1[indices[2]]);

            t2.push_back(ctrlPts2[indices[0]]);
            t2.push_back(ctrlPts2[indices[1]]);
            t2.push_back(ctrlPts2[indices[2]]);

            tRes.push_back(resPts[indices[0]]);
            tRes.push_back(resPts[indices[1]]);
            tRes.push_back(resPts[indices[2]]);

            morphTriangle(img1, img2, alphaBlend, t1, t2, tRes, alpha);
            warpTriangle(img1, warpedDst, t1, tRes);
        }
        
        // create face mask
        cv::Mat mask = cv::Mat::zeros(imgSize, CV_8UC1);
        std::vector<Point> face_outline_pts;
        for (int i = 0; i < 17; ++i) {
            int x = resPts[i].x;
            int y = resPts[i].y;
            if (resPts[28].x > x) x += 2;
            else if (resPts[28].x < x) x -= 2;
            if (resPts[28].y > y) y += 2;
            else if (resPts[28].y < y) y -= 2;
            face_outline_pts.push_back(Point2f(x, y));
        }
        for (int i = 26; i > 20; --i)
            face_outline_pts.push_back(resPts[i]);
        fillConvexPoly(mask, face_outline_pts, Scalar(1.0, 1.0, 1.0), 16, 0);
        std::vector<Point> temp;
        temp.push_back(face_outline_pts[0]);
        for (int i = 21; i > 16; --i)
            temp.push_back(resPts[i]);
        fillConvexPoly(mask, temp, Scalar(1.0, 1.0, 1.0), 16, 0);

        std::vector<Point2f> facePts;
        for (int i = 0; i < 68; ++i)
            facePts.push_back(resPts[i]);
        Rect bbox = boundingRect(facePts);
        Rect paddedBox(bbox.x - 1, bbox.y - 1, bbox.width + 2, bbox.height + 2);

        Mat result = poisson_blending(alphaBlend, warpedDst, mask, paddedBox);

        imwrite(argv[4], result);
        //imshow("Result", result / 255);
        //waitKey(0);

        return 0;
    }
    catch(exception& e) {
        cout << e.what() << endl;
    }
}

