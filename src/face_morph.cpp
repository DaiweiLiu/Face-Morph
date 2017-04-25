#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

using namespace cv;
using namespace dlib;
using namespace std;

/* ============= The idea of using Affine and Warps comes from LearnOpenCv.com ============= */

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri) {
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform(srcTri, dstTri);
    // Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morph_triangle(Mat &img1, Mat &img2, Mat &img, std::vector<Point2f> &t1, std::vector<Point2f> &t2,
                    std::vector<Point2f> &t, float alpha) {

    // Find bounding rectangle/box for each triangle
    Rect r = boundingRect(t);
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by top left corner of the respective boxes
    std::vector<Point2f> t1_rect, t2_rect, t3_rect;
    std::vector<Point> t_rec_int;
    for (int i = 0; i < 3; i++) {
        t1_rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
        t2_rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
        t3_rect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));

        t_rec_int.push_back(Point((int) (t[i].x - r.x), (int) (t[i].y - r.y))); // Used later by fillConvexPoly
    }

    // Get mask by filling triangle
    Mat mask;
    mask = Mat::zeros(r.height, r.width, CV_32FC3);

    fillConvexPoly(mask, t_rec_int, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warp image to small rectangular patches
    Mat img1_rect, img2_rect;
    img1(r1).copyTo(img1_rect);
    img2(r2).copyTo(img2_rect);

    Mat warp_img1, warp_img2;
    warp_img1 = Mat::zeros(r.height, r.width, img1_rect.type());
    warp_img2 = Mat::zeros(r.height, r.width, img2_rect.type());

    applyAffineTransform(warp_img1, img1_rect, t1_rect, t3_rect);
    applyAffineTransform(warp_img2, img2_rect, t2_rect, t3_rect);

    // Alpha blend rectangular patches
    Mat img_rect;
    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2;

    // Copy triangular region of the rectangular patch to the output image
    multiply(img_rect, mask, img_rect);
    multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));
    img(r) = img(r) + img_rect;
}

/* ================================================================================== */

static std::vector<Point2f> get_control_points(dlib::full_object_detection shape, Size size) {
    std::vector<Point2f> landmarks;

    // detect face and get the 68 landmarks
    for (int j = 0; j < shape.num_parts(); ++j) {
        dlib::point p = shape.part(j);
        landmarks.push_back(Point2f(p.x(), p.y()));
    }
    // add hard-code landmarks on forehead TODO: optimization
    Point2f left_eye = (landmarks[37] + landmarks[38] + landmarks[40] + landmarks[41]) * 0.25;
    Point2f right_eye = (landmarks[43] + landmarks[44] + landmarks[46] + landmarks[47]) * 0.25;
    Point2f eye_mid = (left_eye + right_eye) * 0.5;
    Point2f offset = (eye_mid - landmarks[8]) * 0.61;
    Point2f forehead_mid = eye_mid + offset;
    Point2f forehead_left = forehead_mid - (right_eye - left_eye) * 0.79;
    Point2f forehead_right = forehead_mid + (right_eye - left_eye) * 0.79;
    landmarks.push_back(forehead_mid);
    landmarks.push_back(left_eye + offset);
    landmarks.push_back(right_eye + offset);
    landmarks.push_back(forehead_left);
    landmarks.push_back(forehead_right);
    landmarks.push_back(landmarks[0] + (forehead_left - landmarks[0]) * 0.5);
    landmarks.push_back(landmarks[16] + (forehead_right - landmarks[16]) * 0.5);
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

int main(int argc, char **argv) {
    try {
        if (argc != 5) {
            cout << "Call this program like this:" << endl;
            cout << "./face_morph <img1> <img2> <alpha> <img_save_path>" << endl;
            return 0;
        }
        /* arguments
         * argv[1]: <img1>
         * argv[2]: <img2>
         * argv[3]: <alpha>
         * argv[4]: <img_save_path>
         */

        float alpha = std::strtof(argv[3], NULL);

        // Use ML library to generate landmarks
        dlib::frontal_face_detector detector = get_frontal_face_detector();
        dlib::shape_predictor sp;
        deserialize("../static/shape_predictor_68_face_landmarks.dat") >> sp;

        // Get control points on image 1
        Mat img1 = imread(argv[1]);
        img1.convertTo(img1, CV_32F);
        dlib::array2d<rgb_pixel> img1_dlib;
        dlib::load_image(img1_dlib, argv[1]);
        dlib::rectangle bbox1 = detector(img1_dlib)[0];
        dlib::full_object_detection shape1 = sp(img1_dlib, bbox1);
        std::vector<Point2f> control_points1 = get_control_points(shape1, img1.size());
        // Perform Delaunay triangulation on image 1
        Rect rect(0, 0, img1.size().width, img1.size().height);
        Subdiv2D subdiv(rect);
        for (std::vector<Point2f>::iterator it = control_points1.begin(); it != control_points1.end(); it++) {
            subdiv.insert(*it);
        }
        std::vector<Vec6f> triangle_list;
        subdiv.getTriangleList(triangle_list);
        std::vector<Point2f> pts(3);
        std::vector<std::vector<int>> triangle_list_in_index; // Used later to store the triangles by point indices
        for (int i = 0; i < triangle_list.size(); i++) {
            Vec6f t = triangle_list[i];
            std::vector<int> indices(3);
            pts[0] = Point2f(t[0], t[1]);
            pts[1] = Point2f(t[2], t[3]);
            pts[2] = Point2f(t[4], t[5]);
            if (rect.contains(pts[0]) && rect.contains(pts[1]) && rect.contains(pts[2])) {
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < control_points1.size(); k++)
                        if (abs(pts[j].x - control_points1[k].x) < 1 && abs(pts[j].y - control_points1[k].y) < 1) {
                            indices[j] = k;
                        }
                triangle_list_in_index.push_back(indices);
            }
        }

        // Get control points on image 2
        Mat img2 = imread(argv[2]);
        img2.convertTo(img2, CV_32F);
        dlib::array2d<rgb_pixel> img2_dlib;
        dlib::load_image(img2_dlib, argv[2]);
        dlib::rectangle bbox2 = detector(img2_dlib)[0];
        dlib::full_object_detection shape2 = sp(img2_dlib, bbox2);
        std::vector<Point2f> control_points_2 = get_control_points(shape2, img2.size());

        // Set up result image
        Mat result_img;
        std::vector<Point2f> result_points;
        result_img = Mat::zeros(img2.size(), CV_32FC3);

        // compute linearly interpolated control points in result image
        for (int i = 0; i < control_points1.size(); ++i) {
            float x = (1 - alpha) * control_points1[i].x + alpha * control_points_2[i].x;
            float y = (1 - alpha) * control_points1[i].y + alpha * control_points_2[i].y;
            result_points.push_back(Point2f(x, y));
        }

        for (std::vector<int> indices : triangle_list_in_index) {
            std::vector<Point2f> t1, t2, t_res;
            t1.push_back(control_points1[indices[0]]);
            t1.push_back(control_points1[indices[1]]);
            t1.push_back(control_points1[indices[2]]);

            t2.push_back(control_points_2[indices[0]]);
            t2.push_back(control_points_2[indices[1]]);
            t2.push_back(control_points_2[indices[2]]);

            t_res.push_back(result_points[indices[0]]);
            t_res.push_back(result_points[indices[1]]);
            t_res.push_back(result_points[indices[2]]);

            morph_triangle(img1, img2, result_img, t1, t2, t_res, alpha);
        }

        imwrite(argv[4], result_img);
        //imshow("Morphing Result", result / 255.0);
        //waitKey(0);

        return 0;
    }
    catch (exception &e) {
        cout << e.what() << endl;
    }
}

