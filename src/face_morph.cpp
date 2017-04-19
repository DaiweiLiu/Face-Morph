#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

using namespace cv;
using namespace dlib;
using namespace std;

// Draw a single point
static void draw_point( Mat& img, Point2f fp, Scalar color )
{
    circle( img, fp, 2, color, CV_FILLED, CV_AA, 0 );
}
 
// Draw delaunay triangles
static void draw_delaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
{
 
    std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    std::vector<Point> pt(3);
    Size size = img.size();
    cv::Rect rect(0,0, size.width, size.height);
 
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
         
        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}

int main(int argc, char** argv) {
    try {
        dlib::frontal_face_detector detector = get_frontal_face_detector();
        dlib::shape_predictor sp;
        deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

        for (int i = 1; i < argc; ++i) {
            cout << "processing image " << argv[i] << endl;
            dlib::array2d<rgb_pixel> img;
            dlib::load_image(img, argv[i]);
            // pyramid_up(img);
            
            // detect face and get the 68 landmarks
            dlib::rectangle bbox = detector(img)[0];
            dlib::full_object_detection shape = sp(img, bbox);
            cout << ">>> number of parks: " << shape.num_parts() << endl;
            std::vector<cv::Point2f> landmarks;
            for (int j = 0; j < shape.num_parts(); ++j) {
                dlib::point p = shape.part(j);
                landmarks.push_back(cv::Point2f(p.x(), p.y()));
            }
            // add hard-code landmarks on forehead
            Point2f left_eye = (landmarks[37]+landmarks[38]+landmarks[40]+landmarks[41]) * 0.25;
            Point2f right_eye = (landmarks[43]+landmarks[44]+landmarks[46]+landmarks[47]) * 0.25;
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
            cv::Mat cimg = imread(argv[i]);
            Size size = cimg.size();
            landmarks.push_back(Point2f(0, 0));
            landmarks.push_back(Point2f(0, 0.5 * size.height));
            landmarks.push_back(Point2f(0, size.height - 1));
            landmarks.push_back(Point2f(0.5 * size.width, 0));
            landmarks.push_back(Point2f(0.5 * size.width, size.height - 1));
            landmarks.push_back(Point2f(size.width - 1, 0));
            landmarks.push_back(Point2f(size.width - 1, 0.5 * size.height));
            landmarks.push_back(Point2f(size.width - 1, size.height - 1));

            // Perform Delaunay triangulation
            cv::Rect rect(0, 0, size.width, size.height);
            cv::Subdiv2D subdiv(rect);
            for(std::vector<Point2f>::iterator it = landmarks.begin(); it != landmarks.end(); it++) {
                subdiv.insert(*it);
            }

            // Test draw
             Scalar delaunay_color(255,255,255);
             Scalar points_color(0, 0, 255);
             draw_delaunay( cimg, subdiv, delaunay_color );
             for( std::vector<Point2f>::iterator it = landmarks.begin(); it != landmarks.end(); it++) {
                 draw_point(cimg, *it, points_color);
             }
             imshow("Delaunay Triangulation", cimg);
             waitKey(0);

            
        }
        return 0;
    }
    catch(exception& e) {
        cout << e.what() << endl;
    }
}

