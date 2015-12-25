#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace cv;
using namespace std;

map<string, int> conn;
int capLoops = 500000;
vector<int> capTimes;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            string temp = path;
            stringstream templiness(temp);
            string tempname;
            getline(templiness, tempname, '/');
            getline(templiness, tempname, '/');
            if(conn.find(tempname) == conn.end()){
                conn[tempname] = atoi(classlabel.c_str());

                images.push_back(imread(path, 0));
                labels.push_back(atoi(classlabel.c_str()));
            }
        }
    }
}



int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
//        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
//        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
//        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }
    if(argc == 3)
        capLoops = atoi(argv[2]);
    string fn_haar_face = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
    string fn_haar_eye = "/usr/share/opencv/haarcascades/haarcascade_eye.xml";
    string fn_csv = string(argv[1]);
    //int deviceId = atoi(argv[2]);
    int deviceId = 0;
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    // Get the height from the first image.
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    //Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

    model->train(images, labels);

    CascadeClassifier face_cascade;
    face_cascade.load(fn_haar_face);
    CascadeClassifier eye_cascade;
    eye_cascade.load(fn_haar_eye);
    VideoCapture cap(deviceId);
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }

    // init variables for taking attendance
    capTimes.resize(conn.size());
    for(int i = 0; i < conn.size(); i++)
        capTimes[i] = 0;

    // Holds the current frame from the Video device:
    Mat frame;
    for(int cl = 0; cl < capLoops; cl++) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        vector< Rect_<int> > faces;
        face_cascade.detectMultiScale(gray, faces);

        for(int i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);

            vector< Rect_<int> > eyes;
            eye_cascade.detectMultiScale(face, eyes);
            /*if(eyes.size() < 1)
                continue;
                */
            /*for( size_t j = 0; j < eyes.size(); j++){
                Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle(original, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
            }*/
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

            int prediction;
            double confidence;
            model->predict(face_resized, prediction, confidence);
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            string box_text;
            box_text = format( "" );
            //get our name from map
            if(prediction < conn.size())
                for(map<string, int>::iterator ii = conn.begin(); ii!= conn.end(); ii++){
                    if(prediction == (*ii).second){
                        //cout << "[" << cl << "] " << (*ii).second << ": " << (*ii).first <<  endl;
                        box_text.append((*ii).first);
                        capTimes[prediction]++;
                    }
                }
            else
                box_text.append( "Unknown" );
            ostringstream strs;
            strs << confidence;
            string str = strs.str();
            box_text.append(" ");
            box_text.append(str);
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }

        imshow("face_recognizer", original);
        char key = (char) waitKey(20);
        //key = (char) waitKey(20000);
        if(key == 27)
            break;
    }

    //out put the attendance results
    for(map<string, int>::iterator ii = conn.begin(); ii!= conn.end(); ii++){
        if(capTimes[(*ii).second] > capLoops*1/2)
            cout << (*ii).first << "'s here!" << endl;
        else
            cout << (*ii).first << "'s ABSENT!" << endl;
    }
    return 0;
}

