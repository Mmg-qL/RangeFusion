#ifndef DET3DBOX_H
#define DET3DBOX_H

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <ros/package.h>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include <logger.h>
#include <common.h>
#include <NvUffParser.h>       // 用于解析 UFF 模型的头文件
#include <NvUtils.h>           // 一些实用工具的头文件

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <cuda_runtime_api.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <unordered_map>


namespace det3D
{
    struct compareVectors{
        bool operator()(const std::vector<Eigen::Vector3d>& v1, const std::vector<Eigen::Vector3d>& v2) const{
            return std::equal(v1.begin(), v1.end(), v2.begin());
        }
    };

    class det3Dbox{
    private:
        bool init_;
        bool use_real_data_ = true;
        enum class cv_colors{GREEN, BLUE};
        cv::Mat camera_instrinsics_;
        cv::Mat dimension_instrinsics_;
        cv::Mat distance_;

        double fx_, fy_, cx_, cy_;
        double tx_, ty_, tz_;
        std::string trtModelName_;
        std::string initial_file_;
        std::string dimension_file_;
        std::string pkg_loc_;

        static const int OUT_SIZE_ORIENTATION = 1 * 2 * 2;   // Adjust according to your actual dimensions
        static const int OUT_SIZE_CONFIDENCE = 1 * 2;
        static const int OUT_SIZE_DIMENSION = 1 * 3;

        const char* IN_NAME = "input";
        const char* OUT_ORIENTATION = "orientation";
        const char* OUT_CONFIDENCE = "confidence";
        const char* OUT_DIMENSION = "dimension";

        static const int IN_H = 224;
        static const int IN_W = 224;
        static const int BATCH_SIZE = 1;

        nvinfer1::IRuntime* runtime_;
        nvinfer1::ICudaEngine* engine_;
        nvinfer1::IExecutionContext* context_;
        sample::Logger m_logger_;

        Eigen::Vector3f meanRGB_ = {0.485, 0.456, 0.406};
        Eigen::Vector3f stdRGB_ = {0.229, 0.224, 0.225};

        std::vector<double> angle_bins;
        Eigen::MatrixXd proj_matrix;
        Eigen::Matrix3d lidar2camera_matrix;
        Eigen::Matrix3d camera2lidar_matrix;

        char *trtModelStream{ nullptr };

        void drawLines(const cv::Mat& img, const std::vector<cv::Point2i>& points, const cv_colors color, const int &thickness);
        void calcThetaRay(const cv::Mat& img, const std::vector<cv::Point2f>& box2D, double& angle);

        std::vector<std::vector<double>> create_corners(const Eigen::Vector3d& dimension,
                                                        const Eigen::Matrix3d& R,
                                                        const Eigen::Vector3d& location);

        cv::Point2f ProjectPoint(const std::vector<double> &in_point);

        void calc_location(const Eigen::Vector3d& dimension,
                                        const std::vector<cv::Point2f>& box_2d,
                                        const double &alpha, const double &theta_ray,
                                        Eigen::Vector3d& locationOut); 

        void plot3dBox(const cv::Mat& img, 
                const double &yaw,
                const Eigen::Vector3d& dimension,
                const Eigen::Vector3d& center); 

        void plot_regressed_3d_bbox(const cv::Mat& img,
                                    const std::vector<cv::Point2f>& box2D,
                                    const Eigen::Vector3d& dimensions,
                                    const double &alpha,
                                    const double &theta_ray,
                                    Eigen::Vector3d& location,
                                    double &orient);

        void doInference(nvinfer1::IExecutionContext& context, 
                                float* input, 
                                float* orientation_output, 
                                float* confidence_output,
                                float* dimension_output,
                                int batchSize);

        std::vector<double> generateBins(const int &bins);

        void getItem(const float* dimension_in, 
                        const std::string &detectedClass,
                        Eigen::Vector3d& dimension_out);

        Eigen::Matrix3d rotation_matrix(const double &yaw);
        std::tuple<Eigen::Vector3d, Eigen::MatrixXd> lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    

    public:
        det3Dbox();
        ~det3Dbox();
        void init3dboxParameters(const std::string& eigen_path, 
                                           const std::string& initial_file, 
                                           const std::string& dimension_file); 
        void infer3dbox(const cv::Mat &image, 
                          const std::vector<cv::Point2f> &bundingBox2d, 
                          const std::string &detectedClass,
                          Eigen::Vector3d &location,
                          double &orient_angle,
                          Eigen::Vector3d &dimension);

        void formatImage(const cv::Mat& img, const std::vector<cv::Point2f>& box2D, cv::Mat& imgOut);  
        void normalImage2Blob(cv::Mat& img, float* blob); 
    };
}



#endif