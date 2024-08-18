#ifndef RT_DETECT_NODE_H
#define RT_DETECT_NODE_H

#include <memory>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include "rtdetr_det_app/rtdetr_cuda/rtdetr_detect.hpp"
#include "cv_bridge/cv_bridge.h"

#include <opencv2/opencv.hpp>
#include "common/arg_parsing.hpp"
#include "common/cv_cpp_utils.hpp"
#include "rtdetr_det_app/rtdetr_cuda/rtdetr_detect.hpp"

#include <image_transport/image_transport.h>

#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>

#include "common/cv_cpp_utils.hpp"
#include "det3Dbox.h"

#include <tf/tf.h>
#include <tf/transform_listener.h>

class rtDetrNode{
private:
    ros::Subscriber subscribe_image_raw_;
    ros::Publisher publisher_objects_;
    ros::Publisher publisher_imageRect_;
    ros::Publisher publisher_3dimageRect_;
    ros::Publisher publisher_3dobjects_;
    ros::NodeHandle node_handle_;
    ros::NodeHandle private_nh_;
    bool init_;
    std::string engine_path_;
    double det_threshold_;

    std::string engine3dbox_path_;
    std::string intristic_;
    std::string diminsion_param_;

    bool use_real_data_;

    int img_count = 0;
    double curImg_time, preImg_time;
    double imgDiff = 0.0;

    std::unique_ptr<tensorrt_infer::rtdetr_cuda::RTDETRDetect> rtdetr_obj_ptr_;
    std::unique_ptr<det3D::det3Dbox> det3d_obj_ptr_;

    void image_callback(const sensor_msgs::ImageConstPtr& in_image_message);
    void transfer2dTo3d(const cv::Mat &image, 
                                const std_msgs::Header& in_header,
                                const ai::cvUtil::BoxArray &boxes2d, 
                                autoware_msgs::DetectedObjectArray &out_message, 
                                autoware_msgs::DetectedObjectArray &out_3dmessage);
    void cropImage(const sensor_msgs::ImageConstPtr& msg, cv::Mat& cropped_image);

    int image_seq;
    const std::vector<std::string> classlabels_{"pedestrian", "bicycle", "car",
                                            "motorcycle", "airplane", "bus",
                                            "train", "truck", "boat",
                                            "traffic light", "fire hydrant", "stop sign",
                                            "parking meter", "bench", "bird",
                                            "cat", "dog", "horse",
                                            "sheep", "cow", "elephant",
                                            "bear", "zebra", "giraffe",
                                            "backpack", "umbrella", "handbag",
                                            "tie", "suitcase", "frisbee",
                                            "skis", "snowboard", "sports ball",
                                            "kite", "baseball bat", "baseball glove",
                                            "skateboard", "surfboard", "tennis racket",
                                            "bottle", "wine glass", "cup",
                                            "fork", "knife", "spoon",
                                            "bowl", "banana", "apple",
                                            "sandwich", "orange", "broccoli",
                                            "carrot", "hot dog", "pizza",
                                            "donut", "cake", "chair",
                                            "couch", "potted plant", "bed",
                                            "dining table", "toilet", "tv",
                                            "laptop", "mouse", "remote",
                                            "keyboard", "cell phone", "microwave",
                                            "oven", "toaster", "sink",
                                            "refrigerator", "book", "clock",
                                            "vase", "scissors", "teddy bear",
                                            "hair drier", "toothbrush"};
    
    const std::vector<std::string> recongnized_class_{"car", "bus"};
    // const std::vector<std::string> recongnized_class_{"car", "bus", "pedestrian"};

public:
    void run();
    rtDetrNode();
};

#endif