/*
*   Copyright 2024 mmg. All rights reserved.
*   author: mmg(gmm782470390@163.com)
*   Created on: 2024-01-08
*/

#include "rt_detect_node.h"

rtDetrNode::rtDetrNode():
    init_(false),
    use_real_data_(false),
    private_nh_("~")
{
    private_nh_.param<string>("engine_path", engine_path_, "--default");
    private_nh_.param<string>("engine3dbox_path",engine3dbox_path_, "--default");
    private_nh_.param<string>("intristic", intristic_, "--default");
    private_nh_.param<string>("diminsion_param", diminsion_param_, "--default");
    private_nh_.param<double>("det_threshold", det_threshold_, 0.5);

    rtdetr_obj_ptr_.reset(new tensorrt_infer::rtdetr_cuda::RTDETRDetect());
    det3d_obj_ptr_.reset(new det3D::det3Dbox());
}

void rtDetrNode::run()
{  
    CHECK(cudaSetDevice(0)); 
    publisher_objects_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("detection/image_detector/objects", 1);
    publisher_3dobjects_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("detection/image_detector/objects3d", 1);
    publisher_imageRect_ = node_handle_.advertise<sensor_msgs::Image>("/image_rect", 1);
    publisher_3dimageRect_ = node_handle_.advertise<sensor_msgs::Image>("/image_3drect", 1);
    subscribe_image_raw_ = node_handle_.subscribe("/image_raw", 1, &rtDetrNode::image_callback, this);
}

void rtDetrNode::image_callback(const sensor_msgs::ImageConstPtr& in_image_message)
{
    if(!init_){
        rtdetr_obj_ptr_->initParameters(engine_path_, det_threshold_);
        det3d_obj_ptr_->init3dboxParameters(engine3dbox_path_, intristic_, diminsion_param_);
        init_ = true;
        image_seq = 1;
    }else{
        try{
            autoware_msgs::DetectedObjectArray out_message;
            autoware_msgs::DetectedObjectArray out_3dmessage;
            out_message.header = in_image_message->header;
            out_3dmessage.header = in_image_message->header;
            out_3dmessage.header.frame_id = "base_link";

            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(in_image_message, "bgr8");
            cv::Mat cropped_Image;
            if(use_real_data_){
                cropImage(in_image_message, cropped_Image);
            }else{
                cropped_Image = cv_ptr->image;
            }
            ai::cvUtil::Image input_image = ai::cvUtil::cvimg_trans_func(cropped_Image);
            ai::cvUtil::BoxArray det_result = rtdetr_obj_ptr_->forward(input_image);
            transfer2dTo3d(cropped_Image, in_image_message->header, det_result, out_message, out_3dmessage);

            publisher_objects_.publish(out_message);
            publisher_3dobjects_.publish(out_3dmessage);
            // image_seq++;
        }catch(cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }
}

void rtDetrNode::cropImage(const sensor_msgs::ImageConstPtr& msg, cv::Mat& cropped_image){
    try{
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        int img_height = cv_ptr->image.rows;
        int img_width = cv_ptr->image.cols;
        int crop_width = 640;
        int crop_height = 280;
        // std::cout << "image height & width: " << img_height << ", " << img_width << std::endl;

        int x_start = 0, y_start = 0;
        if(x_start >=0 && y_start >= 0 && x_start + crop_width <= img_width && y_start + crop_height <= img_height){
            cv::Rect crop_region(x_start, y_start, crop_width, crop_height);
            cropped_image = cv_ptr->image(crop_region);
        }else{
            ROS_ERROR("requested crop region is out of the image bunds");
        }
    }
    catch(cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void rtDetrNode::transfer2dTo3d(const cv::Mat &image, 
                                const std_msgs::Header& in_header,
                                const ai::cvUtil::BoxArray &boxes2d, 
                                autoware_msgs::DetectedObjectArray &out_message, 
                                autoware_msgs::DetectedObjectArray &out_3dmessage){
    Eigen::Vector3d inferLocation, inferDimension;
    double inferOrient;
    cv::Mat image_detr = image.clone();
    cv::Mat image_3dbox = image.clone();
    int id = 0;
    for(auto &box2d: boxes2d){
        id++;
        std::string name = classlabels_[box2d.class_label];
        auto it = std::find(recongnized_class_.begin(), recongnized_class_.end(), name); 
        if(it == recongnized_class_.end()) continue;

        autoware_msgs::DetectedObject object2d;
        autoware_msgs::DetectedObject object3d;
        object2d.header = in_header;
        object2d.valid = true;
        object2d.pose_reliable = true;
        object2d.id = id;
        object2d.width = abs(box2d.right - box2d.left);
        object2d.height = abs(box2d.top - box2d.bottom);
        object2d.x = box2d.left + 0.5*object2d.width;
        object2d.y = box2d.top + 0.5*object2d.height;
        object2d.score = box2d.confidence;
        object2d.label = name;
        object2d.valid = true;
        out_message.objects.push_back(object2d);

        // 2d detr
        uint8_t b, g, r;
        tie(b, g, r) = ai::utils::random_color(box2d.class_label);
        cv::rectangle(image_detr, cv::Point(box2d.left, box2d.top), 
                      cv::Point(box2d.right, box2d.bottom),cv::Scalar(b, g, r), 1);
        auto caption = cv::format("%s %.2f", name.c_str(), box2d.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::putText(image_detr, caption, cv::Point(box2d.left, box2d.top - 5), 0, 0.5, cv::Scalar(b, g, r), 1, 16);

        // 3d box infer
        std::vector<cv::Point2f> boxpoints = {{box2d.left, box2d.top}, {box2d.right, box2d.bottom}};
        det3d_obj_ptr_->infer3dbox(image_3dbox, boxpoints, name, 
                                   inferLocation, inferOrient, inferDimension);

        // std_msgs::Header tmp_header = in_header;
        // curImg_time = (ros::Time::now()).toSec();
        // if(img_count != 0){
        //     imgDiff += curImg_time - preImg_time;
        // }
        // preImg_time = curImg_time;
        // img_count = 1;
        // tmp_header.stamp = ros::Time().fromSec(imgDiff);

        object3d.header = in_header;
        object3d.header.frame_id = "base_link";
        object3d.valid = true;
        object3d.pose_reliable = true;
        object3d.id = id;

        if(use_real_data_){
            object3d.pose.position.x = inferLocation[2] + 2;
            object3d.pose.position.y = (-inferLocation[0]) - 7.9;
            object3d.pose.position.z = (-inferLocation[1]);
            geometry_msgs::Quaternion q_0 = tf::createQuaternionMsgFromYaw(-(0)); //camera transferTo base_link
            object3d.pose.orientation = q_0;
            object3d.angle = 0;
        }else{
            object3d.pose.position.x = inferLocation[2];
            object3d.pose.position.y = -inferLocation[0];
            object3d.pose.position.z = -inferLocation[1];
            geometry_msgs::Quaternion q_0 = tf::createQuaternionMsgFromYaw(-(inferOrient - M_PI / 2)); //camera transferTo base_link
            object3d.pose.orientation = q_0;
            object3d.angle = -(inferOrient - M_PI / 2);
        }
        object3d.dimensions.x = inferDimension[2];   //CNN output infer sequence is other not xyz are in camera_link 
        object3d.dimensions.y = inferDimension[0];
        object3d.dimensions.z = inferDimension[1];
        object3d.label = name;
        out_3dmessage.objects.push_back(object3d);
        // std::cout << "inferLocation: " <<  object3d.pose.position << std::endl;
        // std::cout << "inferOrient: " << object3d.angle << std::endl;     
    }
    sensor_msgs::ImagePtr output_3dimage = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_3dbox).toImageMsg();
    sensor_msgs::ImagePtr output_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_detr).toImageMsg();

    publisher_3dimageRect_.publish(output_3dimage);
    publisher_imageRect_.publish(output_image);
}