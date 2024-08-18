/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************
 *  v1.0: amc-nu (abrahammonrroy@yahoo.com)
 *
 * range_vision_fusion_node.h
 *
 *  Created on: July, 05th, 2018
 */

#ifndef PROJECT_RANGE_VISION_FUSION_H
#define PROJECT_RANGE_VISION_FUSION_H

#define __APP_NAME__ "range_vision_fusion"

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

#include <ros/ros.h>
#include <ros/package.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Point.h>

#include <jsk_recognition_utils/geo/cube.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "autoware_msgs/DetectedObjectArray.h"

class ROSRangeVisionFusionApp
{
  struct Rectangle {
    double center_x, center_y;  // 中心点坐标
    double width, height;       // 宽度和高度
    double angle;               // 旋转角度
  };
  ros::NodeHandle node_handle_;
  ros::Publisher publisher_fused_objects_;
  ros::Publisher publisher_image_objects_;

  ros::Subscriber intrinsics_subscriber_;
  ros::Subscriber detections_vision_subscriber_;
  ros::Subscriber detections_range_subscriber_;

  message_filters::Subscriber<autoware_msgs::DetectedObjectArray>
  *range_filter_subscriber_, *vision3d_filter_subscriber_; //mmg 2024-01-22

  tf::TransformListener *transform_listener_;
  tf::StampedTransform camera_lidar_tf_;
  tf::Transform cameraTolidar_tf; //mmg 2023-12-05

  cv::Size image_size_;
  cv::Mat camera_instrinsics_;
  cv::Mat distortion_coefficients_;

  cv::Mat image_;
  ros::Subscriber image_subscriber_;

  void ImageCallback(const sensor_msgs::Image::ConstPtr &in_image_msg);

  autoware_msgs::DetectedObjectArray::ConstPtr vision_detections_, range_detections_;

  std::string image_frame_id_;
  std::string boxes_frame_;

  bool processing_;
  bool camera_info_ok_;
  bool camera_lidar_tf_ok_;

  bool use_real_data_;

  float fx_, fy_, cx_, cy_;
  double overlap_threshold_;
  double distance_threshold_;

  double car_width_, car_height_, car_depth_;
  double person_width_, person_height_, person_depth_;
  double truck_width_, truck_depth_, truck_height_;

  size_t empty_frames_;

  typedef
  message_filters::sync_policies::ApproximateTime<autoware_msgs::DetectedObjectArray,
    autoware_msgs::DetectedObjectArray> SyncPolicyT;


  ros::Subscriber range_objects_subscriber_;
  ros::Subscriber vision3d_objects_subscriber_; //mmg 2024-01-22

  message_filters::Synchronizer<SyncPolicyT> *detections_synchronizer_;

  void CheckMinimumDimensions(autoware_msgs::DetectedObject &in_out_object);

  double computeIoU(const Rectangle& rect1, const Rectangle& rect2);  //mmg 2024-01-24

  void RangeDetectionsCallback(const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections);

  void VisionDetectionsCallback(const autoware_msgs::DetectedObjectArray::ConstPtr &in_vision_3ddetections);

  void SyncedDetectionsCallback(const autoware_msgs::DetectedObjectArray::ConstPtr &in_vision_3ddetections,
                                const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections);

  // autoware_msgs::DetectedObjectArray
  // FuseRangeVisionDetections(const autoware_msgs::DetectedObjectArray::ConstPtr &in_vision_detections,
  //                           const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections);
  autoware_msgs::DetectedObjectArray FuseRangeVisionDetections(const autoware_msgs::DetectedObjectArray::ConstPtr &in_3dvision_detections,
                                                              const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections);

  // cv::Point3f TransformPoint(const geometry_msgs::Point &in_point, const tf::StampedTransform &in_transform);
  cv::Point3f TransformPoint(const geometry_msgs::Point &in_point, const tf::Transform &in_transform);

  cv::Point2i ProjectPoint(const cv::Point3f &in_point);

  cv::Rect ProjectDetectionToRect(const autoware_msgs::DetectedObject &in_detection);

  bool IsObjectInImage(const autoware_msgs::DetectedObject &in_detection);

  void TransformRangeToVision(const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections,
                              autoware_msgs::DetectedObjectArray &out_range_detections);
  void VisionFilter(const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections,
                    autoware_msgs::DetectedObjectArray &out_in_cv_range_detections);

  autoware_msgs::DetectedObject TransformObject(const autoware_msgs::DetectedObject &in_detection,
                                                const tf::StampedTransform &in_transform);

  autoware_msgs::DetectedObject MergeObjects(const autoware_msgs::DetectedObject &in_object_a,
                                             const autoware_msgs::DetectedObject &in_object_b);

  void CalculateObjectFeatures(autoware_msgs::DetectedObject &in_out_object,
                               bool in_estimate_pose);

  double GetDistanceToObject(const autoware_msgs::DetectedObject &in_object);
  double CalDistance(const autoware_msgs::DetectedObject &obj1,
                     const autoware_msgs::DetectedObject &obj2);

  void printXY(const autoware_msgs::DetectedObjectArray::ConstPtr& array1,
                                      const autoware_msgs::DetectedObjectArray::ConstPtr& array2);

  /*!
   * Obtains Transformation between two transforms registered in the TF Tree
   * @param in_target_frame
   * @param in_source_frame
   * @return the found transformation in the tree
   */
  tf::StampedTransform
  FindTransform(const std::string &in_target_frame, const std::string &in_source_frame);

  void IntrinsicsCallback(const sensor_msgs::CameraInfo &in_message);

  /*!
   * Reads the config params from the command line
   * @param in_private_handle
   */
  void InitializeROSIo(ros::NodeHandle &in_private_handle);

public:
  void Run();

  ROSRangeVisionFusionApp();
};


#endif //PROJECT_RANGE_VISION_FUSION_H
