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
 */


#include <imm_ukf_pda/imm_ukf_pda.h>
#include <imm_ukf_pda/lonlat2utm.h>


int frame_id = 0;

ImmUkfPda::ImmUkfPda()
  : target_id_(1)
  ,  // assign unique ukf_id_ to each tracking targets
  init_(false),
  imu_init_(false),
  InputMode_(1),
  use_real_data_(false),
  frame_count_(1),
  has_subscribed_vectormap_(false),
  private_nh_("~")
{
  private_nh_.param<std::string>("tracking_frame", tracking_frame_, "world");
  private_nh_.param<int>("life_time_threshold", life_time_threshold_, 8);
  private_nh_.param<double>("gating_threshold", gating_threshold_, 9.22);  //mmg 2023-12-12 9.22 -> 20
  private_nh_.param<double>("gate_probability", gate_probability_, 0.99);
  private_nh_.param<double>("detection_probability", detection_probability_, 0.9);
  private_nh_.param<double>("static_velocity_threshold", static_velocity_threshold_, 0.5);  //mmg 2023-12-14 0.5 ->0.3
  private_nh_.param<int>("static_num_history_threshold", static_num_history_threshold_, 3);
  private_nh_.param<double>("prevent_explosion_threshold", prevent_explosion_threshold_, 1000); //mmg 2023-12-09 1000->2000
  private_nh_.param<double>("merge_distance_threshold", merge_distance_threshold_, 0.5);  //mmg 2023-12-09 0.5->0.7
  private_nh_.param<bool>("use_sukf", use_sukf_, false);

  // for vectormap assisted tracking
  private_nh_.param<bool>("use_vectormap", use_vectormap_, false);
  private_nh_.param<double>("lane_direction_chi_threshold", lane_direction_chi_threshold_, 2.71);
  private_nh_.param<double>("nearest_lane_distance_threshold", nearest_lane_distance_threshold_, 1.0);
  private_nh_.param<std::string>("vectormap_frame", vectormap_frame_, "map");

  // rosparam for benchmark
  private_nh_.param<bool>("is_benchmark", is_benchmark_, true);
  private_nh_.param<std::string>("kitti_data_dir", kitti_data_dir_, "");
  if (is_benchmark_)
  {
    // result_file_path_ = "/home/gmm/file/githup_file/extra_label/res.txt";
    // result_file_path_ = kitti_data_dir_ + "benchmark_results.txt";
    result_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/imm_ukf_pda/benchmark_results.txt";
    // result_file_path_ = "/home/gmm/file/githup_file/py-motmetrics/yourdata/video1_imm_ukf.txt";
    // result_file_path_ = "/home/gmm/file/githup_file/extra_label/video1_imm_ukf.txt";
    std::remove(result_file_path_.c_str());
  }
}

void ImmUkfPda::run()
{
  if(InputMode_ == InputSelect::RangeFusion)  real_object_subscriber_ = node_handle_.subscribe("/detection/fusion_tools/objects", 1, &ImmUkfPda::realcallback, this);
  else if(InputMode_ == InputSelect::Pointpillars) real_object_subscriber_ = node_handle_.subscribe("/detection/lidar_detector/objects", 1, &ImmUkfPda::realcallback, this);
  else if(InputMode_ == InputSelect::Bundingbox3d) real_object_subscriber_ = node_handle_.subscribe("/detection/image_detector/objects3d", 1, &ImmUkfPda::realcallback, this);
  pub_object_array_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("/detection/objects", 1);
}


void ImmUkfPda::realcallback(const autoware_msgs::DetectedObjectArray::ConstPtr& input) 
{
    input_header_ = input->header;  //mmg 2023-12-18
    std::string target_num_path = "/home/gmm/file/githup_file/catkin_autoware/result/target_num/kitti_detect_num.txt";
    std::ofstream numoutfile(target_num_path, std::ofstream::out | std::ofstream::app);
    numoutfile << input_header_.stamp << "," << input->objects.size() << std::endl;
    bool success = updateNecessaryTransform();    //coordinate tranform
    if (!success)
    {
      ROS_INFO("Could not find coordiante transformation");
      return;
    }

    autoware_msgs::DetectedObjectArray transformed_input;
    autoware_msgs::DetectedObjectArray detected_objects_output;
    transformPoseToGlobal(input, transformed_input);  //tranform coordinate
    tracker(transformed_input, detected_objects_output); 
    transformPoseToLocal(detected_objects_output);

    pub_object_array_.publish(detected_objects_output);

    if (is_benchmark_)
    {
      dumpResultText(detected_objects_output);
    }
}

void ImmUkfPda::callback(const autoware_msgs::DetectedObjectArray::ConstPtr& input, 
                         const sensor_msgs::NavSatFix::ConstPtr& gps_msg, 
                         const sensor_msgs::Imu::ConstPtr& imu_msg) //mmg 2023-12-25
{ 
    input_header_ = input->header;  //mmg 2023-12-18
    if(!use_real_data_) gpsCallback(gps_msg, imu_msg);

    bool success = updateNecessaryTransform();    //coordinate tranform
    if (!success)
    {
      ROS_INFO("Could not find coordiante transformation");
      return;
    }

    autoware_msgs::DetectedObjectArray transformed_input;
    autoware_msgs::DetectedObjectArray detected_objects_output;
    transformPoseToGlobal(input, transformed_input);  //tranform coordinate
    tracker(transformed_input, detected_objects_output); 
    // transformPoseToLocal(detected_objects_output);

    pub_object_array_.publish(detected_objects_output);

    if (is_benchmark_)
    {
      dumpResultText(detected_objects_output);
    }
}

//mmg 2023-12-24
void ImmUkfPda::gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& gps_msg, const sensor_msgs::Imu::ConstPtr& imu_msg)
{
    std::string gps_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/gps_file/001.txt";
    std::ofstream gpsoutfile(gps_file_path_, std::ofstream::out | std::ofstream::app);
    double heading;

    if(!gps_init){
      init_latitude = gps_msg->latitude;
      init_longitude = gps_msg->longitude;
      init_altitude = gps_msg->altitude;

      heading = tf::getYaw(imu_msg->orientation); //mmg 2023-12-28
      Eigen::AngleAxisd roto(heading, Eigen::Vector3d::UnitZ());
      porigion_ = roto.toRotationMatrix();
      porigion_.translation() = Eigen::Vector3d(0, 0, 0);

      LonLat2UTM(init_longitude, init_latitude, init_x, init_y);
      gps_init = true;
    }
    else{
      // process the gps data
      double latitude = gps_msg->latitude;
      double longitude = gps_msg->longitude;

      // std::cout << "latitude,longitude,altitude: " << latitude << "," << longitude << std::endl;   
      double position_x, position_y;
      LonLat2UTM(longitude, latitude, position_x, position_y);

      vehicle_x = position_x - init_x;
      vehicle_y = position_y - init_y;
      vehicle_z = gps_msg->altitude - init_altitude;

      heading = tf::getYaw(imu_msg->orientation); //mmg 2023-12-28
      Eigen::AngleAxisd rotnow(heading, Eigen::Vector3d::UnitZ());
      Eigen::Matrix3d rotpnow = rotnow.toRotationMatrix();
      Eigen::Isometry3d p2;
      p2 = rotpnow;
      p2.translation() = Eigen::Vector3d(vehicle_x, vehicle_y, 0);
      translate2origin_ = porigion_.inverse() * p2;
      origin2translate_ = p2.inverse() * porigion_;

      Eigen::Vector3d  p_0, p_1;
      p_0 << 0, 0, 0;
      p_1 = translate2origin_ * p_0;
      gpsoutfile << gps_msg->header.stamp << "," << p_1[0] << "," << p_1[1]  << "," << p_1[2] << "\n";

      Eigen::Isometry3d p3;
      p3 = rotpnow;
      p3.translation() = Eigen::Vector3d(vehicle_x, vehicle_y, 0);

      vehicleHeading_ = heading;
      //mmg 2023-12-28
      tf::Transform lidar2world_transform;
      lidar2world_transform.setOrigin(tf::Vector3(vehicle_x, vehicle_y, vehicle_z));
      double qx = (imu_msg->orientation).x;
      double qy = (imu_msg->orientation).y;
      double qz = (imu_msg->orientation).z;
      double qw = (imu_msg->orientation).w;
      lidar2world_transform.setRotation(tf::Quaternion(qx, qy, qz, qw));
      broadcaster.sendTransform(tf::StampedTransform(lidar2world_transform, ros::Time::now(), "global_frame", "local_frame"));
    }
}

void ImmUkfPda::checkVectormapSubscription()
{
  if (use_vectormap_ && !has_subscribed_vectormap_)
  {
    lanes_ = vmap_.findByFilter([](const vector_map_msgs::Lane& lane) { return true; });
    if (lanes_.empty())
    {
      ROS_INFO("Has not subscribed vectormap");
    }
    else
    {
      has_subscribed_vectormap_ = true;
    }
  }
}

bool ImmUkfPda::updateNecessaryTransform()
{
  bool success = true;
  try
  {
    //wait for transform
    // tf_listener_.waitForTransform(input_header_.frame_id, tracking_frame_, ros::Time(0), ros::Duration(1.0));
    // tf_listener_.lookupTransform(tracking_frame_, input_header_.frame_id, ros::Time(0), local2global_);
    if(use_real_data_){
      tf_listener_.waitForTransform("global_frame", "local_frame", ros::Time(0), ros::Duration(1.0));
      tf_listener_.lookupTransform("global_frame", "local_frame", ros::Time(0), velo_to_global_); //local to global
    }
    else{
      tf_listener_.waitForTransform("world", "base_link", ros::Time(0), ros::Duration(1.0));
      tf_listener_.lookupTransform("world", "base_link", ros::Time(0), velo_to_global_);
    }
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    success = false;
  }
  return success;
}

//mmg 2023-12-18
void ImmUkfPda::imuInitialize(){
    last_imu_time_ = ros::Time(1317625114.098846101);
    current_velocity_.x = 16.7;
    current_velocity_.y = 0;

    current_position_.x = 0.0;
    current_position_.y = 0.0;
    imu_init_ = true;
}


void ImmUkfPda::imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg) {
    // std::ofstream outputfile(result_file_path_, std::ofstream::out | std::ofstream::app);

    if(!imu_init_){
      imuInitialize();
    }

    ros::Time current_time = imu_msg->header.stamp;

    // Calculate time difference
    double dt = (current_time - last_imu_time_).toSec();

    // Integrate linear acceleration to get velocity
    current_velocity_.x += imu_msg->linear_acceleration.x * dt;
    current_velocity_.y += imu_msg->linear_acceleration.y * dt;

    // Integrate velocity to get position
    current_position_.x += current_velocity_.x * dt;
    current_position_.y += current_velocity_.y * dt;

    // outputfile << last_imu_time_ << "," << current_position_.x << "," << current_position_.y << "\n";
    std::cout << "current position: (x,y): " << current_position_.x << "," << current_position_.y << std::endl;
              
    // Update last IMU time
    last_imu_time_ = current_time;
}

//mmg 2023-12-26
void ImmUkfPda::transformPoseToGlobal(const autoware_msgs::DetectedObjectArray::ConstPtr& input,
                                      autoware_msgs::DetectedObjectArray& transformed_input)
{
  /* //mmg 2023-12-26
  transformed_input.header = input_header_;
  for (auto const &object: input.objects)
  {
    geometry_msgs::Pose out_pose = getTransformedPose(object.pose, local2global_);

    autoware_msgs::DetectedObject dd;
    dd.header = input.header;
    dd = object;
    dd.pose = out_pose;

    transformed_input.objects.push_back(dd);
  }*/
  transformed_input.header = input->header;

  for (auto const &object: input->objects)
  {
    geometry_msgs::Pose out_pose = getTransformedPose(object.pose, velo_to_global_);

    autoware_msgs::DetectedObject dd;
    dd.header = input->header;
    dd.header.stamp = velo_to_global_.stamp_;
    dd = object;
    dd.pose = out_pose;
    dd.pose.position.x = out_pose.position.x;
    dd.pose.position.y = out_pose.position.y;

    tf::Quaternion rotation = velo_to_global_.getRotation();
    tf::Matrix3x3 rotationMatrix(rotation);
    double roll, pitch;
    rotationMatrix.getRPY(roll, pitch, car_yaw);

    dd.angle = object.angle + car_yaw;
    transformed_input.objects.push_back(dd);
  }
}

void ImmUkfPda::transformPoseToLocal(autoware_msgs::DetectedObjectArray& detected_objects_output)
{
  /*
  tf::Transform inv_local2global = local2global_.inverse();
  tf::StampedTransform global2local;
  global2local.setData(inv_local2global);
  for (auto& object : detected_objects_output.objects)
  {
    geometry_msgs::Pose out_pose = getTransformedPose(object.pose, global2local);
    object.header = input_header_;
    object.pose = out_pose;
  }
  */
  detected_objects_output.header = input_header_;
  tf::Transform inv_velo_to_global_ = velo_to_global_.inverse();
  tf::StampedTransform global2velo;
  global2velo.setData(inv_velo_to_global_);
  for (auto& object : detected_objects_output.objects)
  {
    geometry_msgs::Pose out_pose = getTransformedPose(object.pose, global2velo);
    object.header = input_header_;
    object.pose = out_pose;
    object.pose.position.x = out_pose.position.x;
    object.pose.position.y = out_pose.position.y;
    object.angle = object.angle - car_yaw;
    object.pose.orientation = tf::createQuaternionMsgFromYaw((object.angle));
  }

}

geometry_msgs::Pose ImmUkfPda::getTransformedPose(const geometry_msgs::Pose& in_pose,
                                                  const tf::StampedTransform& tf_stamp)
{
  tf::Transform transform;
  geometry_msgs::PoseStamped out_pose;
  transform.setOrigin(tf::Vector3(in_pose.position.x, in_pose.position.y, in_pose.position.z));
  transform.setRotation(tf::Quaternion(in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w));
  geometry_msgs::PoseStamped pose_out;
  tf::poseTFToMsg(tf_stamp * transform, out_pose.pose);
  return out_pose.pose;
}

void ImmUkfPda::measurementValidation(const autoware_msgs::DetectedObjectArray& input, UKF& target,
                                      const bool second_init, const Eigen::VectorXd& max_det_z,
                                      const Eigen::MatrixXd& max_det_s,
                                      std::vector<autoware_msgs::DetectedObject>& object_vec,
                                      std::vector<bool>& matching_vec)
{
  // alert: different from original imm-pda filter, here picking up most likely measurement
  // if making it allows to have more than one measurement, you will see non semipositive definite covariance
  bool exists_smallest_nis_object = false;
  double smallest_nis = std::numeric_limits<double>::max();
  int smallest_nis_ind = 0;
  for (size_t i = 0; i < input.objects.size(); i++)
  {
    double x = input.objects[i].pose.position.x;
    double y = input.objects[i].pose.position.y;

    Eigen::VectorXd meas = Eigen::VectorXd(2);
    meas << x, y;
    //max_det_z present prediction of z, meas present mesurement
    Eigen::VectorXd diff = meas - max_det_z;
    double nis = diff.transpose() * max_det_s.inverse() * diff;
    // std::cout << "nis: " << nis << std::endl;

    if (nis < gating_threshold_)
    {
      if (nis < smallest_nis)
      {
        smallest_nis = nis;
        target.object_ = input.objects[i];
        smallest_nis_ind = i;
        exists_smallest_nis_object = true;
      }
    }
  }
  if (exists_smallest_nis_object)
  {
    matching_vec[smallest_nis_ind] = true;
    if (use_vectormap_ && has_subscribed_vectormap_)
    {
      autoware_msgs::DetectedObject direction_updated_object;
      bool use_direction_meas = updateDirection(smallest_nis, target.object_, direction_updated_object, target);
      if (use_direction_meas)
      {
        object_vec.push_back(direction_updated_object);
      }
      else
      {
        object_vec.push_back(target.object_);
      }
    }
    else
    {
      object_vec.push_back(target.object_); //object_vec存储一步预测结果
    }
  }
}

bool ImmUkfPda::updateDirection(const double smallest_nis, const autoware_msgs::DetectedObject& in_object,
                                    autoware_msgs::DetectedObject& out_object, UKF& target)
{
  bool use_lane_direction = false;
  target.is_direction_cv_available_ = false;
  target.is_direction_ctrv_available_ = false;
  bool get_lane_success = storeObjectWithNearestLaneDirection(in_object, out_object);
  if (!get_lane_success)
  {
    return use_lane_direction;
  }
  target.checkLaneDirectionAvailability(out_object, lane_direction_chi_threshold_, use_sukf_);
  if (target.is_direction_cv_available_ || target.is_direction_ctrv_available_)
  {
    use_lane_direction = true;
  }
  return use_lane_direction;
}

bool ImmUkfPda::storeObjectWithNearestLaneDirection(const autoware_msgs::DetectedObject& in_object,
                                                 autoware_msgs::DetectedObject& out_object)
{
  geometry_msgs::Pose lane_frame_pose = getTransformedPose(in_object.pose, tracking_frame2lane_frame_);
  double min_dist = std::numeric_limits<double>::max();

  double min_yaw = 0;
  for (auto const& lane : lanes_)
  {
    vector_map_msgs::Node node = vmap_.findByKey(vector_map::Key<vector_map_msgs::Node>(lane.bnid));
    vector_map_msgs::Point point = vmap_.findByKey(vector_map::Key<vector_map_msgs::Point>(node.pid));
    double distance = std::sqrt(std::pow(point.bx - lane_frame_pose.position.y, 2) +
                                std::pow(point.ly - lane_frame_pose.position.x, 2));
    if (distance < min_dist)
    {
      min_dist = distance;
      vector_map_msgs::Node front_node = vmap_.findByKey(vector_map::Key<vector_map_msgs::Node>(lane.fnid));
      vector_map_msgs::Point front_point = vmap_.findByKey(vector_map::Key<vector_map_msgs::Point>(front_node.pid));
      min_yaw = std::atan2((front_point.bx - point.bx), (front_point.ly - point.ly));
    }
  }

  bool success = false;
  if (min_dist < nearest_lane_distance_threshold_)
  {
    success = true;
  }
  else
  {
    return success;
  }

  // map yaw in rotation matrix representation
  tf::Quaternion map_quat = tf::createQuaternionFromYaw(min_yaw);
  tf::Matrix3x3 map_matrix(map_quat);

  // vectormap_frame to tracking_frame rotation matrix
  tf::Quaternion rotation_quat = lane_frame2tracking_frame_.getRotation();
  tf::Matrix3x3 rotation_matrix(rotation_quat);

  // rotated yaw in matrix representation
  tf::Matrix3x3 rotated_matrix = rotation_matrix * map_matrix;
  double roll, pitch, yaw;
  rotated_matrix.getRPY(roll, pitch, yaw);

  out_object = in_object;
  out_object.angle = yaw;
  return success;
}

void ImmUkfPda::updateTargetWithAssociatedObject(const std::vector<autoware_msgs::DetectedObject>& object_vec,
                                                 UKF& target)
{
  target.lifetime_++;
  if (!target.object_.label.empty() && target.object_.label !="unknown")
  {
    target.label_ = target.object_.label;
  }
  updateTrackingNum(object_vec, target);
  if (target.tracking_num_ == TrackingState::Stable || target.tracking_num_ == TrackingState::Occlusion)
  {
    target.is_stable_ = true;
  }
}

void ImmUkfPda::updateBehaviorState(const UKF& target, const bool use_sukf, autoware_msgs::DetectedObject& object)
{
  if(use_sukf)
  {
    object.behavior_state = MotionModel::CTRV;
  }
  else if (target.mode_prob_cv_ > target.mode_prob_ctrv_ && target.mode_prob_cv_ > target.mode_prob_rm_)
  {
    object.behavior_state = MotionModel::CV;
  }
  else if (target.mode_prob_ctrv_ > target.mode_prob_cv_ && target.mode_prob_ctrv_ > target.mode_prob_rm_)
  {
    object.behavior_state = MotionModel::CTRV;
  }
  else
  {
    object.behavior_state = MotionModel::RM;
  }
}

void ImmUkfPda::initTracker(const autoware_msgs::DetectedObjectArray& input, double timestamp)
{
  for (size_t i = 0; i < input.objects.size(); i++)
  {
    double px = input.objects[i].pose.position.x; //mmg 2023-12-18 add imu position
    double py = input.objects[i].pose.position.y; //mmg 2023-12-18 add imu position
    //mmg 2023-12-11
    double pyaw = input.objects[i].angle;
    Eigen::VectorXd init_meas = Eigen::VectorXd(2);
    init_meas << px, py;

    UKF ukf;
    ukf.initialize(init_meas, timestamp, target_id_, pyaw); //mmg 2023-12-11
    // ukf.initialize(init_meas, timestamp, target_id_);
    targets_.push_back(ukf);
    target_id_++;
  }
  timestamp_ = timestamp;
  init_ = true;
}

void ImmUkfPda::secondInit(UKF& target, const std::vector<autoware_msgs::DetectedObject>& object_vec, double dt)
{
  if (object_vec.size() == 0)
  {
    target.tracking_num_ = TrackingState::Die;
    return;
  }
  // record init measurement for env classification
  target.init_meas_ << target.x_merge_(0), target.x_merge_(1);

  // state update
  double target_x = object_vec[0].pose.position.x;
  double target_y = object_vec[0].pose.position.y;
  double target_diff_x = target_x - target.x_merge_(0);
  double target_diff_y = target_y - target.x_merge_(1);
  double target_yaw = atan2(target_diff_y, target_diff_x);
  // std::cout << "second init target_yaw: " << target_yaw << std::endl;
  // double target_yaw = tf::getYaw(object_vec[0].pose.orientation);  //mmg 2023-12-11
  // double target_yaw = object_vec[0].angle;
  double dist = sqrt(target_diff_x * target_diff_x + target_diff_y * target_diff_y);
  double target_v = dist / dt;

  while (target_yaw > M_PI)
    target_yaw -= 2. * M_PI;
  while (target_yaw < -M_PI)
    target_yaw += 2. * M_PI;

  target.x_merge_(0) = target.x_cv_(0) = target.x_ctrv_(0) = target.x_rm_(0) = target_x;
  target.x_merge_(1) = target.x_cv_(1) = target.x_ctrv_(1) = target.x_rm_(1) = target_y;
  target.x_merge_(2) = target.x_cv_(2) = target.x_ctrv_(2) = target.x_rm_(2) = target_v;
  target.x_merge_(3) = target.x_cv_(3) = target.x_ctrv_(3) = target.x_rm_(3) = target_yaw;

  target.tracking_num_++;
  return;
}

void ImmUkfPda::updateTrackingNum(const std::vector<autoware_msgs::DetectedObject>& object_vec, UKF& target)
{
  if (object_vec.size() > 0)
  {
    if (target.tracking_num_ < TrackingState::Stable)
    {
      target.tracking_num_++;
    }
    else if (target.tracking_num_ == TrackingState::Stable)
    {
      target.tracking_num_ = TrackingState::Stable;
    }
    else if (target.tracking_num_ >= TrackingState::Stable && target.tracking_num_ < TrackingState::Lost)
    {
      target.tracking_num_ = TrackingState::Stable;
    }
    else if (target.tracking_num_ == TrackingState::Lost)
    {
      target.tracking_num_ = TrackingState::Die;
    }
  }
  else
  {
    if (target.tracking_num_ < TrackingState::Stable)
    {
      target.tracking_num_ = TrackingState::Die;
    }
    else if (target.tracking_num_ >= TrackingState::Stable && target.tracking_num_ < TrackingState::Lost)
    {
      target.tracking_num_++;
    }
    else if (target.tracking_num_ == TrackingState::Lost)
    {
      target.tracking_num_ = TrackingState::Die;
    }
  }

  return;
}

bool ImmUkfPda::probabilisticDataAssociation(const autoware_msgs::DetectedObjectArray& input, const double dt,
                                             std::vector<bool>& matching_vec,
                                             std::vector<autoware_msgs::DetectedObject>& object_vec, UKF& target)
{
  double det_s = 0;
  Eigen::VectorXd max_det_z;
  Eigen::MatrixXd max_det_s;
  bool success = true;

  if (use_sukf_)
  {
    max_det_z = target.z_pred_ctrv_;  //预测的车辆状态
    max_det_s = target.s_ctrv_; //预测的协方差
    det_s = max_det_s.determinant();
  }
  else
  {
    // find maxDetS associated with predZ
    target.findMaxZandS(max_det_z, max_det_s);
    det_s = max_det_s.determinant();
  }
  // std::cout << "targets tracking_num_: "<< target.tracking_num_ << ", det_s: " << det_s << std::endl;
  //mmg 2023-12-27
  // prevent ukf not to explode
  // if (std::isnan(det_s) || det_s > prevent_explosion_threshold_)
  if (std::isnan(det_s))
  {
    target.tracking_num_ = TrackingState::Die;
    success = false;
    return success;
  }

  bool is_second_init;
  if (target.tracking_num_ == TrackingState::Init)
  {
    is_second_init = true;
  }
  else
  {
    is_second_init = false;
  }

  // measurement gating
  measurementValidation(input, target, is_second_init, max_det_z, max_det_s, object_vec, matching_vec);   //与预测对比，筛选符合条件的一步target里面的量测
  // std::cout << "after measurementValidation object_vec: " << object_vec.size() << std::endl;

  // std::cout << "before second init target.tracking_num_: " << target.tracking_num_ << std::endl;
  // second detection for a target: update v and yaw
  if (is_second_init)
  {
    //对于通过pda筛选的一步量测进行更新
    secondInit(target, object_vec, dt);
    // std::cout << "after second init target.tracking_num_: " << target.tracking_num_ << std::endl;
    success = false;
    return success;
  }
  updateTargetWithAssociatedObject(object_vec, target);
  // std::cout << "after updateTargetWithAssociatedObject: " << object_vec.size() << std::endl;

  if (target.tracking_num_ == TrackingState::Die)
  {
    success = false;
    return success;
  }
  return success;
}

void ImmUkfPda::makeNewTargets(const double timestamp, const autoware_msgs::DetectedObjectArray& input,
                               const std::vector<bool>& matching_vec)
{
  for (size_t i = 0; i < input.objects.size(); i++)
  {
    if (matching_vec[i] == false)
    {
      double px = input.objects[i].pose.position.x; //mmg 2023-12-18
      double py = input.objects[i].pose.position.y;
      double pyaw = input.objects[i].angle;
      Eigen::VectorXd init_meas = Eigen::VectorXd(2);
      init_meas << px, py;

      UKF ukf;
      ukf.initialize(init_meas, timestamp, target_id_, pyaw); //位置初始化
      // ukf.initialize(init_meas, timestamp, target_id_);  //mmg 2023-12-12
      ukf.object_ = input.objects[i];   //检测框都存入
      targets_.push_back(ukf);
      target_id_++;
    }
  }
}

void ImmUkfPda::staticClassification()
{
  for (size_t i = 0; i < targets_.size(); i++)
  {
    // targets_[i].x_merge_(2) is referred for estimated velocity
    double current_velocity = std::abs(targets_[i].x_merge_(2));
    targets_[i].vel_history_.push_back(current_velocity);
    if (targets_[i].tracking_num_ == TrackingState::Stable && targets_[i].lifetime_ > life_time_threshold_)
    {
      int index = 0;
      double sum_vel = 0;
      double avg_vel = 0;
      for (auto rit = targets_[i].vel_history_.rbegin(); index < static_num_history_threshold_; ++rit)
      {
        index++;
        sum_vel += *rit;
      }
      avg_vel = double(sum_vel / static_num_history_threshold_);

      if(avg_vel < static_velocity_threshold_ && current_velocity < static_velocity_threshold_)
      {
        targets_[i].is_static_ = true;
      }
    }
  }
}

bool
ImmUkfPda::arePointsClose(const geometry_msgs::Point& in_point_a,
                                const geometry_msgs::Point& in_point_b,
                                float in_radius)
{
  return (fabs(in_point_a.x - in_point_b.x) <= in_radius) && (fabs(in_point_a.y - in_point_b.y) <= in_radius);
}

bool
ImmUkfPda::arePointsEqual(const geometry_msgs::Point& in_point_a,
                               const geometry_msgs::Point& in_point_b)
{
  return arePointsClose(in_point_a, in_point_b, CENTROID_DISTANCE);
}

bool
ImmUkfPda::isPointInPool(const std::vector<geometry_msgs::Point>& in_pool,
                          const geometry_msgs::Point& in_point)
{
  for(size_t j = 0; j < in_pool.size(); j++)
  {
    if (arePointsEqual(in_pool[j], in_point))
    {
      return true;
    }
  }
  return false;
}

autoware_msgs::DetectedObjectArray
ImmUkfPda::removeRedundantObjects(const autoware_msgs::DetectedObjectArray& in_detected_objects,
                                  const std::vector<size_t> in_tracker_indices)
{
  if (in_detected_objects.objects.size() != in_tracker_indices.size())
    return in_detected_objects;
  // std::cout << "in removeRedundantObjects target.size(): " << targets_.size() << std::endl;
  autoware_msgs::DetectedObjectArray resulting_objects;
  resulting_objects.header = in_detected_objects.header;

  std::vector<geometry_msgs::Point> centroids;
  //create unique points
  for(size_t i = 0; i < in_detected_objects.objects.size(); i++)
  {
    //object in distance judge
    if(!isPointInPool(centroids, in_detected_objects.objects[i].pose.position))
    {
      centroids.push_back(in_detected_objects.objects[i].pose.position);
    }
  }
  //find each object match track
  std::vector<std::vector<size_t>> matching_objects(centroids.size());
  for(size_t k = 0; k < in_detected_objects.objects.size(); k++)
  {
    const auto& object = in_detected_objects.objects[k];
    for(size_t i = 0; i < centroids.size(); i++)
    {
      if (arePointsClose(object.pose.position, centroids[i], merge_distance_threshold_))
      {
        matching_objects[i].push_back(k);//store index of matched object to this point
        // std::cout << "(i,k): " << i << "," << k << std::endl;
      }
    }
  }

  //get oldest object on each point
  for(size_t i = 0; i < matching_objects.size(); i++)
  {
    size_t oldest_object_index = 0;
    int oldest_lifespan = -1;
    std::string best_label;
    for(size_t j = 0; j < matching_objects[i].size(); j++)
    {
      size_t current_index = matching_objects[i][j];
      int current_lifespan = targets_[in_tracker_indices[current_index]].lifetime_;
      //find the oldest life_time
      if (current_lifespan > oldest_lifespan)
      {
        oldest_lifespan = current_lifespan;
        oldest_object_index = current_index;
      }
      if (!targets_[in_tracker_indices[current_index]].label_.empty() &&
        targets_[in_tracker_indices[current_index]].label_ != "unknown")
      {
        best_label = targets_[in_tracker_indices[current_index]].label_;
      }
    }
    // delete nearby targets except for the oldest target
    for(size_t j = 0; j < matching_objects[i].size(); j++)
    {
      size_t current_index = matching_objects[i][j];
      if(current_index != oldest_object_index)
      {
        targets_[in_tracker_indices[current_index]].tracking_num_= TrackingState::Die;
      }
    }
    
    autoware_msgs::DetectedObject best_object;
    best_object = in_detected_objects.objects[oldest_object_index];
    if (best_label != "unknown" && !best_label.empty())
    {
      best_object.label = best_label;
    }

    resulting_objects.objects.push_back(best_object);
  }

  return resulting_objects;

}

void ImmUkfPda::makeOutput(const autoware_msgs::DetectedObjectArray& input,
                           const std::vector<bool> &matching_vec,
                           autoware_msgs::DetectedObjectArray& detected_objects_output)
{
  autoware_msgs::DetectedObjectArray tmp_objects;
  tmp_objects.header = input.header;
  std::vector<size_t> used_targets_indices;
  // std::cout << "targets size: " << targets_.size() << std::endl;
  for (size_t i = 0; i < targets_.size(); i++)
  {

    double tx = targets_[i].x_merge_(0);
    double ty = targets_[i].x_merge_(1);

    double tv = targets_[i].x_merge_(2);
    // mmg 2023-12-13 the original yaw is unstable, so try to modify
    double tyaw = targets_[i].x_merge_(3);
    // double tyaw = input.objects[i].angle;
    double tyaw_rate = targets_[i].x_merge_(4);

    while (tyaw > M_PI)
      tyaw -= 2. * M_PI;
    while (tyaw < -M_PI)
      tyaw += 2. * M_PI;

    tf::Quaternion q = tf::createQuaternionFromYaw(tyaw);

    autoware_msgs::DetectedObject dd;
    dd = targets_[i].object_;
    // dd.id = targets_[i].ukf_id_; //mmg 2023-12-07
    dd.id = targets_[i].object_.id;
    dd.velocity.linear.x = tv;
    dd.acceleration.linear.y = tyaw_rate;
    dd.velocity_reliable = targets_[i].is_stable_;
    dd.pose_reliable = targets_[i].is_stable_;
    // dd.pose.orientation = input.objects[i].pose.orientation;  //mmg 2023-12-13

    //mmg 2023-12-13
    if (!targets_[i].is_static_ && targets_[i].is_stable_){
            // Aligh the longest side of dimentions with the estimated orientation
      if(targets_[i].object_.dimensions.x < targets_[i].object_.dimensions.y)
        {
          dd.dimensions.x = targets_[i].object_.dimensions.y;
          dd.dimensions.y = targets_[i].object_.dimensions.x;
        }

        dd.pose.position.x = tx;
        dd.pose.position.y = ty;

        // mmg 2023-12-13
        if (!std::isnan(q[0]))
          dd.pose.orientation.x = q[0];
        if (!std::isnan(q[1]))
          dd.pose.orientation.y = q[1];
        if (!std::isnan(q[2]))
          dd.pose.orientation.z = q[2];
        if (!std::isnan(q[3]))
          dd.pose.orientation.w = q[3];
    }
    updateBehaviorState(targets_[i], use_sukf_, dd);

    //tracking_num >= 1&& <4
    //mmg 2023-12-14
    if (targets_[i].is_stable_ || (targets_[i].tracking_num_ >= TrackingState::Init &&
                                   targets_[i].tracking_num_ < TrackingState::Stable))
    {
      tmp_objects.objects.push_back(dd);
      used_targets_indices.push_back(i);
    }
  }
  // std::cout << "tmp_objects.objects: " << tmp_objects.objects.size() << std::endl;
  //mmg 2023-12-13
  detected_objects_output = removeRedundantObjects(tmp_objects, used_targets_indices);
  // detected_objects_output = tmp_objects;
  // std::cout << "the 784 line detected_objects_output size: " << detected_objects_output.objects.size() << std::endl;
}

void ImmUkfPda::removeUnnecessaryTarget()
{
  std::vector<UKF> temp_targets;
  for (size_t i = 0; i < targets_.size(); i++)
  {
    if (targets_[i].tracking_num_ != TrackingState::Die)
    {
      temp_targets.push_back(targets_[i]);
    }
  }
  //清空原始列表元素，并且交换
  std::vector<UKF>().swap(targets_);
  targets_ = temp_targets;
}


void ImmUkfPda::dumpResultText(autoware_msgs::DetectedObjectArray& detected_objects)
{
  std::ofstream outputfile(result_file_path_, std::ofstream::out | std::ofstream::app);
  for (size_t i = 0; i < detected_objects.objects.size(); i++)
  {
    // KITTI tracking benchmark data format:
    // (frame_number,tracked_id, object type, truncation, occlusion, observation angle, x1,y1,x2,y2, h, w, l, cx, cy,
    // cz, yaw)
    // x1, y1, x2, y2 are for 2D bounding box.
    // h, w, l, are for height, width, length respectively
    // cx, cy, cz are for object centroid

    // Tracking benchmark is based on frame_number, tracked_id,
    // bounding box dimentions and object pose(centroid and orientation) from bird-eye view
    
    // outputfile << detected_objects.objects[i].header.stamp << ","
    //         << std::to_string(frame_count_) << "," << std::to_string(detected_objects.objects[i].id) << ","
    //         << std::to_string(detected_objects.objects[i].pose.position.x) << ","
    //         << std::to_string(detected_objects.objects[i].pose.position.y) << ","
    //         << std::to_string(detected_objects.objects[i].pose.position.z) << ","
    //         << std::to_string(detected_objects.objects[i].dimensions.x) << ","
    //         << std::to_string(detected_objects.objects[i].dimensions.y) << ","
    //         << std::to_string(detected_objects.objects[i].dimensions.z) << ","
    //         << "-1" << ","
    //         << "0" << ","
    //         << "0" << ","
    //         << "-1" << "\n";

    // outputfile << std::to_string(frame_count_) << "," << std::to_string(detected_objects.objects[i].id) << ","
    //         << std::to_string(10*detected_objects.objects[i].pose.position.x - 5*detected_objects.objects[i].dimensions.x) << ","
    //         << std::to_string(-10*detected_objects.objects[i].pose.position.y - 5*detected_objects.objects[i].dimensions.y) << ","
    //         << std::to_string(10*detected_objects.objects[i].dimensions.x) << ","
    //         << std::to_string(10*detected_objects.objects[i].dimensions.y) << ","
    //         << "-1" << ","
    //         << "0" << ","
    //         << "0" << ","
    //         << "-1" << "\n";

    //2024.06.13
    // outputfile << detected_objects.objects[i].header.stamp << "," << 1 << ","
    //         << std::to_string(detected_objects.objects[i].pose.position.x - 0.5*detected_objects.objects[i].dimensions.x) << ","
    //         << std::to_string(detected_objects.objects[i].pose.position.y - 0.5*detected_objects.objects[i].dimensions.y) << ","
    //         << std::to_string(detected_objects.objects[i].dimensions.x) << ","
    //         << std::to_string(detected_objects.objects[i].dimensions.y) << ","
    //         << "-1" << ","
    //         << "0" << ","
    //         << "0" << ","
    //         << "-1" << "\n"; 

    // outputfile << detected_objects.objects[i].header.stamp << "," << std::to_string(detected_objects.objects[i].id + 1) << ","
    //         << std::to_string(detected_objects.objects[i].pose.position.x - 0.5*detected_objects.objects[i].dimensions.x) << ","
    //         << std::to_string(-1*detected_objects.objects[i].pose.position.y - 0.5*detected_objects.objects[i].dimensions.y) << ","
    //         << std::to_string(detected_objects.objects[i].dimensions.x) << ","
    //         << std::to_string(detected_objects.objects[i].dimensions.y) << ","
    //         << "-1" << ","
    //         << "0" << ","
    //         << "0" << ","
    //         << "-1" << "\n";    
  }
  frame_count_++;
}

void ImmUkfPda::tracker(const autoware_msgs::DetectedObjectArray& input,
                        autoware_msgs::DetectedObjectArray& detected_objects_output)
{
  frame_id++;
  double timestamp = input.header.stamp.toSec();
  std::vector<bool> matching_vec(input.objects.size(), false);
  if (!init_)
  {
    initTracker(input, timestamp);
    makeOutput(input, matching_vec, detected_objects_output); //寻找存活时间最久的bestobject
    return;
  }

  double dt = (timestamp - timestamp_);
  timestamp_ = timestamp;

  // start UKF process
  for (size_t i = 0; i < targets_.size(); i++)
  {
    targets_[i].is_stable_ = false;
    targets_[i].is_static_ = false;
    if (targets_[i].tracking_num_ == TrackingState::Die)
    {
      continue;
    }
    // prevent ukf not to explode
    // mmg 2023-12-12
    if (targets_[i].p_merge_.determinant() > prevent_explosion_threshold_ ||
        targets_[i].p_merge_(4, 4) > prevent_explosion_threshold_)
    {
      targets_[i].tracking_num_ = TrackingState::Die;
      continue;
    }

    targets_[i].prediction(use_sukf_, has_subscribed_vectormap_, dt);
    // std::cout << "after prediction:!! " << std::endl;

    // std::cout << "targets[" << i << "]: tracking_num_: "<< targets_[i].tracking_num_ << std::endl;
    std::vector<autoware_msgs::DetectedObject> object_vec;
    bool success = probabilisticDataAssociation(input, dt, matching_vec, object_vec, targets_[i]);  //object_vec存储通过pda筛选的量测
    // std::cout << "success: " << success <<  ", 1139 targets[" << i << "]: tracking_num_: "<< targets_[i].tracking_num_ << std::endl;
    //mmg 2023-12-13
    // if (!success)
    // {
    //   continue;
    // }
    if(object_vec.empty() || !success){
      continue;
    }
    targets_[i].update(use_sukf_, detection_probability_, gate_probability_, gating_threshold_, object_vec);
    // std::cout << "the 964 line after targets_[i].update:object_vec: " << object_vec.size() << std::endl;
  }
  // end UKF process
  // std::cout << "end of UKF start makeNewTargets!" << std::endl;
  // making new ukf target for no data association objects
  makeNewTargets(timestamp, input, matching_vec);

  // static dynamic classification
  staticClassification();

  // making output for visualization
  //输出卡尔曼的最终估计结果
  makeOutput(input, matching_vec, detected_objects_output);
  // std::cout << "detected_objects_out: " << detected_objects_output.objects.size() << std::endl;

  // remove unnecessary ukf object
  removeUnnecessaryTarget();
  // std::cout << "end of removeUnnecessaryTarget!" << std::endl;
}
