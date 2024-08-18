#include "range_vision_fusion/range_vision_fusion.h"

int frame_count_ = 1;
int total_object = 0;

cv::Point3f
ROSRangeVisionFusionApp::TransformPoint(const geometry_msgs::Point &in_point, const tf::Transform &in_transform) //按照旋转矩阵旋转point得到cv::point3f格式的点
{
  tf::Vector3 tf_point(in_point.x, in_point.y, in_point.z);
  tf::Vector3 tf_point_t = in_transform * tf_point;
  return cv::Point3f(tf_point_t.x(), tf_point_t.y(), tf_point_t.z());
}

cv::Point2i
ROSRangeVisionFusionApp::ProjectPoint(const cv::Point3f &in_point)
{
  auto u = int(in_point.x * fx_ / in_point.z + cx_);
  auto v = int(in_point.y * fy_ / in_point.z + cy_);

  return cv::Point2i(u, v);
}

autoware_msgs::DetectedObject
ROSRangeVisionFusionApp::TransformObject(const autoware_msgs::DetectedObject &in_detection,
                                         const tf::StampedTransform &in_transform)
{
  autoware_msgs::DetectedObject t_obj = in_detection;

  tf::Vector3 in_pos(in_detection.pose.position.x,
                     in_detection.pose.position.y,
                     in_detection.pose.position.z);
  tf::Quaternion in_quat(in_detection.pose.orientation.x,
                         in_detection.pose.orientation.y,
                         in_detection.pose.orientation.w,
                         in_detection.pose.orientation.z);

  tf::Vector3 in_pos_t = in_transform * in_pos;
  tf::Quaternion in_quat_t = in_transform * in_quat;

  t_obj.pose.position.x = in_pos_t.x();
  t_obj.pose.position.y = in_pos_t.y();
  t_obj.pose.position.z = in_pos_t.z();

  t_obj.pose.orientation.x = in_quat_t.x();
  t_obj.pose.orientation.y = in_quat_t.y();
  t_obj.pose.orientation.z = in_quat_t.z();
  t_obj.pose.orientation.w = in_quat_t.w();

  return t_obj;
}

double ROSRangeVisionFusionApp::computeIoU(const Rectangle& rect1, const Rectangle& rect2){
    auto getRotatedCorners = [](const Rectangle& rect) {
        double angle_rad = M_PI * rect.angle / 180.0;
        double cos_a = std::cos(angle_rad);
        double sin_a = std::sin(angle_rad);

        double w_half = rect.width / 2.0;
        double h_half = rect.height / 2.0;

        double corners_x[4] = {-w_half, w_half, w_half, -w_half};
        double corners_y[4] = {-h_half, -h_half, h_half, h_half};

        for (int i = 0; i < 4; ++i) {
            double x = corners_x[i];
            double y = corners_y[i];
            corners_x[i] = rect.center_x + x * cos_a - y * sin_a;
            corners_y[i] = rect.center_y + x * sin_a + y * cos_a;
        }

        return std::make_pair(corners_x, corners_y);
    };

    auto [rect1_corners_x, rect1_corners_y] = getRotatedCorners(rect1);
    auto [rect2_corners_x, rect2_corners_y] = getRotatedCorners(rect2);

    // 计算交集区域
    double intersection = std::max(0.0, std::min(rect1_corners_x[2], rect2_corners_x[2]) - std::max(rect1_corners_x[0], rect2_corners_x[0])) *
                          std::max(0.0, std::min(rect1_corners_y[3], rect2_corners_y[3]) - std::max(rect1_corners_y[1], rect2_corners_y[1]));

    // 计算并集区域
    double area_rect1 = rect1.width * rect1.height;
    double area_rect2 = rect2.width * rect2.height;
    double area_union = area_rect1 + area_rect2 - intersection;

    // 计算IoU
    double iou = (area_union > 0) ? (intersection / area_union) : 0.0;

    return iou;
}


bool
ROSRangeVisionFusionApp::IsObjectInImage(const autoware_msgs::DetectedObject &in_detection)
{
  cv::Point3f image_space_point = TransformPoint(in_detection.pose.position, cameraTolidar_tf);

  cv::Point2i image_pixel = ProjectPoint(image_space_point);

  return (image_pixel.x >= 0)
         && (image_pixel.x < image_size_.width)
         && (image_pixel.y >= 0)
         && (image_pixel.y < image_size_.height)
         && (image_space_point.z > 0);
}

cv::Rect ROSRangeVisionFusionApp::ProjectDetectionToRect(const autoware_msgs::DetectedObject &in_detection)
{
  cv::Rect projected_box;

  Eigen::Vector3f pos;      //三维检测框的位置
  pos << in_detection.pose.position.x,
    in_detection.pose.position.y,
    in_detection.pose.position.z;

  Eigen::Quaternionf rot(in_detection.pose.orientation.w,   //四元数矩阵
                         in_detection.pose.orientation.x,
                         in_detection.pose.orientation.y,
                         in_detection.pose.orientation.z);

  std::vector<double> dims = {
    in_detection.dimensions.x,
    in_detection.dimensions.y,
    in_detection.dimensions.z
  };

  jsk_recognition_utils::Cube cube(pos, rot, dims);

  Eigen::Affine3f range_vision_tf;
  // tf::transformTFToEigen(camera_lidar_tf_, range_vision_tf);
  tf::transformTFToEigen(cameraTolidar_tf, range_vision_tf);
  jsk_recognition_utils::Vertices vertices = cube.transformVertices(range_vision_tf);

  std::vector<cv::Point> polygon;     //二维检测框点
  for (auto &vertex : vertices)
  {
    cv::Point p = ProjectPoint(cv::Point3f(vertex.x(), vertex.y(), vertex.z()));    //转换到图像坐标系
    polygon.push_back(p);
  }

  projected_box = cv::boundingRect(polygon);

  return projected_box;
}

// void
// ROSRangeVisionFusionApp::TransformRangeToVision(const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections,
//                                                 autoware_msgs::DetectedObjectArray &out_in_cv_range_detections,
//                                                 autoware_msgs::DetectedObjectArray &out_out_cv_range_detections)
// {
//   out_in_cv_range_detections.header = in_range_detections->header;
//   out_in_cv_range_detections.objects.clear();
//   out_out_cv_range_detections.header = in_range_detections->header;
//   out_out_cv_range_detections.objects.clear();
//   for (size_t i = 0; i < in_range_detections->objects.size(); i++)
//   {
//     if (IsObjectInImage(in_range_detections->objects[i]))   //判断点云聚类目标是在图像内还是图像外
//     {
//       out_in_cv_range_detections.objects.push_back(in_range_detections->objects[i]);
//     } else
//     {
//       out_out_cv_range_detections.objects.push_back(in_range_detections->objects[i]);
//     }
//   }
// }

void
ROSRangeVisionFusionApp::TransformRangeToVision(const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections,
                                                autoware_msgs::DetectedObjectArray &out_in_cv_range_detections)
{
  out_in_cv_range_detections.header = in_range_detections->header;
  out_in_cv_range_detections.objects.clear();
  for (size_t i = 0; i < in_range_detections->objects.size(); i++)
  {
      out_in_cv_range_detections.objects.push_back(in_range_detections->objects[i]);
  }
}

void
ROSRangeVisionFusionApp::VisionFilter(const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections,
                                                autoware_msgs::DetectedObjectArray &out_in_cv_range_detections)
{
  out_in_cv_range_detections.header = in_range_detections->header;
  out_in_cv_range_detections.objects.clear();
  for (size_t i = 0; i < in_range_detections->objects.size(); i++)
  {
      double current_vision_distance = GetDistanceToObject(in_range_detections->objects[i]);
      if(current_vision_distance > 90) continue;
      out_in_cv_range_detections.objects.push_back(in_range_detections->objects[i]);
  }
}

void
ROSRangeVisionFusionApp::CalculateObjectFeatures(autoware_msgs::DetectedObject &in_out_object, bool in_estimate_pose)
{

  float min_x = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_z = -std::numeric_limits<float>::max();
  float average_x = 0, average_y = 0, average_z = 0, length, width, height;
  pcl::PointXYZ centroid, min_point, max_point, average_point;

  std::vector<cv::Point2f> object_2d_points;

  pcl::PointCloud<pcl::PointXYZ> in_cloud;
  pcl::fromROSMsg(in_out_object.pointcloud, in_cloud);

  for (const auto &point : in_cloud.points)
  {
    average_x += point.x;
    average_y += point.y;
    average_z += point.z;
    centroid.x += point.x;
    centroid.y += point.y;
    centroid.z += point.z;

    if (point.x < min_x)
      min_x = point.x;
    if (point.y < min_y)
      min_y = point.y;
    if (point.z < min_z)
      min_z = point.z;
    if (point.x > max_x)
      max_x = point.x;
    if (point.y > max_y)
      max_y = point.y;
    if (point.z > max_z)
      max_z = point.z;

    cv::Point2f pt;
    pt.x = point.x;
    pt.y = point.y;
    object_2d_points.push_back(pt);
  }
  min_point.x = min_x;
  min_point.y = min_y;
  min_point.z = min_z;
  max_point.x = max_x;
  max_point.y = max_y;
  max_point.z = max_z;

  if (in_cloud.points.size() > 0)
  {
    centroid.x /= in_cloud.points.size();
    centroid.y /= in_cloud.points.size();
    centroid.z /= in_cloud.points.size();

    average_x /= in_cloud.points.size();
    average_y /= in_cloud.points.size();
    average_z /= in_cloud.points.size();
  }

  average_point.x = average_x;
  average_point.y = average_y;
  average_point.z = average_z;

  length = max_point.x - min_point.x;
  width = max_point.y - min_point.y;
  height = max_point.z - min_point.z;

  geometry_msgs::PolygonStamped convex_hull;
  std::vector<cv::Point2f> hull_points;
  if (object_2d_points.size() > 0)
    cv::convexHull(object_2d_points, hull_points);

  convex_hull.header = in_out_object.header;
  for (size_t i = 0; i < hull_points.size() + 1; i++)
  {
    geometry_msgs::Point32 point;
    point.x = hull_points[i % hull_points.size()].x;
    point.y = hull_points[i % hull_points.size()].y;
    point.z = min_point.z;
    convex_hull.polygon.points.push_back(point);
  }

  for (size_t i = 0; i < hull_points.size() + 1; i++)
  {
    geometry_msgs::Point32 point;
    point.x = hull_points[i % hull_points.size()].x;
    point.y = hull_points[i % hull_points.size()].y;
    point.z = max_point.z;
    convex_hull.polygon.points.push_back(point);
  }

  double rz = 0;
  if (in_estimate_pose)
  {
    cv::RotatedRect box = cv::minAreaRect(hull_points);
    rz = box.angle * 3.14 / 180;
    in_out_object.pose.position.x = box.center.x;
    in_out_object.pose.position.y = box.center.y;
    in_out_object.dimensions.x = box.size.width;
    in_out_object.dimensions.y = box.size.height;
  }

  in_out_object.convex_hull = convex_hull;

  in_out_object.pose.position.x = min_point.x + length / 2;
  in_out_object.pose.position.y = min_point.y + width / 2;
  in_out_object.pose.position.z = min_point.z + height / 2;

  in_out_object.dimensions.x = ((length < 0) ? -1 * length : length);
  in_out_object.dimensions.y = ((width < 0) ? -1 * width : width);
  in_out_object.dimensions.z = ((height < 0) ? -1 * height : height);

  tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);
  tf::quaternionTFToMsg(quat, in_out_object.pose.orientation);
}

autoware_msgs::DetectedObject ROSRangeVisionFusionApp::MergeObjects(const autoware_msgs::DetectedObject &in_object_a,
                                                                    const autoware_msgs::DetectedObject &in_object_b)
{
  autoware_msgs::DetectedObject object_merged;
  object_merged = in_object_b;

  pcl::PointCloud<pcl::PointXYZ> cloud_a, cloud_b, cloud_merged;

  if (!in_object_a.pointcloud.data.empty())
    pcl::fromROSMsg(in_object_a.pointcloud, cloud_a);
  if (!in_object_b.pointcloud.data.empty())
    pcl::fromROSMsg(in_object_b.pointcloud, cloud_b);

  cloud_merged = cloud_a + cloud_b;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_merged, cloud_msg);
  cloud_msg.header = object_merged.pointcloud.header;

  object_merged.pointcloud = cloud_msg;

  return object_merged;

}

// double ROSRangeVisionFusionApp::GetDistanceToObject(const autoware_msgs::DetectedObject &in_object)
// {
//   return sqrt(in_object.dimensions.x * in_object.dimensions.x +
//               in_object.dimensions.y * in_object.dimensions.y +
//               in_object.dimensions.z * in_object.dimensions.z);
// }

double ROSRangeVisionFusionApp::GetDistanceToObject(const autoware_msgs::DetectedObject &in_object)
{
  return sqrt(in_object.pose.position.x * in_object.pose.position.x +
              in_object.pose.position.y * in_object.pose.position.y);
}

double ROSRangeVisionFusionApp::CalDistance(const autoware_msgs::DetectedObject &obj1,
                                            const autoware_msgs::DetectedObject &obj2)
{
  return sqrt((obj1.pose.position.x - obj2.pose.position.x) * (obj1.pose.position.x - obj2.pose.position.x) +
              (obj1.pose.position.y - obj2.pose.position.y) * (obj1.pose.position.y - obj2.pose.position.y));
}

void ROSRangeVisionFusionApp::CheckMinimumDimensions(autoware_msgs::DetectedObject &in_out_object)
{
  if (in_out_object.label == "car")
  {
    if (in_out_object.dimensions.x < car_depth_)
      in_out_object.dimensions.x = car_depth_;
    if (in_out_object.dimensions.y < car_width_)
      in_out_object.dimensions.y = car_width_;
    if (in_out_object.dimensions.z < car_height_)
      in_out_object.dimensions.z = car_height_;
  }
  if (in_out_object.label == "person")
  {
    if (in_out_object.dimensions.x < person_depth_)
      in_out_object.dimensions.x = person_depth_;
    if (in_out_object.dimensions.y < person_width_)
      in_out_object.dimensions.y = person_width_;
    if (in_out_object.dimensions.z < person_height_)
      in_out_object.dimensions.z = person_height_;
  }

  if (in_out_object.label == "truck" || in_out_object.label == "bus")
  {
    if (in_out_object.dimensions.x < truck_depth_)
      in_out_object.dimensions.x = truck_depth_;
    if (in_out_object.dimensions.y < truck_width_)
      in_out_object.dimensions.y = truck_width_;
    if (in_out_object.dimensions.z < truck_height_)
      in_out_object.dimensions.z = truck_height_;
  }
}


void
ROSRangeVisionFusionApp::VisionDetectionsCallback(
  const autoware_msgs::DetectedObjectArray::ConstPtr &in_vision_3ddetections)
{
  if (!processing_ && !in_vision_3ddetections->objects.empty())
  {
    processing_ = true;
    vision_detections_ = in_vision_3ddetections;
    SyncedDetectionsCallback(in_vision_3ddetections, range_detections_);
    processing_ = false;
  }
}

void
ROSRangeVisionFusionApp::RangeDetectionsCallback(
  const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections)
{
  if (!processing_ && !in_range_detections->objects.empty())
  {
    processing_ = true;
    range_detections_ = in_range_detections;
    SyncedDetectionsCallback(vision_detections_, in_range_detections);
    processing_ = false;
  }
}

void ROSRangeVisionFusionApp::printXY(const autoware_msgs::DetectedObjectArray::ConstPtr& array1,
                                      const autoware_msgs::DetectedObjectArray::ConstPtr& array2)
{
  std::string result1_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/XY_figure/obj1.txt";
  std::ofstream outputfile1(result1_file_path_, std::ofstream::out | std::ofstream::app);
  std::string result2_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/XY_figure/obj2.txt";
  std::ofstream outputfile2(result2_file_path_, std::ofstream::out | std::ofstream::app);
  // int amount = std::min(array1->objects.size(), array2->objects.size());
  // double sumX, sumY, sumZ, sumAngle;
  // for(int i = 0; i < amount; ++i){
  //     sumX += (array1->objects[i].pose.position.x - array2->objects[i].pose.position.x);
  //     sumY += (array1->objects[i].pose.position.y - array2->objects[i].pose.position.y);
  //     sumZ += (array1->objects[i].pose.position.z - array2->objects[i].pose.position.z);
  //     sumAngle += (array1->objects[i].angle - array2->objects[i].angle);

  //     outputfile1 << std::to_string(array1->objects[i].id) << ","
  //      << std::to_string(array2->objects[i].id) << ","
  //     << std::to_string(array1->objects[i].pose.position.x - array2->objects[i].pose.position.x) << ","
  //     << std::to_string(array1->objects[i].pose.position.y - array2->objects[i].pose.position.y) << ","
  //     << std::to_string(array1->objects[i].pose.position.z - array2->objects[i].pose.position.z) << ","
  //     << std::to_string(array1->objects[i].dimensions.x - array2->objects[i].dimensions.x) << ","
  //     << std::to_string(array1->objects[i].dimensions.y - array2->objects[i].dimensions.y) << ","
  //     << std::to_string(array1->objects[i].dimensions.z - array2->objects[i].dimensions.z) << ","
  //     << std::to_string(array1->objects[i].angle - array2->objects[i].angle) << ","
  //     << "-1" << ","
  //     << "-1" << ","
  //     << "-1" << ","
  //     << "-1" << "\n";   
  // }
  // std::cout << "Average X, Y, Z, Angle: " << sumX/amount << "," <<  sumY/amount << "," <<  sumZ/amount << "," << sumAngle/amount << std::endl;

  for(int i = 0; i < array1->objects.size(); ++i){
      outputfile1 << array1->objects[i].header.stamp << ","
      << std::to_string(frame_count_) << "," << std::to_string(array1->objects[i].id) << ","
      << std::to_string(array1->objects[i].pose.position.x) << ","
      << std::to_string(array1->objects[i].pose.position.y) << ","
      << std::to_string(array1->objects[i].pose.position.z) << ","
      << std::to_string(array1->objects[i].dimensions.x) << ","
      << std::to_string(array1->objects[i].dimensions.y) << ","
      << std::to_string(array1->objects[i].dimensions.z) << ","
      << std::to_string(array1->objects[i].angle) << ","
      << "-1" << ","
      << "-1" << ","
      << "-1" << ","
      << "-1" << "\n";    
  }

  for(int i = 0; i < array2->objects.size(); ++i){
      outputfile2 << array2->objects[i].header.stamp << ","
      << std::to_string(frame_count_) << "," << std::to_string(array2->objects[i].id) << ","
      << std::to_string(array2->objects[i].pose.position.x) << ","
      << std::to_string(array2->objects[i].pose.position.y) << ","
      << std::to_string(array2->objects[i].pose.position.z) << ","
      << std::to_string(array2->objects[i].dimensions.x) << ","
      << std::to_string(array2->objects[i].dimensions.y) << ","
      << std::to_string(array2->objects[i].dimensions.z) << ","
      << std::to_string(array2->objects[i].angle) << ","
      << "-1" << ","
      << "-1" << ","
      << "-1" << ","
      << "-1" << "\n";    
  }  
}


autoware_msgs::DetectedObjectArray
ROSRangeVisionFusionApp::FuseRangeVisionDetections(
  const autoware_msgs::DetectedObjectArray::ConstPtr &in_3dvision_detections,
  const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections)
{
  autoware_msgs::DetectedObjectArray range_in_cv, vision_obj_array;
  TransformRangeToVision(in_range_detections, range_in_cv);   //点云聚类后的目标是区分图像内和图像外的，将其分别存储
  // printXY(in_3dvision_detections, in_range_detections);
  VisionFilter(in_3dvision_detections, vision_obj_array);   //点云聚类后的目标是区分图像内和图像外的，将其分别存储
  autoware_msgs::DetectedObjectArray fused_objects;
  fused_objects.header = in_range_detections->header;

    //mmg 2023-12-08
  std::string result_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/range_vision_det/range_vision.txt";
  std::ofstream outputfile(result_file_path_, std::ofstream::out | std::ofstream::app);
  std::string box_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/range_vision_det/bbox.txt";
  std::ofstream outputbox(box_file_path_, std::ofstream::out | std::ofstream::app);

  std::vector<std::vector<size_t> > lidar_range_assignments(range_in_cv.objects.size());   //存储图像内点云目标的size
  std::vector<bool> used_range_detections(range_in_cv.objects.size(), false);    //创建size大小的bool类型数组，初始为false
  std::vector<long> vision_range_closest(range_in_cv.objects.size());             //最近目标

  float a1 , a2;
  float d1, d2;
  if(use_real_data_){
    a1 = 0.9, a2 = 0.1;
    d1 = 0.0, d2 = 20.0;
    distance_threshold_ = 7;
  }else{
    a1 = 0.85, a2 = 0.15;
    d1 = 0.0, d2 = 20.0;    
  }

  for (size_t i = 0; i < range_in_cv.objects.size(); i++)     
  {
    const autoware_msgs::DetectedObject range_object = range_in_cv.objects[i];
    cv::Rect range_rect(range_object.pose.position.x, range_object.pose.position.y, 
                          range_object.dimensions.x, range_object.dimensions.y);   //将目标映射到图像上，找到图像中目标的边框，返回一个图像上的四边形

      double range_rect_area = range_rect.area();
      long closest_index = -1;
      double closest_distance = std::numeric_limits<double>::max();

      for (size_t j = 0; j < vision_obj_array.objects.size(); j++)             
      {
        const autoware_msgs::DetectedObject vision_object = vision_obj_array.objects[j];
        
        double distanceDiff = CalDistance(vision_object, range_object);
        double current_distance = distanceDiff;

        cv::Rect vision_rect(vision_object.pose.position.x, vision_object.pose.position.y, vision_object.dimensions.x, vision_object.dimensions.y);
        double vision_rect_area = range_rect.area();
        cv::Rect overlap = range_rect & vision_rect;             //求检测框的交集

        // outputbox << vision_rect.x << "," << vision_rect.y << "," << vision_rect.width << "," << vision_rect.height << ","
        //           << range_rect.x << "," << range_rect.y << "," << range_rect.width << "," << range_rect.height << "\n";

        if(distanceDiff < distance_threshold_ 
        || overlap.area() > range_rect_area * overlap_threshold_ 
        || overlap.area() > vision_rect_area * overlap_threshold_)
        {
          lidar_range_assignments[i].push_back(j);
          // vision_obj_array.objects[j].angle = range_object.angle; 
          vision_obj_array.objects[j].header = range_object.header;
          vision_obj_array.objects[j].id =  range_object.id;
          vision_obj_array.objects[j].pose.orientation = range_object.pose.orientation; 
          vision_obj_array.objects[j].pose.position.x = a1 * range_object.pose.position.x + a2 * vision_object.pose.position.x;
          vision_obj_array.objects[j].pose.position.y = a1 * range_object.pose.position.y + a2 * vision_object.pose.position.y;
          CheckMinimumDimensions(vision_obj_array.objects[j]);     //检查目标的最小尺寸

          //mmg 2023-12-08
          if (current_distance < closest_distance)      //对于每个相机检测的目标，有多个iou>threshold的雷达目标，找到最近距离
          {
            closest_index = j;
            closest_distance = current_distance;
          }
          used_range_detections[i] = true;     //下标的形式存储
        }//end if overlap
      }//end for range_in_cv
      vision_range_closest[i] = closest_index;
    }

  std::vector<bool> used_vision_detections(vision_obj_array.objects.size(), false);

  //存储最近距离的目标
  for (size_t i = 0; i < lidar_range_assignments.size(); i++)
  {
    if (!range_in_cv.objects.empty() && vision_range_closest[i] >= 0)
    {
      used_vision_detections[i] = true;
      fused_objects.objects.push_back(vision_obj_array.objects[vision_range_closest[i]]);
    }
  }

  //mmg 2023-12-08
  for (size_t i = 0; i < used_range_detections.size(); i++)
  {
    if(use_real_data_){
      if (!used_range_detections[i] && range_in_cv.objects[i].pose.position.y < 7 && range_in_cv.objects[i].pose.position.y > -7)     
      {
        fused_objects.objects.push_back(range_in_cv.objects[i]);
      }
    }else{
      if (!used_range_detections[i])     
      {
        fused_objects.objects.push_back(range_in_cv.objects[i]);
      }
    }
  }
  int last_fuse_id = fused_objects.objects.size();
  for (size_t i = 0; i < used_vision_detections.size(); i++)
  {
    int distance = GetDistanceToObject(vision_obj_array.objects[i]);
    if (distance < d2 && distance > d1 && !used_vision_detections[i])     //未满足IoU阈值的对象，没有使用过的相机检测目标
    {
      vision_obj_array.objects[i].id = last_fuse_id++;
      fused_objects.objects.push_back(vision_obj_array.objects[i]);
    }
  }

  //enable merged for visualization
  for (auto &object : fused_objects.objects)
  {
    object.valid = true;
  }

    //mmg 2023-12-08
  for(int i = 0; i < fused_objects.objects.size(); ++i){
      outputfile << fused_objects.objects[i].header.stamp << ","
      << std::to_string(frame_count_) << "," << std::to_string(fused_objects.objects[i].id) << ","
      << std::to_string(fused_objects.objects[i].pose.position.x) << ","
      << std::to_string(fused_objects.objects[i].pose.position.y) << ","
      << std::to_string(fused_objects.objects[i].pose.position.z) << ","
      << std::to_string(fused_objects.objects[i].dimensions.x) << ","
      << std::to_string(fused_objects.objects[i].dimensions.y) << ","
      << std::to_string(fused_objects.objects[i].dimensions.z) << ","
      << std::to_string(fused_objects.objects[i].angle) << ","
      << "-1" << ","
      << "-1" << ","
      << "-1" << ","
      << "-1" << "\n";
  }
  frame_count_++;

  return fused_objects;
}

// autoware_msgs::DetectedObjectArray
// ROSRangeVisionFusionApp::FuseRangeVisionDetections(
//   const autoware_msgs::DetectedObjectArray::ConstPtr &in_3dvision_detections,
//   const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections)
// {
//   autoware_msgs::DetectedObjectArray range_in_cv, range_out_cv;
//   TransformRangeToVision(in_range_detections, range_in_cv, range_out_cv);   //点云聚类后的目标是区分图像内和图像外的，将其分别存储
//   // printXY(in_3dvision_detections, in_range_detections);

//   autoware_msgs::DetectedObjectArray fused_objects;
//   fused_objects.header = in_range_detections->header;

//     //mmg 2023-12-08
//   std::string result_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/range_vision_det/range_vision.txt";
//   std::ofstream outputfile(result_file_path_, std::ofstream::out | std::ofstream::app);
//   std::string box_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/range_vision_det/bbox.txt";
//   std::ofstream outputbox(box_file_path_, std::ofstream::out | std::ofstream::app);

//   std::vector<std::vector<size_t> > vision_range_assignments(in_3dvision_detections->objects.size());   //存储图像内点云目标的size
//   std::vector<bool> used_vision_detections(in_3dvision_detections->objects.size(), false);    //创建size大小的bool类型数组，初始为false
//   std::vector<long> vision_range_closest(in_3dvision_detections->objects.size());             //最近目标

//   for (size_t i = 0; i < in_3dvision_detections->objects.size(); i++)     //相机探测到的目标
//   {
//     const autoware_msgs::DetectedObject vision_object = in_3dvision_detections->objects[i];
//     double current_vision_distance = GetDistanceToObject(vision_object);
//     if(current_vision_distance >= 80){
//       continue;
//     }
//     else{
//       cv::Rect vision_rect(vision_object.pose.position.x, vision_object.pose.position.y, vision_object.dimensions.x, vision_object.dimensions.y);
//       double vision_rect_area = vision_rect.area();
//       long closest_index = -1;
//       double closest_distance = std::numeric_limits<double>::max();

//       for (size_t j = 0; j < range_in_cv.objects.size(); j++)             //激光雷达探测的目标
//       {
//         const autoware_msgs::DetectedObject range_object = range_in_cv.objects[j];
//         double distanceDiff = CalDistance(vision_object, range_object);
//         double current_distance = distanceDiff;

//         cv::Rect range_rect(range_object.pose.position.x, range_object.pose.position.y, 
//                             range_object.dimensions.x, range_object.dimensions.y);   //将目标映射到图像上，找到图像中目标的边框，返回一个图像上的四边形

//         double range_rect_area = range_rect.area();
//         cv::Rect overlap = range_rect & vision_rect;             //求检测框的交集

//         // outputbox << vision_rect.x << "," << vision_rect.y << "," << vision_rect.width << "," << vision_rect.height << ","
//         //           << range_rect.x << "," << range_rect.y << "," << range_rect.width << "," << range_rect.height << "\n";

//         if(distanceDiff < distance_threshold_ 
//         || overlap.area() > range_rect_area * overlap_threshold_ 
//         || overlap.area() > vision_rect_area * overlap_threshold_)
//         {
//           vision_range_assignments[i].push_back(j);
//           range_in_cv.objects[j].score = vision_object.score;     //雷达目标的置信度
//           range_in_cv.objects[j].label = vision_object.label;     //雷达检测目标的标签
//           range_in_cv.objects[j].angle = tf::getYaw(range_in_cv.objects[j].pose.orientation); //mmg 2023-12-16
//           range_in_cv.objects[j].pose.position.x = 0.8 * range_object.pose.position.x + 0.2 * vision_object.pose.position.x;
//           range_in_cv.objects[j].pose.position.y = 0.8 * range_object.pose.position.y + 0.2 * vision_object.pose.position.y;
//           CheckMinimumDimensions(range_in_cv.objects[j]);     //检查目标的最小尺寸

//           //mmg 2023-12-08
//           if (current_distance < closest_distance)      //对于每个相机检测的目标，有多个iou>threshold的雷达目标，找到最近距离
//           {
//             closest_index = j;
//             closest_distance = current_distance;
//           }
//           used_vision_detections[i] = true;     //下标的形式存储
//         }//end if overlap
//       }//end for range_in_cv
//       vision_range_closest[i] = closest_index;
//     }
//   }

//   std::vector<bool> used_range_detections(range_in_cv.objects.size(), false);

//   //存储最近距离的目标
//   for (size_t i = 0; i < vision_range_assignments.size(); i++)
//   {
//     if (!range_in_cv.objects.empty() && vision_range_closest[i] >= 0)
//     {
//       used_range_detections[i] = true;
//       fused_objects.objects.push_back(range_in_cv.objects[vision_range_closest[i]]);
//     }
//   }

//   //mmg 2023-12-08
//   for (size_t i = 0; i < used_vision_detections.size(); i++)
//   {
//     int distance = GetDistanceToObject(in_3dvision_detections->objects[i]);
//     if (!used_vision_detections[i] && distance < 30)     //未满足IoU阈值的对象，没有使用过的相机检测目标
//     {
//       fused_objects.objects.push_back(in_3dvision_detections->objects[i]);
//     }
//   }

//   //enable merged for visualization
//   for (auto &object : fused_objects.objects)
//   {
//     object.valid = true;
//   }

//     //mmg 2023-12-08
//   for(int i = 0; i < fused_objects.objects.size(); ++i){
//       outputfile << fused_objects.objects[i].header.stamp << ","
//       << std::to_string(frame_count_) << "," << std::to_string(fused_objects.objects[i].id) << ","
//       << std::to_string(fused_objects.objects[i].pose.position.x) << ","
//       << std::to_string(fused_objects.objects[i].pose.position.y) << ","
//       << std::to_string(fused_objects.objects[i].pose.position.z) << ","
//       << std::to_string(fused_objects.objects[i].dimensions.x) << ","
//       << std::to_string(fused_objects.objects[i].dimensions.y) << ","
//       << std::to_string(fused_objects.objects[i].dimensions.z) << ","
//       << std::to_string(fused_objects.objects[i].angle) << ","
//       << "-1" << ","
//       << "-1" << ","
//       << "-1" << ","
//       << "-1" << "\n";
//   }
//   frame_count_++;

//   return fused_objects;
// }

void
ROSRangeVisionFusionApp::SyncedDetectionsCallback(        //时间同步
  const autoware_msgs::DetectedObjectArray::ConstPtr &in_vision_3ddetections,
  const autoware_msgs::DetectedObjectArray::ConstPtr &in_range_detections
  )
{
  autoware_msgs::DetectedObjectArray fusion_objects;
  fusion_objects.objects.clear();

  if (empty_frames_ > 5)
  {
    ROS_INFO("[%s] Empty Detections. Make sure the vision and range detectors are running.", __APP_NAME__);
  }

  if (nullptr == in_vision_3ddetections     //均为空
      && nullptr == in_range_detections)
  {
    empty_frames_++;
    return;
  }

  if (nullptr == in_vision_3ddetections     //图像检测为空，雷达检测不为空
      && nullptr != in_range_detections
      && !in_range_detections->objects.empty())
  {
    empty_frames_++;
    // publisher_fused_objects_.publish(in_range_detections);
    return;
  }
  if (nullptr == in_range_detections    //图像检测不为空，雷达检测为空
      && nullptr != in_vision_3ddetections
      && !in_vision_3ddetections->objects.empty())
  {
    // publisher_fused_objects_.publish(in_vision_3ddetections);   //发布检测相机的话题
    empty_frames_++;
    return;
  }

  if (!camera_lidar_tf_ok_)     //相机雷达消息不正常
  {
    camera_lidar_tf_ = FindTransform(image_frame_id_, in_range_detections->header.frame_id);   //camera lidar坐标系转换
  }
  if (
    !camera_lidar_tf_ok_ ||
    !camera_info_ok_)
  {
    ROS_INFO("[%s] Missing Camera-LiDAR TF or CameraInfo", __APP_NAME__);
    return;
  }

  fusion_objects = FuseRangeVisionDetections(in_vision_3ddetections, in_range_detections);    
  publisher_fused_objects_.publish(fusion_objects);
  empty_frames_ = 0;

  vision_detections_ = nullptr;
  range_detections_ = nullptr;

}

void ROSRangeVisionFusionApp::ImageCallback(const sensor_msgs::Image::ConstPtr &in_image_msg)
{
  if (!camera_info_ok_)
    return;
  cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(in_image_msg, "bgr8");
  cv::Mat in_image = cv_image->image;

  cv::Mat undistorted_image;
  cv::undistort(in_image, image_, camera_instrinsics_, distortion_coefficients_);
};

void
ROSRangeVisionFusionApp::IntrinsicsCallback(const sensor_msgs::CameraInfo &in_message)
{
  image_size_.height = in_message.height;
  image_size_.width = in_message.width;
  //读取in_message的内参存入camera_instinsics变量
  camera_instrinsics_ = cv::Mat(3, 3, CV_64F);  //camera_instrinsics : cv::Mat
  for (int row = 0; row < 3; row++)
  {
    for (int col = 0; col < 3; col++)
    {
      camera_instrinsics_.at<double>(row, col) = in_message.K[row * 3 + col];
    }
  }
  //读取in_message的畸变系数参数存入distortion_coefficients_变量
  distortion_coefficients_ = cv::Mat(1, 5, CV_64F);   //畸变系数
  for (int col = 0; col < 5; col++)
  {
    distortion_coefficients_.at<double>(col) = in_message.D[col];
  }
  //焦距和像素中心
  fx_ = static_cast<float>(in_message.P[0]);
  fy_ = static_cast<float>(in_message.P[5]);
  cx_ = static_cast<float>(in_message.P[2]);
  cy_ = static_cast<float>(in_message.P[6]);

  intrinsics_subscriber_.shutdown();
  camera_info_ok_ = true;
  image_frame_id_ = in_message.header.frame_id;
  ROS_INFO("[%s] CameraIntrinsics obtained.", __APP_NAME__);
}

tf::StampedTransform
ROSRangeVisionFusionApp::FindTransform(const std::string &in_target_frame, const std::string &in_source_frame)
{
  tf::StampedTransform transform;

  ROS_INFO("%s - > %s", in_source_frame.c_str(), in_target_frame.c_str());
  camera_lidar_tf_ok_ = false;
  try
  {
    transform_listener_->lookupTransform(in_target_frame, in_source_frame, ros::Time(0), transform);
    camera_lidar_tf_ok_ = true;
    ROS_INFO("[%s] Camera-Lidar TF obtained", __APP_NAME__);
  }
  catch (tf::TransformException &ex)
  {
    ROS_ERROR("[%s] %s", __APP_NAME__, ex.what());
  }

  return transform;
}

void
ROSRangeVisionFusionApp::InitializeROSIo(ros::NodeHandle &in_private_handle)
{
  //get params
  std::string camera_info_src, detected_objects_vision, detected_objects_range, detected_objects_3dvision, min_car_dimensions, min_person_dimensions, min_truck_dimensions;
  std::string fused_topic_str = "/detection/fusion_tools/objects";
  std::string pkg_loc = ros::package::getPath("range_vision_fusion");
  std::string initial_file = pkg_loc + "/cfg/init_params.yaml";
  std::string name_space_str = ros::this_node::getNamespace();
  bool sync_topics = false;
  use_real_data_ = true;

  cv::FileStorage fs(initial_file, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if(!fs.isOpened()){
    std::cout << "[ " + initial_file + "] file opened error!" << std::endl;
    return;
  }

  in_private_handle.param<std::string>("detected_objects_range", detected_objects_range, "/detection/lidar_detector/objects");      //雷达检测发布的话题
  in_private_handle.param<std::string>("detected_objects_3dvision", detected_objects_3dvision, "/detection/image_detector/objects3d");      
  in_private_handle.param<double>("overlap_threshold", overlap_threshold_, 0.5);
  in_private_handle.param<double>("distance_threshold", distance_threshold_, 1);
  in_private_handle.param<std::string>("min_car_dimensions", min_car_dimensions, "[3,2,2]");//w,h,d
  in_private_handle.param<std::string>("min_person_dimensions", min_person_dimensions, "[1,2,1]");
  in_private_handle.param<std::string>("min_truck_dimensions", min_truck_dimensions, "[4,2,2]");
  in_private_handle.param<bool>("sync_topics", sync_topics, false);

  car_width_ = 2;
  car_height_ = 2;
  car_depth_ = 4;

  person_width_ = 0.5;
  person_height_ = 2;
  person_depth_ = 1;

  truck_width_ = 2;
  truck_height_ = 3;
  truck_depth_ = 5.5;

  if (name_space_str != "/")
  {
    if (name_space_str.substr(0, 2) == "//")
    {
      name_space_str.erase(name_space_str.begin());
    }
    camera_info_src = name_space_str + camera_info_src;
  }

  image_size_.height = fs["image_pixel"]["height"];
  image_size_.width = fs["image_pixel"]["width"];
  fs["camera_matrix"] >> camera_instrinsics_ ;  //camera_instrinsics : cv::Mat
  fs["distortion_coefficients"] >> distortion_coefficients_;   //畸变系数

  std::cout << "image_height: " << image_size_.height << ", image_width: " << image_size_.width << std::endl;
  std::cout << "camera_matrix: " << camera_instrinsics_ << std::endl;
  std::cout << "distortion_coefficients: " << distortion_coefficients_ << std::endl;

  //焦距和像素中心
  fx_ = static_cast<float>(camera_instrinsics_.at<double>(0, 0));
  fy_ = static_cast<float>(camera_instrinsics_.at<double>(1, 1));
  cx_ = static_cast<float>(camera_instrinsics_.at<double>(0, 2));
  cy_ = static_cast<float>(camera_instrinsics_.at<double>(1, 2));
  image_frame_id_ = "base_link";
  camera_info_ok_ = true;

  //旋转平移矩阵
  cv::Mat lidar_camera_R;
  cv::Mat lidar_camera_T;
  fs["lidar_to_camera_R"] >> lidar_camera_R;
  fs["lidar_to_camera_T"] >> lidar_camera_T;

  Eigen::Matrix3d R;
  R << lidar_camera_R.at<double>(0, 0), lidar_camera_R.at<double>(0, 1), lidar_camera_R.at<double>(0, 2),
      lidar_camera_R.at<double>(1, 0), lidar_camera_R.at<double>(1, 1), lidar_camera_R.at<double>(1, 2),
      lidar_camera_R.at<double>(2, 0), lidar_camera_R.at<double>(2, 1), lidar_camera_R.at<double>(2, 2);

  Eigen::Quaterniond q(R);
  q.normalize();
  Eigen::Vector3d t(lidar_camera_T.at<double>(0), lidar_camera_T.at<double>(1), lidar_camera_T.at<double>(2));
  tf::Transform transform_lidar_to_camera;
  transform_lidar_to_camera.setOrigin(tf::Vector3(t(0), t(1), t(2)));
  transform_lidar_to_camera.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
  cameraTolidar_tf = transform_lidar_to_camera;

  if (!sync_topics)
  {
    detections_range_subscriber_ = in_private_handle.subscribe(detected_objects_range,
                                                              1,
                                                              &ROSRangeVisionFusionApp::RangeDetectionsCallback,
                                                              this);

    detections_vision_subscriber_ = in_private_handle.subscribe(detected_objects_3dvision,
                                                                1,
                                                                &ROSRangeVisionFusionApp::VisionDetectionsCallback,
                                                                this);
  }
  else{
    vision3d_filter_subscriber_ = new message_filters::Subscriber<autoware_msgs::DetectedObjectArray>(node_handle_,   //mmg 2024-01-22
                                                                                                    detected_objects_3dvision,
                                                                                                    1);
    range_filter_subscriber_ = new message_filters::Subscriber<autoware_msgs::DetectedObjectArray>(node_handle_,
                                                                                                    detected_objects_range,
                                                                                                    1);

    detections_synchronizer_ = new message_filters::Synchronizer<SyncPolicyT>(SyncPolicyT(10),            //mmg 2024-01-22
                                                                            *vision3d_filter_subscriber_,
                                                                            *range_filter_subscriber_);

    detections_synchronizer_->registerCallback(boost::bind(&ROSRangeVisionFusionApp::SyncedDetectionsCallback, this, _1, _2));     
  }                                                                   


  publisher_fused_objects_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>(fused_topic_str, 1);  //创建融合后的话题并发布

}


void
ROSRangeVisionFusionApp::Run()
{
  ros::NodeHandle private_node_handle("~");
  tf::TransformListener transform_listener; //TF库的目的是实现系统中任一个点在所有坐标系之间的坐标变换，也就是说，只要给定一个坐标系下的一个点的坐标，就能获得这个点在其他坐标系下的坐标。

  transform_listener_ = &transform_listener;

  InitializeROSIo(private_node_handle);

  ROS_INFO("[%s] Ready. Waiting for data...", __APP_NAME__);

  ros::spin();    //ROS消息回调处理函数，两者区别在于前者调用后不会再返回，也就是主程序到这儿就不往下执行了

  ROS_INFO("[%s] END", __APP_NAME__);
}

ROSRangeVisionFusionApp::ROSRangeVisionFusionApp()
{
  camera_lidar_tf_ok_ = true;
  camera_info_ok_ = true;
  processing_ = false;
  image_frame_id_ = "base_link";
  if(use_real_data_){
      overlap_threshold_ = 0.15;
  }
  // overlap_threshold_ = 0.15;
  empty_frames_ = 0;
}