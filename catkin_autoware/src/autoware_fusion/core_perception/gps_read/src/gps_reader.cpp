#include "gps_reader.h"

gpsReader::gpsReader()
:use_real_data_(false){

}

gpsReader::~gpsReader(){
    // delete sync1;
    // delete imu_subscriber_;
    // delete gps_subscriber_;
}

void gpsReader::gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& gps_msg, const sensor_msgs::Imu::ConstPtr& imu_msg)
{
    double heading;
    std::string gps_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/gps_file/real_track.txt";
    std::ofstream gpsoutfile(gps_file_path_, std::ofstream::out | std::ofstream::app);
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
      // gpsoutfile << gps_msg->header.stamp << "," << p_1[0] << "," << p_1[1]  << "," << p_1[2] << "\n";
      Eigen::Isometry3d p3;
      p3 = rotpnow;
      p3.translation() = Eigen::Vector3d(vehicle_x, vehicle_y, 0);

      vehicleHeading_ = heading;
      //mmg 2023-12-28
      // std_msgs::Header gps_header = gps_msg->header;
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

void gpsReader::imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg)
{
    // std::string gps_file_path_ = "/home/gmm/file/githup_file/catkin_autoware/result/gps_file/real_track.csv";
    // std::ofstream gpsoutfile(gps_file_path_, std::ofstream::out | std::ofstream::app);

    ros::Time current_time = imu_msg->header.stamp;
    if(!imu_init_){
      last_imu_time_ = current_time;
      imu_init_ = true;
      init_heading = tf::getYaw(imu_msg->orientation) + M_PI / 2;
    }else{
      // Calculate time difference
      init_heading = tf::getYaw(imu_msg->orientation) + M_PI / 2;      
      double dt = (current_time - last_imu_time_).toSec();
      // Integrate velocity to get position
      vehicle_x += 10 * sin(init_heading) * dt;
      vehicle_y += 10 * cos(init_heading) * dt;

      // Update last IMU time
      last_imu_time_ = current_time;
      tf::Transform lidar2world_transform;
      lidar2world_transform.setOrigin(tf::Vector3(vehicle_x, vehicle_y, 0));
      double qx = (imu_msg->orientation).x;
      double qy = (imu_msg->orientation).y;
      double qz = (imu_msg->orientation).z;
      double qw = (imu_msg->orientation).w;
      lidar2world_transform.setRotation(tf::Quaternion(qx, qy, qz, qw));
      // gpsoutfile << vehicle_x << "," << vehicle_y << "," << 0 << "\n";
      broadcaster.sendTransform(tf::StampedTransform(lidar2world_transform, ros::Time::now(), "global_frame", "local_frame"));  
    }
}

void gpsReader::run()
{
    if(use_real_data_){
      real_imu_subscriber_ = node_handle_.subscribe("/imu_sonser_spec/ch104_imu", 1, &gpsReader::imuCallback, this);
    }else{
      gps_subscriber_ = new message_filters::Subscriber<sensor_msgs::NavSatFix>(node_handle_,"/kitti/oxts/gps/fix", 1);
      imu_subscriber_ = new message_filters::Subscriber<sensor_msgs::Imu>(node_handle_, "/kitti/oxts/imu", 1);
      sync1 = new message_filters::Synchronizer<SyncPolicyT1> (SyncPolicyT1(10), *gps_subscriber_, *imu_subscriber_);
      sync1->registerCallback(boost::bind(&gpsReader::gpsCallback, this, _1, _2));
    }
}

