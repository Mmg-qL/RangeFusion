#ifndef GPS_READER_H
#define GPS_READER_H

#include <ros/ros.h>
#include <stdio.h>
#include <fstream>

#include <gps_common/GPSFix.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "lonlat2utm.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

class gpsReader{
private:
    ros::NodeHandle node_handle_;

    message_filters::Subscriber<sensor_msgs::Imu> *imu_subscriber_;  //mmg 2023-12-18
    message_filters::Subscriber<sensor_msgs::NavSatFix> *gps_subscriber_;
    // message_filters::Subscriber<sensor_msgs::Imu> *real_imu_subscriber_;

    ros::Subscriber real_imu_subscriber_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::NavSatFix, sensor_msgs::Imu> SyncPolicyT1; 
    message_filters::Synchronizer<SyncPolicyT1> *sync1; //mmg 2023-12-25

    double init_latitude, init_longitude, init_altitude;
    double vehicleHeading_;
    double oriheading; //mmg 2023-12-29

    ros::Time last_imu_time_;

    double init_x, init_y; //mmg 2023-12-26

    double UTIME_ = 0.0, UTIMN_ = 0.0, Height_ = 0.0; //mmg 2023-12-25
    double vehicle_x = 0.0, vehicle_y = 0.0, vehicle_z = 0.0; //mmg 2023-12-25
    double init_heading;

    tf::TransformBroadcaster broadcaster; 

    Eigen::Isometry3d translate2origin_;  
    Eigen::Isometry3d origin2translate_;  
    Eigen::Isometry3d porigion_;  

    bool gps_init = false;
    bool imu_init_ = false;

    void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& gps_msg, const sensor_msgs::Imu::ConstPtr& imu_msg);
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg);
    bool use_real_data_;

public:
    gpsReader();
    ~gpsReader();
    void run();
};

#endif