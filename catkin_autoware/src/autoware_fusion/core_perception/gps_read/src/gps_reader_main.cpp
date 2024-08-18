#include "gps_reader.h"

int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "time_synch");
    gpsReader app;
    app.run();
    ros::spin();

    return 0;
}
