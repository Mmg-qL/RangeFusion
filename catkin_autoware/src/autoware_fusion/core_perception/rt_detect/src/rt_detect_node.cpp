/*
*   Copyright 2024 mmg. All rights reserved.
*   author: mmg(gmm782470390@163.com)
*   Created on: 2024-01-08
*/

#include "rt_detect_node.h"

int main(int argc, char **argv){
    ros::init(argc, argv, "rt_detect");
    rtDetrNode app;
    app.run();
    ros::spin();
    return 0;
}
