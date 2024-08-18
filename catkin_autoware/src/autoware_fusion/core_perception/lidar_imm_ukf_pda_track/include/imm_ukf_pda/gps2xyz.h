#ifndef GPS2XYZ_H
#define GPS2XYZ_H

#include <nav_msgs/Path.h>
#include <math.h>

/**
 * @brief WGS84 to UTM
 * @details https://blog.csdn.net/saluzi2017/article/details/124884224?spm=1001.2014.3001.5506
 * @param longitude
 *        latitude
 *        x
 *        y
*/

#define EARTH_RADIUS 6378.137;  //earth radius

double rad(const double& d) 
{
	return d * 3.1415926 / 180.0;
}

void GPS2XYZ(const double& init_latitude, const double& init_longitude, const double& latitude, const double& longitude, double& x, double& y){
    double radLat1 ,radLat2, radLong1, radLong2, delta_lat, delta_long;
    radLat1 = rad(init_latitude);
    radLong1 = rad(init_longitude);
    radLat2 = rad(latitude);
    radLong2 = rad(longitude);

    //calculate x
    delta_long = 0;
	delta_lat = radLat2 - radLat1;  //(radLat1,radLong1)-(radLat2,radLong1)

	if(delta_lat > 0){
        x = 2 * asin( sqrt( pow( sin( delta_lat/2 ),2) + cos( radLat1 ) * cos( radLat2)*pow( sin( delta_long/2 ),2 ) ));
    }
    else{
        x = -2 * asin( sqrt( pow( sin( delta_lat/2 ),2) + cos( radLat1 ) * cos( radLat2)*pow( sin( delta_long/2 ),2 ) ));
    }
    x = x * 1000 * EARTH_RADIUS;

    //calculate y
    delta_lat = 0;
    delta_long = radLong2  - radLong1;   //(radLat1,radLong1)-(radLat1,radLong2)
	if(delta_long>0){
        y = 2*asin( sqrt( pow( sin( delta_lat/2 ),2) + cos( radLat2 )*cos( radLat2)*pow( sin( delta_long/2 ),2 ) ) );
    }
	else{
        y = -2*asin( sqrt( pow( sin( delta_lat/2 ),2) + cos( radLat2 )*cos( radLat2)*pow( sin( delta_long/2 ),2 ) ) );
    }
    y = y * 1000 *  EARTH_RADIUS;
}

#endif

