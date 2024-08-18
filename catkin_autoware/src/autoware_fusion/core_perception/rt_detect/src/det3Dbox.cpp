#include "det3Dbox.h"

namespace det3D
{
#include "det3Dbox.h"


    #define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
 
using namespace nvinfer1;

det3Dbox::det3Dbox():
    init_(false){

}

det3Dbox::~det3Dbox(){
    delete[] trtModelStream;
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

void det3Dbox::init3dboxParameters(const std::string& eigen_path, 
                                   const std::string& initial_file, 
                                   const std::string& dimension_file){
    trtModelName_ = eigen_path;
    initial_file_ = initial_file;
    dimension_file_ = dimension_file;
    cv::Mat lidar2camera_R;

    cv::FileStorage fs1(initial_file_, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);

    fs1["camera_matrix"] >> camera_instrinsics_ ;  
    fs1["distance_matrix"] >> distance_;
    fs1["lidar_to_camera_R"] >> lidar2camera_R;

    fx_ = static_cast<float>(camera_instrinsics_.at<double>(0, 0));
    fy_ = static_cast<float>(camera_instrinsics_.at<double>(1, 1));
    cx_ = static_cast<float>(camera_instrinsics_.at<double>(0, 2));
    cy_ = static_cast<float>(camera_instrinsics_.at<double>(1, 2));

    tx_ = static_cast<float>(distance_.at<double>(0));
    ty_ = static_cast<float>(distance_.at<double>(1));
    tz_ = static_cast<float>(distance_.at<double>(2));

    proj_matrix.resize(3, 4);
    proj_matrix.row(0) <<  fx_, 0.0, cx_, tx_;
    proj_matrix.row(1) <<  0.0, fy_, cy_, ty_;
    proj_matrix.row(2) <<  0,  0,  1, tz_;

    // lidar2camera_matrix << lidar2camera_R.at<double>(0, 0), lidar2camera_R.at<double>(0, 1), lidar2camera_R.at<double>(0, 2),
    //                        lidar2camera_R.at<double>(1, 0), lidar2camera_R.at<double>(1, 1), lidar2camera_R.at<double>(1, 2),
    //                        lidar2camera_R.at<double>(2, 0), lidar2camera_R.at<double>(2, 1), lidar2camera_R.at<double>(2, 2);

    // camera2lidar_matrix = lidar2camera_matrix.inverse();
                           
    size_t size{ 0 };
    angle_bins = generateBins(2);
    char *trtModelStream{ nullptr };
    std::ifstream file(trtModelName_, std::ios::binary);
    if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
    }
    runtime_ = createInferRuntime(m_logger_);
    assert(runtime != nullptr);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
    init_ = true;
}

void det3Dbox::normalImage2Blob(cv::Mat& img, float* blob){
    for(int c = 0; c < 3; ++c){
        for(int i = 0; i < img.rows; ++i){
            cv::Vec3b* p1 = img.ptr<cv::Vec3b>(i);
            for(int j = 0; j < img.cols; ++j){
                blob[c * img.cols * img.rows + i * img.cols + j] = (p1[j][c] / 255.0f - meanRGB_[c]) / stdRGB_[c]; 
            }
        }
    }
}

void det3Dbox::formatImage(const cv::Mat& img, const std::vector<cv::Point2f>& box2D, cv::Mat& imgOut){
    // Crop image
    cv::Mat imgInput = img(cv::Range(box2D[0].y, box2D[1].y + 1), cv::Range(box2D[0].x, box2D[1].x + 1));
    cv::resize(imgInput, imgOut, cv::Size(224, 224), 0, 0, cv::INTER_CUBIC);
}


void det3Dbox::getItem(const float* dimension_in, 
                      const std::string &detectedClass,
                      Eigen::Vector3d& dimension_out){

    cv::FileStorage fs2(dimension_file_, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    cv::Mat dimension_matrix;

    if(detectedClass == "car") fs2["car_dimension"] >> dimension_matrix;
    else if(detectedClass == "van") fs2["van_dimension"] >> dimension_matrix;
    else if(detectedClass == "truck") fs2["truck_dimension"] >> dimension_matrix;
    else if(detectedClass == "truck") fs2["truck_dimension"] >> dimension_matrix;
    else if(detectedClass == "pedestrian") fs2["pedestrian_dimension"] >> dimension_matrix;
    else if(detectedClass == "bicycle") fs2["bicycle_dimension"] >> dimension_matrix;
    else fs2["misc_dimension"] >> dimension_matrix;

    float count = static_cast<float>(dimension_matrix.at<double>(0));
    float d1 = static_cast<float>(dimension_matrix.at<double>(1));
    float d2 = static_cast<float>(dimension_matrix.at<double>(2));
    float d3 = static_cast<float>(dimension_matrix.at<double>(3));

    dimension_out << dimension_in[0] + d1 / count,
                     dimension_in[1] + d2 / count,
                     dimension_in[2] + d3 / count;
}

void det3Dbox::infer3dbox(const cv::Mat &image, 
                          const std::vector<cv::Point2f> &bundingBox2d, 
                          const std::string &detectedClass,
                          Eigen::Vector3d &location,
                          double &orient_angle,
                          Eigen::Vector3d &dimension_modify){
    cv::Mat image_data;

    float blob[BATCH_SIZE * 3 * IN_H * IN_W];
    float orientationOut[BATCH_SIZE * 1 * 2 * 2];
    float confidenceOut[BATCH_SIZE * 1 * 2];
    float dimensionOut[BATCH_SIZE * 3];

    formatImage(image, bundingBox2d, image_data);
    normalImage2Blob(image_data, blob);
    doInference(*context_, blob, orientationOut, confidenceOut, dimensionOut, BATCH_SIZE);

    float sin, cos;
    int argmax;
    double theta_ray; 

    if(confidenceOut[0] >= confidenceOut[1]){
        cos = orientationOut[0];
        sin = orientationOut[1];
        argmax = 0;
    }else{
        cos = orientationOut[2];
        sin = orientationOut[3];
        argmax = 1;
    }

    double alpha = std::atan2(sin, cos);
    alpha += angle_bins[argmax];
    alpha -= M_PI;

    calcThetaRay(image, bundingBox2d, theta_ray);
    getItem(dimensionOut, detectedClass, dimension_modify);
    plot_regressed_3d_bbox(image, bundingBox2d, dimension_modify, alpha, theta_ray, location, orient_angle);
}

std::vector<double> det3Dbox::generateBins(const int &bins){
    std::vector<double> angleBins(bins, 0.0);
    double interval = 2.0 * 3.1415926 / bins;

    for (int i = 1; i < bins; ++i) {
        angleBins[i] = i * interval;
    }
    // Center of the bin
    for (auto& angle : angleBins) {
        angle += interval / 2.0;
    }
    return angleBins;
}

void det3Dbox::drawLines(const cv::Mat& img, const std::vector<cv::Point2i>& points, const cv_colors color, const int &thickness){
    int size = points.size();
    for (int i = 0; i < size - 1; ++i) {
        cv::line(img, points[i], points[i + 1], (color == cv_colors::GREEN) ? CV_RGB(0, 255, 0) : CV_RGB(0, 0, 255), thickness);
    }
    // Connect the last and first points
    cv::line(img, points[size - 1], points[0], (color == cv_colors::GREEN) ? CV_RGB(0, 255, 0) : CV_RGB(0, 0, 255), thickness);
}

void det3Dbox::calcThetaRay(const cv::Mat& img, const std::vector<cv::Point2f>& box2D, double& angle){
    int width = img.cols;
    double fovx = 2 * std::atan(static_cast<double>(width) / (2 * proj_matrix(0, 0)));
    double center = (box2D[1].x + box2D[0].x) / 2.0;
    double dx = center - (width / 2);

    int mult = (dx < 0) ? -1 : 1;
    dx = std::abs(dx);
    angle = std::atan((2 * dx * std::tan(fovx / 2)) / width);
    angle = angle * mult;
}

std::vector<std::vector<double>> det3Dbox::create_corners(const Eigen::Vector3d& dimension,
                                                          const Eigen::Matrix3d& R,
                                                          const Eigen::Vector3d& location){
    double dx = dimension[2] / 2;
    double dy = dimension[0] / 2;
    double dz = dimension[1] / 2;

    std::vector<double> x_corners, y_corners, z_corners;

    for (int i : {1, -1}) {
        for (int j : {1, -1}) {
            for (int k : {1, -1}) {
                x_corners.push_back(dx * i);
                y_corners.push_back(dy * j);
                z_corners.push_back(dz * k);
            }
        }
    }

    Eigen::MatrixXd corners(3, 8);
    corners.row(0) = Eigen::Map<Eigen::RowVectorXd>(x_corners.data(), x_corners.size());
    corners.row(1) = Eigen::Map<Eigen::RowVectorXd>(y_corners.data(), y_corners.size());
    corners.row(2) = Eigen::Map<Eigen::RowVectorXd>(z_corners.data(), z_corners.size());

    // Rotate if R is passed in
    if (R != Eigen::Matrix3d::Identity()) {
        corners = R * corners;
    }

    // Shift if location is passed in
    if (location != Eigen::Vector3d::Zero()) {
        corners.colwise() += location;
    }

    std::vector<std::vector<double>> final_corners(8, std::vector<double>(3));

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 3; ++j) {
            final_corners[i][j] = corners(j, i);
        }
    }

    return final_corners;
}

cv::Point2f det3Dbox::ProjectPoint(const std::vector<double> &in_point)
{
  auto u = int(in_point[0] * fx_ / in_point[2] + cx_);
  auto v = int(in_point[1] * fy_ / in_point[2] + cy_);

  return cv::Point2f(u, v);
}


// std::tuple<Eigen::Vector3d, Eigen::MatrixXd> det3Dbox::lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
//     Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
//     Eigen::MatrixXd V = svd.matrixV();
//     Eigen::VectorXd singularValues = svd.singularValues();

//     // Pseudo-inverse of A
//     Eigen::MatrixXd A_inv = V * singularValues.asDiagonal().inverse() * svd.matrixU().transpose();

//     // Least squares solution
//     Eigen::Vector3d x = A_inv * b;
//     Eigen::MatrixXd residual = A * x - b;

//     return std::make_tuple(x, residual);
// }

std::tuple<Eigen::Vector3d, Eigen::MatrixXd> det3Dbox::lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector3d x = svd.solve(b);
    Eigen::MatrixXd residual = A * x - b;
    return std::make_tuple(x, residual);
}

void det3Dbox::calc_location(const Eigen::Vector3d& dimension,
                            const std::vector<cv::Point2f>& box_2d,
                            const double &alpha, const double &theta_ray,
                            Eigen::Vector3d& locationOut)
 {
    // Calculate orientation
    double orient = alpha + theta_ray;
    Eigen::Matrix3d R = rotation_matrix(orient);
    // std::cout << "R: " << R << std::endl;

    // Extract 2D box corners
    double xmin = box_2d[0].x, ymin = box_2d[0].y, xmax = box_2d[1].x, ymax = box_2d[1].y;
    std::vector<double> box_corners = {xmin, ymin, xmax, ymax};

    std::vector<Eigen::Vector3d> left_constraints, right_constraints, top_constraints, bottom_constraints;

    // Using a different coordinate system
    double dx = dimension[2] / 2, dy = dimension[0] / 2, dz = dimension[1] / 2;

    // Determine multipliers based on the relative angle
    int left_mult = 1, right_mult = -1;

    if(alpha < 1.605703 && alpha > 1.535890){
        left_mult = 1;
        right_mult = 1;
    }
    else if(alpha < -1.535890 && alpha > -1.605703){
        left_mult = -1;
        right_mult = -1;
    }
    else if(alpha < 1.570796 && alpha > -1.570796){
        left_mult = -1;
        right_mult = 1;
    }
    if (alpha > 0) {
        left_mult = -1;
        right_mult = 1;
    }
    int switch_mult = (alpha > 0) ? 1 : -1;

    // Generate left and right constraints
    for (int i : {-1, 1}) {
        left_constraints.push_back({left_mult * dx, i * dy, -switch_mult * dz});
        right_constraints.push_back({right_mult * dx, i * dy, switch_mult * dz});
    }

    // Generate top and bottom constraints
    for (int i : {-1, 1}) {
        for (int j : {-1, 1}) {
            top_constraints.push_back({i * dx, -dy, j * dz});
            bottom_constraints.push_back({i * dx, dy, j * dz});
        }
    }

    // Generate all 64 combinations of constraints
    std::vector<std::vector<Eigen::Vector3d>> constraints;
    for (const auto& left : left_constraints) {
        for (const auto& top : top_constraints) {
            for (const auto& right : right_constraints) {
                for (const auto& bottom : bottom_constraints) {
                    constraints.push_back({left, top, right, bottom});
                }
            }
        }
    }

    // Filter out duplicates in constraints
    constraints.erase(std::unique(constraints.begin(), constraints.end(), compareVectors()), constraints.end());

    // Initialize variables for best location estimation
    Eigen::Matrix4d pre_M = Eigen::Matrix4d::Identity();
    double best_error = std::numeric_limits<double>::infinity();
    std::vector<Eigen::Vector3d> best_X;
    
    int count = 1;
    // Loop through each possible constraint and find the best estimate
    for (const auto& constraint : constraints) {
        std::vector<Eigen::Vector3d> X_array = constraint;

        // Create A, b
        Eigen::Matrix<double, 4, 3> A;
        std::array<int, 4> indicies = {0, 1, 0, 1};
        Eigen::Vector4d b;

        for (int row = 0; row < 4; ++row) {
            Eigen::Matrix4d temp_M = pre_M;
            Eigen::Matrix<double, 3, 4> M;
            auto X = X_array[row];

            Eigen::Vector3d RX = R * X;
            temp_M.topRightCorner<3, 1>() = RX;
            M = proj_matrix * temp_M;
            A.row(row) = (M(indicies[row] % 2, Eigen::seq(0, 3)) - box_corners[row] * M(2, Eigen::seq(0, 3))).transpose();
            b(row) = box_corners[row] * M(2, 3) - M(indicies[row], 3);
        }

        Eigen::Vector3d loc;
        Eigen::MatrixXd error;
        std::tie(loc, error) = lstsq(A, b);
        if (error.norm() < best_error) {
            count++;
            locationOut = loc;
            best_error = error.norm();
            best_X = X_array;
        }
     }
}

Eigen::Matrix3d det3Dbox::rotation_matrix(const double &yaw){
    const double pitch = 0;
    const double roll = 0;

    double tx = roll;
    double ty = yaw;
    double tz = pitch;

    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0,
          0, cos(tx), -sin(tx),
          0, sin(tx), cos(tx);

    Eigen::Matrix3d Ry;
    Ry << cos(ty), 0, sin(ty),
          0, 1, 0,
          -sin(ty), 0, cos(ty);

    Eigen::Matrix3d Rz;
    Rz << cos(tz), -sin(tz), 0,
          sin(tz), cos(tz), 0,
          0, 0, 1;

    return Ry;
}


void det3Dbox::plot3dBox(const cv::Mat& img, 
                        const double &yaw,
                        const Eigen::Vector3d& dimension,
                        const Eigen::Vector3d& center){
    
     Eigen::Matrix3d R = rotation_matrix(yaw);  //in camera vision
    std::vector<std::vector<double>> corners = create_corners(dimension, R, center);
    std::vector<cv::Point2f> box_3d;
    for(auto &corner: corners){
        box_3d.push_back(ProjectPoint(corner));
    }
    drawLines(img, {box_3d[0], box_3d[2], box_3d[6], box_3d[4], box_3d[0]}, cv_colors::GREEN, 1);
    drawLines(img, {box_3d[1], box_3d[3], box_3d[7], box_3d[5], box_3d[1]}, cv_colors::GREEN, 1);
    drawLines(img, {box_3d[0], box_3d[4], box_3d[5], box_3d[1], box_3d[3], box_3d[7], box_3d[6], box_3d[2], box_3d[0]}, cv_colors::GREEN, 1);

    std::vector<cv::Point2i> front_mark{box_3d[0], box_3d[1], box_3d[2], box_3d[3]};
    drawLines(img, {front_mark[0], front_mark[3]}, cv_colors::BLUE, 1);
    drawLines(img, {front_mark[1], front_mark[2]}, cv_colors::BLUE, 1);
}


void det3Dbox::plot_regressed_3d_bbox(const cv::Mat& img,
                                    const std::vector<cv::Point2f>& box2D,
                                    const Eigen::Vector3d& dimensions,
                                    const double &alpha,
                                    const double &theta_ray,
                                    Eigen::Vector3d& location,
                                    double &orient){

    calc_location(dimensions, box2D, alpha, theta_ray, location);
    orient = alpha + theta_ray;
    if(use_real_data_){
        orient = M_PI / 2;
        float distance = std::sqrt(location[0] * location[0] + location[1] * location[1]);
        if(distance >= 5 && distance <= 25 && dimensions[2] <= 10 && dimensions[0] <= 3 && dimensions[1] <= 3){
            plot3dBox(img, orient, dimensions, location);
        }
    }else{
        plot3dBox(img, orient, dimensions, location);
    }
    // location = camera2lidar_matrix * location;
}


void det3Dbox::doInference(nvinfer1::IExecutionContext& context, 
                            float* input, 
                            float* orientation_output, 
                            float* confidence_output,
                            float* dimension_output,
                            int batchSize)
{
        const ICudaEngine& engine = context.getEngine();
 
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbBindings() == 2);
        void* buffers[4];
 
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(IN_NAME);
        const int orientationIndex = engine.getBindingIndex(OUT_ORIENTATION);
        const int confidenceIndex = engine.getBindingIndex(OUT_CONFIDENCE);
        const int dimensionIndex = engine.getBindingIndex(OUT_DIMENSION);
        // const int outputIndex = engine.getBindingIndex(OUT_NAME);
 
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[orientationIndex], batchSize * 2 * 2 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[confidenceIndex], batchSize * 1 * 2 * sizeof(float)));
        CHECK(cudaMalloc(&buffers[dimensionIndex], batchSize * 3 * sizeof(float)));
        // CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W /4 * sizeof(float)));
 
        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
 
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(orientation_output, buffers[orientationIndex], batchSize * 1 * 2 * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(confidence_output, buffers[confidenceIndex], batchSize * 1 * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(dimension_output, buffers[dimensionIndex], batchSize * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        // CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
 
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[orientationIndex]));
        CHECK(cudaFree(buffers[confidenceIndex]));
        CHECK(cudaFree(buffers[dimensionIndex]));
}

}
