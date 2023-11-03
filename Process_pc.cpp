#include <iostream>
#include <filesystem>
#include <open3d/Open3D.h>
#include <vector>
#include <string>
#include <algorithm>
#include <armadillo>
#include <sstream>

namespace fs = std::filesystem;

struct Point {
    float x, y, z, intensity;
};

void processPC(std::string inputDir, std::string tempDir) {
    // List all files in the input directory
    std::cout << "Downsampling and Outlier Removal.... " << std::endl;
    std::vector<std::string> fileNames;
    int fileCount = 0; // Initialize a counter for the files

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".ply") {
            ++fileCount; // Increment the file count for each valid file
            fileNames.push_back(entry.path().string());
        }
    }

    std::cout << "Found a total of " << fileCount << " PLY files" << std::endl; // Print the total count

    // Check if the output directory exists, create it if not
    if (!fs::is_directory(tempDir)) {
        fs::create_directory(tempDir);

        // Process each .ply file
        for (const std::string& filePath : fileNames) {
            std::string fileName = fs::path(filePath).filename();
            std::string savePath = fs::path(tempDir) / fileName;

            // Read the point cloud
            auto pcd = open3d::io::CreatePointCloudFromFile(filePath);

            // Uniform downsample
            auto uniDownPcd = pcd->UniformDownSample(8);

            // Remove statistical outlier
            auto [cl, ind] = uniDownPcd->RemoveStatisticalOutliers(5, 1.0);
            auto inlierCloud = uniDownPcd->SelectByIndex(ind);

            // Save the filtered point cloud
            open3d::io::WritePointCloud(savePath, *inlierCloud);
            //std::cout << "Processed Successfully: " << savePath << std::endl; // Print the file path

        }


    } else {
        std::cout << "Apparently, you have preprocessed the input data previously! Please check.\n";
    }
 

}

void splitPC(std::string ply_dir) {
    // Get list of .ply files in the input directory
    std::vector<std::string> file_names;
    std::cout << "Splitting Point Clouds " << std::endl;
    for (const auto &entry : fs::directory_iterator(ply_dir)) {
        if (entry.path().extension() == ".ply") {
            file_names.push_back(entry.path().filename());
        }
    }
    std::sort(file_names.begin(), file_names.end());

    // Check if the data has already been split
    if (file_names[0].find("part1") != std::string::npos || file_names[0].find("part2") != std::string::npos) {
        std::cout << "Apparently, you have split the preprocessed data previously! Please check" << std::endl;
    } else {
        // Process each .ply file
        int fileCount = 0; // Initialize a counter for the files
        for (const std::string &file_name : file_names) {

            // Construct file paths
            fs::path file_path = fs::path(ply_dir) / file_name;
            fs::path save_path_1 = fs::path(ply_dir) / (fs::path(file_name).stem().string() + "_part1.ply");
            fs::path save_path_2 = fs::path(ply_dir) / (fs::path(file_name).stem().string() + "_part2.ply");
            std::vector<fs::path> save_paths = {save_path_1, save_path_2};

            //std::cout << save_path_1 << std::endl;
            //std::cout << save_path_2 << std::endl;


            // Read the point cloud
            auto pcd = open3d::io::CreatePointCloudFromFile(file_path);

            // Get the number of points in the point cloud
            size_t num_points = pcd->points_.size();
            //std::cout << "Number of points: " << num_points << std::endl;

            // Convert points to Eigen matrix
            Eigen::MatrixXd points(pcd->points_.size(), 3);
            for (size_t i = 0; i < pcd->points_.size(); ++i) {
                points.row(i) = pcd->points_[i];
            }

            // Convert points to Armadillo matrix
            arma::mat arma_points(pcd->points_.size(), 3);
            for (size_t i = 0; i < pcd->points_.size(); ++i) {
                // Convert Eigen::Vector3d to std::vector
                std::vector<double> point_vec(3);
                for (int j = 0; j < 3; ++j) {
                    point_vec[j] = pcd->points_[i][j];
                }

                // Convert std::vector to arma::rowvec
                arma_points.row(i) = arma::conv_to<arma::rowvec>::from(point_vec);
            }

            // Print the shape of arma_points
            //std::cout << "Shape of arma_points: " << arma_points.n_rows << " rows x " << arma_points.n_cols << " columns" << std::endl;


            // Split points into two parts based on x-coordinate
            double x_range = 5.13;
            double x_mid_thresh = x_range / 2;
            double x_min_thresh = 0.23;
            double x_max_thresh = 0.77;

            arma::uvec part_1_indices = arma::find(arma_points.col(0) > x_mid_thresh && arma_points.col(0) < x_max_thresh * x_range);
            arma::mat part_1 = arma_points.rows(part_1_indices);
            part_1.col(0) -= x_mid_thresh;

            arma::uvec part_2_indices = arma::find(arma_points.col(0) <= x_mid_thresh && arma_points.col(0) > x_min_thresh * x_range);
            arma::mat part_2 = arma_points.rows(part_2_indices);
            part_2.col(0) -= x_min_thresh * x_range;

            std::vector<arma::mat> inlier_cloud_nps = {part_1, part_2};

            // Print out the shape of part_1 and part_2
            //std::cout << "part_1 has " << part_1.n_rows << " rows and " << part_1.n_cols << " columns." << std::endl;
            //std::cout << "part_2 has " << part_2.n_rows << " rows and " << part_2.n_cols << " columns." << std::endl;
           
           for (size_t i = 0; i < inlier_cloud_nps.size(); ++i) {


              auto splitted_pcd = std::make_shared<open3d::geometry::PointCloud>();


              // Convert arma::mat to Eigen::MatrixXd
              Eigen::MatrixXd new_points(inlier_cloud_nps[i].n_rows, inlier_cloud_nps[i].n_cols);
              for (size_t j = 0; j < inlier_cloud_nps[i].n_rows; ++j) {
                  for (size_t k = 0; k < inlier_cloud_nps[i].n_cols; ++k) {
                      new_points(j, k) = inlier_cloud_nps[i](j, k);
                  }
              }

              // Pass xyz to Open3D.o3d.geometry.PointCloud
              splitted_pcd->points_.resize(new_points.rows());
              for (int j = 0; j < new_points.rows(); ++j) {
                  splitted_pcd->points_[j] = new_points.row(j);
              }


              // Save the point cloud
              open3d::io::WritePointCloud(save_paths[i], *splitted_pcd);
              ++fileCount; // Increment the file count for each saved file file
   
            }


        // Remove original .ply file
        //std::cout <<  file_path << " is deleted " << std::endl;
        fs::remove(file_path);


        }

      std::cout << "Splitted a total of " << fileCount << " PLY files" << std::endl; // Print the total count
    }

}



void convert_ply(const std::string& input_dir, const std::string& output_dir) {
    int file_number = 0;  // Start from 0

    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".ply") {
            std::string input_path = entry.path().string();

            // Create the output filename with zero-padding using std::stringstream
            std::stringstream ss;
            ss << file_number;
            int n_zero = 6;
            std::string index_str = ss.str();
            while (index_str.length() < n_zero) {
                index_str = "0" + index_str;
            }
            std::string output_filename = output_dir + "/" + index_str + ".bin";


            // Read data
            auto point_cloud = open3d::io::CreatePointCloudFromFile(input_path);

            // Convert to DataFrame
            std::vector<Point> points;
            for (int i = 0; i < point_cloud->points_.size(); ++i) {
                Point point;
                point.x = point_cloud->points_.at(i)(0); // Access x
                point.y = point_cloud->points_.at(i)(1); // Access y
                point.z = point_cloud->points_.at(i)(2); // Access z
                point.intensity = 0.0f;
                points.push_back(point);
            }

            // Magnify coordinates
            const float magnify_factor = 20.0f;
            for (auto& point : points) {
                point.x *= magnify_factor;
                point.y *= magnify_factor;
                point.z *= magnify_factor;
            }

            // Record point cloud range
            std::vector<float> x_values, y_values, z_values;
            for (const auto& point : points) {
                x_values.push_back(point.x);
                y_values.push_back(point.y);
                z_values.push_back(point.z);
            }
            const float min_x = *std::min_element(x_values.begin(), x_values.end());
            const float min_y = *std::min_element(y_values.begin(), y_values.end());
            const float min_z = *std::min_element(z_values.begin(), z_values.end());
            const float max_x = *std::max_element(x_values.begin(), x_values.end());
            const float max_y = *std::max_element(y_values.begin(), y_values.end());
            const float max_z = *std::max_element(z_values.begin(), z_values.end());

            // Initialize array to store data
            const std::size_t num_points = points.size();
            std::vector<float> data(num_points * 4);

            // Read data by property
            for (std::size_t i = 0; i < num_points; ++i) {
                data[i * 4 + 0] = points[i].x;
                data[i * 4 + 1] = points[i].y;
                data[i * 4 + 2] = points[i].z;
                data[i * 4 + 3] = points[i].intensity;
            }

            // Save
            std::ofstream output_file(output_filename, std::ios::binary);
            output_file.write(reinterpret_cast<const char*>(data.data()), num_points * sizeof(float) * 4);
            output_file.close();
      

            ++file_number;
        }
    }

    std::cout << "PLY to BIN Conversion completed for " << file_number << " files" << std::endl;
}

int main(int argc, char** argv) {
    // Get arguments from the user
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " --input-dir <input_folder> --output-dir <output_folder>\n";
        return 1;
    }

    std::string inputDir;
    std::string outputDir;

    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "--input-dir") {
            inputDir = argv[i + 1];
        } else if (arg == "--output-dir") {
            outputDir = argv[i + 1];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    const std::string temp_dir = "/content/temp/";  // Directory to save output files

    processPC(inputDir, temp_dir);
    
    splitPC(temp_dir);

    // Create the output directory if it doesn't exist
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directory(outputDir);
    }
    convert_ply(temp_dir, outputDir);


    return 0;
}
