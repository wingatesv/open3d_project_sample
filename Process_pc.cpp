#include <iostream>
#include <filesystem>
#include <open3d/Open3D.h>
#include <vector>
#include <string>
#include <algorithm>
#include <armadillo>
#include <sstream>
#include "json.hpp"
#include <regex>

namespace fs = std::filesystem;

struct Point {
    float x, y, z, intensity;
};

/**
 * Process Point Clouds: Downsample and Remove Outliers
 *
 * This function processes point cloud data in PLY format. It downsamples the point clouds and
 * removes statistical outliers, saving the filtered point clouds in the specified output directory.
 *
 * @param inputDir     The directory containing input PLY files.
 * @param tempDir      The output directory for processed point clouds.
 */

void processPC(const std::string& inputDir, const std::string& tempDir) {
    try {
        // Check if the input directory exists
        if (!std::filesystem::is_directory(inputDir)) {
            std::cerr << "Error: Input directory does not exist." << std::endl;
            return;
        }

        std::cout << "Downsampling and Outlier Removal.... " << std::endl;

        // List all PLY files in the input directory
        std::vector<std::string> fileNames;
        int fileCount = 0; // Initialize a counter for the files

        for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".ply") {
                ++fileCount; // Increment the file count for each valid file
                fileNames.push_back(entry.path().string());
            }
        }

        if (fileCount == 0) {
            std::cout << "No PLY files found in the input directory." << std::endl;
            return;
        }

        std::cout << "Found a total of " << fileCount << " PLY files" << std::endl; // Print the total count

        // Check if the output directory exists, create it if not
        if (!std::filesystem::is_directory(tempDir)) {
            std::filesystem::create_directory(tempDir);

            // Process each .ply file
            for (const std::string& filePath : fileNames) {
                std::string fileName = std::filesystem::path(filePath).filename();
                std::string savePath = std::filesystem::path(tempDir) / fileName;

                // Read the point cloud
                auto pcd = open3d::io::CreatePointCloudFromFile(filePath);

                // Check if the point cloud was successfully read
                if (!pcd) {
                    std::cerr << "Error reading the point cloud from file: " << filePath << std::endl;
                    continue; // Skip to the next file
                }

                // Uniform downsample
                auto uniDownPcd = pcd->UniformDownSample(8);

                // Remove statistical outlier
                auto [cl, ind] = uniDownPcd->RemoveStatisticalOutliers(5, 1.0);
                auto inlierCloud = uniDownPcd->SelectByIndex(ind);

                // Save the filtered point cloud
                if (open3d::io::WritePointCloud(savePath, *inlierCloud)) {
                    //std::cout << "Processed Successfully: " << savePath << std::endl;
                } else {
                    std::cerr << "Error saving the processed point cloud to: " << savePath << std::endl;
                }
            }
        } else {
            std::cout << "The output directory already exists. Please remove or specify a different tempDir." << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "An exception occurred: " << ex.what() << std::endl;
    }
}


/**
 * Split Point Clouds and Labels
 *
 * This function processes point cloud data in PLY format, splits the data into two parts based on x-coordinate,
 * and updates associated JSON labels. The resulting parts are saved with new filenames and paths.
 *
 * @param ply_dir     The directory containing input PLY files.
 * @param label_dir   The directory containing associated JSON label files.
 */
void splitPC(std::string ply_dir, std::string label_dir) {
    try {
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
                try {
                    // Construct file paths
                    fs::path file_path = fs::path(ply_dir) / file_name;
                    fs::path save_path_1 = fs::path(ply_dir) / (fs::path(file_name).stem().string() + "_part1.ply");
                    fs::path save_path_2 = fs::path(ply_dir) / (fs::path(file_name).stem().string() + "_part2.ply");
                    std::vector<fs::path> save_paths = {save_path_1, save_path_2};

                    // Read the point cloud
                    auto pcd = open3d::io::CreatePointCloudFromFile(file_path);

                    if (!pcd) {
                        std::cerr << "Error reading the point cloud from file: " << file_name << std::endl;
                        continue; // Skip to the next file
                    }

                    // Get the number of points in the point cloud
                    size_t num_points = pcd->points_.size();

                    // Convert points to Eigen matrix
                    Eigen::MatrixXd points(pcd->points_.size(), 3);
                    for (size_t i = 0; i < pcd->points_.size(); ++i) {
                        points.row(i) = pcd->points_[i];
                    }

                    // Convert points to Armadillo matrix
                    arma::mat arma_points(pcd->points_.size(), 3);
                    for (size_t i = 0; i < pcd->points_.size(); ++i) {
                        std::vector<double> point_vec(3);
                        for (int j = 0; j < 3; ++j) {
                            point_vec[j] = pcd->points_[i][j];
                        }
                        arma_points.row(i) = arma::conv_to<arma::rowvec>::from(point_vec);
                    }

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

                    // Process associated JSON label files
                    try {
                        std::string ori_label_path = label_dir + "/" + file_name.substr(0, file_name.find_last_of(".")) + ".json";
                        std::string label_path_1 = label_dir + "/" + fs::path(file_name).stem().string() + "_part1.json";
                        std::string label_path_2 = label_dir + "/" + fs::path(file_name).stem().string() + "_part2.json";
                        std::vector<std::string> label_paths = {label_path_1, label_path_2};

                        nlohmann::json data;
                        std::ifstream ori_label_file(ori_label_path);
                        ori_label_file >> data;

                        auto &objs = data["objects"];
                        nlohmann::json objs_part_1;
                        nlohmann::json objs_part_2;

                        for (auto &obj : objs) {
                            double x_value = obj["centroid"]["x"].get<double>();
                            if (x_value > x_mid_thresh) {
                                obj["centroid"]["x"] = x_value - x_mid_thresh;
                                objs_part_1.push_back(obj);
                            } else {
                                obj["centroid"]["x"] = x_value - x_min_thresh * x_range;
                                objs_part_2.push_back(obj);
                            }
                        }

                        for (size_t i = 0; i < label_paths.size(); ++i) {
                            nlohmann::json new_data = data;
                            new_data["objects"] = (i == 0) ? objs_part_1 : objs_part_2;
                            std::string original_filename = new_data["filename"].get<std::string>();
                            std::string new_filename = std::regex_replace(original_filename, std::regex("\\.ply$"), "_part" + std::to_string(i + 1) + ".ply");
                            new_data["filename"] = new_filename;
                            std::string path = new_data["path"].get<std::string>();
                            std::string new_path = std::regex_replace(path, std::regex("\\.ply$"), "_part" + std::to_string(i + 1) + ".ply");
                            new_data["path"] = new_path;

                            std::ofstream label_file(label_paths[i]);
                            label_file << new_data.dump(4);
                        }

                        fs::remove(ori_label_path);
                    } catch (const std::exception &e) {
                        std::cerr << "Error processing label for file: " << file_name << " - " << e.what() << std::endl;
                    }

                    for (size_t i = 0; i < inlier_cloud_nps.size(); ++i) {
                        auto splitted_pcd = std::make_shared<open3d::geometry::PointCloud>();

                        Eigen::MatrixXd new_points(inlier_cloud_nps[i].n_rows, inlier_cloud_nps[i].n_cols);
                        for (size_t j = 0; j < inlier_cloud_nps[i].n_rows; ++j) {
                            for (size_t k = 0; k < inlier_cloud_nps[i].n_cols; ++k) {
                                new_points(j, k) = inlier_cloud_nps[i](j, k);
                            }
                        }

                        splitted_pcd->points_.resize(new_points.rows());
                        for (int j = 0; j < new_points.rows(); ++j) {
                            splitted_pcd->points_[j] = new_points.row(j);
                        }

                        open3d::io::WritePointCloud(save_paths[i], *splitted_pcd);
                        ++fileCount;
                    }

                    fs::remove(file_path);
                } catch (const std::exception &e) {
                    std::cerr << "Error processing label or point cloud: " << e.what() << std::endl;
                }
            }

            std::cout << "Splitted a total of " << fileCount << " PLY files" << std::endl;
        }
    } catch (const std::exception &ex) {
        std::cerr << "An exception occurred: " << ex.what() << std::endl;
    }
}




struct Object {
    double x, y, z, dx, dy, dz, rot;
    int cls;
};

/**
 * JSON to Text Conversion
 *
 * This function converts object information from a JSON file to a text file with magnified values and class labels.
 *
 * @param json_file_path  The path to the input JSON file.
 * @param magnifyFactor   The factor to magnify object dimensions and coordinates.
 * @param output_filename The path to the output text file.
 */
void convertJsonToText(const std::string& json_file_path, const float& magnifyFactor, const std::string& output_filename) {
    try {
        nlohmann::json json_data;

        // Read the JSON file
        std::ifstream jsonFile(json_file_path);
        jsonFile >> json_data;

        std::vector<Object> objects;

        // Extract object information from JSON
        for (const auto& obj : json_data["objects"]) {
            Object object;
            object.x = obj["centroid"]["x"].get<double>() * magnifyFactor;
            object.y = obj["centroid"]["y"].get<double>() * magnifyFactor;
            object.z = obj["centroid"]["z"].get<double>() * magnifyFactor;
            object.dx = obj["dimensions"]["length"].get<double>() * magnifyFactor;
            object.dy = obj["dimensions"]["width"].get<double>() * magnifyFactor;
            object.dz = obj["dimensions"]["height"].get<double>() * magnifyFactor;
            object.rot = obj["rotations"]["z"].get<double>();
            object.cls = (obj["name"].get<std::string>() == "good") ? 0 : 1;
            objects.push_back(object);
        }

        // Check if objects were successfully extracted
        if (objects.empty()) {
            std::cerr << "No objects found in the JSON file: " << json_file_path << std::endl;
            return;
        }

        // Create the output file
        std::ofstream outputFile(output_filename);

        // Write object information to the text file
        for (const auto& object : objects) {
            outputFile << object.x << " " << object.y << " " << object.z << " " << object.dx << " " << object.dy << " "
                       << object.dz << " " << object.rot << " " << object.cls << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "An exception occurred: " << ex.what() << std::endl;
    }
}


/**
 * Convert PLY and JSON to Text
 *
 * This function converts PLY files to BIN files and their associated JSON files to TXT files with magnified values.
 *
 * @param input_dir        The directory containing the PLY and JSON files.
 * @param output_dir       The directory to store the output BIN files.
 * @param label_dir        The directory containing the JSON label files.
 * @param output_label_dir The directory to store the output TXT label files.
 */
void convert_ply_and_json_to_text(const std::string& input_dir, const std::string& output_dir, const std::string& label_dir, const std::string& output_label_dir) {
    try {
        int file_number = 0;  // Start from 0

        // Create the output directory if it doesn't exist
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directory(output_dir);
        }

        // Create the output label directory if it doesn't exist
        if (!std::filesystem::exists(output_label_dir)) {
            std::filesystem::create_directory(output_label_dir);
        }

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
                std::string label_filename = output_label_dir + "/" + index_str + ".txt";

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

                // Save BIN file
                std::ofstream output_file(output_filename, std::ios::binary);
                output_file.write(reinterpret_cast<const char*>(data.data()), num_points * sizeof(float) * 4);
                output_file.close();

                // Process associated JSON label files
                std::string json_file_path = label_dir + "/" + entry.path().stem().string() + ".json";
                convertJsonToText(json_file_path, magnify_factor, label_filename);

                ++file_number;
            }
        }

        std::cout << "PLY to BIN and JSON to TXT Conversion completed for " << file_number << " files" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "An exception occurred: " << ex.what() << std::endl;
    }
}


/**
 * Main Function
 *
 * This function processes PLY and JSON files from the specified directories and converts them to BIN and TXT formats.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 on successful execution, 1 on error.
 */
int main(int argc, char** argv) {
    try {
        // Get arguments from the user
        if (argc != 9) {
            std::cerr << "Usage: " << argv[0] << " --input-dir <input_folder> --output-dir <output_folder> --label-dir <label_folder> --output-label-dir <output_label_folder>\n";
            return 1;
        }

        std::string inputDir;
        std::string outputDir;
        std::string labelDir;
        std::string outputLabelDir;

        for (int i = 1; i < argc; i += 2) {
            std::string arg = argv[i];
            if (arg == "--input-dir") {
                inputDir = argv[i + 1];
            } else if (arg == "--output-dir") {
                outputDir = argv[i + 1];
            } else if (arg == "--label-dir") {
                labelDir = argv[i + 1];
            } else if (arg == "--output-label-dir") {
                outputLabelDir = argv[i + 1];
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
        }

        const std::string temp_dir = "/content/temp/";  // Directory to save output files

        processPC(inputDir, temp_dir);

        splitPC(temp_dir, labelDir);

        convert_ply_and_json_to_text(temp_dir, outputDir, labelDir, outputLabelDir);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "An exception occurred: " << ex.what() << std::endl;
        return 1;
    }
}

