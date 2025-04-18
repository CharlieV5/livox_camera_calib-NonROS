#pragma once
#ifndef _CALIBRATION_H_
#define _CALIBRATION_H_

#include "common.h"
#include <Eigen/Core>
#include <fstream>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <sstream>

#include <stdio.h>
#include <string>
#include <time.h>
#include <unordered_map>

class Calibration
{
public:
	enum ProjectionType
	{
		DEPTH,
		INTENSITY,
		BOTH
	};
	enum Direction
	{
		UP,
		DOWN,
		LEFT,
		RIGHT
	};

	Calibration(const std::string& image_file, const std::string& pcd_file, const std::string& calib_config_file, const std::string& camera_yaml);

	bool loadCameraConfig(const std::string& camera_file);
	bool loadCalibConfig(const std::string& config_file);
	bool loadConfig(const std::string& configFile);
	bool checkFov(const cv::Point2d& p);

	void colorCloud(const Vector6d& extrinsic_params, const int density,
		const cv::Mat& rgb_img,
		const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
		pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& color_cloud);

	void edgeDetector(const int& canny_threshold, const int& edge_threshold,
		const cv::Mat& src_img, cv::Mat& edge_img,
		pcl::PointCloud<pcl::PointXYZ>::Ptr& edge_cloud);

	void projection(const Vector6d& extrinsic_params,
		const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
		const ProjectionType projection_type, const bool is_fill_img,
		cv::Mat& projection_img);

	void calcLine(const std::vector<Plane>& plane_list, const double voxel_size,
		const Eigen::Vector3d origin,
		std::vector<pcl::PointCloud<pcl::PointXYZI>>& line_cloud_list);

	cv::Mat fillImg(const cv::Mat& input_img, const Direction first_direct, const Direction second_direct);

	void buildPnp(const Vector6d& extrinsic_params, const int dis_threshold,
		const bool show_residual,
		const pcl::PointCloud<pcl::PointXYZ>::Ptr& cam_edge_cloud_2d,
		const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
		std::vector<PnPData>& pnp_list);
	
	void buildVPnp(const Vector6d& extrinsic_params, const int dis_threshold,
			const bool show_residual,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr& cam_edge_cloud_2d,
			const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d,
			std::vector<VPnPData>& pnp_list);

	cv::Mat getConnectImg(const int dis_threshold,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_edge_cloud);

	cv::Mat getProjectionImg(const Vector6d& extrinsic_params);

	void initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
		const float voxel_size,
		std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map);

	void LiDAREdgeExtraction(
		const std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map,
		const float ransac_dis_thre, const int plane_size_threshold,
		pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d);

	void calcDirection(const std::vector<Eigen::Vector2d>& points, Eigen::Vector2d& direction);
	// void calcResidual(const Vector6d &extrinsic_params,
	//                   const std::vector<VPnPData> vpnp_list,
	//                   std::vector<float> &residual_list);
	// void calcCovarance(const Vector6d &extrinsic_params,
	//                    const VPnPData &vpnp_point, const float pixel_inc,
	//                    const float range_inc, const float degree_inc,
	//                    Eigen::Matrix2f &covarance);


public:
	int rgb_edge_minLen_ = 200;
	int rgb_canny_threshold_ = 20;
	int min_depth_ = 2.5;
	int max_depth_ = 80;
	int plane_max_size_ = 5;
	float detect_line_threshold_ = 0.02;
	int line_number_ = 0;
	int color_intensity_threshold_ = 5;
	Eigen::Vector3d adjust_euler_angle_;

	
	//  相机内参
	int m_cam_model = 0; // 0 : pinhole, 1 : fisheye
	int width_, height_;
	cv::Mat camera_matrix_;
	cv::Mat dist_coeffs_;

	// 初始外参
	cv::Mat init_extrinsic_;
	// 初始旋转矩阵
	Eigen::Matrix3d init_rotation_matrix_;
	// 初始平移向量
	Eigen::Vector3d init_translation_vector_;

	int is_use_custom_msg_;
	float voxel_size_ = 1.0;
	float down_sample_size_ = 0.02;
	float ransac_dis_threshold_ = 0.02;
	float plane_size_threshold_ = 60;
	float theta_min_;
	float theta_max_;
	float direction_theta_min_;
	float direction_theta_max_;
	float min_line_dis_threshold_ = 0.03;
	float max_line_dis_threshold_ = 0.06;

	string image_path_;
	cv::Mat rgb_image_;
	cv::Mat image_;
	cv::Mat grey_image_;
	// 裁剪后的灰度图像
	cv::Mat cut_grey_image_;


	// 存储从pcd/bag处获取的原始点云
	pcl::PointCloud<pcl::PointXYZI>::Ptr raw_lidar_cloud_;

	// 存储平面交接点云
	pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud_;
	std::vector<int> plane_line_number_;
	// 存储RGB图像边缘点的2D点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_cloud_;
	// 存储LiDAR Depth/Intensity图像边缘点的2D点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_edge_cloud_;
};

#endif