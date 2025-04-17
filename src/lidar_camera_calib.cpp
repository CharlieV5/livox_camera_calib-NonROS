#include "ceres/ceres.h"
#include "common.h"
#include "Calibration.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include <pcl/io/pcd_io.h>
#include <filesystem>

#include "calib_cost.h"
#include <yaml-cpp/yaml.h>

using namespace std;


void roughCalib(Calibration& calib, Vector6d& calib_params, double search_resolution, int max_iter);

int main(int argc, char** argv)
{
	string cam_int_yaml_file = argv[1];
	string calib_config_file = argv[2];

	// Data path
	string image_file;
	string pcd_file;
	string result_file;

	fs::path dir = fs::path(cam_int_yaml_file).parent_path();

	std::vector<fs::path> files = get_files_by_extensions(dir.string(), {"png", "pcd"});
	if (files.size() != 2)
	{
		return 0;
	}

    for (auto file : files)
	{
		if (file.extension() == ".png")
			image_file = file.generic_string();
        if (file.extension() == ".pcd")
			pcd_file = file.generic_string();
	}


	double width;
	double height;

	// Calib config
	bool use_rough_calib = false;
	// instrins matrix
	Eigen::Matrix3d inner;
	// Distortion coefficient
	Eigen::Vector4d distor;
	Eigen::Vector4d quaternion;
	Eigen::Vector3d transation;

	auto calib_yaml = YAML::LoadFile(calib_config_file);
	use_rough_calib = calib_yaml["use_rough_calib"].as<bool>();

	Calibration calib(image_file, pcd_file, calib_config_file, cam_int_yaml_file);
	
	Eigen::Vector3d init_euler_angle = calib.init_rotation_matrix_.eulerAngles(2, 1, 0);
	Eigen::Vector3d init_transation = calib.init_translation_vector_;

	Vector6d calib_params;
	calib_params << 
		init_euler_angle(0), init_euler_angle(1), init_euler_angle(2),
		init_transation(0), init_transation(1), init_transation(2);

	std::vector<PnPData> pnp_list;
	std::vector<VPnPData> vpnp_list;

	printf("Finish prepare!\n");
	Eigen::Matrix3d R;
	Eigen::Vector3d T;
	for (int i=0; i<3; i++)
	{
		for (int j = 0; j < 3; j++)
		inner(i,j) = calib.camera_matrix_.at<double>(i, j);
	}

	for (int i = 0; i < 4; i++)
	{
		distor(i,0) = calib.dist_coeffs_.at<double>(i,0);
	}

	R = calib.init_rotation_matrix_;
	T = calib.init_translation_vector_;
	std::cout << "Initial rotation matrix:" << std::endl
		<< calib.init_rotation_matrix_ << std::endl;
	std::cout << "Initial translation:"
		<< calib.init_translation_vector_.transpose() << std::endl;
	bool use_vpnp = true;
	Eigen::Vector3d euler = R.eulerAngles(2, 1, 0);
	calib_params[0] = euler[0];
	calib_params[1] = euler[1];
	calib_params[2] = euler[2];
	calib_params[3] = T[0];
	calib_params[4] = T[1];
	calib_params[5] = T[2];

	cv::Mat init_img = calib.getProjectionImg(calib_params);
	cv::imshow("Initial extrinsic", init_img);

	cv::waitKey(1000);

	if (use_rough_calib)
	{
		roughCalib(calib, calib_params, DEG2RAD(0.1), 50);
	}
	cv::Mat test_img = calib.getProjectionImg(calib_params);
	cv::imshow("After rough extrinsic", test_img);
	cv::waitKey(1000);
	int iter = 0;
	// Maximum match distance threshold: 15 pixels
	// If initial extrinsic lead to error over 15 pixels, the algorithm will not
	// work
	int dis_threshold = 30;
	bool opt_flag = true;

	// Iteratively reducve the matching distance threshold
	for (dis_threshold = 30; dis_threshold > 10; dis_threshold -= 1)
	{
		// For each distance, do twice optimization
		for (int cnt = 0; cnt < 2; cnt++)
		{
			if (use_vpnp)
			{
				calib.buildVPnp(calib_params, dis_threshold, true,
					calib.rgb_egde_cloud_, calib.plane_line_cloud_,
					vpnp_list);
			}
			else
			{
				calib.buildPnp(calib_params, dis_threshold, true,
					calib.rgb_egde_cloud_, calib.plane_line_cloud_,
					pnp_list);
			}
			std::cout << "Iteration:" << iter++ << " Dis:" << dis_threshold
				<< " pnp size:" << vpnp_list.size() << std::endl;

			cv::Mat projection_img = calib.getProjectionImg(calib_params);
			cv::imshow("Optimization", projection_img);
			cv::waitKey(100);
			Eigen::Vector3d euler_angle(calib_params[0], calib_params[1], calib_params[2]);
			Eigen::Matrix3d opt_init_R;

			opt_init_R = Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitZ()) *
				Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitX());
			Eigen::Quaterniond q(opt_init_R);
			Eigen::Vector3d ori_t = T;

			double ext[7];
			ext[0] = q.x();
			ext[1] = q.y();
			ext[2] = q.z();
			ext[3] = q.w();
			ext[4] = T[0];
			ext[5] = T[1];
			ext[6] = T[2];
			Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
			Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

			ceres::LocalParameterization* q_parameterization =
				new ceres::EigenQuaternionParameterization();
			ceres::Problem problem;

			problem.AddParameterBlock(ext, 4, q_parameterization);
			problem.AddParameterBlock(ext + 4, 3);
			if (use_vpnp)
			{
				for (auto val : vpnp_list)
				{
					ceres::CostFunction* cost_function;
					cost_function = vpnp_calib::Create(val, inner, distor);
					problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
				}
			}
			else
			{
				for (auto val : pnp_list)
				{
					ceres::CostFunction* cost_function;
					cost_function = pnp_calib::Create(val, inner, distor);
					problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
				}
			}
			ceres::Solver::Options options;
			options.preconditioner_type = ceres::JACOBI;
			options.linear_solver_type = ceres::SPARSE_SCHUR;
			options.minimizer_progress_to_stdout = true;
			options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;
			Eigen::Matrix3d rot = m_q.toRotationMatrix();
			euler_angle = rot.eulerAngles(2, 1, 0);
			// std::cout << rot << std::endl;
			// std::cout << m_t << std::endl;
			calib_params[0] = euler_angle[0];
			calib_params[1] = euler_angle[1];
			calib_params[2] = euler_angle[2];
			calib_params[3] = m_t(0);
			calib_params[4] = m_t(1);
			calib_params[5] = m_t(2);
			R = rot;
			T[0] = m_t(0);
			T[1] = m_t(1);
			T[2] = m_t(2);
			Eigen::Quaterniond opt_q(R);
			std::cout << "q_dis:" << RAD2DEG(opt_q.angularDistance(q))
				<< " ,t_dis:" << (T - ori_t).norm() << std::endl;

		}
		if (!opt_flag)
		{
			break;
		}
	}

	R = Eigen::AngleAxisd(calib_params[0], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(calib_params[1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(calib_params[2], Eigen::Vector3d::UnitX());
	std::ofstream outfile(result_file);
	for (int i = 0; i < 3; i++)
	{
		outfile << R(i, 0) << "," << R(i, 1) << "," << R(i, 2) << "," << T[i]
			<< std::endl;
	}
	outfile << 0 << "," << 0 << "," << 0 << "," << 1 << std::endl;
	cv::Mat opt_img = calib.getProjectionImg(calib_params);
	cv::imshow("Optimization result", opt_img);

	fs::path new_dir(dir.string() + "/result");
	fs::create_directories(new_dir);

	string out_img = new_dir.string() + "/opt.png";
	cv::imwrite(out_img, opt_img);
	cv::waitKey(1000);
	Eigen::Matrix3d init_rotation;
	init_rotation << 0, -1.0, 0, 0, 0, -1.0, 1, 0, 0;
	Eigen::Matrix3d adjust_rotation;
	adjust_rotation = init_rotation.inverse() * R;
	Eigen::Vector3d adjust_euler = adjust_rotation.eulerAngles(2, 1, 0);

	pcl::PointCloud<pcl::PointXYZRGBL>::Ptr rgb_cloud2(new pcl::PointCloud<pcl::PointXYZRGBL>);
	calib.colorCloud(calib_params, 5, calib.image_, calib.raw_lidar_cloud_, rgb_cloud2);

	auto fs_pcd = std::filesystem::path(pcd_file);
	string str_dir = fs_pcd.parent_path().string();
	string str_filename = fs_pcd.filename().string() + "-rgb";	
	string out_pcd = new_dir.string() + "/" + str_filename + ".pcd";

	pcl::io::savePCDFileBinaryCompressed(out_pcd, *rgb_cloud2);

	std::cout << "finish" << std::endl;
	getchar();

	return 0;
}

void roughCalib(Calibration& calibra, Vector6d& calib_params,
	double search_resolution, int max_iter)
{
	float match_dis = 25;
	Eigen::Vector3d fix_adjust_euler(0, 0, 0);
	for (int n = 0; n < 2; n++)
	{
		for (int round = 0; round < 3; round++)
		{
			Eigen::Matrix3d rot;
			rot = Eigen::AngleAxisd(calib_params[0], Eigen::Vector3d::UnitZ()) *
				Eigen::AngleAxisd(calib_params[1], Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(calib_params[2], Eigen::Vector3d::UnitX());
			// std::cout << "init rot" << rot << std::endl;
			float min_cost = 1000;
			for (int iter = 0; iter < max_iter; iter++)
			{
				Eigen::Vector3d adjust_euler = fix_adjust_euler;
				adjust_euler[round] = fix_adjust_euler[round] +
					pow(-1, iter) * int(iter / 2) * search_resolution;
				Eigen::Matrix3d adjust_rotation_matrix;
				adjust_rotation_matrix =
					Eigen::AngleAxisd(adjust_euler[0], Eigen::Vector3d::UnitZ()) *
					Eigen::AngleAxisd(adjust_euler[1], Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(adjust_euler[2], Eigen::Vector3d::UnitX());
				Eigen::Matrix3d test_rot = rot * adjust_rotation_matrix;
				// std::cout << "adjust_rotation_matrix " << adjust_rotation_matrix
				//           << std::endl;
				Eigen::Vector3d test_euler = test_rot.eulerAngles(2, 1, 0);
				// std::cout << "test euler: " << test_euler << std::endl;
				Vector6d test_params;
				test_params << test_euler[0], test_euler[1], test_euler[2],
					calib_params[3], calib_params[4], calib_params[5];
				std::vector<VPnPData> vpnp_list;
				calibra.buildVPnp(test_params, match_dis, false,
					calibra.rgb_egde_cloud_, calibra.plane_line_cloud_,
					vpnp_list);
				float cost = (calibra.plane_line_cloud_->size() - vpnp_list.size()) *
					1.0 / calibra.plane_line_cloud_->size();
				if (cost < min_cost)
				{
					std::cout << "Rough calibration min cost:" << cost << std::endl;
					min_cost = cost;
					calib_params[0] = test_params[0];
					calib_params[1] = test_params[1];
					calib_params[2] = test_params[2];
					calibra.buildVPnp(calib_params, match_dis, true,
						calibra.rgb_egde_cloud_, calibra.plane_line_cloud_,
						vpnp_list);
					cv::Mat projection_img = calibra.getProjectionImg(calib_params);
					cv::imshow("Rough Optimization", projection_img);
					cv::waitKey(50);
				}
			}
		}
	}
}
