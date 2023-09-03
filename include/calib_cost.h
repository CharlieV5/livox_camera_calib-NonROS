#pragma once
#ifndef _CALIB_COST_H_
#define _CALIB_COST_H_

#include "Calibration.h"
#include "ceres/ceres.h"
#include "common.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include <pcl/io/pcd_io.h>
#include <filesystem>

// Normal pnp solution
class pnp_calib {
public:
  // instrins matrix
  Eigen::Matrix3d m_inner;
  // Distortion coefficient
  Eigen::Vector4d m_distor;

  pnp_calib(PnPData p, const Eigen::Matrix3d& inner, const Eigen::Vector4d& distor)
  { 
    pd = p; 
    m_inner = inner;
    m_distor = distor;
  }

  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const {
    Eigen::Matrix<T, 3, 3> innerT = m_inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = m_distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
           distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
           distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);
    return true;
  }
  static ceres::CostFunction *Create(PnPData p, const Eigen::Matrix3d& inner, const Eigen::Vector4d& distor) {
    return (new ceres::AutoDiffCostFunction<pnp_calib, 2, 4, 3>(
      new pnp_calib(p, inner, distor)));
  }

private:
  PnPData pd;
};

// pnp calib with direction vector
class vpnp_calib {
public:
  // instrins matrix
  Eigen::Matrix3d m_inner;
  // Distortion coefficient
  Eigen::Vector4d m_distor;

  vpnp_calib(VPnPData p, const Eigen::Matrix3d& inner, const Eigen::Vector4d& distor)
  { 
    pd = p; 
    m_inner = inner;
    m_distor = distor;
  }

  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const {
    Eigen::Matrix<T, 3, 3> innerT = m_inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = m_distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
           distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
           distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    if (T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0)) {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
    } else {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
      Eigen::Matrix<T, 2, 2> I =
          Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
      Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
      Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();
      Eigen::Matrix<T, 2, 2> V = n * nt;
      V = I - V;
      Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
      R.coeffRef(0, 0) = residuals[0];
      R.coeffRef(1, 0) = residuals[1];
      R = V * R;
      // Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<float, 2,
      // 2>::Zero().cast<T>(); R.coeffRef(0, 0) = residuals[0];
      // R.coeffRef(1, 1) = residuals[1]; R = V * R * V.transpose();
      residuals[0] = R.coeffRef(0, 0);
      residuals[1] = R.coeffRef(1, 0);
    }
    return true;
  }
  static ceres::CostFunction *Create(VPnPData p, const Eigen::Matrix3d& inner, const Eigen::Vector4d& distor) {
    return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 4, 3>(
        new vpnp_calib(p,inner, distor)));
  }

private:
  VPnPData pd;
};

#endif