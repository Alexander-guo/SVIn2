/*
 * @file implementation/DoubleSphereCamera.hpp
 * @brief Header implementation file for the DoubleSphereCamera class, in reference to Basalt's implementation of the
 * double sphere camera model.
 * @author Jinyuan Guo (alexguo@udel.edu)
 */

#include <vector>
#include <cmath>
#include "okvis/cameras/DoubleSphereCamera.hpp"

// \brief okvis Main namespace of this package.
namespace okvis {
// \brief cameras Namespace for camera-related functionality.
namespace cameras {

template <class DISTORTION_T>
DoubleSphereCamera<DISTORTION_T>::DoubleSphereCamera(int imageWidth,
                                                     int imageHeight,
                                                     double focalLengthU,
                                                     double focalLengthV,
                                                     double imageCenterU,
                                                     double imageCenterV,
                                                     double xi,
                                                     double alpha,
                                                     const distortion_t& distortion,
                                                     uint64_t id)
    : CameraBase(imageWidth, imageHeight, id),
      distortion_(distortion),
      fu_(focalLengthU),
      fv_(focalLengthV),
      cu_(imageCenterU),
      cv_(imageCenterV),
      xi_(xi),
      alpha_(alpha) {
  intrinsics_[0] = fu_;      //< focalLengthU
  intrinsics_[1] = fv_;      //< focalLengthV
  intrinsics_[2] = cu_;      //< imageCenterU
  intrinsics_[3] = cv_;      //< imageCenterV
  intrinsics_[4] = xi_;      //< xi
  intrinsics_[5] = alpha_;   //< alpha
  one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
  one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
  fu_over_fv_ = fu_ / fv_;   //< fu_ / fv_
}

// overwrite all intrinsics - use with caution !
template <class DISTORTION_T>
bool DoubleSphereCamera<DISTORTION_T>::setIntrinsics(const Eigen::VectorXd& intrinsics) {
  if (intrinsics.cols() != NumIntrinsics) {
    return false;
  }
  intrinsics_ = intrinsics;
  fu_ = intrinsics[0];     //< focalLengthU
  fv_ = intrinsics[1];     //< focalLengthV
  cu_ = intrinsics[2];     //< imageCenterU
  cv_ = intrinsics[3];     //< imageCenterV
  xi_ = intrinsics[4];     //< xi
  alpha_ = intrinsics[5];  //< alpha
  distortion_.setParameters(intrinsics.tail<distortion_t::NumDistortionIntrinsics>());
  one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
  one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
  fu_over_fv_ = fu_ / fv_;   //< fu_ / fv_
  return true;
}

template <class DISTORTION_T>
void DoubleSphereCamera<DISTORTION_T>::getIntrinsics(Eigen::VectorXd& intrinsics) const {  // NOLINT
  intrinsics = intrinsics_;
  Eigen::VectorXd distortionIntrinsics;
  if (distortion_t::NumDistortionIntrinsics > 0) {
    distortion_.getParameters(distortionIntrinsics);
    intrinsics.tail<distortion_t::NumDistortionIntrinsics>() = distortionIntrinsics;
  }
}

//////////////////////////////////////////
// Methods to project points

// Projects a Euclidean point to a 2d image point (projection).
template <class DISTORTION_T>
CameraBase::ProjectionStatus DoubleSphereCamera<DISTORTION_T>::project(const Eigen::Vector3d& point,
                                                                       Eigen::Vector2d* imagePoint) const {
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];

  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;

  const double r2 = xx + yy;

  const double d1_2 = r2 + zz;
  const double d1 = std::sqrt(d1_2);

  const double w1 = alpha_ > 0.5 ? (1.0 - alpha_) / alpha_ : alpha_ / (1.0 - alpha_);
  const double w2 = (w1 + xi_) / std::sqrt(2.0 * w1 * xi_ + xi_ * xi_ + 1.0);

  if (z <= -w2 * d1) {
    return CameraBase::ProjectionStatus::Behind;
  }

  const double k = xi_ * d1 + z;
  const double kk = k * k;

  const double d2_2 = r2 + kk;
  const double d2 = std::sqrt(d2_2);

  const double norm = alpha_ * d2 + (1.0 - alpha_) * k;

  const double mx = x / norm;
  const double my = y / norm;

  Eigen::Vector2d imagePointUndistorted;
  imagePointUndistorted[0] = mx;
  imagePointUndistorted[1] = my;

  // distortion
  Eigen::Vector2d imagePoint2;
  if (!distortion_.distort(imagePointUndistorted, &imagePoint2)) {
    return CameraBase::ProjectionStatus::Invalid;
  }

  // scale and offset
  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (!CameraBase::isInImage(*imagePoint)) {
    return CameraBase::ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return CameraBase::ProjectionStatus::Masked;
  }
  return CameraBase::ProjectionStatus::Successful;
}

// Projects a Euclidean point to a 2d image point (projection).
template <class DISTORTION_T>
CameraBase::ProjectionStatus DoubleSphereCamera<DISTORTION_T>::project(const Eigen::Vector3d& point,
                                                                       Eigen::Vector2d* imagePoint,
                                                                       Eigen::Matrix<double, 2, 3>* pointJacobian,
                                                                       Eigen::Matrix2Xd* intrinsicsJacobian) const {
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];

  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;

  const double r2 = xx + yy;

  const double d1_2 = r2 + zz;
  const double d1 = std::sqrt(d1_2);

  const double w1 = alpha_ > 0.5 ? (1.0 - alpha_) / alpha_ : alpha_ / (1.0 - alpha_);
  const double w2 = (w1 + xi_) / std::sqrt(2.0 * w1 * xi_ + xi_ * xi_ + 1.0);

  if (z <= -w2 * d1) {
    return CameraBase::ProjectionStatus::Behind;
  }

  const double k = xi_ * d1 + z;
  const double kk = k * k;

  const double d2_2 = r2 + kk;
  const double d2 = std::sqrt(d2_2);

  const double norm = alpha_ * d2 + (1.0 - alpha_) * k;

  const double mx = x / norm;
  const double my = y / norm;

  Eigen::Vector2d imagePointUndistorted;
  imagePointUndistorted[0] = mx;
  imagePointUndistorted[1] = my;

  Eigen::Matrix<double, 2, 3> pointJacobianProjection;
  Eigen::Matrix2Xd intrinsicsJacobianProjection;
  Eigen::Matrix2d distortionJacobian;
  Eigen::Matrix2Xd intrinsicsJacobianDistortion;
  Eigen::Vector2d imagePoint2;

  bool distortionSuccess;
  if (intrinsicsJacobian) {
    // get both Jacobians
    intrinsicsJacobian->resize(2, NumIntrinsics);

    distortionSuccess =
        distortion_.distort(imagePointUndistorted, &imagePoint2, &distortionJacobian, &intrinsicsJacobianDistortion);

    // compute the intrinsics Jacobian
    const double norm2 = norm * norm;
    const double tmp4 = (alpha_ - 1.0 - alpha_ * k / d2) * d1 / norm2;
    const double tmp5 = (k - d2) / norm2;

    intrinsicsJacobianProjection.resize(2, NumIntrinsics);
    intrinsicsJacobianProjection.setZero();
    intrinsicsJacobianProjection(0, 0) = mx;
    intrinsicsJacobianProjection(0, 2) = 1.0;
    intrinsicsJacobianProjection(1, 1) = my;
    intrinsicsJacobianProjection(1, 3) = 1.0;

    intrinsicsJacobianProjection(0, 4) = fu_ * x * tmp4;
    intrinsicsJacobianProjection(1, 4) = fv_ * y * tmp4;

    intrinsicsJacobianProjection(0, 5) = fu_ * x * tmp5;
    intrinsicsJacobianProjection(1, 5) = fv_ * y * tmp5;

    // We have to add both together
    // out_J = J_proj_intr + J_dist_intr
    // The previous okvis code did this:
    *intrinsicsJacobian = intrinsicsJacobianProjection;

    if (distortion_t::NumDistortionIntrinsics > 0) {
      intrinsicsJacobian->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
          Eigen::Vector2d(fu_, fv_).asDiagonal() * intrinsicsJacobianDistortion;  // chain rule
    }
  } else {
    // only get point Jacobian
    distortionSuccess = distortion_.distort(imagePointUndistorted, &imagePoint2, &distortionJacobian);
  }

  // compute the point Jacobian in any case
  const double norm2 = norm * norm;
  const double xy = x * y;
  const double tt2 = xi_ * z / d1 + 1.0;

  const double d_norm_d_r2 = (xi_ * (1.0 - alpha_) / d1 + alpha_ * (xi_ * k / d1 + 1.0) / d2) / norm2;

  const double tmp2 = ((1.0 - alpha_) * tt2 + alpha_ * k * tt2 / d2) / norm2;

  pointJacobianProjection(0, 0) = 1.0 / norm - xx * d_norm_d_r2;
  pointJacobianProjection(1, 0) = -xy * d_norm_d_r2;

  pointJacobianProjection(0, 1) = -xy * d_norm_d_r2;
  pointJacobianProjection(1, 1) = 1.0 / norm - yy * d_norm_d_r2;

  pointJacobianProjection(0, 2) = -x * tmp2;
  pointJacobianProjection(1, 2) = -y * tmp2;

  Eigen::Matrix<double, 2, 3>& J = *pointJacobian;
  J = Eigen::Vector2d(fu_, fv_).asDiagonal() * distortionJacobian * pointJacobianProjection;

  // scale and offset
  (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
  (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

  if (!distortionSuccess) {
    return CameraBase::ProjectionStatus::Invalid;
  }
  if (!CameraBase::isInImage(*imagePoint)) {
    return CameraBase::ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return CameraBase::ProjectionStatus::Masked;
  }
  return CameraBase::ProjectionStatus::Successful;
}

// Projects a Euclidean point to a 2d image point (projection).
template <class DISTORTION_T>
CameraBase::ProjectionStatus DoubleSphereCamera<DISTORTION_T>::projectWithExternalParameters(
    const Eigen::Vector3d& point,
    const Eigen::VectorXd& parameters,
    Eigen::Vector2d* imagePoint,
    Eigen::Matrix<double, 2, 3>* pointJacobian,
    Eigen::Matrix2Xd* intrinsicsJacobian) const {
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];

  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;

  const double r2 = xx + yy;

  const double d1_2 = r2 + zz;
  const double d1 = std::sqrt(d1_2);

  // parse parameters into human readable form
  const double fu = parameters[0];
  const double fv = parameters[1];
  const double cu = parameters[2];
  const double cv = parameters[3];
  const double xi = parameters[4];
  const double alpha = parameters[5];

  const double w1 = alpha > 0.5 ? (1.0 - alpha) / alpha : alpha / (1.0 - alpha);
  const double w2 = (w1 + xi) / std::sqrt(2.0 * w1 * xi + xi * xi + 1.0);

  if (z <= -w2 * d1) {
    return CameraBase::ProjectionStatus::Behind;
  }

  const double k = xi * d1 + z;
  const double kk = k * k;

  const double d2_2 = r2 + kk;
  const double d2 = std::sqrt(d2_2);

  const double norm = alpha * d2 + (1.0 - alpha) * k;

  const double mx = x / norm;
  const double my = y / norm;

  Eigen::VectorXd distortionParameters;
  if (distortion_t::NumDistortionIntrinsics > 0) {
    distortionParameters = parameters.template tail<distortion_t::NumDistortionIntrinsics>();
  }

  Eigen::Vector2d imagePointUndistorted(mx, my);

  Eigen::Matrix<double, 2, 3> pointJacobianProjection;
  Eigen::Matrix2Xd intrinsicsJacobianProjection;
  Eigen::Matrix2d distortionJacobian;
  Eigen::Matrix2Xd intrinsicsJacobianDistortion;
  Eigen::Vector2d imagePoint2;

  bool distortionSuccess;
  if (intrinsicsJacobian) {
    // get both Jacobians
    intrinsicsJacobian->resize(2, NumIntrinsics);

    distortionSuccess = distortion_.distortWithExternalParameters(
        imagePointUndistorted, distortionParameters, &imagePoint2, &distortionJacobian, &intrinsicsJacobianDistortion);

    // compute the intrinsics Jacobian
    const double norm2 = norm * norm;
    const double tmp4 = (alpha - 1.0 - alpha * k / d2) * d1 / norm2;
    const double tmp5 = (k - d2) / norm2;

    intrinsicsJacobianProjection.resize(2, NumIntrinsics);
    intrinsicsJacobianProjection.setZero();
    intrinsicsJacobianProjection(0, 0) = mx;
    intrinsicsJacobianProjection(0, 2) = 1.0;
    intrinsicsJacobianProjection(1, 1) = my;
    intrinsicsJacobianProjection(1, 3) = 1.0;

    intrinsicsJacobianProjection(0, 4) = fu * x * tmp4;
    intrinsicsJacobianProjection(1, 4) = fv * y * tmp4;

    intrinsicsJacobianProjection(0, 5) = fu * x * tmp5;
    intrinsicsJacobianProjection(1, 5) = fv * y * tmp5;

    *intrinsicsJacobian = intrinsicsJacobianProjection;

    if (distortion_t::NumDistortionIntrinsics > 0) {
      intrinsicsJacobian->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
          Eigen::Vector2d(fu, fv).asDiagonal() * intrinsicsJacobianDistortion;  // chain rule
    }
  } else {
    // only get point Jacobian
    distortionSuccess = distortion_.distortWithExternalParameters(
        imagePointUndistorted, distortionParameters, &imagePoint2, &distortionJacobian);
  }

  // compute the point Jacobian, if requested
  if (pointJacobian) {
    const double norm2 = norm * norm;
    const double xy = x * y;
    const double tt2 = xi * z / d1 + 1.0;

    const double d_norm_d_r2 = (xi * (1.0 - alpha) / d1 + alpha * (xi * k / d1 + 1.0) / d2) / norm2;

    const double tmp2 = ((1.0 - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;

    pointJacobianProjection(0, 0) = 1.0 / norm - xx * d_norm_d_r2;
    pointJacobianProjection(1, 0) = -xy * d_norm_d_r2;

    pointJacobianProjection(0, 1) = -xy * d_norm_d_r2;
    pointJacobianProjection(1, 1) = 1.0 / norm - yy * d_norm_d_r2;

    pointJacobianProjection(0, 2) = -x * tmp2;
    pointJacobianProjection(1, 2) = -y * tmp2;

    Eigen::Matrix<double, 2, 3>& J = *pointJacobian;
    J = Eigen::Vector2d(fu, fv).asDiagonal() * distortionJacobian * pointJacobianProjection;
  }

  // scale and offset
  (*imagePoint)[0] = fu * imagePoint2[0] + cu;
  (*imagePoint)[1] = fv * imagePoint2[1] + cv;

  if (!distortionSuccess) {
    return CameraBase::ProjectionStatus::Invalid;
  }
  if (!CameraBase::isInImage(*imagePoint)) {
    return CameraBase::ProjectionStatus::OutsideImage;
  }
  if (CameraBase::isMasked(*imagePoint)) {
    return CameraBase::ProjectionStatus::Masked;
  }
  return CameraBase::ProjectionStatus::Successful;
}

// Projects Euclidean points to 2d image points (projection) in a batch.
template <class DISTORTION_T>
void DoubleSphereCamera<DISTORTION_T>::projectBatch(const Eigen::Matrix3Xd& points,
                                                    Eigen::Matrix2Xd* imagePoints,
                                                    std::vector<CameraBase::ProjectionStatus>* stati) const {
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector3d point = points.col(i);
    Eigen::Vector2d imagePoint;
    CameraBase::ProjectionStatus status = project(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    if (stati) stati->push_back(status);
  }
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
template <class DISTORTION_T>
CameraBase::ProjectionStatus DoubleSphereCamera<DISTORTION_T>::projectHomogeneous(const Eigen::Vector4d& point,
                                                                                  Eigen::Vector2d* imagePoint) const {
  Eigen::Vector3d head = point.head<3>();
  if (point[3] < 0) {
    return project(-head, imagePoint);
  } else {
    return project(head, imagePoint);
  }
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
template <class DISTORTION_T>
CameraBase::ProjectionStatus DoubleSphereCamera<DISTORTION_T>::projectHomogeneous(
    const Eigen::Vector4d& point,
    Eigen::Vector2d* imagePoint,
    Eigen::Matrix<double, 2, 4>* pointJacobian,
    Eigen::Matrix2Xd* intrinsicsJacobian) const {
  Eigen::Vector3d head = point.head<3>();
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  CameraBase::ProjectionStatus status;
  if (point[3] < 0) {
    status = project(-head, imagePoint, &pointJacobian3, intrinsicsJacobian);
  } else {
    status = project(head, imagePoint, &pointJacobian3, intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

// Projects a point in homogenous coordinates to a 2d image point (projection).
template <class DISTORTION_T>
CameraBase::ProjectionStatus DoubleSphereCamera<DISTORTION_T>::projectHomogeneousWithExternalParameters(
    const Eigen::Vector4d& point,
    const Eigen::VectorXd& parameters,
    Eigen::Vector2d* imagePoint,
    Eigen::Matrix<double, 2, 4>* pointJacobian,
    Eigen::Matrix2Xd* intrinsicsJacobian) const {
  Eigen::Vector3d head = point.head<3>();
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  CameraBase::ProjectionStatus status;
  if (point[3] < 0) {
    status = projectWithExternalParameters(-head, parameters, imagePoint, &pointJacobian3, intrinsicsJacobian);
  } else {
    status = projectWithExternalParameters(head, parameters, imagePoint, &pointJacobian3, intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

// Projects points in homogenous coordinates to 2d image points (projection) in a batch.
template <class DISTORTION_T>
void DoubleSphereCamera<DISTORTION_T>::projectHomogeneousBatch(const Eigen::Matrix4Xd& points,
                                                               Eigen::Matrix2Xd* imagePoints,
                                                               std::vector<ProjectionStatus>* stati) const {
  const int numPoints = points.cols();
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector4d point = points.col(i);
    Eigen::Vector2d imagePoint;
    CameraBase::ProjectionStatus status = projectHomogeneous(point, &imagePoint);
    imagePoints->col(i) = imagePoint;
    if (stati) stati->push_back(status);
  }
}

//////////////////////////////////////////
// Methods to backproject points

// Back-project a 2d image point into Euclidean space (direction vector).
template <class DISTORTION_T>
bool DoubleSphereCamera<DISTORTION_T>::backProject(const Eigen::Vector2d& imagePoint,
                                                   Eigen::Vector3d* direction) const {
  // unscale and center
  Eigen::Vector2d imagePoint2;
  imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
  imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

  // undistort
  Eigen::Vector2d undistortedImagePoint;
  bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint);

  const double mx = undistortedImagePoint[0];
  const double my = undistortedImagePoint[1];

  const double r2 = mx * mx + my * my;

  bool is_valid = !(alpha_ > 0.5 && (r2 >= 1.0 / (2.0 * alpha_ - 1.0)));

  const double xi2_2 = alpha_ * alpha_;
  const double xi1_2 = xi_ * xi_;

  const double sqrt2 = std::sqrt(1.0 - (2.0 * alpha_ - 1.0) * r2);

  const double norm2 = alpha_ * sqrt2 + 1.0 - alpha_;

  const double mz = (1.0 - xi2_2 * r2) / norm2;
  const double mz2 = mz * mz;

  const double norm1 = mz2 + r2;
  const double sqrt1 = std::sqrt(mz2 + (1.0 - xi1_2) * r2);
  const double k = (mz * xi_ + sqrt1) / norm1;

  (*direction)[0] = k * mx;
  (*direction)[1] = k * my;
  (*direction)[2] = k * mz - xi_;

  return success && is_valid;
}

// Back-project a 2d image point into Euclidean space (direction vector).
template <class DISTORTION_T>
inline bool DoubleSphereCamera<DISTORTION_T>::backProject(const Eigen::Vector2d& imagePoint,
                                                          Eigen::Vector3d* direction,
                                                          Eigen::Matrix<double, 3, 2>* pointJacobian) const {
  // unscale and center
  Eigen::Vector2d imagePoint2;
  imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
  imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

  // undistort
  Eigen::Vector2d undistortedImagePoint;
  Eigen::Matrix2d pointJacobianUndistortion;
  bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint, &pointJacobianUndistortion);

  const double mx = undistortedImagePoint[0];
  const double my = undistortedImagePoint[1];

  const double r2 = mx * mx + my * my;

  bool is_valid = !(alpha_ > 0.5 && (r2 >= 1.0 / (2.0 * alpha_ - 1.0)));

  const double xi2_2 = alpha_ * alpha_;
  const double xi1_2 = xi_ * xi_;

  const double sqrt2 = std::sqrt(1.0 - (2.0 * alpha_ - 1.0) * r2);

  const double norm2 = alpha_ * sqrt2 + 1.0 - alpha_;

  const double mz = (1.0 - xi2_2 * r2) / norm2;
  const double mz2 = mz * mz;

  const double norm1 = mz2 + r2;
  const double sqrt1 = std::sqrt(mz2 + (1.0 - xi1_2) * r2);
  const double k = (mz * xi_ + sqrt1) / norm1;

  (*direction)[0] = k * mx;
  (*direction)[1] = k * my;
  (*direction)[2] = k * mz - xi_;

  const double norm2_2 = norm2 * norm2;
  const double norm1_2 = norm1 * norm1;

  const double d_mz_d_r2 = (0.5 * alpha_ - xi2_2) * (r2 * xi2_2 - 1.0) / (sqrt2 * norm2_2) - xi2_2 / norm2;

  const double d_mz_d_mx = 2.0 * mx * d_mz_d_r2;
  const double d_mz_d_my = 2.0 * my * d_mz_d_r2;

  const double d_k_d_r2 = (xi_ * d_mz_d_r2 + 0.5 / sqrt1 * (2.0 * mz * d_mz_d_r2 + 1.0 - xi1_2)) / norm1 -
                          (mz * xi_ + sqrt1) * (2.0 * mz * d_mz_d_r2 + 1.0) / norm1_2;

  const double d_k_d_mx = d_k_d_r2 * 2.0 * mx;
  const double d_k_d_my = d_k_d_r2 * 2.0 * my;

  Eigen::Matrix<double, 3, 2> J_unproj;
  J_unproj(0, 0) = mx * d_k_d_mx + k;
  J_unproj(1, 0) = my * d_k_d_mx;
  J_unproj(2, 0) = mz * d_k_d_mx + k * d_mz_d_mx;

  J_unproj(0, 1) = mx * d_k_d_my;
  J_unproj(1, 1) = my * d_k_d_my + k;
  J_unproj(2, 1) = mz * d_k_d_my + k * d_mz_d_my;

  // Jacobian w.r.t. imagePoint
  Eigen::Matrix2d outProjectJacobian = Eigen::Matrix2d::Zero();
  outProjectJacobian(0, 0) = one_over_fu_;
  outProjectJacobian(1, 1) = one_over_fv_;

  (*pointJacobian) = J_unproj * pointJacobianUndistortion * outProjectJacobian;  // chain rule

  return success && is_valid;
}

// Back-project 2d image points into Euclidean space (direction vectors).
template <class DISTORTION_T>
bool DoubleSphereCamera<DISTORTION_T>::backProjectBatch(const Eigen::Matrix2Xd& imagePoints,
                                                        Eigen::Matrix3Xd* directions,
                                                        std::vector<bool>* success) const {
  const int numPoints = imagePoints.cols();
  directions->row(3) = Eigen::VectorXd::Ones(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector2d imagePoint = imagePoints.col(i);
    Eigen::Vector3d point;
    bool suc = backProject(imagePoint, &point);
    if (success) success->push_back(suc);
    directions->col(i) = point;
  }
  return true;
}

// Back-project a 2d image point into homogeneous point (direction vector).
template <class DISTORTION_T>
bool DoubleSphereCamera<DISTORTION_T>::backProjectHomogeneous(const Eigen::Vector2d& imagePoint,
                                                              Eigen::Vector4d* direction) const {
  Eigen::Vector3d ray;
  bool success = backProject(imagePoint, &ray);
  direction->template head<3>() = ray;
  (*direction)[4] = 1.0;  // arbitrary
  return success;
}

// Back-project a 2d image point into homogeneous point (direction vector).
template <class DISTORTION_T>
bool DoubleSphereCamera<DISTORTION_T>::backProjectHomogeneous(const Eigen::Vector2d& imagePoint,
                                                              Eigen::Vector4d* direction,
                                                              Eigen::Matrix<double, 4, 2>* pointJacobian) const {
  Eigen::Vector3d ray;
  Eigen::Matrix<double, 3, 2> pointJacobian3;
  bool success = backProject(imagePoint, &ray, &pointJacobian3);
  direction->template head<3>() = ray;
  (*direction)[4] = 1.0;  // arbitrary
  pointJacobian->template bottomRightCorner<1, 2>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<3, 2>() = pointJacobian3;
  return success;
}

// Back-project 2d image points into homogeneous points (direction vectors).
template <class DISTORTION_T>
bool DoubleSphereCamera<DISTORTION_T>::backProjectHomogeneousBatch(const Eigen::Matrix2Xd& imagePoints,
                                                                   Eigen::Matrix4Xd* directions,
                                                                   std::vector<bool>* success) const {
  const int numPoints = imagePoints.cols();
  directions->row(3) = Eigen::VectorXd::Ones(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    Eigen::Vector2d imagePoint = imagePoints.col(i);
    Eigen::Vector3d point;
    bool suc = backProject(imagePoint, &point);
    if (success) success->push_back(suc);
    directions->template block<3, 1>(0, i) = point;
  }
  return true;
}

}  // namespace cameras
}  // namespace okvis
