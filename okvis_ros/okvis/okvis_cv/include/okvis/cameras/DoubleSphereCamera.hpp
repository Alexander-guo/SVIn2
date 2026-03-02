/**
 * @file cameras/DoubleSphereCamera.hpp
 * @brief Header file for the DoubleSphereCamera class.
 * @author Jinyuan Guo (alexguo@udel.edu)
 */

#ifndef INCLUDE_OKVIS_CAMERAS_DOUBLESPIHERECAMERA_HPP_
#define INCLUDE_OKVIS_CAMERAS_DOUBLESPIHERECAMERA_HPP_

#include <stdint.h>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>  // Code that causes warning goes here
#pragma GCC diagnostic pop
#include "okvis/cameras/CameraBase.hpp"
#include "okvis/cameras/DistortionBase.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

/// \class DoubleSphereCamera<DISTORTION_T>
/// \brief This implements a standard double sphere camera projection model.
/// \tparam DISTORTION_T the distortion type, NOTE only support NoDistortion! e.g. okvis::cameras::NoDistortion
template <class DISTORTION_T>
class DoubleSphereCamera : public CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef DISTORTION_T distortion_t;  ///< Makes the distortion type accessible.

  /// \brief Constructor that will figure out the type of distortion
  /// @param[in] imageWidth The width in pixels.
  /// @param[in] imageHeight The height in pixels.
  /// @param[in] focalLengthU The horizontal focal length in pixels.
  /// @param[in] focalLengthV The vertical focal length in pixels.
  /// @param[in] imageCenterU The horizontal centre in pixels.
  /// @param[in] imageCenterV The vertical centre in pixels.
  /// @param[in] xi The distance between the centers of the two virtual spheres in range [-1, 1].
  /// @param[in] alpha The parameter controls the relative scaling and distortions in range [0, 1], 
  ///            representing the projection from the second sphere onto the final image plane,
  ///            where alpha=0 corresponds to the unified projection model and alpha=1 corresponds to the shifted pinhole model.
  /// @param[in] distortion The distortion object to be used.
  /// @param[in] id Assign a generic ID, if desired.
  DoubleSphereCamera(int imageWidth,
                     int imageHeight,
                     double focalLengthU,
                     double focalLengthV,
                     double imageCenterU,
                     double imageCenterV,
                     double xi,
                     double alpha,
                     const distortion_t& distortion,
                     uint64_t id = -1);

  /// \brief Destructor.
  ~DoubleSphereCamera() {}
  static const int NumProjectionIntrinsics = 6;  ///< optimisable projection intrinsics
  static const int NumIntrinsics =
      NumProjectionIntrinsics + distortion_t::NumDistortionIntrinsics;  ///< total number of intrinsics

  /// \brief Get the focal length along the u-dimension.
  /// \return The horizontal focal length in pixels.
  double focalLengthU() const { return fu_; }

  /// \brief Get the focal length along the v-dimension.
  /// \return The vertical focal length in pixels.
  double focalLengthV() const { return fv_; }

  /// \brief Get the image centre along the u-dimension.
  /// \return The horizontal centre in pixels.
  double imageCenterU() const { return cu_; }

  /// \brief Get the focal image centre along the v-dimension.
  /// \return The vertical centre in pixels.
  double imageCenterV() const { return cv_; }

  /// \brief Get the xi parameter for the double sphere model.
  /// \return The xi parameter, representing the distance between the centers of the two virtual spheres.
  double xi() const { return xi_; }

  /// \brief Get the alpha parameter for the double sphere model.
  /// \return The alpha parameter, representing the projection from the second sphere onto the final image plane.
  double alpha() const { return alpha_; }

  /// \brief Get the intrinsics as a concatenated vector.
  /// \return The intrinsics as a concatenated vector.
  inline void getIntrinsics(Eigen::VectorXd& intrinsics) const;  // NOLINT

  /// \brief overwrite all intrinsics - use with caution !
  /// \param[in] intrinsics The intrinsics as a concatenated vector.
  inline bool setIntrinsics(const Eigen::VectorXd& intrinsics);

  /// \brief Get the total number of intrinsics.
  /// \return Number of intrinsics parameters.
  inline int noIntrinsicsParameters() const { return NumIntrinsics; }

  //////////////////////////////////////////////////////////////
  /// \name Methods to project points
  /// @{

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Euclidean coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline CameraBase::ProjectionStatus project(const Eigen::Vector3d& point, Eigen::Vector2d* imagePoint) const;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline CameraBase::ProjectionStatus project(const Eigen::Vector3d& point,
                                              Eigen::Vector2d* imagePoint,
                                              Eigen::Matrix<double, 2, 3>* pointJacobian,
                                              Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline CameraBase::ProjectionStatus projectWithExternalParameters(const Eigen::Vector3d& point,
                                                                    const Eigen::VectorXd& parameters,
                                                                    Eigen::Vector2d* imagePoint,
                                                                    Eigen::Matrix<double, 2, 3>* pointJacobian,
                                                                    Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const;

  /// \brief Projects Euclidean points to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in Euclidean coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void projectBatch(const Eigen::Matrix3Xd& points,
                           Eigen::Matrix2Xd* imagePoints,
                           std::vector<CameraBase::ProjectionStatus>* stati) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Homogeneous coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline CameraBase::ProjectionStatus projectHomogeneous(const Eigen::Vector4d& point,
                                                         Eigen::Vector2d* imagePoint) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline CameraBase::ProjectionStatus projectHomogeneous(const Eigen::Vector4d& point,
                                                         Eigen::Vector2d* imagePoint,
                                                         Eigen::Matrix<double, 2, 4>* pointJacobian,
                                                         Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline CameraBase::ProjectionStatus projectHomogeneousWithExternalParameters(
      const Eigen::Vector4d& point,
      const Eigen::VectorXd& parameters,
      Eigen::Vector2d* imagePoint,
      Eigen::Matrix<double, 2, 4>* pointJacobian = NULL,
      Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const;

  /// \brief Projects points in homogenous coordinates to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in homogeneous coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void projectHomogeneousBatch(const Eigen::Matrix4Xd& points,
                                      Eigen::Matrix2Xd* imagePoints,
                                      std::vector<CameraBase::ProjectionStatus>* stati) const;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to backproject points
  /// @{

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The Euclidean direction vector.
  /// @return     true on success.
  inline bool backProject(const Eigen::Vector2d& imagePoint, Eigen::Vector3d* direction) const;

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The Euclidean direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function  w.r.t. the point.
  /// @return     true on success.
  inline bool backProject(const Eigen::Vector2d& imagePoint,
                          Eigen::Vector3d* direction,
                          Eigen::Matrix<double, 3, 2>* pointJacobian) const;

  /// \brief Back-project 2d image points into Euclidean space (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The Euclidean direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectBatch(const Eigen::Matrix2Xd& imagePoints,
                               Eigen::Matrix3Xd* directions,
                               std::vector<bool>* success) const;

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The homogeneous point as direction vector.
  /// @return     true on success.
  inline bool backProjectHomogeneous(const Eigen::Vector2d& imagePoint, Eigen::Vector4d* direction) const;

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The homogeneous point as direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function.
  /// @return     true on success.
  inline bool backProjectHomogeneous(const Eigen::Vector2d& imagePoint,
                                     Eigen::Vector4d* direction,
                                     Eigen::Matrix<double, 4, 2>* pointJacobian) const;

  /// \brief Back-project 2d image points into homogeneous points (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The homogeneous points as direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectHomogeneousBatch(const Eigen::Matrix2Xd& imagePoints,
                                          Eigen::Matrix4Xd* directions,
                                          std::vector<bool>* success) const;
  /// @}

  /// \brief get a test instance
  static std::shared_ptr<CameraBase> createTestObject() {
    return std::shared_ptr<CameraBase>(
        new DoubleSphereCamera(624, 624, 230.73, 230.64, 316.215, 317.275, 0.0076174882446198786, 0.5824467900419464, 
            distortion_t::testObject()));
  }
  /// \brief get a test instance
  static DoubleSphereCamera testObject() {
    return DoubleSphereCamera(624,
                              624,
                              230.73,
                              230.64,
                              316.215,
                              317.275,
                              0.0076174882446198786,
                              0.5824467900419464,
                              distortion_t::testObject());
  };

  /// \brief Obtain the projection type
  std::string type() const { return "DoubleSphereCamera<" + distortion_.type() + ">"; }

  /// \brief Obtain the projection type
  const std::string distortionType() const { return distortion_.type(); }

 protected:
  /// \brief No default constructor.
  DoubleSphereCamera() = delete;

  distortion_t distortion_;  ///< the distortion to be used

  Eigen::Matrix<double, NumIntrinsics, 1> intrinsics_;  ///< summary of all intrinsics parameters
  double fu_;                                           ///< focalLengthU
  double fv_;                                           ///< focalLengthV
  double cu_;                                           ///< imageCenterU
  double cv_;                                           ///< imageCenterV
  double xi_;                                           ///< xi parameter for the double sphere model
  double alpha_;                                        ///< alpha parameter for the double sphere model
  double one_over_fu_;                                  ///< 1.0 / fu_
  double one_over_fv_;                                  ///< 1.0 / fv_
  double fu_over_fv_;                                   ///< fu_ / fv_
};

}  // namespace cameras
}  // namespace okvis

#include "implementation/DoubleSphereCamera.hpp"

#endif /* INCLUDE_OKVIS_CAMERAS_DOUBLESPEHERECAMERA_HPP_ */