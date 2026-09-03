#include "pose_graph/Keyframe.h"

#include <glog/logging.h>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "okvis/cameras/DoubleSphereCamera.hpp"
#include "okvis/cameras/NoDistortion.hpp"
#include "pose_graph/Parameters.h"
#include "utils/UtilsOpenCV.h"

namespace {

struct LoopClosureFunnelRecord {
  int current_kf_id = -1;
  int64_t current_timestamp = 0;
  int candidate_kf_id = -1;
  int64_t candidate_timestamp = 0;
  std::string camera_model;
  std::string pnp_model;
  size_t tracked_points = 0;
  size_t candidate_keypoints = 0;
  size_t descriptor_matches = 0;
  int brief_hamming_threshold = 80;
  int min_correspondences = 0;
  bool pnp_attempted = false;
  bool pnp_solver_succeeded = false;
  bool pnp_exception = false;
  size_t pnp_inliers = 0;
  int pnp_iterations = 0;
  double pnp_reprojection_threshold = 0.0;
  double relative_yaw_deg = std::numeric_limits<double>::quiet_NaN();
  double relative_translation_m = std::numeric_limits<double>::quiet_NaN();
  double max_yaw_deg = 25.0;
  double max_position_m = 15.0;
  bool yaw_gate_passed = false;
  bool position_gate_passed = false;
  bool accepted = false;
  std::string rejection_reason;
};

void appendLoopClosureFunnelRecord(const std::string& debug_output_path,
                                   const LoopClosureFunnelRecord& record) {
  std::ofstream output(debug_output_path + "/loop_closure_funnel.csv", std::ios::app);
  if (!output.is_open()) {
    LOG(ERROR) << "Could not append loop-closure funnel diagnostics in " << debug_output_path;
    return;
  }

  output << std::setprecision(17) << record.current_kf_id << ',' << record.current_timestamp << ','
         << record.candidate_kf_id << ',' << record.candidate_timestamp << ',' << record.camera_model << ','
         << record.pnp_model << ',' << record.tracked_points << ',' << record.candidate_keypoints << ','
         << record.descriptor_matches << ',' << record.brief_hamming_threshold << ',' << record.min_correspondences << ','
         << record.pnp_attempted << ',' << record.pnp_solver_succeeded << ',' << record.pnp_exception << ','
         << record.pnp_inliers << ',' << record.pnp_iterations << ',' << record.pnp_reprojection_threshold << ','
         << record.relative_yaw_deg << ',' << record.relative_translation_m << ',' << record.max_yaw_deg << ','
         << record.max_position_m << ',' << record.yaw_gate_passed << ',' << record.position_gate_passed << ','
         << record.accepted << ',' << record.rejection_reason << '\n';
}

class PixelReprojectionAbsolutePoseSacProblem
    : public opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem {
 public:
  using Base = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
  using Model = Base::model_t;

  PixelReprojectionAbsolutePoseSacProblem(Base::adapter_t& adapter,
                                          const okvis::cameras::CameraBase& camera,
                                          const std::vector<cv::Point2f>& image_points)
      : Base(adapter, Base::KNEIP), camera_(camera), image_points_(image_points) {}

  void getSelectedDistancesToModel(const Model& model,
                                   const std::vector<int>& indices,
                                   std::vector<double>& scores) const override {
    const Eigen::Matrix3d R_c_w = model.leftCols<3>().transpose();
    const Eigen::Vector3d T_w_c = model.col(3);
    scores.reserve(scores.size() + indices.size());

    for (const int index : indices) {
      const Eigen::Vector3d point_c = R_c_w * (_adapter.getPoint(index) - T_w_c);
      Eigen::Vector2d projected;
      const auto projection_status = camera_.project(point_c, &projected);
      if (projection_status != okvis::cameras::CameraBase::ProjectionStatus::Successful ||
          !projected.allFinite()) {
        scores.push_back(std::numeric_limits<double>::infinity());
        continue;
      }

      const Eigen::Vector2d observed(image_points_[index].x, image_points_[index].y);
      scores.push_back((projected - observed).squaredNorm());
    }
  }

 private:
  const okvis::cameras::CameraBase& camera_;
  const std::vector<cv::Point2f>& image_points_;
};

}  // namespace

const int Keyframe::TH_HIGH = 100;
const int Keyframe::TH_LOW = 50;
const size_t Keyframe::briskDetectionOctaves_ = 0;                ///< The set number of brisk octaves.
const double Keyframe::briskDetectionThreshold_ = 40.0;           ///< The set BRISK detection threshold.
const double Keyframe::briskDetectionAbsoluteThreshold_ = 800;    ///< The set BRISK absolute detection threshold.
const size_t Keyframe::briskDetectionMaximumKeypoints_ = 300;     ///< The set maximum number of keypoints.
const bool Keyframe::briskDescriptionRotationInvariance_ = true;  ///< The set rotation invariance setting.
const bool Keyframe::briskDescriptionScaleInvariance_ = false;    ///< The set scale invariance setting.
const double Keyframe::briskMatchingThreshold_ = 80.0;            ///< The set BRISK matching threshold.

template <typename Derived>
static void reduceVector(std::vector<Derived>& v, std::vector<uchar> status) {  // NOLINT
  int j = 0;
  for (int i = 0; i < static_cast<int>(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

Keyframe::Keyframe(Timestamp _time_stamp,
                   std::vector<Eigen::Vector3i>& _point_ids,
                   int _index,
                   Eigen::Vector3d& _svin_T_w_i,
                   Eigen::Matrix3d& _svin_R_w_i,
                   std::vector<cv::Mat>& _images,
                   std::vector<cv::Point3f>& _point_3d,
                   std::vector<cv::KeyPoint>& _point_2d_uv,
                   std::vector<size_t>& _point_camera_indices,
                   std::map<Keyframe*, int>& KFcounter,
                   int _sequence,
                   BriefVocabulary* vocBrief,
                   const Parameters& params,
                   const bool is_vio_keyframe)
    : params_(params) {
  time_stamp = _time_stamp;

  // @Reloc
  index = _index;
  svin_T_w_i = _svin_T_w_i;
  svin_R_w_i = _svin_R_w_i;
  T_w_i = svin_T_w_i;
  R_w_i = svin_R_w_i;
  origin_svin_T = svin_T_w_i;
  origin_svin_R = svin_R_w_i;
  images.reserve(_images.size());
  for (const cv::Mat& source_image : _images) images.push_back(source_image.clone());
  image = images.empty() ? cv::Mat() : images.front();

  const size_t camera_count = images.size();
  camera_point_3d.resize(camera_count);
  camera_point_2d_uv.resize(camera_count);
  camera_point_ids.resize(camera_count);
  for (size_t i = 0; i < _point_3d.size() && i < _point_2d_uv.size() &&
                     i < _point_ids.size() && i < _point_camera_indices.size();
       ++i) {
    const size_t camera_index = _point_camera_indices[i];
    if (camera_index >= camera_count) continue;
    camera_point_3d[camera_index].push_back(_point_3d[i]);
    camera_point_2d_uv[camera_index].push_back(_point_2d_uv[i]);
    camera_point_ids[camera_index].push_back(_point_ids[i]);
  }
  // Keep the established camera-0 representation untouched for the active
  // loop-closure path. Other camera views remain available for
  // multicamera global-map visualization.
  if (camera_count > 0) {
    point_3d = camera_point_3d[0];
    point_2d_uv = camera_point_2d_uv[0];
    point_ids_ = camera_point_ids[0];
  }

  has_loop = false;
  loop_index = -1;
  has_fast_point = false;
  loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence = _sequence;
  is_vio_keyframe_ = is_vio_keyframe;

  if (is_vio_keyframe_) computeWindowBRIEFPoint();
  voc = vocBrief;
  computeBoW();
  KFcounter_ = KFcounter;  // for Covisibility graph
  updateConnections();     // for Covisibility graph

  computeBRIEFPoint();

  if (!params.debug_mode_) {
    image.release();
    for (cv::Mat& camera_image : images) camera_image.release();
  }
}

Keyframe::Keyframe(int64_t _time_stamp,
                   int _index,
                   Eigen::Vector3d& _svin_T_w_i,
                   Eigen::Matrix3d& _svin_R_w_i,
                   std::map<Keyframe*, int>& KFcounter,
                   int _sequence,
                   const Parameters& params,
                   const bool is_vio_keyframe)
    : params_(params) {
  time_stamp = _time_stamp;

  index = _index;
  svin_T_w_i = _svin_T_w_i;
  svin_R_w_i = _svin_R_w_i;
  T_w_i = svin_T_w_i;
  R_w_i = svin_R_w_i;
  origin_svin_T = svin_T_w_i;
  origin_svin_R = svin_R_w_i;

  has_loop = false;
  loop_index = -1;
  has_fast_point = false;
  loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence = _sequence;
  KFcounter_ = KFcounter;  // for Covisibility graph

  is_vio_keyframe_ = is_vio_keyframe;
  updateConnections();  // for Covisibility graph
}

double Keyframe::brisk_distance(const cv::Mat& a, const cv::Mat& b) {
  const unsigned char* pa = a.ptr<unsigned char>();
  const unsigned char* pb = b.ptr<unsigned char>();
  // number_of_128_bit_words or number_of_col, L = 48
  return static_cast<double>(brisk::Hamming::PopcntofXORed(pa, pb, 3 /*48 / 16*/));
}

void Keyframe::computeBRISKPoint() {
  // for searchByDescriptor to create new BRISK keypoints and descriptors
  std::shared_ptr<cv::FeatureDetector> detector(
      new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(Keyframe::briskDetectionThreshold_,
                                                                         Keyframe::briskDetectionOctaves_,
                                                                         Keyframe::briskDetectionAbsoluteThreshold_,
                                                                         Keyframe::briskDetectionMaximumKeypoints_));

  std::shared_ptr<cv::DescriptorExtractor> extractor(new brisk::BriskDescriptorExtractor(
      Keyframe::briskDescriptionRotationInvariance_, Keyframe::briskDescriptionScaleInvariance_));

  detector->detect(image, brisk_keypoints);

  if (!window_keypoints.empty()) {
    extractor->compute(image, window_keypoints, window_brisk_descriptors);
  } else {
    std::cout << "window keypoints are empty. This is a problem!!" << std::endl;
  }
  extractor->compute(image, brisk_keypoints, brisk_descriptors);

  std::cout << "Size of Brisk keypoints: " << brisk_keypoints.size() << std::endl;
}

void Keyframe::computeBoW() {
  if (bowVec.empty() || featVec.empty()) {
    // Feature vector associate features with nodes in the 4th level (from leaves up)
    // We assume the vocabulary tree has 6 levels, change the 4 otherwise
    voc->transform(brief_descriptors, bowVec);
  }
}

void Keyframe::updateConnections() {
  if (KFcounter_.empty() && is_vio_keyframe_) {
    // std::cout << "KFcounter is empty for KF: " << index << " This SHOULDN't be happening except 1st frame."
    // << std::endl;
    return;
  }

  // std::cout<<"Weights for observed keyframes in Kf: "<< this->index << std::endl;
  int th_weight = 20;  // TODO(Sharmin): Move it to the Config file
  for (std::map<Keyframe*, int>::iterator mit = KFcounter_.begin(); mit != KFcounter_.end(); mit++) {
    if (mit->second > th_weight) {
      mConnectedKeyFrameWeights.insert(std::make_pair(mit->first, mit->second));
      // std::cout << "Observed Kf: " << mit->first->index << " with weight(common MapPoint): " << mit->second
      //           << std::endl;
    }
  }
}

// Note Keypoints found by okvis_estimator
void Keyframe::computeWindowBRIEFPoint() {
  BriefExtractor extractor(params_.brief_pattern_file_.c_str());

  window_keypoints = point_2d_uv;

  extractor(image, window_keypoints, window_brief_descriptors);

  for (int i = 0; i < static_cast<int>(window_keypoints.size()); i++) {
    Eigen::Vector3d tmp_p;

    project_normal(Eigen::Vector2d(window_keypoints[i].pt.x, window_keypoints[i].pt.y), tmp_p);

    cv::KeyPoint tmp_norm;
    tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    window_keypoints_norm.push_back(tmp_norm);
  }
}

void Keyframe::project_normal(Eigen::Vector2d kp, Eigen::Vector3d& point3d) const {
  const float invfx = 1.0f / params_.camera_calibration_.focal_length_.x();
  const float invfy = 1.0f / params_.camera_calibration_.focal_length_.y();

  const float u = kp[0];
  const float v = kp[1];
  point3d[0] = (u - params_.camera_calibration_.principal_point_.x()) * invfx;
  point3d[1] = (v - params_.camera_calibration_.principal_point_.y()) * invfy;
  point3d[2] = 1.0;
}

void Keyframe::computeBRIEFPoint() {
  BriefExtractor extractor(params_.brief_pattern_file_.c_str());
  const int fast_th = 20;  // corner detector response threshold
  if (1) {
    cv::FAST(image, keypoints, fast_th, true);
  } else {
    std::vector<cv::Point2f> tmp_pts;
    cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
    for (int i = 0; i < static_cast<int>(tmp_pts.size()); i++) {
      cv::KeyPoint key;
      key.pt = tmp_pts[i];
      keypoints.push_back(key);
    }
  }
  extractor(image, keypoints, brief_descriptors);

  for (int i = 0; i < static_cast<int>(keypoints.size()); i++) {
    Eigen::Vector3d tmp_p;

    project_normal(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);

    cv::KeyPoint tmp_norm;
    tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    keypoints_norm.push_back(tmp_norm);
  }
}

void BriefExtractor::operator()(const cv::Mat& im,
                                std::vector<cv::KeyPoint>& keys,
                                std::vector<DVision::BRIEF256::bitset>& descriptors) const {
  m_brief.compute(im, keys, descriptors);
}

bool Keyframe::matchBrisk(const cv::Mat& window_descriptor,
                          const cv::Mat& descriptors_old,
                          const std::vector<cv::KeyPoint>& keypoints_old,
                          cv::Point2f& best_match) {
  cv::Point2f best_pt;
  int bestDist = 256;
  int bestIndex = -1;
  for (size_t i = 0; i < descriptors_old.rows; i++) {
    double dis = brisk_distance(window_descriptor, descriptors_old.row(i));
    if (dis < bestDist) {
      bestDist = dis;
      bestIndex = i;
    }
  }
  // printf("best dist %d", bestDist);
  if (bestIndex != -1 && bestDist < briskMatchingThreshold_) {
    best_match = keypoints_old[bestIndex].pt;
    return true;
  } else {
    return false;
  }
}

void Keyframe::searchByBRISKDescriptor(std::vector<cv::Point2f>& matched_2d_old,
                                       std::vector<uchar>& status,
                                       const cv::Mat& descriptors_old,
                                       const std::vector<cv::KeyPoint>& keypoints_old) {
  for (size_t i = 0; i < window_brisk_descriptors.rows; i++) {
    cv::Point2f pt(0.f, 0.f);
    if (matchBrisk(window_brisk_descriptors.row(i), descriptors_old, keypoints_old, pt))
      status.push_back(1);
    else
      status.push_back(0);
    matched_2d_old.push_back(pt);
  }
}

bool Keyframe::searchInAera(const DVision::BRIEF256::bitset window_descriptor,
                            const std::vector<DVision::BRIEF256::bitset>& descriptors_old,
                            const std::vector<cv::KeyPoint>& keypoints_old,
                            const std::vector<cv::KeyPoint>& keypoints_old_norm,
                            cv::Point2f& best_match,
                            cv::Point2f& best_match_norm) {
  cv::Point2f best_pt;
  int bestDist = 128;
  int bestIndex = -1;
  for (int i = 0; i < static_cast<int>(descriptors_old.size()); i++) {
    int dis = HammingDis(window_descriptor, descriptors_old[i]);
    if (dis < bestDist) {
      bestDist = dis;
      bestIndex = i;
    }
  }
  // printf("best dist %d", bestDist);
  // hard-coded hamming distance threshold for BRIEF256 here
  if (bestIndex != -1 && bestDist < 80) {
    best_match = keypoints_old[bestIndex].pt;
    best_match_norm = keypoints_old_norm[bestIndex].pt;
    return true;
  } else {
    return false;
  }
}

void Keyframe::searchByBRIEFDes(std::vector<cv::Point2f>& matched_2d_old,
                                std::vector<cv::Point2f>& matched_2d_old_norm,
                                std::vector<uchar>& status,
                                const std::vector<DVision::BRIEF256::bitset>& descriptors_old,
                                const std::vector<cv::KeyPoint>& keypoints_old,
                                const std::vector<cv::KeyPoint>& keypoints_old_norm) {
  for (int i = 0; i < static_cast<int>(window_brief_descriptors.size()); i++) {
    cv::Point2f pt(0.f, 0.f);
    cv::Point2f pt_norm(0.f, 0.f);
    if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
      status.push_back(1);
    else
      status.push_back(0);
    matched_2d_old.push_back(pt);
    matched_2d_old_norm.push_back(pt_norm);
  }
}

bool Keyframe::PnPRANSAC(const std::vector<cv::Point2f>& matched_2d_old,
                         const std::vector<cv::Point3f>& matched_3d,
                         std::vector<uchar>& status,
                         Eigen::Vector3d& PnP_T_old,
                         Eigen::Matrix3d& PnP_R_old,
                         bool* threw_exception) {
  status.assign(matched_2d_old.size(), 0);
  if (threw_exception) *threw_exception = false;
  if (matched_2d_old.size() != matched_3d.size()) return false;

  if (params_.camera_calibration_.projection_type_ == "double_sphere") {
    try {
      using DoubleSphereCamera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>;
      const CameraCalibration& calibration = params_.camera_calibration_;
      const DoubleSphereCamera camera(calibration.image_dimension_.x(),
                                      calibration.image_dimension_.y(),
                                      calibration.focal_length_.x(),
                                      calibration.focal_length_.y(),
                                      calibration.principal_point_.x(),
                                      calibration.principal_point_.y(),
                                      calibration.double_sphere_params_.x(),
                                      calibration.double_sphere_params_.y(),
                                      okvis::cameras::NoDistortion());

      opengv::bearingVectors_t bearing_vectors;
      opengv::points_t world_points;
      std::vector<cv::Point2f> valid_image_points;
      std::vector<size_t> original_indices;
      bearing_vectors.reserve(matched_2d_old.size());
      world_points.reserve(matched_3d.size());
      valid_image_points.reserve(matched_2d_old.size());
      original_indices.reserve(matched_2d_old.size());

      for (size_t i = 0; i < matched_2d_old.size(); ++i) {
        Eigen::Vector3d bearing;
        const Eigen::Vector2d image_point(matched_2d_old[i].x, matched_2d_old[i].y);
        if (!camera.backProject(image_point, &bearing) || !bearing.allFinite() || bearing.norm() == 0.0) continue;

        bearing_vectors.push_back(bearing.normalized());
        world_points.emplace_back(matched_3d[i].x, matched_3d[i].y, matched_3d[i].z);
        valid_image_points.push_back(matched_2d_old[i]);
        original_indices.push_back(i);
      }

      if (bearing_vectors.size() < 4) return false;

      opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors, world_points);
      using SacProblem = PixelReprojectionAbsolutePoseSacProblem;
      opengv::sac::Ransac<SacProblem> ransac;
      ransac.sac_model_ = std::make_shared<SacProblem>(adapter, camera, valid_image_points);
      const double pixel_threshold = params_.loop_closure_params_.pnp_reprojection_thresh;
      ransac.threshold_ = pixel_threshold * pixel_threshold;
      ransac.max_iterations_ = static_cast<int>(params_.loop_closure_params_.pnp_ransac_iterations);
      ransac.probability_ = 0.99;

      const bool solver_succeeded = ransac.computeModel();
      if (!solver_succeeded || !ransac.model_coefficients_.allFinite()) return false;

      for (const int inlier : ransac.inliers_) status[original_indices[inlier]] = 1;
      PnP_R_old = ransac.model_coefficients_.leftCols<3>();
      PnP_T_old = ransac.model_coefficients_.col(3);
      return true;
    } catch (const std::exception& exception) {
      LOG(WARNING) << "Double Sphere PnP failed with exception: " << exception.what();
      if (threw_exception) *threw_exception = true;
      return false;
    }
  }

  cv::Mat r, rvec, t, tmp_r;

  cv::Mat K = (cv::Mat_<double>(3, 3) << params_.camera_calibration_.focal_length_.x(),
               0,
               params_.camera_calibration_.principal_point_.x(),
               0,
               params_.camera_calibration_.focal_length_.y(),
               params_.camera_calibration_.principal_point_.y(),
               0,
               0,
               1.0);

  // std::cout << "Camera Matrix: " << K << std::endl;
  // std::cout << "distortion coeffs: " << params_.camera_calibration_.distortion_coefficients_ << std::endl;

  Eigen::Matrix3d R_inital;
  Eigen::Vector3d P_inital;

  Eigen::Matrix3d R_w_c = origin_svin_R;
  Eigen::Vector3d T_w_c = origin_svin_T;

  R_inital = R_w_c.inverse();
  P_inital = -(R_inital * T_w_c);

  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, t);

  cv::Mat inliers;

  // bjoshi
  // Temporary fix for https://github.com/opencv/opencv/issues/17799
  // This is a bug in opencv. The bug is fixed in opencv master branch.

  bool pnp_solver_succeeded = false;
  try {
    pnp_solver_succeeded = solvePnPRansac(matched_3d,
                                          matched_2d_old,
                                          K,
                                          params_.camera_calibration_.distortion_coefficients_,
                                          rvec,
                                          t,
                                          false,
                                          params_.loop_closure_params_.pnp_ransac_iterations,
                                          params_.loop_closure_params_.pnp_reprojection_thresh,
                                          0.99,
                                          inliers);
  } catch (const cv::Exception& e) {
    // std::cout << "Caught exception in PnPRANSAC:" << e.what() << std::endl;
    if (threw_exception) *threw_exception = true;
    inliers.setTo(cv::Scalar(0));
  }

  for (int i = 0; i < inliers.rows; i++) {
    int n = inliers.at<int>(i);
    status[n] = 1;
  }

  cv::Rodrigues(rvec, r);
  Eigen::Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();
  Eigen::Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(t, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);

  PnP_R_old = R_w_c_old;
  PnP_T_old = T_w_c_old;
  return pnp_solver_succeeded;
}

bool Keyframe::findConnection(Keyframe* old_kf) {
  LoopClosureFunnelRecord diagnostic;
  diagnostic.current_kf_id = index;
  diagnostic.current_timestamp = time_stamp;
  diagnostic.candidate_kf_id = old_kf->index;
  diagnostic.candidate_timestamp = old_kf->time_stamp;
  diagnostic.camera_model = params_.camera_calibration_.projection_type_;
  diagnostic.pnp_model = diagnostic.camera_model == "double_sphere" ? "opengv_bearing_ds" : "opencv_pinhole";
  diagnostic.tracked_points = point_3d.size();
  diagnostic.candidate_keypoints = old_kf->keypoints.size();
  diagnostic.min_correspondences = params_.loop_closure_params_.min_correspondences;
  diagnostic.pnp_iterations = params_.loop_closure_params_.pnp_ransac_iterations;
  diagnostic.pnp_reprojection_threshold = params_.loop_closure_params_.pnp_reprojection_thresh;
  diagnostic.max_yaw_deg = params_.loop_closure_params_.max_yaw_diff;
  diagnostic.max_position_m = params_.loop_closure_params_.max_position_diff;

  if (!old_kf->is_vio_keyframe_) {
    diagnostic.rejection_reason = "old_keyframe_not_vio";
    if (params_.loopClosureDiagnosticsEnabled()) {
      appendLoopClosureFunnelRecord(params_.debug_output_path_, diagnostic);
    }
    return false;
  }

  std::vector<cv::KeyPoint> matched_2d_cur;
  std::vector<cv::Point2f> matched_2d_old;
  std::vector<cv::Point2f> matched_2d_old_norm;
  std::vector<cv::Point3f> matched_3d;
  std::vector<Eigen::Vector3i> matched_ids;  // Reloc
  std::vector<uchar> status;

  matched_3d = point_3d;
  matched_2d_cur = point_2d_uv;
  matched_ids = point_ids_;

  if (params_.debug_mode_) {
    cv::Mat old_img = UtilsOpenCV::DrawCircles(old_kf->image, old_kf->keypoints);
    cv::Mat cur_image = UtilsOpenCV::DrawCircles(image, point_2d_uv);
    std::string loop_candidate_directory = params_.debug_output_path_ + "/loop_candidates/";
    std::string filename = loop_candidate_directory + "loop_candidate_" + std::to_string(index) + "_" +
                           std::to_string(old_kf->index) + ".jpg";
    UtilsOpenCV::showImagesSideBySide(cur_image, old_img, "loop closing candidates", false, true, filename);
  }

  searchByBRIEFDes(matched_2d_old,
                   matched_2d_old_norm,
                   status,
                   old_kf->brief_descriptors,
                   old_kf->keypoints,
                   old_kf->keypoints_norm);
  reduceVector(matched_2d_old, status);
  reduceVector(matched_3d, status);
  reduceVector(matched_2d_cur, status);
  reduceVector(matched_2d_old_norm, status);
  reduceVector(matched_ids, status);
  status.clear();
  diagnostic.descriptor_matches = matched_2d_cur.size();

  if (params_.debug_mode_) {
    cv::Mat corners_match_image =
        UtilsOpenCV::DrawCornersMatches(image, matched_2d_cur, old_kf->image, matched_2d_old, true);
    std::string dscriptor_match_dir = params_.debug_output_path_ + "/descriptor_matched/";
    std::string filename = dscriptor_match_dir + "descriptor_match_" + std::to_string(index) + "_" +
                           std::to_string(old_kf->index) + ".jpg";
    UtilsOpenCV::writeCompressedDebugImage(filename, corners_match_image);
  }

  // std::cout << "Size Before RANSAC: " << matched_2d_cur.size() << std::endl;

  // opengv::transformation_t T_w_c_old;
  // if (LoopClosureUtils::geometricVerificationNister(
  //         matched_2d_cur, matched_2d_old, status, params_.loop_closure_params_.min_correspondences, &T_w_c_old)) {
  //   reduceVector(matched_2d_old, status);
  //   reduceVector(matched_3d, status);
  //   reduceVector(matched_2d_cur, status);
  //   reduceVector(matched_2d_old_norm, status);
  //   reduceVector(matched_ids, status);
  //   status.clear();

  //   if (params_.debug_image_) {
  //     cv::Mat corners_match_image =
  //         UtilsOpenCV::DrawCornersMatches(image, matched_2d_cur, old_kf->image, matched_2d_old, true);
  //     std::string dscriptor_match_dir = pkg_path + "/output_logs/geometric_verification/";
  //     std::string filename = dscriptor_match_dir + "geometric_verification_" + std::to_string(index) + "_" +
  //                            std::to_string(old_kf->index) + ".jpg";
  //     UtilsOpenCV::writeCompressedDebugImage(filename, corners_match_image);
  //   }
  // } else {
  //   return false;
  // }

  Eigen::Vector3d PnP_T_old;
  Eigen::Matrix3d PnP_R_old;
  Eigen::Vector3d relative_t;
  Eigen::Quaterniond relative_q;
  double relative_yaw;
  cv::Mat pnp_verified_image;

  const auto savePnpVerifiedImage = [&](const std::string& decision, bool accepted) {
    if (!params_.debug_mode_ || pnp_verified_image.empty()) {
      return;
    }

    cv::Mat notation(90, pnp_verified_image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    putText(notation,
            "current frame: " + std::to_string(index),
            cv::Point2f(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0),
            3);
    putText(notation,
            "previous frame: " + std::to_string(old_kf->index),
            cv::Point2f(20 + pnp_verified_image.cols / 2, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0),
            3);
    putText(notation,
            "decision: " + decision,
            cv::Point2f(20, 72),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            accepted ? cv::Scalar(0, 128, 0) : cv::Scalar(0, 0, 255),
            3);
    cv::vconcat(notation, pnp_verified_image, pnp_verified_image);

    const std::string classification = accepted ? "passed" : "rejected";
    const std::string output_dir = params_.debug_output_path_ + "/pnp_verified/" + classification + "/";
    const std::string filename = output_dir + "pnp_verified_" + std::to_string(index) + "_" +
                                 std::to_string(old_kf->index) + ".jpg";
    UtilsOpenCV::writeCompressedDebugImage(filename, pnp_verified_image);
  };

  if (static_cast<int>(matched_2d_cur.size()) > params_.loop_closure_params_.min_correspondences) {
    diagnostic.pnp_attempted = true;
    diagnostic.pnp_solver_succeeded =
        PnPRANSAC(matched_2d_old, matched_3d, status, PnP_T_old, PnP_R_old, &diagnostic.pnp_exception);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_ids, status);
    status.clear();
    diagnostic.pnp_inliers = matched_2d_cur.size();

    if (params_.debug_mode_) {
      pnp_verified_image =
          UtilsOpenCV::DrawCornersMatches(image, matched_2d_cur, old_kf->image, matched_2d_old, true);
    }
  }

  // std::cout<< "Size after RANSAC "<< matched_2d_cur.size() << std::endl;

  if (static_cast<int>(matched_2d_cur.size()) > params_.loop_closure_params_.min_correspondences) {
    relative_t = PnP_R_old.transpose() * (origin_svin_T - PnP_T_old);
    relative_q = PnP_R_old.transpose() * origin_svin_R;

    relative_yaw = Utils::normalizeAngle(Utils::R2ypr(origin_svin_R).x() - Utils::R2ypr(PnP_R_old).x());
    diagnostic.relative_yaw_deg = relative_yaw;
    diagnostic.relative_translation_m = relative_t.norm();
    diagnostic.yaw_gate_passed = abs(relative_yaw) < params_.loop_closure_params_.max_yaw_diff;
    diagnostic.position_gate_passed = relative_t.norm() < params_.loop_closure_params_.max_position_diff;

    if (diagnostic.yaw_gate_passed && diagnostic.position_gate_passed) {
      if (params_.debug_mode_) {
        cv::Mat loop_image =
            UtilsOpenCV::DrawCornersMatches(image, matched_2d_cur, old_kf->image, matched_2d_old, true);
        cv::Mat notation(50, loop_image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        putText(notation,
                "current frame: " + std::to_string(index),
                cv::Point2f(20, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(255),
                3);

        putText(
            notation,
            "previous frame: " + std::to_string(old_kf->index) + " matches: " + std::to_string(matched_2d_cur.size()),
            cv::Point2f(20 + loop_image.cols / 2, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(255),
            3);
        cv::vconcat(notation, loop_image, loop_image);
        std::string pnp_verified_dir = params_.debug_output_path_ + "/loop_closure/";
        std::string filename =
            pnp_verified_dir + "loop_closure_" + std::to_string(index) + "_" + std::to_string(old_kf->index) + ".jpg";
        UtilsOpenCV::writeCompressedDebugImage(filename, loop_image);
        std::string loop_closure_stats = params_.debug_output_path_ + "/loop_closure.txt";
        std::ofstream loop_closure_file(loop_closure_stats, std::ios::app);
        loop_closure_file.setf(std::ios::fixed, std::ios::floatfield);
        Eigen::Vector3d relative_ypr = Utils::R2ypr(relative_q.toRotationMatrix());
        loop_closure_file.precision(9);
        loop_closure_file << index << " " << time_stamp << " " << old_kf->index << " " << old_kf->time_stamp << " "
                          << relative_t.x() << " " << relative_t.y() << " " << relative_t.z() << " "
                          << relative_ypr.transpose() << std::endl;
        loop_closure_file.close();
      }
      has_loop = true;
      loop_index = old_kf->index;
      loop_info << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(),
          relative_q.z(), relative_yaw;
      diagnostic.accepted = true;
      diagnostic.rejection_reason = "accepted";
      savePnpVerifiedImage(diagnostic.rejection_reason, true);
      if (params_.loopClosureDiagnosticsEnabled()) {
        appendLoopClosureFunnelRecord(params_.debug_output_path_, diagnostic);
      }
      return true;
    }
  }

  if (!diagnostic.pnp_attempted) {
    diagnostic.rejection_reason = "insufficient_descriptor_matches";
  } else if (diagnostic.pnp_inliers <= static_cast<size_t>(params_.loop_closure_params_.min_correspondences)) {
    diagnostic.rejection_reason = diagnostic.pnp_exception
                                      ? "pnp_exception"
                                      : (diagnostic.pnp_solver_succeeded ? "insufficient_pnp_inliers"
                                                                        : "pnp_solver_failed");
  } else if (!diagnostic.yaw_gate_passed && !diagnostic.position_gate_passed) {
    diagnostic.rejection_reason = "yaw_and_position_gates_failed";
  } else if (!diagnostic.yaw_gate_passed) {
    diagnostic.rejection_reason = "yaw_gate_failed";
  } else {
    diagnostic.rejection_reason = "position_gate_failed";
  }
  savePnpVerifiedImage(diagnostic.rejection_reason, false);
  if (params_.loopClosureDiagnosticsEnabled()) {
    appendLoopClosureFunnelRecord(params_.debug_output_path_, diagnostic);
  }

  return false;
}

int Keyframe::HammingDis(const DVision::BRIEF256::bitset& a, const DVision::BRIEF256::bitset& b) {
  DVision::BRIEF256::bitset xor_of_bitset = a ^ b;
  int dis = xor_of_bitset.count();
  return dis;
}

void Keyframe::getSVInPose(Eigen::Vector3d& _T_w_i, Eigen::Matrix3d& _R_w_i) {
  _T_w_i = svin_T_w_i;
  _R_w_i = svin_R_w_i;
}

void Keyframe::getPose(Eigen::Vector3d& _T_w_i, Eigen::Matrix3d& _R_w_i) {
  _T_w_i = T_w_i;
  _R_w_i = R_w_i;
}

void Keyframe::updatePose(const Eigen::Vector3d& _T_w_i, const Eigen::Matrix3d& _R_w_i) {
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
}

void Keyframe::updateSVInPose(const Eigen::Vector3d& _T_w_i, const Eigen::Matrix3d& _R_w_i) {
  svin_T_w_i = _T_w_i;
  svin_R_w_i = _R_w_i;
  T_w_i = svin_T_w_i;
  R_w_i = svin_R_w_i;
}

Eigen::Vector3d Keyframe::getLoopRelativeT() { return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2)); }

Eigen::Quaterniond Keyframe::getLoopRelativeQ() {
  return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double Keyframe::getLoopRelativeYaw() { return loop_info(7); }

void Keyframe::updateLoop(Eigen::Matrix<double, 8, 1>& _loop_info) {
  if (abs(_loop_info(7)) < 30.0 && Eigen::Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    // printf("update loop info\n");
    loop_info = _loop_info;
  }
}

BriefExtractor::BriefExtractor(const std::string& pattern_file) {
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened()) throw std::string("Could not open file ") + pattern_file;

  std::vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}

// void Keyframe::setRelocalizationPCLCallback(const PointCloudCallback& pcl_callback) {
//   relocalization_pcl_callback_ = pcl_callback;
// }
