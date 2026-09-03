#include "pose_graph/PoseGraph.h"

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <fstream>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <list>
#include <map>
#include <thread>
#include <set>
#include <string>

#include "pose_graph/Pose3DError.h"
#include "utils/UtilsOpenCV.h"

namespace {

struct MulticameraLoopCandidate {
  size_t current_camera = 0;
  size_t historical_camera = 0;
  size_t rank = 0;
  int keyframe_id = -1;
  double score = 0.0;
  double score_threshold = 0.0;
  bool score_passed = false;
  Keyframe* keyframe = nullptr;
  Keyframe::CameraPairDiagnostic verification;
};

std::string cameraPairDirectory(size_t current_camera, size_t historical_camera) {
  return "cam" + std::to_string(current_camera) + "cam" +
         std::to_string(historical_camera);
}

std::string cameraPairLabel(size_t current_camera, size_t historical_camera) {
  return "cam " + std::to_string(current_camera) + " -> cam " +
         std::to_string(historical_camera);
}

cv::Mat addMulticameraDebugBanner(const cv::Mat& image,
                                  const std::string& camera_pair,
                                  int current_id,
                                  int historical_id,
                                  const std::string& decision = "",
                                  const std::string& inlier_summary = "") {
  if (image.empty()) return image;
  const int banner_height = 55 + (decision.empty() ? 0 : 40) + (inlier_summary.empty() ? 0 : 40);
  cv::Mat banner(banner_height,
                 image.cols,
                 CV_8UC3,
                 cv::Scalar(255, 255, 255));
  cv::putText(banner,
              camera_pair + "   current: " + std::to_string(current_id) +
                  "   historical: " + std::to_string(historical_id),
              cv::Point(20, 36),
              cv::FONT_HERSHEY_SIMPLEX,
              0.9,
              cv::Scalar(0, 0, 0),
              2);
  if (!decision.empty()) {
    cv::putText(banner,
                "decision: " + decision,
                cv::Point(20, 77),
                cv::FONT_HERSHEY_SIMPLEX,
                0.9,
                decision == "accepted" ? cv::Scalar(0, 128, 0) : cv::Scalar(0, 0, 255),
                2);
  }
  if (!inlier_summary.empty()) {
    cv::putText(banner,
                inlier_summary,
                cv::Point(20, decision.empty() ? 77 : 117),
                cv::FONT_HERSHEY_SIMPLEX,
                0.9,
                cv::Scalar(0, 0, 0),
                2);
  }
  cv::Mat output;
  cv::vconcat(banner, image, output);
  return output;
}

std::string multicameraRejectionReason(const Keyframe::CameraPairDiagnostic& verification,
                                       int min_correspondences) {
  if (!verification.pnp_attempted) return "insufficient_descriptor_matches";
  if (verification.pnp_exception) return "pnp_exception";
  if (!verification.pnp_solver_succeeded) return "pnp_solver_failed";
  if (verification.pnp_inliers <= static_cast<size_t>(min_correspondences)) {
    return "insufficient_pnp_inliers";
  }
  if (!verification.yaw_gate_passed && !verification.position_gate_passed) {
    return "yaw_and_position_gates_failed";
  }
  if (!verification.yaw_gate_passed) return "yaw_gate_failed";
  if (!verification.position_gate_passed) return "position_gate_failed";
  return "accepted";
}

void appendDBoWFunnelRecord(const Keyframe* keyframe,
                            int frame_index,
                            float min_score,
                            const DBoW2::QueryResults& results,
                            int selected_candidate_id,
                            const std::string& decision) {
  if (!keyframe->params_.loopClosureDiagnosticsEnabled()) return;

  const double score_threshold = 0.60 * min_score;
  int passing_results = 0;
  int best_passing_id = -1;
  double best_passing_score = 0.0;
  for (const auto& result : results) {
    if (result.Score > score_threshold) {
      passing_results++;
      if (best_passing_id == -1 || result.Score > best_passing_score) {
        best_passing_id = result.Id;
        best_passing_score = result.Score;
      }
    }
  }

  const int temporal_gap = selected_candidate_id >= 0 ? frame_index - selected_candidate_id : -1;
  std::ofstream output(keyframe->params_.debug_output_path_ + "/loop_closure_dbow_funnel.csv", std::ios::app);
  if (!output.is_open()) return;
  output << std::setprecision(17) << frame_index << ',' << keyframe->time_stamp << ',' << min_score << ','
         << score_threshold << ',' << frame_index - 50 << ',' << results.size() << ',' << passing_results << ','
         << best_passing_id << ',' << best_passing_score << ',' << selected_candidate_id << ',' << temporal_gap << ','
         << decision << '\n';
}

}  // namespace

PoseGraph::PoseGraph() {
  earliest_loop_index = -1;
  t_drift = Eigen::Vector3d(0, 0, 0);
  yaw_drift = 0;
  r_drift = Eigen::Matrix3d::Identity();
  w_t_svin = Eigen::Vector3d(0, 0, 0);
  w_r_svin = Eigen::Matrix3d::Identity();
  global_index = 0;
  sequence_cnt = 0;
  sequence_loop.push_back(0);
  base_sequence = 1;
  is_fast_localization_ = true;
}

PoseGraph::~PoseGraph() { shutdown(); }

void PoseGraph::set_fast_relocalization(const bool fast_relocalization) { is_fast_localization_ = fast_relocalization; }

void PoseGraph::setBriefVocAndDB(BriefVocabulary* vocabulary, BriefDatabase database) {
  voc = vocabulary;
  db = database;
}

void PoseGraph::startOptimizationThread(bool vio_only_optimization) {
  shutdown_requested_ = false;
  if (vio_only_optimization) {
    t_optimization = std::thread(&PoseGraph::optimize4DoFPoseGraph, this);
  } else {
    t_optimization = std::thread(&PoseGraph::optimize6DoFPoseGraph, this);
  }
}

void PoseGraph::shutdown() {
  shutdown_requested_ = true;
  if (t_optimization.joinable()) t_optimization.join();
}

void PoseGraph::addKFToPoseGraph(Keyframe* cur_kf, bool flag_detect_loop) {
  // shift to base frame
  Eigen::Vector3d svin_P_cur;
  Eigen::Matrix3d svin_R_cur;
  if (sequence_cnt != cur_kf->sequence) {
    sequence_cnt++;
    sequence_loop.push_back(0);
    w_t_svin = Eigen::Vector3d(0, 0, 0);
    w_r_svin = Eigen::Matrix3d::Identity();

    {
      std::lock_guard<std::mutex> l(driftMutex_);
      t_drift = Eigen::Vector3d(0, 0, 0);
      r_drift = Eigen::Matrix3d::Identity();
    }
  }

  cur_kf->getSVInPose(svin_P_cur, svin_R_cur);
  svin_P_cur = w_r_svin * svin_P_cur + w_t_svin;
  svin_R_cur = w_r_svin * svin_R_cur;
  cur_kf->updateSVInPose(svin_P_cur, svin_R_cur);
  cur_kf->index = global_index;
  global_index++;
  std::set<Keyframe*> loopCandidates;
  int loop_index = -1;

  if (flag_detect_loop) {  // at least 50 KF has been passed
    if (cur_kf->params_.loop_closure_params_.multicamera_enabled &&
        cur_kf->camera_brief_descriptors.size() > 1u) {
      loop_index = detectMulticameraLoop(cur_kf, cur_kf->index);
    } else {
      loop_index = detectLoop(cur_kf, cur_kf->index);
    }
  } else {
    db.add(cur_kf->brief_descriptors);
  }

  if (loop_index != -1) {
    Keyframe* old_kf = getKFPtr(loop_index);
    const bool connection_found = cur_kf->has_loop || cur_kf->findConnection(old_kf);
    if (connection_found) {
      if (earliest_loop_index > loop_index || earliest_loop_index == -1) earliest_loop_index = loop_index;

      Eigen::Vector3d w_P_old, w_P_cur, svin_P_cur;
      Eigen::Matrix3d w_R_old, w_R_cur, svin_R_cur;
      old_kf->getSVInPose(w_P_old, w_R_old);  // old_kf replaced by min_loop_kf
      cur_kf->getSVInPose(svin_P_cur, svin_R_cur);

      Eigen::Vector3d relative_t;
      Eigen::Quaterniond relative_q;
      relative_t = cur_kf->getLoopRelativeT();
      relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
      w_P_cur = w_R_old * relative_t + w_P_old;
      w_R_cur = w_R_old * relative_q;
      double shift_yaw;
      Eigen::Matrix3d shift_r;
      Eigen::Vector3d shift_t;
      shift_yaw = Utils::R2ypr(w_R_cur).x() - Utils::R2ypr(svin_R_cur).x();
      shift_r = Utils::ypr2R(Eigen::Vector3d(shift_yaw, 0, 0));
      shift_t = w_P_cur - w_R_cur * svin_R_cur.transpose() * svin_P_cur;
      // shift svin pose of whole sequence to the world frame
      if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0) {
        w_r_svin = shift_r;
        w_t_svin = shift_t;
        svin_P_cur = w_r_svin * svin_P_cur + w_t_svin;
        svin_R_cur = w_r_svin * svin_R_cur;
        cur_kf->updateSVInPose(svin_P_cur, svin_R_cur);
        std::list<Keyframe*>::iterator it = keyframelist.begin();
        for (; it != keyframelist.end(); it++) {
          if ((*it)->sequence == cur_kf->sequence) {
            Eigen::Vector3d svin_P_cur;
            Eigen::Matrix3d svin_R_cur;
            (*it)->getSVInPose(svin_P_cur, svin_R_cur);
            svin_P_cur = w_r_svin * svin_P_cur + w_t_svin;
            svin_R_cur = w_r_svin * svin_R_cur;
            (*it)->updateSVInPose(svin_P_cur, svin_R_cur);
          }
        }
        sequence_loop[cur_kf->sequence] = 1;
      }
      std::lock_guard<std::mutex> l(optimizationMutex_);
      optimizationBuffer_.push(cur_kf->index);
    }
  }

  {
    std::lock_guard<std::mutex> l(kflistMutex_);
    Eigen::Vector3d P;
    Eigen::Matrix3d R;
    cur_kf->getSVInPose(P, R);
    P = r_drift * P + t_drift;
    R = r_drift * R;
    cur_kf->updatePose(P, R);
    Eigen::Quaterniond Q{R};

    std::pair<Eigen::Vector3d, Eigen::Vector3d> loop_info;
    loop_info.first = Eigen::Vector3d::Zero();
    loop_info.second = Eigen::Vector3d::Zero();

    if (cur_kf->has_loop) {
      Keyframe* connected_KF = getKFPtr(cur_kf->loop_index);
      Eigen::Vector3d connected_P, P0;
      Eigen::Matrix3d connected_R, R0;
      connected_KF->getPose(connected_P, connected_R);
      cur_kf->getPose(P0, R0);
      loop_info = {P0, connected_P};
    }

    keyframelist.push_back(cur_kf);

    std::pair<Timestamp, Eigen::Matrix4d> pose;
    pose.first = cur_kf->time_stamp;
    pose.second.block<3, 3>(0, 0) = R;
    pose.second.block<3, 1>(0, 3) = P;
    CHECK(keyframe_pose_callback_);
    keyframe_pose_callback_(pose, loop_info);
  }
}

Keyframe* PoseGraph::getKFPtr(int index) {
  std::list<Keyframe*>::iterator it = keyframelist.begin();
  for (; it != keyframelist.end(); it++) {
    if ((*it)->index == index) break;
  }
  if (it != keyframelist.end())
    return *it;
  else
    return NULL;
}

int PoseGraph::detectLoop(Keyframe* keyframe, int frame_index) {
  cv::Mat compressed_image;

  if (keyframe->bowVec.empty() || keyframe->featVec.empty()) {
    // Feature vector associate features with nodes in the 4th level (from leaves up)
    // We assume the vocabulary tree has 6 levels, change the 4 otherwise
    voc->transform(keyframe->brief_descriptors, keyframe->bowVec);
  }

  float min_score = 1.0;
  for (std::map<Keyframe*, int>::iterator mit = keyframe->mConnectedKeyFrameWeights.begin();
       mit != keyframe->mConnectedKeyFrameWeights.end();
       mit++) {
    // BowVector neigh_vec;

    if (mit->first->bowVec.empty() || mit->first->featVec.empty())
      voc->transform(mit->first->brief_descriptors, mit->first->bowVec);

    float score = voc->score(keyframe->bowVec, mit->first->bowVec);
    // std::cout << "Score in neigh frames: " << score << " with Id: " << mit->first->index << std::endl;
    if (score < min_score) min_score = score;
  }

  // std::cout<< "Min BoW Score: "<< min_score << std::endl;

  // first query; then add this frame into database!
  DBoW2::QueryResults ret;
  db.query(keyframe->bowVec, ret, 4, frame_index - 50);
  db.add(keyframe->brief_descriptors);

  if (keyframe->params_.loop_closure_params_.multicamera_diagnostics &&
      keyframe->params_.loopClosureDiagnosticsEnabled()) {
    runMulticameraDiagnostics(keyframe, frame_index, 0.60F * min_score);
  }

  bool find_loop = false;
  cv::Mat loop_result;

  for (unsigned int i = 0; i < ret.size(); i++) {
    if (ret[i].Score > 0.60 * min_score) {
      find_loop = true;
      // std::cout<< "Query KF: "<< frame_index<< " candidate kf: "<< ret[i].Id << std::endl;
    }
  }

  // Note:  Returns depending on the highest score
  if (find_loop && frame_index > 50) {
    int best_index = -1;
    float best_score = 0.0;
    for (unsigned int i = 0; i < ret.size(); i++) {
      if (best_index == -1 || (ret[i].Score > best_score && ret[i].Score > 0.60 * min_score)) {
        best_index = ret[i].Id;
        best_score = ret[i].Score;
      }
    }
    appendDBoWFunnelRecord(keyframe, frame_index, min_score, ret, best_index, "candidate_selected");
    return best_index;
  } else {
    std::string decision;
    if (frame_index <= 50) {
      decision = "warmup";
    } else if (ret.empty()) {
      decision = "no_query_results";
    } else {
      decision = "score_threshold_failed";
    }
    appendDBoWFunnelRecord(keyframe, frame_index, min_score, ret, -1, decision);
    return -1;
  }
}

int PoseGraph::detectMulticameraLoop(Keyframe* keyframe, int frame_index) {
  const size_t camera_count = keyframe->camera_brief_descriptors.size();
  if (camera_count < 2u) return detectLoop(keyframe, frame_index);

  if (multicamera_diagnostic_databases_.empty()) {
    multicamera_diagnostic_databases_.resize(camera_count);
    for (BriefDatabase& camera_database : multicamera_diagnostic_databases_) {
      camera_database.setVocabulary(*voc, false, 0);
    }
  }
  if (multicamera_diagnostic_databases_.size() != camera_count) {
    LOG(ERROR) << "Camera count changed inside multicamera loop closure; falling back to camera 0.";
    return detectLoop(keyframe, frame_index);
  }

  std::vector<MulticameraLoopCandidate> candidates;
  for (size_t current_camera = 0; current_camera < camera_count; ++current_camera) {
    const auto& current_descriptors = keyframe->camera_brief_descriptors[current_camera];
    if (current_descriptors.empty()) continue;

    DBoW2::BowVector current_bow = keyframe->camera_bow_vectors[current_camera];
    if (current_bow.empty()) voc->transform(current_descriptors, current_bow);

    float min_neighbor_score = 1.0F;
    for (const auto& connection : keyframe->mConnectedKeyFrameWeights) {
      const Keyframe* neighbor = connection.first;
      if (neighbor == nullptr || current_camera >= neighbor->camera_bow_vectors.size()) continue;
      DBoW2::BowVector neighbor_bow = neighbor->camera_bow_vectors[current_camera];
      if (neighbor_bow.empty() && current_camera < neighbor->camera_brief_descriptors.size() &&
          !neighbor->camera_brief_descriptors[current_camera].empty()) {
        voc->transform(neighbor->camera_brief_descriptors[current_camera], neighbor_bow);
      }
      if (!neighbor_bow.empty()) {
        min_neighbor_score = std::min(min_neighbor_score,
                                      static_cast<float>(voc->score(current_bow, neighbor_bow)));
      }
    }
    const double score_threshold = 0.60 * min_neighbor_score;

    for (size_t historical_camera = 0; historical_camera < camera_count; ++historical_camera) {
      DBoW2::QueryResults results;
      multicamera_diagnostic_databases_[historical_camera].query(
          current_bow, results, 4, frame_index - 50);
      for (size_t rank = 0; rank < results.size(); ++rank) {
        MulticameraLoopCandidate candidate;
        candidate.current_camera = current_camera;
        candidate.historical_camera = historical_camera;
        candidate.rank = rank + 1u;
        candidate.keyframe_id = results[rank].Id;
        candidate.score = results[rank].Score;
        candidate.score_threshold = score_threshold;
        candidate.score_passed = frame_index > 50 && results[rank].Score > score_threshold;
        candidates.push_back(candidate);
      }
    }
  }

  // Query every ordered camera pair before adding this synchronized rig frame.
  // This preserves one database entry per camera and keeps every DBoW ID equal
  // to the pose-graph keyframe ID.
  for (size_t camera = 0; camera < camera_count; ++camera) {
    const DBoW2::EntryId entry_id =
        multicamera_diagnostic_databases_[camera].add(keyframe->camera_brief_descriptors[camera]);
    CHECK_EQ(static_cast<int>(entry_id), frame_index)
        << "Per-camera DBoW database lost rig-keyframe ID alignment for camera " << camera;
  }

  // Keyframes are never deleted, so these pointers remain valid after the
  // short locked snapshot. Do not hold the list lock during matching or PnP.
  {
    std::lock_guard<std::mutex> lock(kflistMutex_);
    for (MulticameraLoopCandidate& candidate : candidates) {
      if (candidate.score_passed) candidate.keyframe = getKFPtr(candidate.keyframe_id);
    }
  }

  std::vector<size_t> verification_indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].score_passed && candidates[i].keyframe != nullptr) {
      verification_indices.push_back(i);
    }
  }

  if (!verification_indices.empty()) {
    std::atomic<size_t> next_job{0u};
    const size_t worker_count = std::min(
        verification_indices.size(),
        static_cast<size_t>(std::max(1, keyframe->params_.loop_closure_params_.multicamera_matching_threads)));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&]() {
        while (true) {
          const size_t job = next_job.fetch_add(1u);
          if (job >= verification_indices.size()) return;
          MulticameraLoopCandidate& candidate = candidates[verification_indices[job]];
          candidate.verification = keyframe->diagnoseCameraPair(candidate.keyframe,
                                                                candidate.current_camera,
                                                                candidate.historical_camera);
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
  }

  MulticameraLoopCandidate* winner = nullptr;
  for (MulticameraLoopCandidate& candidate : candidates) {
    if (!candidate.verification.accepted) continue;
    if (winner == nullptr ||
        candidate.verification.pnp_inliers > winner->verification.pnp_inliers ||
        (candidate.verification.pnp_inliers == winner->verification.pnp_inliers &&
         candidate.score > winner->score)) {
      winner = &candidate;
    }
  }

  if (keyframe->params_.debug_mode_) {
    for (const MulticameraLoopCandidate& candidate : candidates) {
      if (!candidate.score_passed || candidate.keyframe == nullptr ||
          candidate.current_camera >= keyframe->images.size() ||
          candidate.current_camera >= keyframe->camera_point_2d_uv.size() ||
          candidate.historical_camera >= candidate.keyframe->camera_keypoints.size() ||
          candidate.historical_camera >= candidate.keyframe->images.size()) {
        continue;
      }
      const cv::Mat& current_image = keyframe->images[candidate.current_camera];
      const cv::Mat& historical_image = candidate.keyframe->images[candidate.historical_camera];
      if (current_image.empty() || historical_image.empty()) continue;

      const std::string pair_directory =
          cameraPairDirectory(candidate.current_camera, candidate.historical_camera);
      const std::string pair_label =
          cameraPairLabel(candidate.current_camera, candidate.historical_camera);
      const std::string filename_stem =
          "cur_" + std::to_string(frame_index) + "_hist_" +
          std::to_string(candidate.keyframe_id) + "_rank_" +
          std::to_string(candidate.rank) + ".jpg";

      const std::string candidate_directory =
          keyframe->params_.debug_output_path_ + "/loop_candidates/" + pair_directory;
      std::filesystem::create_directories(candidate_directory);
      cv::Mat current_features = UtilsOpenCV::DrawCircles(
          current_image, keyframe->camera_point_2d_uv[candidate.current_camera]);
      cv::Mat historical_features = UtilsOpenCV::DrawCircles(
          historical_image, candidate.keyframe->camera_keypoints[candidate.historical_camera]);
      cv::Mat candidate_image = UtilsOpenCV::concatenateTwoImages(current_features, historical_features);
      candidate_image = addMulticameraDebugBanner(candidate_image,
                                                   pair_label,
                                                   frame_index,
                                                   candidate.keyframe_id);
      UtilsOpenCV::writeCompressedDebugImage(candidate_directory + "/" + filename_stem,
                                             candidate_image);

      const std::string descriptor_directory =
          keyframe->params_.debug_output_path_ + "/descriptor_matched/" + pair_directory;
      std::filesystem::create_directories(descriptor_directory);
      cv::Mat descriptor_image = UtilsOpenCV::DrawCornersMatches(
          current_image,
          candidate.verification.descriptor_current_points,
          historical_image,
          candidate.verification.descriptor_historical_points,
          true);
      descriptor_image = addMulticameraDebugBanner(descriptor_image,
                                                    pair_label,
                                                    frame_index,
                                                    candidate.keyframe_id);
      UtilsOpenCV::writeCompressedDebugImage(descriptor_directory + "/" + filename_stem,
                                             descriptor_image);

      if (candidate.verification.pnp_attempted) {
        const std::string decision = multicameraRejectionReason(
            candidate.verification, keyframe->params_.loop_closure_params_.min_correspondences);
        const std::string classification = candidate.verification.accepted ? "passed" : "rejected";
        const std::string pnp_directory = keyframe->params_.debug_output_path_ +
                                          "/pnp_verified/" + pair_directory + "/" + classification;
        std::filesystem::create_directories(pnp_directory);
        cv::Mat pnp_image = UtilsOpenCV::DrawCornersMatches(
            current_image,
            candidate.verification.pnp_current_points,
            historical_image,
            candidate.verification.pnp_historical_points,
            true);
        pnp_image = addMulticameraDebugBanner(pnp_image,
                                              pair_label,
                                              frame_index,
                                              candidate.keyframe_id,
                                              decision,
                                              "inlier#: " + std::to_string(candidate.verification.pnp_inliers) +
                                                  ", required inlier#: " +
                                                  std::to_string(keyframe->params_.loop_closure_params_.min_correspondences + 1));
        UtilsOpenCV::writeCompressedDebugImage(pnp_directory + "/" + filename_stem,
                                               pnp_image);
      }
    }
  }

  if (keyframe->params_.loopClosureDiagnosticsEnabled()) {
    std::ofstream output(keyframe->params_.debug_output_path_ +
                             "/loop_closure_multicamera_shadow.csv",
                         std::ios::app);
    if (output.is_open()) {
      for (const MulticameraLoopCandidate& candidate : candidates) {
        const auto& verification = candidate.verification;
        output << std::setprecision(17) << frame_index << ',' << keyframe->time_stamp << ','
               << candidate.current_camera << ',' << candidate.historical_camera << ','
               << candidate.rank << ',' << candidate.keyframe_id << ',' << candidate.score << ','
               << candidate.score_threshold << ',' << candidate.score_passed << ','
               << (candidate.score_passed && candidate.keyframe != nullptr) << ','
               << verification.tracked_points << ',' << verification.descriptor_matches << ','
               << verification.pnp_solver_succeeded << ',' << verification.pnp_exception << ','
               << verification.pnp_inliers << ',' << verification.relative_yaw_deg << ','
               << verification.relative_translation_m << ',' << verification.yaw_gate_passed << ','
               << verification.position_gate_passed << ',' << verification.accepted << ','
               << (&candidate == winner) << '\n';
      }
    }
  }

  if (winner == nullptr) return -1;
  keyframe->acceptCameraPairConnection(winner->keyframe, winner->verification);
  if (keyframe->params_.debug_mode_) {
    const std::string pair_directory =
        cameraPairDirectory(winner->current_camera, winner->historical_camera);
    const std::string pair_label =
        cameraPairLabel(winner->current_camera, winner->historical_camera);
    const std::string loop_directory =
        keyframe->params_.debug_output_path_ + "/loop_closure/" + pair_directory;
    std::filesystem::create_directories(loop_directory);
    cv::Mat loop_image = UtilsOpenCV::DrawCornersMatches(
        keyframe->images[winner->current_camera],
        winner->verification.pnp_current_points,
        winner->keyframe->images[winner->historical_camera],
        winner->verification.pnp_historical_points,
        true);
    loop_image = addMulticameraDebugBanner(loop_image,
                                           pair_label,
                                           frame_index,
                                           winner->keyframe_id,
                                           "accepted",
                                           "inlier#: " + std::to_string(winner->verification.pnp_inliers) +
                                               ", required inlier#: " +
                                               std::to_string(keyframe->params_.loop_closure_params_.min_correspondences + 1));
    UtilsOpenCV::writeCompressedDebugImage(
        loop_directory + "/cur_" + std::to_string(frame_index) + "_hist_" +
            std::to_string(winner->keyframe_id) + ".jpg",
        loop_image);

    std::ofstream loop_file(keyframe->params_.debug_output_path_ + "/loop_closure.txt",
                            std::ios::app);
    if (loop_file.is_open()) {
      const Eigen::Vector3d relative_ypr =
          Utils::R2ypr(winner->verification.relative_q.toRotationMatrix());
      loop_file << std::fixed << std::setprecision(9) << frame_index << '\t'
                << keyframe->time_stamp << '\t' << winner->keyframe_id << '\t'
                << winner->keyframe->time_stamp << '\t'
                << winner->verification.relative_t.x() << '\t'
                << winner->verification.relative_t.y() << '\t'
                << winner->verification.relative_t.z() << '\t' << relative_ypr.x() << '\t'
                << relative_ypr.y() << '\t' << relative_ypr.z() << "\tcam " << winner->current_camera
                << " -> cam " << winner->historical_camera << '\n';
    }
  }
  LOG(INFO) << "Accepted multicamera loop " << frame_index << " -> " << winner->keyframe_id
            << " using camera " << winner->current_camera << " -> " << winner->historical_camera
            << " with " << winner->verification.pnp_inliers << " PnP inliers";
  return winner->keyframe_id;
}

void PoseGraph::runMulticameraDiagnostics(Keyframe* keyframe,
                                          int frame_index,
                                          float score_threshold) {
  const size_t camera_count = keyframe->camera_brief_descriptors.size();
  if (camera_count < 2) return;

  if (multicamera_diagnostic_databases_.empty()) {
    multicamera_diagnostic_databases_.resize(camera_count);
    for (BriefDatabase& camera_database : multicamera_diagnostic_databases_) {
      camera_database.setVocabulary(*voc, false, 0);
    }
  }
  if (multicamera_diagnostic_databases_.size() != camera_count) {
    LOG(ERROR) << "Camera count changed inside multicamera loop-closure diagnostics.";
    return;
  }

  std::ofstream output(keyframe->params_.debug_output_path_ +
                           "/loop_closure_multicamera_shadow.csv",
                       std::ios::app);
  if (!output.is_open()) {
    LOG(ERROR) << "Could not append multicamera loop-closure shadow diagnostics.";
    return;
  }

  for (size_t current_camera = 0; current_camera < camera_count; ++current_camera) {
    const auto& current_descriptors = keyframe->camera_brief_descriptors[current_camera];
    if (current_descriptors.empty()) continue;
    DBoW2::BowVector current_bow = keyframe->camera_bow_vectors[current_camera];
    if (current_bow.empty()) voc->transform(current_descriptors, current_bow);

    for (size_t historical_camera = 0; historical_camera < camera_count; ++historical_camera) {
      DBoW2::QueryResults results;
      multicamera_diagnostic_databases_[historical_camera].query(
          current_bow, results, 4, frame_index - 50);
      for (size_t rank = 0; rank < results.size(); ++rank) {
        const bool score_passed = results[rank].Score > score_threshold && frame_index > 50;
        const bool verification_attempted = score_passed;
        Keyframe::CameraPairDiagnostic diagnostic;
        if (verification_attempted) {
          Keyframe* historical_keyframe = nullptr;
          {
            std::lock_guard<std::mutex> lock(kflistMutex_);
            historical_keyframe = getKFPtr(results[rank].Id);
          }
          if (historical_keyframe != nullptr) {
            diagnostic = keyframe->diagnoseCameraPair(
                historical_keyframe, current_camera, historical_camera);
          }
        }
        output << std::setprecision(17) << frame_index << ',' << keyframe->time_stamp << ','
               << current_camera << ',' << historical_camera << ',' << rank + 1 << ','
               << results[rank].Id << ',' << results[rank].Score << ',' << score_threshold << ','
               << score_passed << ',' << verification_attempted << ',' << diagnostic.tracked_points << ','
               << diagnostic.descriptor_matches << ',' << diagnostic.pnp_solver_succeeded << ','
               << diagnostic.pnp_exception << ',' << diagnostic.pnp_inliers << ','
               << diagnostic.relative_yaw_deg << ',' << diagnostic.relative_translation_m << ','
               << diagnostic.yaw_gate_passed << ',' << diagnostic.position_gate_passed << ','
               << diagnostic.accepted << ",0\n";
      }
    }
  }

  // One entry per rig keyframe is added to every per-camera database so that
  // DBoW entry IDs remain identical to pose-graph keyframe IDs.
  for (size_t camera = 0; camera < camera_count; ++camera) {
    const DBoW2::EntryId entry_id =
        multicamera_diagnostic_databases_[camera].add(keyframe->camera_brief_descriptors[camera]);
    CHECK_EQ(static_cast<int>(entry_id), frame_index)
        << "Per-camera diagnostic DBoW database lost rig-keyframe ID alignment for camera " << camera;
  }
}

void PoseGraph::optimize4DoFPoseGraph() {
  while (!shutdown_requested_) {
    int cur_index = -1;

    {
      std::lock_guard<std::mutex> l(optimizationMutex_);
      while (!optimizationBuffer_.empty()) {
        cur_index = optimizationBuffer_.front();
        optimizationBuffer_.pop();
      }
    }
    if (cur_index != -1) {
      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.max_num_iterations = 10;
      // options.trust_region_strategy_type = ceres::DOGLEG;
      options.logging_type = ceres::SILENT;
      options.minimizer_progress_to_stdout = false;

      ceres::Solver::Summary summary;
      ceres::LossFunction* loss_function;
      loss_function = new ceres::HuberLoss(0.1);

      kflistMutex_.lock();
      Keyframe* cur_kf = getKFPtr(cur_index);

      int max_length = cur_index + 1;

      double t_array[max_length][3];
      Eigen::Quaterniond q_array[max_length];  // NOLINT
      double euler_array[max_length][3];       // NOLINT
      double sequence_array[max_length];       // NOLINT

      ceres::Manifold* angle_manifold = new ceres::AutoDiffManifold<YawAngleFunctor, 1, 1>;

      std::list<Keyframe*>::iterator it;

      int i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < earliest_loop_index) continue;
        (*it)->local_index = i;
        Eigen::Quaterniond tmp_q;
        Eigen::Matrix3d tmp_r;
        Eigen::Vector3d tmp_t;
        (*it)->getSVInPose(tmp_t, tmp_r);
        tmp_q = tmp_r;
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i] = tmp_q;

        Eigen::Vector3d euler_angle = Utils::R2ypr(tmp_q.toRotationMatrix());
        euler_array[i][0] = euler_angle.x();
        euler_array[i][1] = euler_angle.y();
        euler_array[i][2] = euler_angle.z();

        sequence_array[i] = (*it)->sequence;

        problem.AddParameterBlock(euler_array[i], 1, angle_manifold);
        problem.AddParameterBlock(t_array[i], 3);

        if ((*it)->index <= earliest_loop_index) {
          problem.SetParameterBlockConstant(euler_array[i]);
          problem.SetParameterBlockConstant(t_array[i]);
        }

        // add edge
        // adding sequential egde. Fixed sized window of length 4 serves as covisibility
        for (int j = 1; j < 3; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            Eigen::Vector3d euler_conncected = Utils::R2ypr(q_array[i - j].toRotationMatrix());
            Eigen::Vector3d relative_t(t_array[i][0] - t_array[i - j][0],
                                       t_array[i][1] - t_array[i - j][1],
                                       t_array[i][2] - t_array[i - j][2]);
            relative_t = q_array[i - j].inverse() * relative_t;
            double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
            ceres::CostFunction* cost_function = FourDOFError::Create(relative_t.x(),
                                                                      relative_t.y(),
                                                                      relative_t.z(),
                                                                      relative_yaw,
                                                                      euler_conncected.y(),
                                                                      euler_conncected.z());
            problem.AddResidualBlock(
                cost_function, NULL, euler_array[i - j], t_array[i - j], euler_array[i], t_array[i]);
          }
        }

        // add loop edge

        if ((*it)->has_loop) {
          assert((*it)->loop_index >= earliest_loop_index);
          int connected_index = getKFPtr((*it)->loop_index)->local_index;
          Eigen::Vector3d euler_conncected = Utils::R2ypr(q_array[connected_index].toRotationMatrix());
          Eigen::Vector3d relative_t;
          relative_t = (*it)->getLoopRelativeT();
          double relative_yaw = (*it)->getLoopRelativeYaw();
          ceres::CostFunction* cost_function = FourDOFWeightError::Create(
              relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw, euler_conncected.y(), euler_conncected.z());
          problem.AddResidualBlock(cost_function,
                                   loss_function,
                                   euler_array[connected_index],
                                   t_array[connected_index],
                                   euler_array[i],
                                   t_array[i]);
        }

        if ((*it)->index == cur_index) break;
        i++;
      }
      kflistMutex_.unlock();

      ceres::Solve(options, &problem, &summary);

      {
        std::lock_guard<std::mutex> l(kflistMutex_);
        i = 0;
        for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
          if ((*it)->index < earliest_loop_index) continue;
          Eigen::Quaterniond tmp_q;
          tmp_q = Utils::ypr2R(Eigen::Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
          Eigen::Vector3d tmp_t = Eigen::Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
          Eigen::Matrix3d tmp_r = tmp_q.toRotationMatrix();
          (*it)->updatePose(tmp_t, tmp_r);

          if ((*it)->index == cur_index) break;
          i++;
        }

        Eigen::Vector3d cur_t, svin_t;
        Eigen::Matrix3d cur_r, svin_r;
        cur_kf->getPose(cur_t, cur_r);
        cur_kf->getSVInPose(svin_t, svin_r);
        {
          std::lock_guard<std::mutex> l(driftMutex_);
          yaw_drift = Utils::R2ypr(cur_r).x() - Utils::R2ypr(svin_r).x();
          r_drift = Utils::ypr2R(Eigen::Vector3d(yaw_drift, 0, 0));
          t_drift = cur_t - r_drift * svin_t;
        }

        it++;
        for (; it != keyframelist.end(); it++) {
          Eigen::Vector3d P;
          Eigen::Matrix3d R;
          (*it)->getSVInPose(P, R);
          P = r_drift * P + t_drift;
          R = r_drift * R;
          (*it)->updatePose(P, R);
        }
      }
      updatePath();
      if (loop_closure_optimization_callback_) {
        Keyframe* last_kf = keyframelist.back();
        loop_closure_optimization_callback_(last_kf->time_stamp);
      }
    }
    std::chrono::milliseconds duration(500);
    std::this_thread::sleep_for(duration);
  }
}

void PoseGraph::optimize6DoFPoseGraph() {
  while (!shutdown_requested_) {
    int cur_index = -1;
    int first_looped_index = -1;

    {
      std::lock_guard<std::mutex> l(optimizationMutex_);
      while (!optimizationBuffer_.empty()) {
        cur_index = optimizationBuffer_.front();
        first_looped_index = earliest_loop_index;
        optimizationBuffer_.pop();
      }
    }
    if (cur_index != -1) {
      // clang-format off
      Eigen::Matrix<double, 6, 6> relative_pose_sqrt_information, loop_closure_sqrt_information;
      relative_pose_sqrt_information << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 20.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 20.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 100.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 100.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 57.3;

      loop_closure_sqrt_information << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 20.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 20.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 100.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 100.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 100.0;
      // clang-format on

      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);

      kflistMutex_.lock();
      Keyframe* cur_kf = getKFPtr(cur_index);

      int kMaxLength = cur_index + 1;

      Eigen::Vector3d t_array[kMaxLength];
      Eigen::Quaterniond q_array[kMaxLength];  // NOLINT
      double sequence_array[kMaxLength];       // NOLINT

      ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

      std::list<Keyframe*>::iterator it;

      int i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index) continue;
        (*it)->local_index = i;
        Eigen::Quaterniond tmp_q;
        Eigen::Matrix3d tmp_r;
        Eigen::Vector3d tmp_t;
        (*it)->getSVInPose(tmp_t, tmp_r);
        tmp_q = tmp_r;
        t_array[i] = tmp_t;
        q_array[i] = tmp_q;
        sequence_array[i] = (*it)->sequence;

        problem.AddParameterBlock(q_array[i].coeffs().data(), 4, quaternion_manifold);
        problem.AddParameterBlock(t_array[i].data(), 3);

        if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
          problem.SetParameterBlockConstant(t_array[i].data());
          problem.SetParameterBlockConstant(q_array[i].coeffs().data());
        }

        // add edge
        // adding sequential egde. Fixed sized window of length 4 serves as covisibility
        for (int j = 1; j < 5; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            Eigen::Quaterniond relative_q = q_array[i - j].inverse() * q_array[i];
            Eigen::Vector3d relative_t = q_array[i - j].inverse() * (t_array[i] - t_array[i - j]);
            ceres::Pose3d relative_pose(relative_t, relative_q);
            ceres::CostFunction* cost_function =
                ceres::PoseGraph3dErrorTerm::Create(relative_pose, relative_pose_sqrt_information);

            problem.AddResidualBlock(cost_function,
                                     NULL,
                                     t_array[i - j].data(),
                                     q_array[i - j].coeffs().data(),
                                     t_array[i].data(),
                                     q_array[i].coeffs().data());
          }
        }

        // add loop edge

        if ((*it)->has_loop) {
          assert((*it)->loop_index >= first_looped_index);
          Eigen::Vector3d relative_t = (*it)->getLoopRelativeT();
          Eigen::Quaterniond relative_q = (*it)->getLoopRelativeQ();
          ceres::Pose3d relative_pose(relative_t, relative_q);
          ceres::CostFunction* cost_function =
              ceres::PoseGraph3dErrorTerm::Create(relative_pose, loop_closure_sqrt_information);

          int connected_index = getKFPtr((*it)->loop_index)->local_index;
          problem.AddResidualBlock(cost_function,
                                   loss_function,
                                   t_array[connected_index].data(),
                                   q_array[connected_index].coeffs().data(),
                                   t_array[i].data(),
                                   q_array[i].coeffs().data());
        }

        if ((*it)->index == cur_index) break;
        i++;
      }
      kflistMutex_.unlock();

      ceres::Solve(options, &problem, &summary);

      {
        std::lock_guard<std::mutex> l(kflistMutex_);
        i = 0;
        for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
          if ((*it)->index < first_looped_index) continue;
          (*it)->updatePose(t_array[i], q_array[i].toRotationMatrix());

          if ((*it)->index == cur_index) break;
          i++;
        }

        Eigen::Vector3d cur_t, svin_t;
        Eigen::Matrix3d cur_r, svin_r;
        cur_kf->getPose(cur_t, cur_r);
        cur_kf->getSVInPose(svin_t, svin_r);
        {
          std::lock_guard<std::mutex> l(driftMutex_);
          r_drift = cur_r.transpose() * svin_r;
          yaw_drift = Utils::R2ypr(r_drift).x();
          t_drift = cur_t - r_drift * svin_t;
        }

        it++;
        for (; it != keyframelist.end(); it++) {
          Eigen::Vector3d P;
          Eigen::Matrix3d R;
          (*it)->getSVInPose(P, R);
          P = r_drift * P + t_drift;
          R = r_drift * R;
          (*it)->updatePose(P, R);
        }
      }
      updatePath();
      if (loop_closure_optimization_callback_) {
        Keyframe* last_kf = keyframelist.back();
        loop_closure_optimization_callback_(last_kf->time_stamp);
      }
    }
  }
}

void PoseGraph::updatePath() {
  std::lock_guard<std::mutex> l(kflistMutex_);

  std::list<Keyframe*>::iterator it;

  std::vector<std::pair<Timestamp, Eigen::Matrix4d>> loop_closure_path;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> loop_closure_edges;

  for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
    Eigen::Vector3d P;
    Eigen::Matrix3d R;
    (*it)->getPose(P, R);
    Eigen::Quaterniond Q{R};

    std::pair<Timestamp, Eigen::Matrix4d> pose;
    pose.first = (*it)->time_stamp;
    pose.second.block<3, 3>(0, 0) = R;
    pose.second.block<3, 1>(0, 3) = P;
    loop_closure_path.push_back(pose);

    if ((*it)->has_loop) {
      Keyframe* connected_KF = getKFPtr((*it)->loop_index);
      Eigen::Vector3d connected_P;
      Eigen::Matrix3d connected_R;
      connected_KF->getPose(connected_P, connected_R);
      (*it)->getPose(P, R);
      loop_closure_edges.push_back({P, connected_P});
    }
  }

  CHECK(loop_closure_callback_);
  loop_closure_callback_(loop_closure_path, loop_closure_edges);
}

void PoseGraph::updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1>& _loop_info) {
  Keyframe* kf = getKFPtr(index);
  kf->updateLoop(_loop_info);
  if (abs(_loop_info(7)) < 30.0 && Eigen::Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    if (is_fast_localization_) {
      Keyframe* old_kf = getKFPtr(kf->loop_index);
      Eigen::Vector3d w_P_old, w_P_cur, svin_P_cur;
      Eigen::Matrix3d w_R_old, w_R_cur, svin_R_cur;
      old_kf->getPose(w_P_old, w_R_old);
      kf->getSVInPose(svin_P_cur, svin_R_cur);

      Eigen::Vector3d relative_t;
      Eigen::Quaterniond relative_q;
      relative_t = kf->getLoopRelativeT();
      relative_q = (kf->getLoopRelativeQ()).toRotationMatrix();
      w_P_cur = w_R_old * relative_t + w_P_old;
      w_R_cur = w_R_old * relative_q;
      double shift_yaw;
      Eigen::Matrix3d shift_r;
      Eigen::Vector3d shift_t;
      shift_yaw = Utils::R2ypr(w_R_cur).x() - Utils::R2ypr(svin_R_cur).x();
      shift_r = Utils::ypr2R(Eigen::Vector3d(shift_yaw, 0, 0));
      shift_t = w_P_cur - w_R_cur * svin_R_cur.transpose() * svin_P_cur;

      {
        std::lock_guard<std::mutex> l(driftMutex_);
        yaw_drift = shift_yaw;
        r_drift = shift_r;
        t_drift = shift_t;
      }
    }
  }
}

void PoseGraph::setLoopClosureOptimizationCallback(const EventCallback& optimization_finish_callback) {
  loop_closure_optimization_callback_ = optimization_finish_callback;
}

void PoseGraph::setKeyframePoseCallback(const KeframeWithLoopClosureCallback& keyframe_pose_callback) {
  keyframe_pose_callback_ = keyframe_pose_callback;
}

void PoseGraph::setLoopClosureCallback(const PathWithLoopClosureCallback& loop_closure_callback) {
  loop_closure_callback_ = loop_closure_callback;
}
