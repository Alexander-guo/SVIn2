#!/usr/bin/env bash

# Run the five Insta360 X5 fisheye datasets described in
# /home/ws/run_insta360_dataset.md.  By default all sequences are run in the
# documented order.  Pass sequence numbers (for example, "2 4") to run only a
# subset, or use --dry-run to perform preflight checks without starting ROS.

set -uo pipefail

WORKSPACE=/home/ws
DATA_ROOT=/home/data/Insta360/X5/test/pro_case_uw
POSE_GRAPH_DIR="${WORKSPACE}/src/SVIn2/pose_graph"
TRAJECTORY_SOURCE_DIR="${POSE_GRAPH_DIR}/svin_results"
POINTCLOUD_SOURCE_DIR="${POSE_GRAPH_DIR}/reconstruction_results"
DEBUG_LINK="${POSE_GRAPH_DIR}/debug_output"
CONFIG_SOURCE_DIR="${WORKSPACE}/install/okvis_ros/share/okvis_ros/config"
PLAYBACK_RATE=0.5
FINAL_CALLBACK_WAIT=15
SERVICE_WAIT_TIMEOUT=90
SHUTDOWN_TIMEOUT=180

# label|sequence directory|bag directory relative to sequence|launch file|start offset|config file
DATASETS=(
  "stavronikita_shipwreck|barbados26_01/2026_01_27_stavronikita|bag/insta360/shipwreck|svin_insta360X5_rear.launch.py|0|config_insta360X5_rear_ds_air.yaml"
  "kechries|greece26_06/2026_06_11_kechries|bag/insta360/LRV_20260611_155413|svin_insta360X5_front.launch.py|0|config_insta360X5_front_ds_air.yaml"
  "otochha|mexico26_02/2026_02_28_OtochHa|bag/insta360/LRV_20260228_114846_01|svin_insta360X5_front.launch.py|0|config_insta360X5_front_ds_air.yaml"
  "tajma_ha|mexico26_02/2026_02_22_Tajma_Ha|bag/insta360/LRV_20260222_103116_01|svin_insta360X5_front.launch.py|3|config_insta360X5_front_ds_air.yaml"
  "escondido|mexico26_02/2026_02_21_Escondido|bag/insta360/LRV_20260221_130838|svin_insta360X5_front.launch.py|0|config_insta360X5_front_ds_air.yaml"
)

DRY_RUN=0
LIST_ONLY=0
SELECTED=()
LAUNCH_PID=
ORIGINAL_DEBUG_KIND=absent
ORIGINAL_DEBUG_TARGET=
DEBUG_STATE_REMEMBERED=0

usage() {
  cat <<'EOF'
Usage: run_insta360_dataset.sh [--dry-run] [--list] [SEQUENCE ...]

Run all five sequences in the documented order, or only the numbered
sequences supplied on the command line.

  --dry-run  Validate inputs and print the commands without running ROS
  --list     List the configured sequences and exit
  -h, --help Show this help

Examples:
  run_insta360_dataset.sh --dry-run
  run_insta360_dataset.sh 1
  run_insta360_dataset.sh 2 4 5
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

list_datasets() {
  local index entry label sequence_rel bag_rel launch_file start_offset config_name
  for index in "${!DATASETS[@]}"; do
    IFS='|' read -r label sequence_rel bag_rel launch_file start_offset config_name <<< "${DATASETS[$index]}"
    printf '%d: %-24s %s/%s (%s, config=%s, offset=%ss)\n' \
      "$((index + 1))" "$label" "$DATA_ROOT/$sequence_rel" "$bag_rel" \
      "$launch_file" "$config_name" "$start_offset"
  done
}

for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    --list) LIST_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    1|2|3|4|5) SELECTED+=("$argument") ;;
    *) die "unknown argument: $argument" ;;
  esac
done

if (( LIST_ONLY )); then
  list_datasets
  exit 0
fi
if (( ${#SELECTED[@]} == 0 )); then
  SELECTED=(1 2 3 4 5)
fi

preflight() {
  local number entry label sequence_rel bag_rel launch_file start_offset config_name bag_dir config_source

  [[ -d "$POSE_GRAPH_DIR" ]] || die "pose_graph directory not found: $POSE_GRAPH_DIR"
  for number in "${SELECTED[@]}"; do
    entry=${DATASETS[$((number - 1))]}
    IFS='|' read -r label sequence_rel bag_rel launch_file start_offset config_name <<< "$entry"
    bag_dir="${DATA_ROOT}/${sequence_rel}/${bag_rel}"
    [[ -f "${bag_dir}/metadata.yaml" ]] || die "ROS bag metadata not found: ${bag_dir}/metadata.yaml"
    [[ -f "${WORKSPACE}/src/SVIn2/okvis_ros/launch/${launch_file}" ]] || \
      die "launch file not found: $launch_file"
    config_source="${CONFIG_SOURCE_DIR}/${config_name}"
    [[ -f "$config_source" ]] || die "runtime config file not found: $config_source"
  done

  if (( ! DRY_RUN )); then
    [[ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]] || \
      die "ROS setup not found: /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
    [[ -f "${WORKSPACE}/install/setup.bash" ]] || \
      die "workspace setup not found: ${WORKSPACE}/install/setup.bash"
    command -v setsid >/dev/null || die "setsid is required"
    command -v stdbuf >/dev/null || die "stdbuf is required"
  fi
}

remember_debug_link() {
  if [[ -L "$DEBUG_LINK" ]]; then
    ORIGINAL_DEBUG_KIND=symlink
    ORIGINAL_DEBUG_TARGET=$(readlink "$DEBUG_LINK")
  elif [[ -e "$DEBUG_LINK" ]]; then
    die "$DEBUG_LINK already exists and is not a symlink; refusing to replace it"
  fi
  DEBUG_STATE_REMEMBERED=1
}

restore_debug_link() {
  if [[ -L "$DEBUG_LINK" ]]; then
    unlink "$DEBUG_LINK"
  elif [[ -e "$DEBUG_LINK" ]]; then
    echo "WARNING: $DEBUG_LINK changed into a non-symlink; leaving it untouched" >&2
    return
  fi
  if [[ "$ORIGINAL_DEBUG_KIND" == symlink ]]; then
    ln -s "$ORIGINAL_DEBUG_TARGET" "$DEBUG_LINK"
  fi
}

stop_launch() {
  if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "Cleanup: stopping ROS launch process group $LAUNCH_PID"
    kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
    sleep 3
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
  fi
  LAUNCH_PID=
}

cleanup() {
  stop_launch
  if (( ! DRY_RUN && DEBUG_STATE_REMEMBERED )); then
    restore_debug_link
  fi
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

latest_new_file() {
  local directory=$1 pattern=$2 marker=$3
  find "$directory" -maxdepth 1 -type f -name "$pattern" -newer "$marker" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

run_dataset() {
  local number=$1 entry label sequence_rel bag_rel launch_file start_offset config_name
  local sequence_dir bag_dir timestamp result_dir run_tag marker
  local okvis_log bag_log experiment_log service_log trajectory_service_log
  local debug_target config_source config_snapshot bag_status service_status launch_status service_ready
  local latest_trajectory_txt latest_trajectory_ply latest_pointcloud
  local -a bag_command

  entry=${DATASETS[$((number - 1))]}
  IFS='|' read -r label sequence_rel bag_rel launch_file start_offset config_name <<< "$entry"
  sequence_dir="${DATA_ROOT}/${sequence_rel}"
  bag_dir="${sequence_dir}/${bag_rel}"
  config_source="${CONFIG_SOURCE_DIR}/${config_name}"
  timestamp=$(date +%Y%m%d_%H%M%S)
  result_dir="${sequence_dir}/fisheyeSVIn2_results/${timestamp}"
  run_tag="${label}_insta360X5_${timestamp}"
  marker="${result_dir}/.run_started"
  okvis_log="${result_dir}/${run_tag}_okvis_ros.log"
  bag_log="${result_dir}/${run_tag}_rosbag_play.log"
  experiment_log="${result_dir}/${run_tag}_experiment.log"
  service_log="${result_dir}/${run_tag}_save_pointcloud_service.log"
  trajectory_service_log="${result_dir}/${run_tag}_save_trajectory_service.log"
  debug_target="${result_dir}/debug_output"
  config_snapshot="${result_dir}/${config_name}"
  bag_command=(ros2 bag play "$bag_dir" --rate "$PLAYBACK_RATE")
  if [[ "$start_offset" != 0 ]]; then
    bag_command+=(--start-offset "$start_offset")
  fi

  if (( DRY_RUN )); then
    echo "[$number/5] $label"
    echo "  result: $result_dir"
    echo "  debug:  $debug_target"
    echo "  config: $config_source -> $config_snapshot"
    printf '  launch: ros2 launch okvis_ros %q okvis_config:=%q\n' \
      "$launch_file" "$config_snapshot"
    printf '  bag:   '; printf ' %q' "${bag_command[@]}"; printf '\n'
    return 0
  fi

  mkdir -p "${sequence_dir}/fisheyeSVIn2_results" || \
    die "cannot create result root: ${sequence_dir}/fisheyeSVIn2_results"
  mkdir "$result_dir" || die "result directory already exists or cannot be created: $result_dir"
  mkdir "$debug_target" || die "cannot create debug output directory: $debug_target"
  touch "$marker" || die "cannot create run marker: $marker"
  exec 3>&1 4>&2
  exec > >(tee -a "$experiment_log") 2>&1

  # Snapshot the dereferenced package-share config before launching, then make
  # ROS use that snapshot so the archived YAML is exactly what the run reads.
  cp -L "$config_source" "$config_snapshot" || \
    die "failed to snapshot runtime config: $config_source"

  echo "===== DATASET $number/5: $label ====="
  echo "RUN_TAG=$run_tag"
  echo "RESULT_DIR=$result_dir"
  echo "BAG_DIR=$bag_dir"
  echo "PLAYBACK_RATE=$PLAYBACK_RATE"
  echo "START_OFFSET=$start_offset"
  echo "LAUNCH_FILE=$launch_file"
  echo "CONFIG_FILE=$config_source"
  echo "CONFIG_SNAPSHOT=$config_snapshot"
  echo "STARTED_AT=$(date --iso-8601=seconds)"

  if [[ -L "$DEBUG_LINK" ]]; then
    unlink "$DEBUG_LINK"
  elif [[ -e "$DEBUG_LINK" ]]; then
    die "$DEBUG_LINK changed into a non-symlink; refusing to replace it"
  fi
  ln -s "$debug_target" "$DEBUG_LINK"
  echo "DEBUG_OUTPUT=$debug_target"

  setsid stdbuf -oL -eL ros2 launch okvis_ros "$launch_file" \
    "okvis_config:=$config_snapshot" \
    > >(tee -a "$okvis_log") 2>&1 &
  LAUNCH_PID=$!
  echo "LAUNCH_PID=$LAUNCH_PID PGID=$LAUNCH_PID"

  service_ready=0
  for ((waited = 0; waited < SERVICE_WAIT_TIMEOUT; waited++)); do
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      wait "$LAUNCH_PID" || true
      die "launch exited before /save_pointcloud became available"
    fi
    if ros2 service list 2>/dev/null | grep -qx /save_pointcloud; then
      service_ready=1
      break
    fi
    sleep 1
  done
  (( service_ready == 1 )) || die "/save_pointcloud unavailable after ${SERVICE_WAIT_TIMEOUT}s"

  echo "Starting bag playback at ${PLAYBACK_RATE}x speed (offset ${start_offset}s)"
  stdbuf -oL -eL "${bag_command[@]}" 2>&1 | tee -a "$bag_log"
  bag_status=${PIPESTATUS[0]}
  echo "Bag playback exited with status $bag_status at $(date --iso-8601=seconds)"
  (( bag_status == 0 )) || return "$bag_status"

  echo "Waiting ${FINAL_CALLBACK_WAIT}s for final callbacks"
  sleep "$FINAL_CALLBACK_WAIT"

  ros2 service call /save_pointcloud std_srvs/srv/Trigger 2>&1 | tee "$service_log"
  service_status=${PIPESTATUS[0]}
  (( service_status == 0 )) || die "/save_pointcloud failed with status $service_status"

  # The node also saves on shutdown; this explicit call is an extra safeguard.
  ros2 service call /save_trajectory std_srvs/srv/Trigger 2>&1 | tee "$trajectory_service_log"
  service_status=${PIPESTATUS[0]}
  (( service_status == 0 )) || die "/save_trajectory failed with status $service_status"

  echo "Stopping launch process group gracefully"
  kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
  for ((waited = 0; waited < SHUTDOWN_TIMEOUT; waited++)); do
    kill -0 "$LAUNCH_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "WARNING: launch ignored SIGINT for ${SHUTDOWN_TIMEOUT}s; sending SIGTERM"
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
  fi
  wait "$LAUNCH_PID"
  launch_status=$?
  LAUNCH_PID=
  echo "Launch exited with status $launch_status"

  latest_trajectory_txt=$(latest_new_file "$TRAJECTORY_SOURCE_DIR" 'svin_*.txt' "$marker")
  latest_pointcloud=$(latest_new_file "$POINTCLOUD_SOURCE_DIR" 'pointcloud_*.ply' "$marker")
  [[ -n "$latest_trajectory_txt" ]] || die "no new trajectory TXT was produced"
  latest_trajectory_ply=${latest_trajectory_txt%.txt}.ply
  [[ -s "$latest_trajectory_txt" ]] || die "trajectory TXT is empty: $latest_trajectory_txt"
  [[ -s "$latest_trajectory_ply" ]] || die "matching trajectory PLY is missing: $latest_trajectory_ply"
  [[ -n "$latest_pointcloud" && -s "$latest_pointcloud" ]] || die "no new point cloud was produced"

  cp -p "$latest_trajectory_txt" "${result_dir}/${run_tag}_trajectory.txt" || \
    die "failed to copy trajectory TXT into the result directory"
  cp -p "$latest_trajectory_ply" "${result_dir}/${run_tag}_trajectory.ply" || \
    die "failed to copy trajectory PLY into the result directory"
  cp -p "$latest_pointcloud" "${result_dir}/${run_tag}_pointcloud.ply" || \
    die "failed to copy point cloud into the result directory"
  sha256sum "${result_dir}/${run_tag}_trajectory.txt" \
    "${result_dir}/${run_tag}_trajectory.ply" \
    "${result_dir}/${run_tag}_pointcloud.ply"
  echo "FINISHED_AT=$(date --iso-8601=seconds)"
  echo "EXPERIMENT_SUCCESS=$label"

  exec 1>&3 2>&4
  exec 3>&- 4>&-
}

preflight

if (( DRY_RUN )); then
  echo "Dry run only; ROS will not be launched and no result directories will be created."
else
  # ROS/colcon setup files reference optional variables that may be unset.
  set +u
  # shellcheck disable=SC1091
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  ros_setup_status=$?
  # shellcheck disable=SC1091
  source "${WORKSPACE}/install/setup.bash"
  workspace_setup_status=$?
  set -u
  (( ros_setup_status == 0 )) || die "failed to source ROS setup"
  (( workspace_setup_status == 0 )) || die "failed to source workspace setup"
  remember_debug_link
  if ros2 service list 2>/dev/null | grep -qx /save_pointcloud; then
    die "/save_pointcloud already exists; stop the active SVIn2 run before starting this batch"
  fi
fi

for sequence_number in "${SELECTED[@]}"; do
  run_dataset "$sequence_number"
  run_status=$?
  (( run_status == 0 )) || die "dataset $sequence_number failed with status $run_status"
done

echo "Completed ${#SELECTED[@]} selected dataset(s)."
