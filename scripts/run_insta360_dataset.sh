#!/usr/bin/env bash

# Run the Insta360 X5 fisheye dataset configurations based on
# /home/ws/run_insta360_dataset.md. By default all sequences are run in order.
# Pass sequence numbers (for example, "2 4") to run only a subset, or use
# --dry-run to perform preflight checks without starting ROS.

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
CLEANUP_INT_TIMEOUT=5
CLEANUP_TERM_TIMEOUT=3
CLEANUP_KILL_TIMEOUT=2
WATCHDOG_INT_TIMEOUT=3
WATCHDOG_TERM_TIMEOUT=2
BAG_STOP_TIMEOUT=3

# label|sequence directory|bag directory relative to sequence|launch file|start offset|config file|camera mode
DATASETS=(
  "stavronikita_shipwreck|barbados26_01/2026_01_27_stavronikita|bag/insta360/shipwreck|svin_insta360X5_rear.launch.py|0|config_insta360X5_rear_ds_air.yaml|rear"
  "kechries|greece26_06/2026_06_11_kechries|bag/insta360/LRV_20260611_155413|svin_insta360X5_front.launch.py|0|config_insta360X5_front_ds_air.yaml|front"
  "otochha|mexico26_02/2026_02_28_OtochHa|bag/insta360/LRV_20260228_114846_01|svin_insta360X5_front.launch.py|0|config_insta360X5_front_ds_air.yaml|front"
  "tajma_ha|mexico26_02/2026_02_22_Tajma_Ha|bag/insta360/LRV_20260222_103116_01|svin_insta360X5_front.launch.py|3|config_insta360X5_front_ds_air.yaml|front"
  "escondido|mexico26_02/2026_02_21_Escondido|bag/insta360/LRV_20260221_130838|svin_insta360X5_front.launch.py|0|config_insta360X5_front_ds_air.yaml|front"
  "otochha|mexico26_02/2026_02_28_OtochHa|bag/insta360/LRV_20260228_114846_01|svin_insta360X5_dual.launch.py|0|config_insta360X5_dual_ds_air.yaml|dual"
  "escondido|mexico26_02/2026_02_21_Escondido|bag/insta360/LRV_20260221_130838|svin_insta360X5_dual.launch.py|0|config_insta360X5_dual_ds_air.yaml|dual"
  "tajma_ha|mexico26_02/2026_02_22_Tajma_Ha|bag/insta360/LRV_20260222_103116_01|svin_insta360X5_dual.launch.py|3|config_insta360X5_dual_ds_air.yaml|dual"
)
DATASET_COUNT=${#DATASETS[@]}

DRY_RUN=0
LIST_ONLY=0
SELECTED=()
LAUNCH_PID=
LAUNCH_SESSION_ID=
LAUNCH_WATCHDOG_PID=
LAUNCH_STATUS=0
BAG_PID=
ORIGINAL_DEBUG_KIND=absent
ORIGINAL_DEBUG_TARGET=
DEBUG_STATE_REMEMBERED=0

usage() {
  cat <<'EOF'
Usage: run_insta360_dataset.sh [--dry-run] [--list] [SEQUENCE ...]

Run all eight sequences in the documented order, or only the numbered
sequences supplied on the command line.

  --dry-run  Validate inputs and print the commands without running ROS
  --list     List the configured sequences and exit
  -h, --help Show this help

Examples:
  run_insta360_dataset.sh --dry-run
  run_insta360_dataset.sh 1
  run_insta360_dataset.sh 2 4 5 8
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

list_datasets() {
  local index entry label sequence_rel bag_rel launch_file start_offset config_name camera_mode
  for index in "${!DATASETS[@]}"; do
    IFS='|' read -r label sequence_rel bag_rel launch_file start_offset config_name camera_mode <<< "${DATASETS[$index]}"
    printf '%d: %-24s %s/%s (%s, mode=%s, config=%s, offset=%ss)\n' \
      "$((index + 1))" "$label" "$DATA_ROOT/$sequence_rel" "$bag_rel" \
      "$launch_file" "$camera_mode" "$config_name" "$start_offset"
  done
}

for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    --list) LIST_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ "$argument" =~ ^[0-9]+$ ]] && (( argument >= 1 && argument <= DATASET_COUNT )); then
        SELECTED+=("$argument")
      else
        die "unknown argument or dataset number out of range 1-${DATASET_COUNT}: $argument"
      fi
      ;;
  esac
done

if (( LIST_ONLY )); then
  list_datasets
  exit 0
fi
if (( ${#SELECTED[@]} == 0 )); then
  for ((number = 1; number <= DATASET_COUNT; number++)); do
    SELECTED+=("$number")
  done
fi

preflight() {
  local number entry label sequence_rel bag_rel launch_file start_offset config_name camera_mode
  local bag_dir config_source

  [[ -d "$POSE_GRAPH_DIR" ]] || die "pose_graph directory not found: $POSE_GRAPH_DIR"
  for number in "${SELECTED[@]}"; do
    entry=${DATASETS[$((number - 1))]}
    IFS='|' read -r label sequence_rel bag_rel launch_file start_offset config_name camera_mode <<< "$entry"
    bag_dir="${DATA_ROOT}/${sequence_rel}/${bag_rel}"
    [[ -f "${bag_dir}/metadata.yaml" ]] || die "ROS bag metadata not found: ${bag_dir}/metadata.yaml"
    [[ -f "${WORKSPACE}/src/SVIn2/okvis_ros/launch/${launch_file}" ]] || \
      die "launch file not found: $launch_file"
    config_source="${CONFIG_SOURCE_DIR}/${config_name}"
    [[ -f "$config_source" ]] || die "runtime config file not found: $config_source"
    case "$camera_mode" in
      rear|front|dual) ;;
      *) die "unsupported camera mode for dataset $number: $camera_mode" ;;
    esac
  done

  if (( ! DRY_RUN )); then
    [[ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]] || \
      die "ROS setup not found: /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
    [[ -f "${WORKSPACE}/install/setup.bash" ]] || \
      die "workspace setup not found: ${WORKSPACE}/install/setup.bash"
    command -v setsid >/dev/null || die "setsid is required"
    command -v stdbuf >/dev/null || die "stdbuf is required"
    command -v pkill >/dev/null || die "pkill is required"
    command -v ps >/dev/null || die "ps is required"
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

session_alive() {
  local session_id=$1
  [[ -n "$session_id" ]] && ps -s "$session_id" -o stat= 2>/dev/null | \
    grep -qv '^[[:space:]]*Z'
}

wait_for_session() {
  local session_id=$1 timeout=$2 waited
  for ((waited = 0; waited < timeout; waited++)); do
    session_alive "$session_id" || return 0
    sleep 1
  done
  ! session_alive "$session_id"
}

launch_session_alive() {
  session_alive "${LAUNCH_SESSION_ID:-}"
}

wait_for_launch_session() {
  wait_for_session "${LAUNCH_SESSION_ID:-}" "$1"
}

runner_alive() {
  local runner_pid=$1
  ps -p "$runner_pid" -o stat= 2>/dev/null | grep -qv '^[[:space:]]*Z'
}

launch_watchdog() {
  local runner_pid=$1 session_id=$2

  # Ctrl-C targets the runner's terminal process group. The watchdog must
  # survive it so it can clean up if Bash exits before running its EXIT trap.
  trap '' HUP INT QUIT TERM
  while session_alive "$session_id"; do
    if ! runner_alive "$runner_pid"; then
      echo "Watchdog: runner exited; stopping ROS launch session $session_id" >&2
      pkill -INT -s "$session_id" 2>/dev/null || true
      if ! wait_for_session "$session_id" "$WATCHDOG_INT_TIMEOUT"; then
        pkill -TERM -s "$session_id" 2>/dev/null || true
        if ! wait_for_session "$session_id" "$WATCHDOG_TERM_TIMEOUT"; then
          pkill -KILL -s "$session_id" 2>/dev/null || true
        fi
      fi
      return
    fi
    sleep 1
  done
}

process_alive() {
  local pid=$1
  ps -p "$pid" -o stat= 2>/dev/null | grep -qv '^[[:space:]]*Z'
}

stop_bag() {
  local waited
  [[ -n "${BAG_PID:-}" ]] || return 0

  if process_alive "$BAG_PID"; then
    echo "Stopping rosbag playback process $BAG_PID"
    kill -INT "$BAG_PID" 2>/dev/null || true
    for ((waited = 0; waited < BAG_STOP_TIMEOUT; waited++)); do
      process_alive "$BAG_PID" || break
      sleep 1
    done
    if process_alive "$BAG_PID"; then
      kill -TERM "$BAG_PID" 2>/dev/null || true
      sleep 1
    fi
    if process_alive "$BAG_PID"; then
      kill -KILL "$BAG_PID" 2>/dev/null || true
    fi
  fi
  wait "$BAG_PID" 2>/dev/null || true
  BAG_PID=
}

stop_launch() {
  local int_timeout=${1:-$CLEANUP_INT_TIMEOUT}

  [[ -n "${LAUNCH_PID:-}" ]] || return 0
  echo "Stopping ROS launch session ${LAUNCH_SESSION_ID:-unknown}"

  if launch_session_alive; then
    pkill -INT -s "$LAUNCH_SESSION_ID" 2>/dev/null || true
    if ! wait_for_launch_session "$int_timeout"; then
      echo "WARNING: ROS processes ignored SIGINT; sending SIGTERM"
      pkill -TERM -s "$LAUNCH_SESSION_ID" 2>/dev/null || true
      if ! wait_for_launch_session "$CLEANUP_TERM_TIMEOUT"; then
        echo "WARNING: ROS processes ignored SIGTERM; sending SIGKILL"
        pkill -KILL -s "$LAUNCH_SESSION_ID" 2>/dev/null || true
        if ! wait_for_launch_session "$CLEANUP_KILL_TIMEOUT"; then
          echo "WARNING: launch session $LAUNCH_SESSION_ID still appears alive after SIGKILL" >&2
        fi
      fi
    fi
  fi

  wait "$LAUNCH_PID" 2>/dev/null
  LAUNCH_STATUS=$?
  if [[ -n "${LAUNCH_WATCHDOG_PID:-}" ]]; then
    wait "$LAUNCH_WATCHDOG_PID" 2>/dev/null || true
  fi
  LAUNCH_PID=
  LAUNCH_SESSION_ID=
  LAUNCH_WATCHDOG_PID=
  return 0
}

cleanup() {
  local exit_status=$?
  trap '' HUP INT TERM
  stop_bag
  stop_launch
  if (( ! DRY_RUN && DEBUG_STATE_REMEMBERED )); then
    restore_debug_link
  fi
  return "$exit_status"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

latest_new_file() {
  local directory=$1 pattern=$2 marker=$3
  find "$directory" -maxdepth 1 -type f -name "$pattern" -newer "$marker" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

run_dataset() {
  local number=$1 entry label sequence_rel bag_rel launch_file start_offset config_name camera_mode
  local sequence_dir bag_dir timestamp result_dir run_tag marker
  local okvis_log bag_log experiment_log service_log trajectory_service_log
  local debug_target config_source config_snapshot bag_status service_status launch_status service_ready
  local latest_trajectory_txt latest_trajectory_ply latest_pointcloud
  local -a bag_command

  entry=${DATASETS[$((number - 1))]}
  IFS='|' read -r label sequence_rel bag_rel launch_file start_offset config_name camera_mode <<< "$entry"
  sequence_dir="${DATA_ROOT}/${sequence_rel}"
  bag_dir="${sequence_dir}/${bag_rel}"
  config_source="${CONFIG_SOURCE_DIR}/${config_name}"
  timestamp=$(date +%Y%m%d_%H%M%S)
  result_dir="${sequence_dir}/fisheyeSVIn2_results/${timestamp}_${camera_mode}"
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
    echo "[$number/${DATASET_COUNT}] $label (${camera_mode})"
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

  echo "===== DATASET $number/${DATASET_COUNT}: $label (${camera_mode}) ====="
  echo "RUN_TAG=$run_tag"
  echo "RESULT_DIR=$result_dir"
  echo "BAG_DIR=$bag_dir"
  echo "PLAYBACK_RATE=$PLAYBACK_RATE"
  echo "START_OFFSET=$start_offset"
  echo "LAUNCH_FILE=$launch_file"
  echo "CAMERA_MODE=$camera_mode"
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
  LAUNCH_SESSION_ID=$LAUNCH_PID
  launch_watchdog "$BASHPID" "$LAUNCH_SESSION_ID" &
  LAUNCH_WATCHDOG_PID=$!
  echo "LAUNCH_PID=$LAUNCH_PID SESSION_ID=$LAUNCH_SESSION_ID WATCHDOG_PID=$LAUNCH_WATCHDOG_PID"

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
  stdbuf -oL -eL "${bag_command[@]}" > >(tee -a "$bag_log") 2>&1 &
  BAG_PID=$!
  wait "$BAG_PID"
  bag_status=$?
  BAG_PID=
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

  stop_launch "$SHUTDOWN_TIMEOUT"
  launch_status=$LAUNCH_STATUS
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
