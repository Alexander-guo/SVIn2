from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.substitutions import ThisLaunchFileDir
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
  # Resolve full path to config file
  okvis_config_rel = LaunchConfiguration('okvis_config').perform(context)
  launch_file_dir = ThisLaunchFileDir().perform(context)
  abs_okvis_config_path = os.path.abspath(os.path.join(launch_file_dir, okvis_config_rel))

  print(f"\n[INFO] Full path to okvis_config: {abs_okvis_config_path}\n")

  # OKVIS node
  okvis_node = Node(
    package='okvis_ros',
    executable='okvis_node',
    name='okvis_node',
    parameters=[{
      'config_filename': abs_okvis_config_path,
      'mesh_file': 'firefly.dae',
      'paired_compressed_images': True,
      'paired_compressed_camera0_topic': '/insta360/front/image_raw/compressed',
      'paired_compressed_camera1_topic': '/insta360/rear/image_raw/compressed',
      'paired_compressed_queue_size': 100,
      'paired_compressed_log_counters': LaunchConfiguration('paired_compressed_log_counters')
    }],
    remappings=[
      ('/camera0', '/cam0/image_raw'),
      ('/camera1', '/cam1/image_raw'),
      ('/imu', '/insta360/imu')
    ]
  )

  # Pose Graph node
  pose_graph_node = Node(
    package='pose_graph',
    executable='pose_graph_node',
    name='pose_graph_node',
    condition=IfCondition(LaunchConfiguration('use_pose_graph')),
    parameters=[{
      'config_file': abs_okvis_config_path,
      'output_directory': LaunchConfiguration('pose_graph_output_directory'),
      'multicamera_loop_closure_diagnostics': LaunchConfiguration(
        'multicamera_loop_closure_diagnostics')
    }]
  )

  # RViz node
  rviz_node = Node(
    package='rviz2',
    executable='rviz2',
    name='rviz',
    condition=IfCondition(LaunchConfiguration('use_rviz')),
    arguments=['-d', os.path.join(
      FindPackageShare('okvis_ros').perform(context),
      'rviz_config/svin_multicam.rviz')],
    output='screen'
  )

  return [okvis_node, pose_graph_node, rviz_node]


def generate_launch_description():
  # Declare launch argument for config path
  config_arg = DeclareLaunchArgument(
    'okvis_config',
    default_value=PathJoinSubstitution([
      FindPackageShare('okvis_ros'),
      'config',
      'config_insta360X5_dual_ds_air.yaml'
    ])
  )

  use_pose_graph_arg = DeclareLaunchArgument(
    'use_pose_graph',
    default_value='true'
  )

  use_rviz_arg = DeclareLaunchArgument(
    'use_rviz',
    default_value='true'
  )

  paired_compressed_log_counters_arg = DeclareLaunchArgument(
    'paired_compressed_log_counters',
    default_value='false'
  )

  multicamera_loop_closure_diagnostics_arg = DeclareLaunchArgument(
    'multicamera_loop_closure_diagnostics',
    default_value='false'
  )

  pose_graph_output_directory_arg = DeclareLaunchArgument(
    'pose_graph_output_directory',
    default_value=''
  )

  # Global mapping uses the original images to colorize each camera's
  # landmarks. Dual-camera OKVIS decodes these streams internally, so expose
  # both on the raw topics expected by the pose graph as well.
  front_uncompressor_node = Node(
    package='okvis_ros',
    executable='uncompress_image',
    name='front_uncompressor',
    condition=IfCondition(LaunchConfiguration('use_pose_graph')),
    output='screen',
    parameters=[{
      'compressed_img_topic': '/insta360/front/image_raw/compressed',
      'ouput_img_topic': '/cam0/image_raw'
    }]
  )

  rear_uncompressor_node = Node(
    package='okvis_ros',
    executable='uncompress_image',
    name='rear_uncompressor',
    condition=IfCondition(LaunchConfiguration('use_pose_graph')),
    output='screen',
    parameters=[{
      'compressed_img_topic': '/insta360/rear/image_raw/compressed',
      'ouput_img_topic': '/cam1/image_raw'
    }]
  )

  return LaunchDescription([
    config_arg,
    use_pose_graph_arg,
    use_rviz_arg,
    paired_compressed_log_counters_arg,
    multicamera_loop_closure_diagnostics_arg,
    pose_graph_output_directory_arg,
    front_uncompressor_node,
    rear_uncompressor_node,
    OpaqueFunction(function=launch_setup)
  ])
