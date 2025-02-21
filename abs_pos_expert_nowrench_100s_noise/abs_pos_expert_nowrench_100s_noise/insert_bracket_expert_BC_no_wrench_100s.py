import numpy as np
import pickle
import rospy
import tf2_ros
import time
from cv_bridge import CvBridge
from datetime import datetime
from controller_manager_msgs.srv import (
    ListControllers,
    ListControllersRequest,
    SwitchController,
    SwitchControllerRequest,
)
from enum import Enum
from metalman_core.common.util import (
    matrix_to_pose_msg,
    matrix_to_pose_stamped,
    matrix_to_transform_stamped,
    transform_to_matrix,
    transform_to_arrays,
)
from geometry_msgs.msg import PoseStamped, TransformStamped, WrenchStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger, TriggerRequest
from tqdm import tqdm, trange
import os
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import torch
from metalman_core.insertion.RL.abs_pos_expert_nowrench_100s.Model import *
from metalman_core.insertion.expert_ML.training_dataset.read_n_vis_pickle_motion_wrench import PickleDataReader
from gymnasium.spaces import Box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# SETUP
CONTACT_THRESHOLD = 10


class DataCollector:
    def __init__(
        self,
        service_n_topic_prefix: str,
        zero_ftsensor_service: str,
        filtered_wrench_topic: str,
        target_frame_topic: str,
        controller_to_stop: str,
        controller_to_start: str,
        list_controller_service: str,
        switch_controller_service: str,
        base_link_frame: str,
        end_effector_frame: str,
        start_pose_translation_noise_range,
        start_pose_translation_orientation_range,
        frequency: int,
        timeout: float,
        filename: str,
    ):
        # Argument parameters
        self.service_n_topic_prefix = service_n_topic_prefix
        self.list_controller_service = list_controller_service
        self.switch_controller_service = switch_controller_service
        self.controller_to_start = controller_to_start
        self.controller_to_stop = controller_to_stop

        # ft_sensor
        self.zero_ftsensor_service = zero_ftsensor_service

        # Wrench
        self.filtered_wrench = np.zeros(6)
        self.filtered_wrench_sub = rospy.Subscriber(
            filtered_wrench_topic, WrenchStamped, self.filtered_wrench_callback
        )

        # JOINT
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback
        )

        # Frame
        self.target_frame_topic = service_n_topic_prefix + target_frame_topic
        self.target_frame_pub = rospy.Publisher(self.target_frame_topic, PoseStamped, queue_size=1)


        self.base_link = base_link_frame
        self.end_effector_frame = end_effector_frame

        self.start_pose_translation_noise_range = start_pose_translation_noise_range
        self.start_pose_translation_orientation_range = (
            start_pose_translation_orientation_range
        )

        self.frequency = frequency
        self.timeout = timeout
        self.filename = filename

        # Transforms
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.goal_pose = self.get_goal_pose(5)
        self.observation_space = Box(low = -np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype= np.float64)
        self.reader = PickleDataReader()

        self.model_path = "/home/davintj/ws_raise/src/metalman/metalman_core/src/metalman_core/insertion/RL/abs_pos_expert_nowrench_100s/runs_FFBC_model_epoch_30.pth"
        self.model = FFBC()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")

    def call_FT_zero_calib(self, service):
        print("Waiting for zero FT sensor Service")
        rospy.wait_for_service(self.service_n_topic_prefix + self.zero_ftsensor_service)

        s = rospy.ServiceProxy(service, Trigger)
        resp = s.call(TriggerRequest())

        if resp.success == True:
            print("FT sensor Calibrated!")
        else:
            print("FT sensor calibration failed!")
    def joint_state_callback(self, msg):
        self.joint_position = list(msg.position)
        self.joint_velocity = list(msg.velocity)
        self.joint_effort = list(msg.effort)
    def filtered_wrench_callback(self, msg):
        self.filtered_wrench[0] = msg.wrench.force.x
        self.filtered_wrench[1] = msg.wrench.force.y
        self.filtered_wrench[2] = msg.wrench.force.z
        self.filtered_wrench[3] = msg.wrench.torque.x
        self.filtered_wrench[4] = msg.wrench.torque.y
        self.filtered_wrench[5] = msg.wrench.torque.z
    def get_goal_pose(self, wait_for_transform=0.05):
        return transform_to_matrix(
            self.tf_buffer.lookup_transform(
                self.base_link,
                "gt",
                rospy.Time(),
                rospy.Duration(wait_for_transform),
            ).transform
        )
    def get_current_end_effector_pose(self, wait_for_transform=0.05):
        return transform_to_matrix(
            self.tf_buffer.lookup_transform(
                self.base_link,
                self.end_effector_frame,
                rospy.Time(),
                rospy.Duration(wait_for_transform),
            ).transform
        )

    def call_list_controllers(self, service):
        print("Waiting for the List Controller service")
        rospy.wait_for_service(service)
        print("List Controller service available")

        s = rospy.ServiceProxy(service, ListControllers)
        resp = s.call(ListControllersRequest())
        return resp

    def matrix_to_pose(self, matrix):
        translation = matrix[:3, 3]
        rotation = Rotation.from_matrix(matrix[:3, :3])
        quaternion = rotation.as_quat()
        return np.hstack((translation, quaternion))

    def pose_to_matrix(self, pose):
        translation = pose[:3]
        quaternion = pose[3:]

        # Create the rotation matrix from the quaternion
        rotation = Rotation.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()

        # Create the 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation

        return transformation_matrix

    def call_switch_controllers(
            self,
            service,
            list,
            start=[],
            stop=[],
            strict=SwitchControllerRequest.STRICT,
            num_tries=3,
            sleep=0.2,
    ):
        rospy.wait_for_service(service)
        if any(c.name == start[0] and c.state == "running" for c in list.controller):
            print(f"{start[0]} is already running!")
            return True
        else:
            for i in range(num_tries):
                s = rospy.ServiceProxy(service, SwitchController)
                response = s.call(SwitchControllerRequest([], stop, strict, False, 0.0))
                if response.ok == 1:
                    print(f"{stop[0]} stopped!")
                else:
                    print("Error while switching the controller!")
                    rospy.sleep(sleep)
                    continue

                response = s.call(
                    SwitchControllerRequest(start, [], strict, False, 0.0)
                )
                if response.ok == 1:
                    print(f"{start[0]} started!")
                    return True
                else:
                    print("Error while switching the controller!")
                    rospy.sleep(sleep)
                    continue
            return False

    def predict(self, input_data, model):
        model.eval()
        inputs = torch.tensor(input_data).float()
        with torch.no_grad():
            outputs = model(inputs)
        return outputs.numpy()

    def expert_insertion(self):
        rospy.sleep(1)
        self.call_FT_zero_calib(self.zero_ftsensor_service)
        self.call_FT_zero_calib(self.zero_ftsensor_service)
        rospy.sleep(1)
        initial_pose_offset = np.eye(4)
        initial_pose_offset[:3, 3] = np.random.uniform(
            self.start_pose_translation_noise_range[0],
            self.start_pose_translation_noise_range[1],
        )
        initial_pose_offset[:3, :3] = Rotation.from_euler(
            "xyz",
            np.random.uniform(
                self.start_pose_translation_orientation_range[0],
                self.start_pose_translation_orientation_range[1],
            ),
        ).as_matrix()
        initial_pose = np.copy(self.goal_pose) @ (initial_pose_offset)

        initial_pose_goal_frame = np.linalg.inv(self.goal_pose).dot(initial_pose)

        current_pose = self.get_current_end_effector_pose()
        rel_curr_to_goal = np.linalg.inv(self.goal_pose) @ current_pose  # 4x4 matrix
        rel_curr_to_goal_quat = self.matrix_to_pose(rel_curr_to_goal)
        self.model_input = np.tile(rel_curr_to_goal_quat, (100, 1)).flatten()

        """ Put moveaway pose which is just offset in one axis """
        move_away_offset = np.eye(4)
        move_away_offset[:3, 3] = [0, 0, -0.03]
        move_away_pose = np.copy(self.goal_pose) @ move_away_offset

        class State(Enum):
            SWITCH_CONTROLLER = 0
            BC = 1
            SUCCESS = 2

        self.state = State.SWITCH_CONTROLLER

        def loop(timer_event):
            current_pose = self.get_current_end_effector_pose()
            target_frame = current_pose
            if self.state == State.SWITCH_CONTROLLER:

                list = self.call_list_controllers(self.list_controller_service)
                response = self.call_switch_controllers(
                    service=self.switch_controller_service,
                    list=list,
                    start=[self.controller_to_start],
                    stop=[self.controller_to_stop],
                )

                if response == True:
                    print(f"{self.state} completed!")
                    self.call_FT_zero_calib(self.zero_ftsensor_service)
                    self.call_FT_zero_calib(self.zero_ftsensor_service)
                    rospy.sleep(4)
                    self.state = self.state.BC


                    self.state = self.state.BC
                    print(f"{self.state} is next!")

            elif self.state == State.BC:
                current_pose = self.get_current_end_effector_pose()
                # target_frame = current_pose
                rel_curr_to_goal = np.linalg.inv(self.goal_pose) @ current_pose # 4x4 matrix
                rel_curr_to_goal_quat = self.matrix_to_pose(rel_curr_to_goal)
                self.model_input = np.roll(self.model_input, shift=-7)  # Shift left by 7 elements # This might not be the correct way to do this
                self.model_input[-7:] = rel_curr_to_goal_quat
                pred_act= self.predict(self.model_input, self.model)
                # Here, pred act is also 700, we want to extract pred_act[:7]
                pred_pose = pred_act[:7]
                predicted_curr_to_goal_matrix = self.pose_to_matrix(pred_pose)
                pseudo_target = self.goal_pose @ predicted_curr_to_goal_matrix
                target_frame= pseudo_target
                # Now model input has to shift, delete the first 7 element out of model input and then append pred_pose to the last 7

                # Change it to target_frame:
                # print("input", current_pose)
                # print("Pred", pseudo_target)
                print("Model Input", self.model_input)




            self.target_frame_pub.publish(
                matrix_to_pose_stamped(target_frame, self.base_link)
            )
            # print(matrix_to_pose_stamped(target_frame, self.base_link))
            # print(self.get_current_end_effector_pose())
            # print(initial_pose)
            # print(self.goal_pose)
            print(self.state)
            self.tf_broadcaster.sendTransform(
                matrix_to_transform_stamped(target_frame, self.base_link, "expert_target")
            )

        timer = rospy.Timer(rospy.Duration(1.0 / self.frequency), loop)
        start_time = time.time()
        while self.state != State.SUCCESS:
            rospy.sleep(1.0 / self.frequency)
            if time.time() - start_time > self.timeout:
                timer.shutdown()
                return [], [], [], [], [], [], [], []

        timer.shutdown()


if __name__ == "__main__":
    rospy.init_node("collect_stereo_data")

    data = DataCollector(
        rospy.get_param("~service_n_topic_prefix"),
        rospy.get_param("~zero_ftsensor_service"),
        rospy.get_param("~filtered_wrench_topic"),
        rospy.get_param("~target_frame_topic"),
        rospy.get_param("~controller_to_stop"),
        rospy.get_param("~controller_to_start"),
        rospy.get_param("~list_controller_service"),
        rospy.get_param("~switch_controller_service"),
        rospy.get_param("~base_link_frame"),
        rospy.get_param("~end_effector_frame"),
        rospy.get_param("~start_pose_translation_noise_range"),
        rospy.get_param("~start_pose_translation_orientation_range"),
        rospy.get_param("~frequency"),
        rospy.get_param("~timeout"),
        rospy.get_param("~filename"),
    )
    data.expert_insertion()
    rospy.spin()
