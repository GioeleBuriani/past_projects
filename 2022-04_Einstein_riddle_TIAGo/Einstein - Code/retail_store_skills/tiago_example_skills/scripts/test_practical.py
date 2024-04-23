#!/usr/bin/env python

import rospy
import time
import geometry_msgs.msg
from move_base import MoveBase

from look_to_point import LookToPoint

from plan_arm_ik import PlanArmIk
from visualization_msgs.msg import Marker as VisMarker

from gripper_control import GripperControl

rospy.init_node("example_task")

move = False
look = False
arm = True
gripper = False

if move:
    move_base = MoveBase()
    pose_goal = geometry_msgs.msg.Pose()

    pose_goal.position.x = 1.0
    pose_goal.position.y = 0.0
    pose_goal.position.z = 0.0


    pose_goal.orientation.x = 0.0
    pose_goal.orientation.y = 0.0
    pose_goal.orientation.z = 0.0
    pose_goal.orientation.w = 1.0

    move_base.run(pose_goal)

if look:
    head_control = LookToPoint()
    point = geometry_msgs.msg.Point()
    point.x = 1.0
    point.y = 0.5
    point.z = 0.5
    head_control.run(point)

if arm:
    armgoal_marker_pub = rospy.Publisher("armgoal_marker", VisMarker, queue_size=0)
    arm_right = PlanArmIk('right', armgoal_marker_pub)
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x = 0.707
    pose_goal.orientation.y = 0.0
    pose_goal.orientation.z = 0.0
    pose_goal.orientation.w = 0.707
    pose_goal.position.x = 0.3
    pose_goal.position.y = -0.4
    pose_goal.position.z = 0.90

    arm_right.run(pose_goal)

if gripper:
    gripper_right = GripperControl('right')
    gripper_right.run('close')
    rospy.sleep(2)
    gripper_right.run('open')