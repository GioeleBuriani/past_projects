#!/usr/bin/env python

# The present script creates a ros node which receives a sentence
# from a given topic and connects with the action server tts_to_soundplay
# which will take charge of reproducing the sound of the sentence received.
# As described in the tts_to_soundplay.py file, the server makes use of the 
# soundplay library to achieve this.
#
# Author:
#   Ruben Martin Rodriguez, group 10

import rospy
import actionlib

#import PAL Robotics custom headers for connecting to th
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from std_msgs.msg import String

# Subscriber and client class definition
class MySubscriber:
	#Constructor
    def __init__(self):
    	# Create the subscriber and client objects
        self.subscription = rospy.Subscriber('/text_to_say', String, self.listener_callback)
        self.client = actionlib.SimpleActionClient('tts_to_soundplay', TtsAction)
        # Wait for the server to run
        rospy.loginfo("Waiting for Server")
        self.client.wait_for_server()
        rospy.loginfo("Reached Server")
        # Create the goal object for communication
        self.goal = TtsGoal()

    # Callback function of the subscriber
    def listener_callback(self, msg):
    	# Collect the data and fill in the goal object data
    	text = msg.data
    	print("---")
    	self.goal.rawtext.text = text
    	self.goal.rawtext.lang_id = "EN"
    	# Send the goal object to the speaking action client and get 
    	# the feedback using the function defined below
    	self.client.send_goal(self.goal, feedback_cb=feedbackCb)
    	# Get the result of the action and display an informative message
    	self.client.wait_for_result()
    	res = self.client.get_result()
    	print("text: " + res.text)
    	print("warning/error msgs: " + res.msg)
    	print("---")


# Small function for returning an informative message about the events taking place
# at the action server.
def event(event):
	return {
	1 : 'TTS_EVENT_INITIALIZATION',
	2 : 'TTS_EVENT_SHUTDOWN',
	4 : 'TTS_EVENT_SYNCHRONIZATION',
	8 : 'TTS_EVENT_FINISHED_PLAYING_UTTERANCE',
	16 : 'TTS_EVENT_MARK',
	32 : 'TTS_EVENT_STARTED_PLAYING_WORD',
	64 : 'TTS_EVENT_FINISHED_PLAYING_PHRASE',
	128 : 'TTS_EVENT_FINISHED_PLAYING_SENTENCE'
	}[event]

# Callback function for the feedback of the action server 
def feedbackCb(feedback):
	print("event type: " + event(feedback.event_type))
	print("timestamp: " + str(feedback.timestamp))
	print("current word: " + feedback.text_said)
	print("next word: " + feedback.next_word)
	print("-")


if __name__ == '__main__':
	# Initialize the node and subscriber and let ros spin
	rospy.init_node('tts_client')
	my_sub = MySubscriber()
	rospy.spin()
		
