/*Example node of a standard publisher for testing the correct functioning of the
GUI's textbox.
Author:
  Ruben Martin Rodriguez, group 10*/

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>

int main(int argc, char **argv)
{
  // Initiate the ros node
  ros::init(argc, argv, "talker");
  ros::NodeHandle n("~");

  // Create the publisher object by advertising to the corresponding topic
  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("/error_topic", 1000);

  // Setup the loop rate for publishing messages
  ros::Rate loop_rate(10);

  int count = 0;
  while (ros::ok())
  {
    // Create the message object to be sent 
    std_msgs::String msg;

    // Define the contents of the message
    std::stringstream ss;
    ss << "hello world " << count;
    msg.data = ss.str();

    // Show the contents on screen for verifying the correct functioning
    ROS_INFO("%s", msg.data.c_str());

    // Publish the message
    chatter_pub.publish(msg);

    // Spin and handle the publishing rate
    ros::spinOnce();
    loop_rate.sleep();
    ++count;
  }
  return 0;
}
