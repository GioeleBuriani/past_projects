#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include "custom_msgs/Param_list.h"
#include "custom_msgs/is_human.h"
#include <stdlib.h>
#include <sstream>
#include "dyn_reconf_mdp/param_client.h"
#include "std_msgs/String.h"

class TiagoGoAssistant {
private:

  int current_state;
  bool tried_interaction;
  ParamClient client_;
  dyn_reconf_mdp::tiago_mdpConfig config_;
  ros::Subscriber sub; // ROS Subscriber 
  ros::Publisher pub;
  ros::ServiceClient service_client; 
  actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac; // ROS ActionClient

public:
  TiagoGoAssistant(ros::NodeHandle *nh) : ac("move_base", true), client_(nh) {
      
    // Wait for the action server to come up so that we can begin processing goals.
    ROS_INFO("Waiting for the move_base action server to come up.");
    ac.waitForServer();
    ROS_INFO("Action server started.");
    
    sub = nh->subscribe("/params", 1, &TiagoGoAssistant::callback, this);
    ROS_INFO("Subscriber started.");
    pub = nh->advertise<std_msgs::String>("/text_to_say", 1);
    service_client = nh->serviceClient<custom_msgs::is_human>("detect_human");
    current_state = 0;

    tried_interaction = false;
  }
   
  // Function for handling the base movement of the robot to a given location 
  bool move_to_goal(double xPos, double yPos, double zOr, double wOr) {

    // Create a new goal to send to move_base
    move_base_msgs::MoveBaseGoal goal;
    
    //set up the frame parameters
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();

    // Send a goal location to the robot
    goal.target_pose.pose.position.x = xPos;
    goal.target_pose.pose.position.y = yPos;
    goal.target_pose.pose.position.z =  0.0;
    goal.target_pose.pose.orientation.x = 0.0;
    goal.target_pose.pose.orientation.y = 0.0;
    goal.target_pose.pose.orientation.z = zOr;
    goal.target_pose.pose.orientation.w = wOr;

    ac.sendGoal(goal);
    ac.waitForResult();

    // Get the final state of the action
    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
      ROS_INFO("Moving COMPLETED!");
      return true;
    }
    else {
      // Handle the movement failure
      ROS_INFO("Moving FAILED!");
      return false;
    }
  }

  // Function for determining the next action of the system
  bool direct(){
    // Only resume action when operator said so
    if (config_.order_req && config_.resume) {

      ROS_INFO("Sending robot for order-request");
      
      // (1) Move robot to checkout location
      if(current_state == 0) {
        bool success = move_to_goal(0.88, 3.4, 0.707, 0.707);
        // If the action succeeded, then return true and change internal variable for performing next action
        if (success) {
          ROS_INFO("Robot is at checkout.");
          current_state = 1;
          return true;
        }else {
          current_state = 0;
          if(!tried_interaction){
            tried_interaction = true;
            interact(0);
          }else{
            tried_interaction = false;
            config_.resume = false;
            config_.level = 2;
            config_.info_msg = "An obstacle is impeding the product picking!";
            client_.setConfig(config_);
            client_.sendConfig();
          ros::Duration(0.15).sleep();
          }
          
          return false;
        }
      }
      // If the robot is at the checkout location, start pick and place
      else if (current_state == 1) {
        // (2) Do PDDL things upon detected product
        if (config_.product != -1){

            std::ostringstream out;  
            out << "rosparam set place_product" << config_.product;
            system(out.str().c_str());
            system("roslaunch planning_control update_goals.launch");

            ROS_INFO("Robot has COMPLETED the order-request!");

            config_.charge = true;
            config_.order_req = false;
            client_.setConfig(config_);
            client_.sendConfig();
            current_state = 0;
            return true;
        }else{
            config_.resume = false;
            config_.level = 1;
            config_.info_msg = "No products to pick! Place correctly the product or send robot back to charge.";
            client_.setConfig(config_);
            client_.sendConfig();
            return false;
        }
      }
    } 
    else if(config_.scan && config_.resume) {
      
      // Paramters for scanning shelves
      float x = -2.5;
      float y = -1.5;
      float z_orientation = 1.0;

      // Perform scaning shelves 
      ROS_INFO("Sending robot to scan shelves ...");

      move_to_goal(x, y, z_orientation, 0); // shelf 1
      y += 1.0;
      move_to_goal(x, y, z_orientation, 0); // shelf 2
      y += 1.0;
      move_to_goal(x, y, z_orientation, 0); // shelf 3
      y += 1.0;
      move_to_goal(x, y, z_orientation, 0); // shelf 4

      // Completed
      ROS_INFO("Robot has COMPLETED to scan the shelves!");
      config_.scan = false;
      config_.info_msg = "TIAGo free!";
      client_.setConfig(config_);
      client_.sendConfig();
      ros::Duration(0.15).sleep();
    } 
    else if (config_.charge && config_.resume) {
      
      // Move robot to charging station location 
      move_to_goal(0, 0, 0, 1); 
      
      // Completed
      ROS_INFO("Robot is back at charging station!");
      config_.charge = false;
      config_.info_msg = "TIAGo free!";
      client_.setConfig(config_);
      client_.sendConfig();
      current_state = 0;
    }
  }

  void interact(int sentence){
    custom_msgs::is_human srv;
    srv.request.req = true;

    // Communicate with the service for determining the presence of a human
    
    if(service_client.call(srv)){
      if(srv.response.is_human){
          
          ROS_INFO("Human detected!");

          // If a human is detected, interact with it
          std_msgs::String text;
          
          if (sentence == 0) { 
            text.data = "Hello, would you be so kind to step aside so that I can pick this product?";
          }
          
          pub.publish(text);
          ros::Duration(15).sleep();
      }
    }else{
      ROS_INFO("Did not get response from service.");
    }
  }
  
  // Callback function for handling the current information of the reconfigurable server
  void callback(custom_msgs::Param_list msg) {
    // Update the fields of the internal parameter value attribute
    config_.order_req = msg.order_req;
    config_.resume = msg.resume;
    config_.scan = msg.scan;
    config_.charge = msg.charge;
    config_.level = msg.level;
    config_.product = msg.product;
    config_.info_msg = msg.info_msg;
    
    // Send instructions to the robot according to the received information
    direct();
  }
};

 
int main (int argc, char **argv) {
  // Connect to ROS
  ros::init(argc, argv, "tiago_director"); 
  ros::NodeHandle nh;

  TiagoGoAssistant tiago(&nh);
  
  ros::spin();
}
