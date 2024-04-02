#include<ros/ros.h> 
#include "std_msgs/UInt32MultiArray.h"
#include "custom_msgs/Param_list.h"
#include <stdlib.h>
#include "dyn_reconf_mdp/param_client.h"
#include "custom_msgs/is_human.h"


class MySub{
private:
    ParamClient client_;
    dyn_reconf_mdp::tiago_mdpConfig config_;
    ros::Subscriber subscriber_object;
    ros::Subscriber sub;
public:
    MySub(ros::NodeHandle *n):client_(n){
        sub = n->subscribe("/params", 1, &MySub::callback_server, this);
        subscriber_object = n->subscribe("/aruco_marker_publisher/markers_list", 1, &MySub::Callback, this); 
    }

    void Callback(const std_msgs::UInt32MultiArray& message_holder) 
    { 
        std::vector<unsigned int, std::allocator<unsigned int>> detection_array = message_holder.data;

        ROS_INFO_STREAM("The number of detected items is " << detection_array.size());

        for (const auto& item_id: detection_array){
            ROS_INFO_STREAM("Detected item number is " << item_id);
        }

        if(detection_array.size() > 0){
            if(config_.product != detection_array[0]){
                config_.product = detection_array[0];
                client_.setConfig(config_);
                client_.sendConfig();
                ROS_INFO("Product is updated!");
            }
        }

        if(detection_array.size() == 0){
            if(config_.product != -1){
                config_.product = -1;
                client_.setConfig(config_);
                client_.sendConfig();
                ROS_INFO("Product is updated!");
            }
        }
    } 

    void callback_server(custom_msgs::Param_list msg) {
          
        // Update the fields of the internal parameter value attribute
        config_.order_req = msg.order_req;
        config_.resume = msg.resume;
        config_.scan = msg.scan;
        config_.charge = msg.charge;
        config_.level = msg.level;
        config_.product = msg.product;
        config_.info_msg = msg.info_msg;

        
    }
};

bool detect(custom_msgs::is_human::Request &req, custom_msgs::is_human::Response &res) {
    res.is_human = true;
}

int main(int argc, char **argv) 
{ 
  ros::init(argc,argv,"detection"); //name this node 
  
  ros::NodeHandle n; // need this to establish communications with our new node 

  ros::ServiceServer human_detector = n.advertiseService("detect_human", detect);

  MySub listener(&n); 

  ros::spin();  

  return 0; 
} 