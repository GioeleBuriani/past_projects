/* The present script corresponds to the source file of the class used for
 the publishing and updating of information of the dynamically reconfigurable 
 node. 
Author:
  Ruben Martin Rodriguez, group 10*/

#include "dyn_reconf_mdp/dyn_pub.h"

DynPub::DynPub(){}

DynPub::~DynPub(){}

// Callback function of the dynamic reconfigure server, it will display an
// informative message about the changes in parameter values and update the internal
// variable
void DynPub::configCb(dyn_reconf_mdp::tiago_mdpConfig &config, uint32_t level){
	ROS_INFO("Updated parameters: order_req %s, resume %s, scan %s, charge %s, level %d, product %d, info_msg %s", 
		config.order_req?"True":"False",
		config.resume?"True":"False",
		config.scan?"True":"False",
		config.charge?"True":"False",
		config.level,
		config.product,
		config.info_msg.c_str());

	this->config_.order_req = config.order_req;
	this->config_.resume = config.resume;
	this->config_.scan = config.scan;
	this->config_.charge = config.charge;
	this->config_.level = config.level;
	this->config_.product = config.product;
	this->config_.info_msg = config.info_msg;
}

// Using a publisher created on the "parent" node, publish the updated values of the 
// parameters using a message type in accordance to the parameter list
void DynPub::publish(ros::Publisher *pub){
	custom_msgs::Param_list list;
	list.order_req = this->config_.order_req;
	list.resume = this->config_.resume;
	list.scan = this->config_.scan;
	list.charge = this->config_.charge;
	list.level = this->config_.level;
	list.product = this->config_.product;
	list.info_msg = this->config_.info_msg;
	pub->publish(list);
}

// Getter of the class' internal configuration attribute
dyn_reconf_mdp::tiago_mdpConfig DynPub::getData(){
	return this->config_;
}
