/* The present script corresponds to the source file of the class used as
 client for the dynamically reconfigurable node.
Author:
  Ruben Martin Rodriguez, group 10*/

#include "dyn_reconf_mdp/param_client.h"

// Parametrized constructor, intializes the client and gets the
// initial values of the parameters
ParamClient::ParamClient(ros::NodeHandle *nh) : client_("server"){
	nh->getParam("server/order_req", this->config_.order_req);
	nh->getParam("server/level", this->config_.level);
	nh->getParam("server/info_msg", this->config_.info_msg);
	nh->getParam("server/scan", this->config_.scan);
	nh->getParam("server/charge", this->config_.charge);
	nh->getParam("server/level", this->config_.level);
	nh->getParam("server/product", this->config_.product);
}

// Explicit default constructor, initializes the client object
ParamClient::ParamClient() : client_("server"){}

// Initialization function for getting the parameters, required for
// integration with the GUI node
void ParamClient::initialize(ros::NodeHandlePtr nh){
	nh->getParam("server/order_req", this->config_.order_req);
	nh->getParam("server/level", this->config_.level);
	nh->getParam("server/info_msg", this->config_.info_msg);
	nh->getParam("server/scan", this->config_.scan);
	nh->getParam("server/charge", this->config_.charge);
	nh->getParam("server/level", this->config_.level);
	nh->getParam("server/product", this->config_.product);
}

// Destructor
ParamClient::~ParamClient(){}

// Store the passed configuration in the internal variable
void ParamClient::setConfig(dyn_reconf_mdp::tiago_mdpConfig config){
	this->config_ = config;
}

// Send the configuration of the parameters to the dynamically 
// reconfigurable node
void ParamClient::sendConfig(){
	this->client_.setConfiguration(this->config_);
}

// Getter of the class' internal variable
dyn_reconf_mdp::tiago_mdpConfig ParamClient::getData(){
	return this->config_;
}
