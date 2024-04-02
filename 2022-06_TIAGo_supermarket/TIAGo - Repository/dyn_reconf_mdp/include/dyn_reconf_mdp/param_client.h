/* The present script corresponds to the header file of the class used as
 client for the dynamically reconfigurable node. It povides with a templated 
 way for communicating with the server in this same package, facilitating 
 its integration in other packages. 
Author:
  Ruben Martin Rodriguez, group 10*/

#ifndef PARAM_CLIENT_H
#define PARAM_CLIENT_H

#include "ros/ros.h"
#include "dynamic_reconfigure/client.h"
#include "dyn_reconf_mdp/tiago_mdpConfig.h"

class ParamClient{
public:
	// Parametrized constructor for normal usage
	ParamClient(ros::NodeHandle *nh);

	// Default constructor and initialization function for integration with the GUI node
	ParamClient();
	void initialize(ros::NodeHandlePtr nh);
	
	// Destructor
	~ParamClient();
	
	// Member function used for setting a specific configuration within the class which
	// will be later on sent to the parameter node/server
	void setConfig(dyn_reconf_mdp::tiago_mdpConfig config);
	
	// Member function used for sending the currently stored configuration to the 
	// parameter node
	void sendConfig();
	
	// Getter used for retrieving the current configuration stored within the class instance
	dyn_reconf_mdp::tiago_mdpConfig getData();
private:
	// Variable containing a configuration of the parameters
	dyn_reconf_mdp::tiago_mdpConfig config_;
	
	// Dynamic reconfigure client object required for communication
	dynamic_reconfigure::Client<dyn_reconf_mdp::tiago_mdpConfig> client_;
};

#endif
