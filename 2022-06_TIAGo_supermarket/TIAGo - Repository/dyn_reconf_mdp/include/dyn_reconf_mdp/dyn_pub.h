/* The present script corresponds to the header file of the class used for
 the publishing and updating of information of the dynamically reconfigurable 
 node. The class will take charge of updating the parameter values upon request
 from the client(s) through a callback, update the internal variable, and publish
 the updated parameters through a topic to notify all listening nodes. 
Author:
  Ruben Martin Rodriguez, group 10*/

#ifndef DYN_PUB_H
#define DYN_PUB_H

#include "ros/ros.h"
#include "dynamic_reconfigure/server.h"
#include "dyn_reconf_mdp/tiago_mdpConfig.h"
#include "custom_msgs/Param_list.h"
#include "stdlib.h"

class DynPub{
public:
	// Default constructor and destructor
	DynPub();
	~DynPub();
	
	// Callback function used for handling a change in the parameters, will
	// take as parameters the updated configuration and the level of the parameter
	// states (see documentation on dynamic reconfigure for its exact meaning)
	void configCb(dyn_reconf_mdp::tiago_mdpConfig &config, uint32_t level);

	// Function used for publishing the updated values of the parameters
	// Takes as argument the publisher created in the node that contains
	// the instance of the class
	void publish(ros::Publisher *pub);
	
	// Getter of the internal configuration
	dyn_reconf_mdp::tiago_mdpConfig getData();
private:
	// Configuration attribute to store the values of the parameters
	dyn_reconf_mdp::tiago_mdpConfig config_;
};

#endif