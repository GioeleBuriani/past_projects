/* The present script corresponds to the node definition of a sample client
of the dynamically reconfigurable node. It will change the value of one of 
the parameters every 4 seconds.
Author:
  Ruben Martin Rodriguez, group 10*/

#include <ros/ros.h>
#include "dyn_reconf_mdp/param_client.h"


int main(int argc, char **argv)
{
	ros::init(argc, argv, "client_test");
   	ros::NodeHandle n;

   	ParamClient client(&n);

	ros::Rate r(0.25);

	dyn_reconf_mdp::tiago_mdpConfig config;

	while (n.ok()){
      config = client.getData();
      config.order_req = !config.order_req;
      if(config.level == 1){
      	config.level = 0;
      }else{
      	config.level = 1;
      }

      client.setConfig(config);
      client.sendConfig();
      ros::spinOnce();
      r.sleep();
	}
	return 0;
}