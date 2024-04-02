/* The present script corresponds to the definition of the dynamically
reconfigurable node. By using a dyn_pub object, the node will update 
the values of the dynamically reconfigurable parameters upon request and
will publish the updated values. 
Author:
  Ruben Martin Rodriguez, group 10*/

#include <ros/ros.h>
#include "dyn_reconf_mdp/dyn_pub.h"
#include "custom_msgs/Param_list.h"


int main(int argc, char **argv) {

   ros::init(argc, argv, "server");
   ros::NodeHandle n;

   //Initialize the class for handling the server and publishing
   DynPub *dyn_class = new DynPub();

   // Initialize the dynamic reconfigure server
   dynamic_reconfigure::Server<dyn_reconf_mdp::tiago_mdpConfig> server;

   // Initialize the dynamic reconfigure callback representation function
   dynamic_reconfigure::Server<dyn_reconf_mdp::tiago_mdpConfig>::CallbackType f;
   
   //Bind the callback function to the callback of the dynamic reconfigure object
   f = boost::bind(&DynPub::configCb, dyn_class, _1, _2);
   
   //Bind the callback object to the server object
   server.setCallback(f);

   // Create a publisher to advertise the updated parameter values
   ros::Publisher pub = n.advertise<custom_msgs::Param_list>("/params", 10);

   ros::Rate r(1000);

   // Let the node spin and publish the updated values
   ROS_INFO("Spinning node");
   while (n.ok()){
      dyn_class->publish(&pub);

      ros::spinOnce();
      r.sleep();
   }
   delete dyn_class;
   return 0;
 }
