/* The present script corresponds to the header file of the GUI's
main window. It contains the decaration of the class with its attributes
and member function, as well as the slots for communicating between
GUI widgets.
Author:
  Ruben Martin Rodriguez, group 10*/

#ifndef MY_TERMINAL_H
#define MY_TERMINAL_H

#include <QWidget>
#include <ros/ros.h>
#include <qtimer.h>
#include <std_msgs/String.h>
#include "custom_msgs/Param_list.h"
#include "dyn_reconf_mdp/param_client.h"
#include <stdlib.h>

//Required to specify that the class will fall under the Ui namespace
namespace Ui {
class MyTerminal;
}

//Make the terminal inherit from the QWidget class
class MyTerminal : public QWidget
{
  //Required in all Qt class declarations
  Q_OBJECT

public:
  // Constructor and desctructor declaration
  explicit MyTerminal(QWidget *parent = nullptr);
  ~MyTerminal();
  //Create a callback for the subscriber
  void errorCb(const std_msgs::String::ConstPtr& msg);
  //Create a callback for the parameter subscriber topic to update 
  // the parameter values and change the UI appearance
  void paramCb(const custom_msgs::Param_list config);

public slots:
  //Function for handling the spinning of the subscriber on Qt's demand
  void spinOnce();

private slots:
  // Slots for handling user interaction with widgets (button)
  // The former is triggered upon start button click and the latter
  // upon resume operation button click
  void on_start_clicked();
  void on_resumeButton_clicked();
  void on_chargeButton_clicked();
  void on_emergencyButton_clicked();
  void on_scanButton_clicked();

private:
  //Define the pointer to the UI object employed by Qt to access the objects defined in
  //my_terminal.ui
  Ui::MyTerminal *ui;

  //Define the ros_timer object that will handle the spin of the subscriber
  QTimer *ros_timer;

  //Create the pointer to the node handle of the node and the subscriber and publisher objects
  ros::NodeHandlePtr nh_;
  ros::Subscriber my_sub_;
  ros::Subscriber param_sub_;
  ros::Publisher my_pub_;
  ParamClient client_;
  dyn_reconf_mdp::tiago_mdpConfig config_;
};

#endif // MY_TERMINAL_H
