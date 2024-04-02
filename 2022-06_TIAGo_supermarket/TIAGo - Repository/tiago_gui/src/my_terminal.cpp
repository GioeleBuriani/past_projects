/* The present script corresponds to the source file of the GUI's
main window. It contains the definition of the member functions.
Author:
  Ruben Martin Rodriguez, group 10*/

#include "my_terminal.h"
#include "ui_my_terminal.h"

//Constructor class with standard argument for ui initialization, that is, call
// to the parent class constructor QWidget and ui
MyTerminal::MyTerminal(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::MyTerminal)
{
  //Required by Qt to initialize any GUI
  ui->setupUi(this);

  //Initialization of the ROS pointer
  nh_.reset(new ros::NodeHandle("~"));
  client_.initialize(nh_);

  // setup the timer that will signal ros commands when to happen, this is required
  // to prevent the window from freezing upon ros::spin. It creates a QTimer obect that 
  // will be connected to the spinOnce function defined below every 100ms
  ros_timer = new QTimer(this);
  connect(ros_timer, SIGNAL(timeout()), this, SLOT(spinOnce()));
  ros_timer->start(100);  // set the rate to 100ms

  // setup subscriber to the error topic
  my_sub_ = nh_->subscribe<std_msgs::String>("/error_topic", 1, &MyTerminal::errorCb, this);
  param_sub_ = nh_->subscribe<custom_msgs::Param_list>("/params", 1, &MyTerminal::paramCb, this);

  // publish a message on the indications channel
  my_pub_ = nh_->advertise<std_msgs::String>("indications",1);

  // Change font size of the text
  ui->error_panel->setStyleSheet("font: 16pt;");
}

// Class destructor
MyTerminal::~MyTerminal()
{
  //Delete the pointers
  delete ui;
  delete ros_timer;
}

// Callback-like function used by the timer to spin ros once on every call and 
// prevent the window from freezing upon ros::spin usage
void MyTerminal::spinOnce(){
  //Verify whether the node is still active and spin or close the application otherwise
  if(ros::ok()){
    ros::spinOnce();
  }
  else
      QApplication::quit();
}

// Callback function for the subscriber, takes in an input string which will determine
// the message to be output to the screen and other GUI actions
void MyTerminal::errorCb(const std_msgs::String::ConstPtr& msg){
  // set the received message in the terminal
  ui->error_panel->setText(QString::fromStdString(msg->data.c_str()));
}

// Callback of the parameter subscriber topic to update and handle the parameter information
void MyTerminal::paramCb(const custom_msgs::Param_list config){
  // Update the fields of the internal parameter value attribute
  config_.order_req = config.order_req;
  config_.resume = config.resume;
  config_.scan = config.scan;
  config_.charge = config.charge;
  config_.level = config.level;
  config_.product = config.product;
  config_.info_msg = config.info_msg;

  // Depending on the status of the order request and level parameters, change the led color
  if(config_.level == 2){
    ui->led->setState(ui->led->StateError);
  }else if(config_.level == 1){
    ui->led->setState(ui->led->StateWarning);
  }else if(config.order_req || config.scan || config.charge){
    ui->led->setState(ui->led->StateOkBlue);
  }else{
    ui->led->setState(ui->led->StateOk);
  }

  // Display the informative message on screen
  ui->error_panel->setText(QString::fromStdString(config_.info_msg.c_str()));
}


// Function used for handling the clicking of the start button, for now, it only toggles
// the output message and the colour of the led
void MyTerminal::on_start_clicked()
{
  if(!config_.order_req && config_.level < 2){
    dyn_reconf_mdp::tiago_mdpConfig temp_config = config_;
    temp_config.order_req = true;
    temp_config.info_msg = "TIAGo working!";
    client_.setConfig(temp_config);
    client_.sendConfig();
  }
}


// Function used for handling the clicking of the Resume operation button,
// for now it only changes the parameter value and the displayed message
void MyTerminal::on_resumeButton_clicked()
{
  if(!config_.resume && config_.level < 2){
    dyn_reconf_mdp::tiago_mdpConfig temp_config = config_;
    temp_config.resume = true;
    temp_config.level = 0;
    temp_config.info_msg = "TIAGo working!";
    client_.setConfig(temp_config);
    client_.sendConfig();
  }
}

// Function used for handling the clicking of the Resume operation button,
// for now it only changes the corresponding parameter and prints an informative message
void MyTerminal::on_chargeButton_clicked()
{
  if(!config_.charge && config_.level < 2){
    dyn_reconf_mdp::tiago_mdpConfig temp_config = config_;
    temp_config.charge = true;
    temp_config.order_req = false;
    temp_config.scan = false;
    temp_config.resume = true;
    temp_config.product = -1;
    temp_config.level = 0;
    temp_config.info_msg = "Going to charging station!";
    client_.setConfig(temp_config);
    client_.sendConfig();
  }
}

// Function for handling emergency button click
void MyTerminal::on_emergencyButton_clicked()
{
  dyn_reconf_mdp::tiago_mdpConfig temp_config = config_;
  temp_config.scan = false;
  temp_config.order_req = false;
  temp_config.resume = false;
  temp_config.charge = false;
  temp_config.level = 2;
  temp_config.info_msg = "Emergency STOP! Assistance is required! The state of the robot is unknown!";
  client_.setConfig(temp_config);
  client_.sendConfig();
  system("rosnode list | grep 'tiago_director' | xargs -r rosnode kill");
  system("rosnode list | grep 'rosplan' | xargs -r rosnode kill");
}

// Function for handling scan button click
void MyTerminal::on_scanButton_clicked()
{
    if(!config_.scan && config_.level < 2){
    dyn_reconf_mdp::tiago_mdpConfig temp_config = config_;
    temp_config.scan = true;
    temp_config.info_msg = "Scanning!";
    client_.setConfig(temp_config);
    client_.sendConfig();
  }
}