/*Main node of the GUI, will create a terminal object and let it run.
Author:
  Ruben Martin Rodriguez, group 10*/
#include <QApplication>
#include "my_terminal.h"


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "hello_gui_node");
  QApplication a(argc, argv);

  MyTerminal w;

  // set the window title
  w.setWindowTitle(QString::fromStdString("TIAGo Terminal"));

  w.show();
  return a.exec();
}
