/* The present script corresponds to the source file of the GUI's
custom LED widget. It contains the definition of the member functions.
Author:
  Ruben Martin Rodriguez, group 10*/

#include "qledlabel.h"
#include <QDebug>

// Set the size of the widget (square, so both height and width)
static const int SIZE = 80;
// declare the different colors of the led as strings with a gradient
static const QString greenSS = QString("color: white;border-radius: %1;background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:1, y2:1, stop:0 rgba(20, 252, 7, 255), stop:1 rgba(25, 134, 5, 255));").arg(SIZE/2);
static const QString redSS = QString("color: white;border-radius: %1;background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:0.92, y2:0.988636, stop:0 rgba(255, 12, 12, 255), stop:0.869347 rgba(103, 0, 0, 255));").arg(SIZE/2);
static const QString orangeSS = QString("color: white;border-radius: %1;background-color: qlineargradient(spread:pad, x1:0.232, y1:0.272, x2:0.98, y2:0.959773, stop:0 rgba(255, 113, 4, 255), stop:1 rgba(91, 41, 7, 255))").arg(SIZE/2);
static const QString blueSS = QString("color: white;border-radius: %1;background-color: qlineargradient(spread:pad, x1:0.04, y1:0.0565909, x2:0.799, y2:0.795, stop:0 rgba(203, 220, 255, 255), stop:0.41206 rgba(0, 115, 255, 255), stop:1 rgba(0, 49, 109, 255));").arg(SIZE/2);

// Constructor, calls to the parent class through a higher level pointer
QLedLabel::QLedLabel(QWidget *parent) :
    QLabel(parent)
{
    // Set the state to ok by default
    setState(StateOk);
    // Set the dimensions of the widget upon construction
    setFixedSize(SIZE, SIZE);
}

// State handling function, will change the color of the LED according to the passed state
void QLedLabel::setState(State state)
{
    switch (state) {
    case StateOk:
        setStyleSheet(greenSS);
    break;
    case StateOkBlue:
        setStyleSheet(blueSS);
    break;
    case StateWarning:
      setStyleSheet(orangeSS);
    break;
    case StateError:
      setStyleSheet(redSS);
    break;

    }
    // Change the internal attribute accordingly for external object access
    state_ = state;
}

// Alternative function taking in a boolean, if it is true, the state is OK, 
// otherwise, error state
void QLedLabel::setState(bool state)
{
    setState(state ? StateOk : StateError);
}

// Getter function for the state attribute
int QLedLabel::getState(){
  return state_;
}
