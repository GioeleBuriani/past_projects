/* The present script corresponds to the header file of the GUI's
custom LED widget. It contains the decaration of the class with its attributes
and member function, as well as the slots for communicating between
GUI widgets.
Author:
  Ruben Martin Rodriguez, group 10*/

#ifndef QLEDLABEL_H
#define QLEDLABEL_H

// Import parent class
#include <QLabel>

// Define the class inheriting from QLabel
class QLedLabel : public QLabel
{
    // Required for all Qt objects
    Q_OBJECT
public:
    // Explicit constructor as required by Qt
    explicit QLedLabel(QWidget *parent = 0);

    // Enumeration of components, each of them will be associated an
    // integer value starting from 0 and can be externally treated as such
    // By making the enumeration public, it is more intuitive to handle the
    // states, rather than numerically
    enum State{
        StateOk,
        StateOkBlue,
        StateWarning,
        StateError
    };


signals:

public slots:
    // Declarations of the functions for changing the state of the led
    void setState(State state);
    void setState(bool state);
    // Function for getting the current state of the led, casted into integer
    int getState();

private:
    int state_;
};

#endif // QLEDLABEL_H
