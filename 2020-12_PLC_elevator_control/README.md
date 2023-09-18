# PLC control for an elevator

<br>
<img src="./Elevator%20-%20Sim%20environment.png" width="300">  
<br>

**Location**: Alma Mater Studiorum – Università di Bologna  
**Period**: Sep 2020 - Dec 2020  
**Collaborators**: Vittorio Galletti

## Context
Within the Control systems technologies course, we created the logic control scheme for a PLC controlling a simulated elevator using the Codesys software. The program was written with a combination of SFC and ST languages and allowed the system to manage multiple calls and perform all the required actions (move to the correct floor, open doors, signal emergencies,…) in the correct sequence.

## Project Description
The project revolves around developing a control system for an elevator, focusing on quick response, emergency handling, fault detection, and ensuring safety. The elevator's operational intricacies reveal multiple states, sensor interactions, and decision-making layers. The objective is to conceive an elevator mechanism that offers seamless functionality in routine situations and is equipped for unpredictable scenarios.

Initially, the system addressed emergency situations with the Doors GA, stopping the elevator when the emergency button is pressed during door operations. However, this approach only handled door-related emergencies. The introduction of the Move GA ensured the elevator would halt at the nearest deck during movement, further enhancing safety protocols. Yet, unforeseen circumstances, like sensor malfunctions, posed threats to the system’s efficiency.

To tackle sensor faults, a comprehensive fault-handling mechanism was instituted. The Move GA played a pivotal role, utilizing a Fault Detection step to diagnose anomalies based on sensor sequences. Still, this only tackled the identification part. The system's reaction to these faults, especially distinguishing between deck and ramp faults, demanded further refinements. Thus, additional branches in the Move GA were introduced to dictate the elevator’s behaviour post-fault detection, ensuring its course remains unimpeded and adheres to safety standards.

While the Move GA addressed immediate reactions, the Policy layer was formulated to ensure the system's future interactions would discount any malfunctioning components. For instance, it actively avoided calls to compromised decks. An innovative feature was the full-stop mechanism, which halts the elevator operations after two simultaneous faults, underscoring the project's emphasis on safety.

However, implementing these algorithms had its challenges. Arrays were adopted to streamline floor management and priority settings, but the considerations for time measurement in fault detection introduced uncertainties. These emanated from potential motor control variations and other mechanical delays, demanding alternative solutions.

In essence, the Elevator Control System project meticulously incorporated layers of decision-making processes, emergency handling techniques, fault detection and management, and advanced algorithmic functions. It exemplified a delicate balance between rigorous safety protocols, user experience, and technical innovation. Through iterative refinements, from the Doors GA to the Policy layer, the system evolved into a holistic solution that promises safety, efficiency, and adaptability.

## Files
- **Elevator - Project Guidelines.pdf**: The document describing the project guidelines and containing a picture of the simulation interface 
- **Elevator - Final Report.pdf**: The final report for the project
- **Elevator - Project Code.pdf**: The project code file to be run with the Codesys software
- **Elevator - Sim environment.png**: Image of the simulation environment
