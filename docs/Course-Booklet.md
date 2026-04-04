```
Course Code: 20973
```
# Workshop in Autonomous Systems Simulation

```
Course Booklet fall 2026
```

- Dear Student
- Workshop Description..............................................................................................................
- Project Definition......................................................................................................................
   - Topic selection..................................................................................................................
   - Project Presentations (design review) by mid-semester:..................................................
   - Final Submission (by August 31st):..................................................................................
   - Grading Criteria:................................................................................................................
- project suggestions................................................................................................................
   - Projects for CARLA (Autonomous Driving Simulation)...................................................
      - 1. Intelligent Traffic Control System............................................................................
      - 2. Self-Driving Vehicle in Mixed Traffic........................................................................
      - 3. Emergency Vehicle Navigation...............................................................................
      - 4. Smart Parking Assistant..........................................................................................
      - 5. Weather-Adaptive Autonomous Driving..................................................................
      - 6. Safe Pedestrian Interaction.....................................................................................
      - 7. Cooperative Multi-Vehicle Navigation.....................................................................
      - 8. Reinforcement Learning for Lane Changing............................................................
      - 9. Platooning of Autonomous Trucks...........................................................................
      - 10. Smart Bus System for Public Transport................................................................
      - 11. Simulation for an Autonomous Taxi System with On-Demand Dispatch...............
   - Projects for MuJoCo (Robotics and Physics-Based Simulation).....................................
      - 1. Quadruped Robot Terrain Adaptation......................................................................
      - 2. Dexterous Robotic Arm for Object Manipulation.....................................................
      - 3. Human-Robot Collaboration in Industrial Tasks......................................................
      - 4. Balancing and Locomotion for Humanoid Robots...................................................
      - 5. Robotic Exoskeleton for Assisted Walking..............................................................
      - 6. Drone Precision Landing on Moving Platforms.......................................................
      - 7. Soft Robotic Gripper for Fragile Objects.................................................................
      - 8. Autonomous Warehouse Logistics..........................................................................
      - 9. AI-Controlled Robotic Soccer Players.....................................................................


## Dear Student

Welcome to the Autonomous Systems Simulation Workshop. We are pleased to have you
join us in this course, where you will explore simulation frameworks, develop autonomous
system functionalities, and evaluate their performance in diverse scenarios.
This booklet provides you with essential information about the workshop, including the
course schedule, activities, and project guidelines.
A dedicated course website is available, where you can find additional learning materials.
The website serves as a communication platform for interacting with the teaching staff and
fellow students. Additionally, you can access online learning resources and the university
library services at the following links:
● Course Website:
● Library Services: [http://www.openu.ac.il/Library](http://www.openu.ac.il/Library)
● Online Learning Platform: [http://telem.openu.ac.il](http://telem.openu.ac.il)
If needed, you may schedule meetings via email at roiwe@openu.ac.il
**For students studying abroad:** Despite the physical distance, we strive to maintain close
connections and provide support as needed. All essential course details are included in this
booklet and on the course website. We highly recommend utilizing the online platform and all
available learning resources.
**Wishing you an enjoyable and productive learning experience!
Roi Wexler
Course Coordinator**


## Workshop Description..............................................................................................................

This workshop focuses on the simulation of autonomous systems, including autonomous
vehicles, drones, and robots.
The primary goal is to familiarize students with an advanced simulation framework that
enables the implementation of autonomous components, the addition of new functionalities,
and the evaluation of these components' performance under diverse scenarios.
Each student will have the opportunity to choose a suitable simulation platform based on
their interests and the guidance of their supervisor.
Simulation platforms available for use in this workshop include:
● CARLA
● MuJoCo
As a key part of the workshop, students will undertake a project—either individually or in
pairs—where they will design, develop, and document an autonomous system simulation.
The project will require applying programming principles, defining simulation scenarios, and
setting performance evaluation metrics.
Through this experience, students will develop technical proficiency in autonomous systems,
problem-solving skills, and the ability to critically assess system behavior under diverse
conditions.
**Presentation:**
Each student will prepare a presentation of approximately 30 minutes on his chosen project
and present it to the rest of the students.
In addition, a copy of the presentation slides and a summary of the material must be
submitted during the session.
presentation topics will be approved during the first three weeks of the semester.
**Projects:**
The workshop includes the development of a final project, utilizing the technologies learned
in the course.
The project must adhere to programming principles.
Students may choose a topic from a list of suggested projects or propose their own ideas.


It is highly recommended working on this assignment in pairs. Working together will allow
you to brainstorm ideas and tackle the tasks more efficiently. **If you need help finding a
partner, try posting on the course forum or consult with your instructor**.
Should you choose to complete the assignment individually, the scope will be adjusted
accordingly for a single person during the approval of the project plan.
Towards the mid of the semester, students will submit a project plan including the
engineering solution (design) and present it to the class. A detailed explanation of this task
can be found later in this manual.
The deadline for the final project submission is August 31st. By this date, students are
required to submit their projects and schedule a 15–30 minute presentation meeting with the
instructor.


Grading Policy:
● Project Plan Presentation: 35%
● Project Implementation: 65%

Code Quality (^) 25%
SOLID Principles Implementation (^) 15%
Reliability, Error Handling & System
Robustness

#### 15%

```
Project Structure & Design
Implementation
```
#### 15%

#### Documentation & Developer Experience 10%

#### Meeting KPIs and Performance Goals 20%

```
Self-Evaluation & Reflective Analysis
```
#### 5%

#### (bonus)


**Course Materials:**
● CARLA Documentation
● MuJoCo Documentation
● Additional readings and resources will be provided throughout the course.
**Recommended Reads**
● S.O.L.I.D Principles
● The Clean Architecture - Clean Coder Blog
● Software Design Patterns & Architecture
● Kanban Guide for Software Development will become relevant after the design
solution is approved.
**Instructors Contact Information:
●** Lior Kaster lior.kastel@openu.ac.il
● Nadav Beno nadavb@openu.ac.il
● Roy Rachmany roy.rachmany@openu.ac.il
**Coordinator contact Information:**
● Roi Wexler roiwe@openu.ac.il
**Note;** This is an English-language course that fulfills the credit requirements in English. This
means that all course materials, presentations, and communication will be conducted in
English.


## Project Definition......................................................................................................................

In this workshop, students will develop a project involving the simulation of autonomous
systems, such as self-driving vehicles, drones, or robots.
Students may choose a topic from the provided list of project suggestions or propose their
own, subject to approval.
Projects could be submitted in pairs and must be completed according to the following
schedule:

### Topic selection..................................................................................................................

Three weeks into the semester, each student must select a project topic, get it approved by
the instructor, and register for one of the scheduled meetings to present it to the class.
Once approved, you will be expected to set up a Kanban board in Trello (link will be provided
by the instructor) to manage your project's backlog, map out your development milestones,
and populate it with detailed tickets that reflect your execution plan and technical tasks.

### Project Presentations (design review) by mid-semester:..................................................

By mid-semester, students will present their project plan. The presentation will be focused on
an explanation of the system design choices. Students must demonstrate a deep

### understanding of their project and be ready to answer related questions.

This session will simulate a professional design review meeting, where you will be expected
to present the following:
● **Project Requirements** – Clear definition of functional and technical requirements,
including constraints and expected outcomes.
● **System Architecture and Key Components** – Overview of the system structure
solutions, main modules, and their interactions. You must include a clear architectural
diagram and a sequence chart.
**Design Alternatives:** Present at least two valid architectural approaches. Compare
them and justify your final choice by explaining the **trade-offs** and the **technical
rationale** behind your decision.
● **Description of Simulation Scenarios** – Explanation of the real-world situations
being simulated.
● **Performance Metrics** – Definition of key performance indicators (KPIs) for
evaluating the system’s efficiency and accuracy.
● **Challenges and Risks** – Identification of potential technical or conceptual
challenges and proposed solutions.


```
● Preliminary Prototype/Demo – If possible, a basic implementation of core
functionalities or a visualization of the system.
● Evaluation Criteria – How the system’s success will be measured
● Timeline – Breakdown of individual contributions, milestones, and a work plan for
completing the project.
```
### Final Submission (by August 31st):..................................................................................

The project submission must include the following documentation:
● A design document outlining the architecture, key components, simulation scenarios,
performance metrics, and all other elements from the Initial Planning Phase.
● A README file explaining how to install and run the system
● The code itself

### Grading Criteria:................................................................................................................

The project will account for 65% of the final grade, evaluated based on system design,
stability, scenarios and tests plan, and bug resilience. The final grade also includes project
documentation and presentation. A minimum final score of 60 is required to complete the
course successfully.


## project suggestions................................................................................................................

### Projects for CARLA (Autonomous Driving Simulation)...................................................

#### 1. Intelligent Traffic Control System............................................................................

Simulate an autonomous traffic management system that adapts in real time to congestion,
optimizing traffic signals and rerouting vehicles for efficiency (project can focus on a part of
this system).

#### 2. Self-Driving Vehicle in Mixed Traffic........................................................................

Develop a scenario where an autonomous car navigates through a mix of self-driving and
human-driven vehicles, handling unpredictable behaviors safely.

#### 3. Emergency Vehicle Navigation...............................................................................

Design an AI system that allows autonomous emergency vehicles (ambulances, fire trucks)
to navigate through heavy traffic by coordinating with other self-driving emergency cars and
the traffic around them, including traffic lights.

#### 4. Smart Parking Assistant..........................................................................................

Simulate an autonomous parking system where vehicles detect available spots and perform
complex parking maneuvers in real-world conditions.

#### 5. Weather-Adaptive Autonomous Driving..................................................................

Test a self-driving car in various weather conditions, such as heavy rain, fog, or snow, and
implement AI-based adjustments for visibility and traction control.

#### 6. Safe Pedestrian Interaction.....................................................................................

Develop an AI model that enables self-driving cars to predict pedestrian behavior, stop safely
at crosswalks, and interact intelligently in dense urban environments (You can choose to
focus on one component).

#### 7. Cooperative Multi-Vehicle Navigation.....................................................................

Simulate a group of autonomous vehicles coordinating their movements at intersections and
highways to improve traffic flow and reduce congestion (You can choose to focus on one
component).


#### 8. Reinforcement Learning for Lane Changing............................................................

Use reinforcement learning to train a self-driving car to execute safe and efficient lane
changes on highways, considering traffic speed and density (You can choose to focus on
one component).

#### 9. Platooning of Autonomous Trucks...........................................................................

Develop a system where multiple autonomous trucks drive in close proximity to improve fuel
efficiency and minimize aerodynamic drag (You can choose to focus on one component).

#### 10. Smart Bus System for Public Transport................................................................

Simulate a network of autonomous buses that adjust routes based on passenger demand,
traffic conditions, and real-time scheduling constraints (You can choose to focus on one
component).

#### 11. Simulation for an Autonomous Taxi System with On-Demand Dispatch...............

Develop a simulation of an autonomous taxi fleet management system, where the nearest
available vehicle is dispatched based on passenger requests. The system will optimize route
planning to ensure the fastest and most efficient journey to the destination (You can choose
to focus on one component).

### Projects for MuJoCo (Robotics and Physics-Based Simulation).....................................

#### 1. Quadruped Robot Terrain Adaptation......................................................................

Simulate a four-legged robot (like a robotic dog) navigating rough terrain, learning to adjust
its gait dynamically based on terrain feedback, or by implementing pretrained algorithm.

#### 2. Dexterous Robotic Arm for Object Manipulation.....................................................

Develop a robotic arm capable of precise manipulation tasks such as stacking, sorting, or
assembling objects using reinforcement learning, or by implementing pretrained algorithm.

#### 3. Human-Robot Collaboration in Industrial Tasks......................................................

Simulate an industrial environment where robots and human workers collaborate safely,
focusing on motion prediction and adaptive responses (you can choose to focus on few
pre-determined scenarios)

#### 4. Balancing and Locomotion for Humanoid Robots...................................................

Implement a reinforcement learning model to enable a humanoid robot to maintain balance
and walk across uneven surfaces.


#### 5. Robotic Exoskeleton for Assisted Walking..............................................................

Simulate a wearable robotic exoskeleton that adapts to a human’s movement, providing
support for rehabilitation or mobility enhancement.

#### 6. Drone Precision Landing on Moving Platforms.......................................................

Develop an autonomous drone that can land accurately on a moving platform, incorporating
real-time sensor fusion and control algorithms.

#### 7. Soft Robotic Gripper for Fragile Objects.................................................................

Design a soft robotic gripper that can handle delicate objects like glassware or fruits without
causing damage, optimizing grip force and surface adaptation.

#### 8. Autonomous Warehouse Logistics..........................................................................

Develop a logistics system where multiple autonomous robots manage inventory, transport
goods, and avoid collisions in a dynamic warehouse environment (You can choose to focus
on one component).

#### 9. AI-Controlled Robotic Soccer Players.....................................................................

Create a team of robotic soccer players trained using reinforcement learning to improve
passing, goal-scoring, and teamwork strategies (You can choose to focus on one component
like the goal keeper).


