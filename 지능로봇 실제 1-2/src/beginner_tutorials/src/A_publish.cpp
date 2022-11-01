#include "ros/ros.h"
#include <random>
#include "std_msgs/Int64.h"

int main(int argc, char **argv)
{

  ros::init(argc, argv, "node_1");

  ros::NodeHandle n;

  ros::Publisher chatter_pub = n.advertise<std_msgs::Int64>("A", 1000);

  ros::Rate loop_rate(10);

  std_msgs::Int64 msg;
  int rand =0;
  std::random_device rd;   // non-deterministic generator
  std::mt19937 gen(rd());  // to seed mersenne twister.
  std::uniform_int_distribution<> dist(1, 10);

  while (ros::ok())
  {
   
    rand = dist(gen);
    msg.data = rand;
    std::cout<<msg.data<<std::endl; 
    chatter_pub.publish(msg);
    loop_rate.sleep();
  }


  return 0;
}