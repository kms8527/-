#include "ros/ros.h"
#include "std_msgs/Int64.h"
#include "std_msgs/Float64.h"
/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
// %Tag(CALLBACK)%


class cavg{


    private:
    ros::NodeHandle n;
    ros::Subscriber sub_C;
    ros::Publisher pub_D;
    double cnt;
    double avg;

    public:
    cavg(const ros::NodeHandle& nh)
    {
        n = nh;
        sub_C = n.subscribe("C", 1, &cavg::C_Callback,this);
        pub_D = n.advertise<std_msgs::Float64>("D",1);
        cnt = 0;
        avg =0;
        
    }    

    void C_Callback(const std_msgs::Int64::ConstPtr& msg)
    {
        std_msgs::Float64 pub_msg;
        cnt++;
        if(cnt>1)
            avg = (avg + msg->data/(cnt-1)) * (cnt-1)/cnt;
        else
            avg = msg->data;

        pub_msg.data = avg;
        pub_D.publish(pub_msg);
        std::cout<<"cnt : "<<cnt << ", avg :" << avg<<std::endl;

    }        
        
};


// %EndTag(CALLBACK)%

int main(int argc, char **argv)
{

    ros::init(argc, argv, "node_4");
    ros::NodeHandle n;
    ros::Rate loop_rate(10);

    cavg cavg(n);
    
// %Tag(SPIN)%
    ros::spin();
    return 0;
}