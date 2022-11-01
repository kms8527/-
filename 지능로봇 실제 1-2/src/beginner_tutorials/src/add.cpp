#include "ros/ros.h"
#include "std_msgs/Int64.h"

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
// %Tag(CALLBACK)%


class cadd{


    private:
    ros::NodeHandle n;
    ros::Subscriber sub_A;
    ros::Subscriber sub_B;
    ros::Publisher pub_C;
    int A;
    int B;

    public:
    cadd(const ros::NodeHandle& nh)
    {
        n = nh;
        sub_A = n.subscribe("A", 1, &cadd::A_Callback,this);
        sub_B = n.subscribe("B", 1, &cadd::B_Callback,this);
        pub_C = n.advertise<std_msgs::Int64>("C",1);
        A= -1;
        B= -1;       
        
    }    
    void A_Callback(const std_msgs::Int64::ConstPtr& msg)
    {
        this->A = msg->data;
    }
    void B_Callback(const std_msgs::Int64::ConstPtr& msg)
    {
        this->B = msg->data;
        add();
    }        
    int add(){
        std_msgs::Int64 msg;
        
        if( A !=-1 && B !=-1){
            msg.data = A+B;
            pub_C.publish(msg);
            std::cout<<msg.data<<std::endl;
        }
        
    }
        
};


// %EndTag(CALLBACK)%

int main(int argc, char **argv)
{

    ros::init(argc, argv, "node_3");
    ros::NodeHandle n;
    ros::Rate loop_rate(10);

    cadd cadd(n);
    
// %Tag(SPIN)%
    ros::spin();
    return 0;
}