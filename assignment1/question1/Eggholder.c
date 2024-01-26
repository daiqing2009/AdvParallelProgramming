/* the algorithm to write eggholder function */
#include <math.h>


float Eggholder(float x, float y){
    float bias_y = y +47;

    return -bias_y*sin(sqrt(fabs(x/2+bias_y)))- x*sin(sqrt(fabs(x-bias_y)));
}
