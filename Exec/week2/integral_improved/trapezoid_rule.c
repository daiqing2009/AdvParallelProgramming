/* trapezoid_rule.c */

extern float f(float x); /* function for integration */

float Trapezoid(float a, float b, int n, float h)
{
    float integral;   /* result of integration  */
    float x;
    int i;
  
    integral = (f(a) + f(b))/2.0;
  
    x = a;
    for ( i = 1; i <= n-1; i++ ) 
    {
        x = x + h;
        integral = integral + f(x);
    }
  
    return integral*h;
} 
