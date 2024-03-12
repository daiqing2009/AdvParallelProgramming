## Question 1
Write a parallel MPI program which generates a sequence of N random numbers, each between 0 and 1, on each of its p processes.

Then implement a sorting algorithm where each process sorts the sequence of numbers it has, then distributes parts of it to the other processes so that at the end process 0 has the random numbers between 0 and 1/p, process 1 has numbers between 1/p and 2/p, and so on until the last process has numbers between (p-1)/p and 1

## Logic of the program
### frame work of overall program
The overall framework of the program can be as simple as following:
1. Initialize the sequence of floating numbers and sort within process
2. Distribute numbers to corresponding processes according to the requirement
3. Merge distributed number from different processes and print the result
### key code segment
There are major challenges I encountered or trickes i used in coding
* utilized the merchanism of 
``
* use merge on 
## Performance analysis
### Thoeratical Calculation 

### Lab Experiment