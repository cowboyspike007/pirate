def array_advance(A):
    fail_path=dict()
    path=[0]
    is_possible=False
    step_no=[1]*len(A)

    while len(path)!=0:
        index=path[-1]
        path.pop()
        while A[index]!=0:
            if(index in fail_path):
                break
            path.append(index)
            index+=A[index]
            if(index>=len(A)-1):
                path.append(len(A)-1)
                is_possible=True
                return is_possible, path
        while (len(path)!=0 and step_no[path[-1]]>A[path[-1]]):
            fail_path[path[-1]]=-1
            path.pop()
        if len(path)!=0:
            step_no[path[-1]]+=1
            next_index=path[-1]+step_no[path[-1]]-1
            while len(path)!=0 and (next_index in fail_path or A[next_index]==0):
                step_no[path[-1]]+=1
                next_index=path[-1]+step_no[path[-1]]-1
                if step_no[path[-1]]>A[path[-1]]:
                    fail_path[path[-1]]=-1
                    path.pop()
            if len(path)!=0:
                path.append(next_index)
    return is_possible, path 
        
        
        


# True: Possible to navigate to last index in A:
# Moves: 1,3,2
A = [3, 3, 1, 0, 2, 2, 0, 1]
is_possible, path=array_advance(A)
if is_possible:
    print("it is possible and the path is: ")
    i=0
    for i in range(len(path)-2):
        print(path[i+1]-path[i],end="  ")
        i+=1
    print(path[i+1]-path[i])
else:
    print("not possible")

# False: Not possible to navigate to last index in A:
A = [3, 2, 0, 0, 2, 0, 1]
is_possible, path=array_advance(A)
if is_possible:
    print("it is possible and the path is: ")
    i=0
    for i in range(len(path)-2):
        print(path[i+1]-path[i],end="  ")
        i+=1
    print(path[i+1]-path[i])
else:
    print("not possible")
