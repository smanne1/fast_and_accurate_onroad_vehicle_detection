

List = np.ones((MaxLaneNum, ExpLaneNum))
for i in range(MaxLaneNum):
    for j in range(ExpLaneNum):
        if (Count_ref[i] > 0) & (Enable[j] == 1):
            List[i, j] = abs(Line[0, j].transpose() - Rep_ref[0,i]) + abs(Line[1, j].transpose() - Rep_ref[1, i]) * 200
        
    
#Find the best matches between the current lines and those in the repository.
#Match_dis  = intmax('int16')*ones(1, MaxLaneNum, 'int16');
Match_dis  = np.ones((1, MaxLaneNum))
Match_list = np.zeros((2, MaxLaneNum))
for i in range(ExpLaneNum):
    if i > 1:
        #Reset the row and column where the minimum element lies on.
        List[rowInd, :] = np.ones((1, ExpLaneNum))
        List[:, colInd] = np.ones((MaxLaneNum, 1))
    
    # In the 1st iteration, find minimum element (corresponds to
    # best matching targets) in the distance matrix. Then, use the
    # updated distance matrix where the minimun elements and their
    # corresponding rows and columns have been reset.
    Val = List.min()
    rowInd,colInd = np.unravel_index(a.argmin(), List.shape) 
    Match_dis[i]    = Val
    Match_list[:,i] = [rowInd,colInd].transpose()

# Update reference target list.
# If a line in the repository matches with an input line, replace
# it with the input one and increase the count number by one;
# otherwise, reduce the count number by one. The count number is
# then saturated.
Count_ref = Count_ref - 1
for i in range(ExpLaneNum):
    if Match_dis(i) > TrackThreshold:
        # Insert in an unused space in the reference target list
        NewInd = Count_ref.argmin()
        Rep_ref[:, NewInd] = Line[:, Match_list[1, i]]
        Count_ref[NewInd] = Count_ref[NewInd] + 2;
    else:
        # Update the reference list
        Rep_ref[:, Match_list[0, i]] = Line[:, Match_list[1, i]]
        Count_ref[Match_list[0, i]] = Count_ref[Match_list[0, i]] + 2
    
Count_ref[Count_ref < 0] = 0
Count_ref[Count_ref > CountUpperThresh] = CountUpperThresh
