from utils.types import MaskToken


class VotingStatus : 
    limitVote = 100 
    initVote = 2 
    
    def __init__(self) : 
        self.votes = { 
            MaskToken.FaceNotFound : 0 , 
            MaskToken.Masked : 0 , 
            MaskToken.NotMasked : 0
        } 
        
        self.voteResult = MaskToken.UnKnown  
        self.voteCount = VotingStatus.initVote
        
    def vote(self, mtoken) -> MaskToken : 
        if self.votes[mtoken] < VotingStatus.limitVote : 
            self.votes[mtoken] += 1 
            
        if self.voteResult != mtoken : 
            if self.voteCount > self.votes[mtoken] : 
                self.voteCount = self.votes[mtoken]
                self.voteResult = mtoken   
        
        return self.voteResult 
    
    def show(self) : 
        print("FaceNoteFound\t : {}".format(self.votes[MaskToken.FaceNotFound]))        
        print("Masked       \t : {}".format(self.votes[MaskToken.Masked]))        
        print("NotMasked    \t : {}".format(self.votes[MaskToken.NotMasked]))        


class VotingSystem : 
    def __init__(self) : 
        self.vstatus = {}
        
    def vote(self, tid, mtoken) -> MaskToken : 
        if self.vstatus.get(tid) is None : 
            self.vstatus[tid] = VotingStatus() 
        
        result = self.vstatus[tid].vote(mtoken)
        return result 
        
    def show(self, tid) : 
        if self.vstatus.get(tid) is None : 
            print("no such ID_{} ".format(tid))
        else : 
            print("== TID_{} result ==".format(tid))
            self.vstatus[tid].show()
            print("===================")