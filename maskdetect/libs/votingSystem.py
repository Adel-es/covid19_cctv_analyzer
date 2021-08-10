from typing import MutableMapping
from utils.types import MaskToken


class VotingStatus : 
    limitVote = 100 
    initVote = 2 
    conNFLimit = 5
    
    def __init__(self) : 
        self.votes = { 
            MaskToken.UnKnown : VotingStatus.initVote, 
            MaskToken.Masked : 0 , 
            MaskToken.NotMasked : 0
        } 
        self.continuedNF = 0 
        self.voteResult = MaskToken.UnKnown  
        
        
    def vote(self, mtoken) -> MaskToken :
        
        # this is unusuall cases -> NotNear or UnKnown don't need a vote. 
        if mtoken == MaskToken.NotNear : 
            return self.voteResult 
        if mtoken == MaskToken.UnKnown : 
            return self.voteResult
        
        # faceNotFound -> +1 continuedNF 
        if mtoken == MaskToken.FaceNotFound : 
            self.continuedNF += 1; 
            if self.continuedNF >= VotingStatus.conNFLimit : 
                return MaskToken.FaceNotFound; 
            return self.voteResult 
        
        # others  
        self.continuedNF = 0 
        if self.votes[mtoken] < VotingStatus.limitVote : 
            self.votes[mtoken] += 1 
        else :  # if voting overflows the limit, divide by 2 to continue, and give opportunity for others,, 
            self.votes[MaskToken.Masked] = self.votes[MaskToken.Masked] / 2
            self.votes[MaskToken.NotMasked] = self.votes[MaskToken.NotMasked] / 2 
            
        if self.voteResult != mtoken : 
            if  self.votes[self.voteResult] < self.votes[mtoken] : 
                self.voteResult = mtoken 
        return self.voteResult 
    
    def show(self) : 
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