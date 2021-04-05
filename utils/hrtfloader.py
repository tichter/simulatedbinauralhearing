import scipy.io

class HRTFLoader():
    """
    General class to load different HRTF databases.
    Currently, only OldenburgDB implemented
    """
    def __init__(self,ED=False,BTE=False,add_ele=False,doublepolar=False):
        self.hrtfs = None
        self.azilabels = None
        self.elevationlabels = None

        self.returnHRTFs = None
        self.returnLabels = None
        self.returnLabelsEle = None

        self.doublepolar = doublepolar

        if ED:
            self.loadKEMAR(micloc="ED")
            if add_ele:
                self.setReturnAllFrontal()
            else:
                self.setReturnAzimuthFrontal()
        if BTE:
            self.loadKEMAR(micloc="BTEM")
            if add_ele:
                self.setReturnAllFrontal()
            else:
                self.setReturnAzimuthFrontal()

    def __len__(self):
        return len(self.returnHRTFs)

    def __getitem__(self,idx):
        return self.returnHRTFs[idx],self.returnLabels[idx]

    def setReturnAllFrontal(self):
        self.returnHRTFs = []
        self.returnLabels = []
        self.returnLabelsEle = []

        for i,azilabel in enumerate(self.azilabels):
            if self.elevationlabels[i] < 90:#remove 90 degree
                if azilabel >= -90 and azilabel <= 90:
                    #print("E{} - A{}".format(self.elevationlabels[i],azilabel))
                    self.returnHRTFs.append(self.hrtfs[:,i].T)
                    self.returnLabels.append(azilabel)
                    self.returnLabelsEle.append(self.elevationlabels[i])

    def setReturnAzimuthFrontal(self):
        self.returnHRTFs = []
        self.returnLabels = []
        for i,azilabel in enumerate(self.azilabels):
            if self.elevationlabels[i] == 0:
                if azilabel >= -90 and azilabel <= 90:
                    self.returnHRTFs.append(self.hrtfs[:,i].T)
                    self.returnLabels.append(azilabel)

    def getAzimuthFrontal(self):
        self.setReturnAzimuthFrontal()
        return self.returnHRTFs,self.returnLabels

    def loadKEMAR(self,micloc="ED"):
        """
        Function to load the correct OldenburgDB HRTFs into a general dataformat
        """
        if self.hrtfs is None:
            print("Loading HRTFs from OldenburgDB. Mic Location: {}".format(micloc))

            if micloc is "ED":
                KEMARmat = scipy.io.loadmat("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils\\KEMAR-ED.mat")
                print("Kemar eardrum loaded.")
            elif micloc is "BTEF":
                KEMARmat = scipy.io.loadmat("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils\\KEMAR-BTE_fr.mat")
            elif micloc is "BTEM":
                KEMARmat = scipy.io.loadmat("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils\\KEMAR-BTE_mid.mat")
            elif micloc is "BTER":
                KEMARmat = scipy.io.loadmat("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils\\KEMAR-BTE_rear.mat")
            else:
                print("Cannot find the right HRTF matfile in the OldenburgDB.")

            if self.doublepolar:
                azelmat = scipy.io.loadmat("D:\\1_Uni\\0_Master\\5_CI-Thesis\\05FinalCode\\utils\\doublepolarOlHeaDcoords.mat")
                locs = [azelmat['ReturnAz'].T[0],azelmat['ReturnEle'].T[0]]
            else:
                locs = KEMARmat['M_directions'] # [azimuth,elevation]
            hrtfs = KEMARmat['M_data'] # [256,azimuth,elevation]

            self.azilabels = locs[0]
            self.elevationlabels = locs[1]
            self.hrtfs = hrtfs

            #azelmat = scipy.io.loadmat("D:\\1_Uni\\0_Master\\5_CI-Thesis\\01Deliverable\\Code\\utils\\doublepolarOlHeaDcoords.mat")
            #print(azelmat['ReturnAz'].T)
            #print(KEMARmat['M_directions'][0])
            #print(self.azilabels)
            #input("hi")
