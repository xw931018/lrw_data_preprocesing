from psfdataset import PSFDataset, transforms
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

def getEncoder(data):
    encoder= LabelEncoder()
    categories = list(data.keys())
    encoder.fit(categories)
    return encoder

def generate_iterator(data, key = 'train', refLength = 29):

    encoder = getEncoder(data)
    iter_list = []
    x = 0
    for word, keypointsOneWord in data.items():
        

        keypointsList = keypointsOneWord[key]
        num_of_samples = len(keypointsList)
        
        ## There are some sample whose length is smaller than 29. We need to either delete it, 
        ## or extend it to 29 length for now. But this can be resolved if signature transform is introduced
        for i in range(len(keypointsList)):
             singleSample = keypointsList[i]
             if len(singleSample) < refLength:
                 singleSample = np.array(list(singleSample) + [singleSample[-1]] * (refLength - len(singleSample)))
                 keypointsList[i] = singleSample
        iter_list = iter_list + list(zip(keypointsList, np.array(list(encoder.transform([word])) * num_of_samples)))
    return iter(iter_list)


def rotate(video_array):
    
    n = len(video_array)
    
    #loop over all of the video
    
    video_array = np.array(video_array).copy()
    for i in range(n):
        
        a = video_array[i] #i-th video
        
        #loop over all of the frames
        
        for j in range(a.shape[0]):
            
            """
            From now on we will work with the j-th frame of the i-th video a[j,:,:].
            
            Let us recall that the 30-th landmark is the lower extreme of the nose while 
            the 27-th landmark is the upper extreme of the nose.
            
            The main idea of this rotation function is to firstly translate all of the landmarks
            in the reference system with origin in a[j,30,:]. Afterwards, we look at a[j,27,:]
            as a vector in R^2 in this new system and we apply a rotation to all of the
            landmarks so that the first coordinate of a[j,27,:] becomes 0.
            
            This rotation can be achieved by defining an R^2 rotation matrix of angle theta (which takes the form 
            [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]) where theta is the angle between a[j,27,:]
            and the y axis.
            """
                      
            a[j,:,0] = a[j,:,0] - a[j,2,0]
            a[j,:,1] = a[j,:,1] - a[j,2,1]
            
            cos = a[j,1,1]/np.sqrt(a[j,1,0]**2 + a[j,1,1]**2) #cos(theta)
            sin = a[j,1,0]/np.sqrt(a[j,1,0]**2 + a[j,1,1]**2)
            
            
            
            for l in range(a.shape[1]):
                
                rot = np.array([[cos,-sin],[sin,cos]])
                
                a[j,l,:] = rot@a[j,l,:]
        
        
    return video_array

def flip_image(video_array):
    '''This functions flips the landmarks about an axis.'''
    a = video_array #i-th video
        
    b = np.copy(a)

    for j in range(a.shape[0]):

        for l in range(a.shape[1]):

            b[j,l,0] = -a[j,l,0]

    return b

# Training set augmentation 

def flip_train_augmentation(video_data): 
    '''The function returns a dictionary equal to the raw data, but the training set 
    for each word in the dictionary is augmented with its flipped version.'''
    modified_data = video_data.copy()
    for i in video_data:
        augmented_list = []
        for j in range(len(video_data[i]['train'])):
            augmented_list.append(flip_image(video_data[i]['train'][j]))
        modified_data[i]['train'] =  modified_data[i]['train'] + augmented_list
    return modified_data
    
def average_rot(video_array):
    
    n = len(video_array)
    '''This function calculates the angle of rotation for each video in each frame. 
    The result can be used to build histograms as in the following cell. '''
    #loop over all of the video
    theta = []
    for i in range(n):
        
        a = video_array[i].copy() #i-th video
        
        #loop over all of the frames
        
        for j in range(a.shape[0]):
                                  
            a[j,:,0] = a[j,:,0] - a[j,2,0]
            a[j,:,1] = a[j,:,1] - a[j,2,1]
            
            cos = a[j,1,1]/np.sqrt(a[j,1,0]**2 + a[j,1,1]**2) #cos(theta)
            sin = a[j,1,0]/np.sqrt(a[j,1,0]**2 + a[j,1,1]**2) #sin(theta)
            
            theta.append(np.arcsin(sin))
    
    return theta
   
def normalize_based_on_first_frame(video_array):
    '''For an array of videos we normalize all frames based on the first frame for each video.'''
    video_array = video_array.copy()
    for i in range(len(video_array)):
        maximum = np.max(video_array[i][0])
        minimum = np.min(video_array[i][0])
        video_array[i] = (video_array[i]-minimum)/(maximum-minimum)
    return video_array

class LinearClassifierNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        torch.nn.Module.__init__(self)
        self.linear1 = torch.nn.Linear(input_dim, 2048)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.output = torch.nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.output(x)
        return x
    
def trainModelWithSpecificDataDict(dataDict, transforms, ModelClass = LinearClassifierNet, device = 'cpu', EPOCHS = 20):
    trainingset = PSFDataset(transform = transforms) 
    testset = PSFDataset(transform = transforms)
    valset  = PSFDataset(transform = transforms)
    
    selected_n_classes = len(dataDict.keys())
    trainingset.fill_from_iterator(generate_iterator(dataDict, 'train'))
    testset.fill_from_iterator(generate_iterator(dataDict, 'test'))
    valset.fill_from_iterator(generate_iterator(dataDict, 'val'))
    
    print("Number of trainingset elements:", len(trainingset))
    print("Number of testset elements", len(testset))
    print("Dimension of feature vector:", trainingset.get_data_dimension())
    
    device = torch.device('cpu')
    BATCH_SIZE = 8
    LR = 0.0001
    EPOCHS = EPOCHS
    
    training_loader = torch.utils.data.DataLoader(trainingset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)
    model = ModelClass(input_dim=trainingset.get_data_dimension(),
                                output_dim=selected_n_classes) # selected_n_classes = 10 (maximum 500)
    model.to(device=device, dtype=torch.double)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    def train_network(epoch):
        model.train()
        cumulated_loss = 0.0
        for i, data in tqdm(enumerate(training_loader), desc="Epoch " + str(epoch), leave=False):
            inputs, labels = data[0].to(device, dtype = torch.double), data[1].to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            cumulated_loss += loss.item()
        return (cumulated_loss / len(training_loader))

    def test_network():
        cumulated_outputs = np.array([])
        cumulated_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device, dtype = torch.double), data[1].to(device, dtype=torch.long)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                cumulated_loss += loss.item()

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                cumulated_outputs = np.concatenate((cumulated_outputs, outputs.cpu().numpy()), axis=None)
            test_loss = cumulated_loss / len(test_loader)
            return test_loss, cumulated_outputs
    
    # Print initial accuracy before training
    __, outputs = test_network()
    acc = accuracy_score(testset.get_labels(), outputs)
    print("Initial accuracy:", acc)

    for epoch in trange(EPOCHS, desc="Training"):
        train_loss = train_network(epoch)
        test_loss, outputs = test_network()

        acc = accuracy_score(testset.get_labels(), outputs)
        print("train_loss:", train_loss, "\ttest_loss:", test_loss, "\tAccuracy:", acc)
    fig = plt.figure(figsize=(10, 10))
    
    cm = confusion_matrix(testset.get_labels(), outputs)
    ax = fig.add_subplot(111)
    cm_display = ConfusionMatrixDisplay(cm,
                                        display_labels=dataDict.keys()).plot(ax=ax,
                                                                       xticks_rotation="vertical")

    
    return model, outputs