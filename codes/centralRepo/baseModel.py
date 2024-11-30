#!/usr/bin/env python
# coding: utf-8

"""
     The Base model for any deep learning analysis.
     This class should provide following functionalities for any deep learning
     module
         1. train() -> Train the model
         2. predict() -> Evaluate the train, validation, and test performance
         3. Create train and validation graphs
         4. Run over CPU/ GPU (if available)
     This class needs following things to run:
         1. net -> The architecture of the network. It should inherit nn.Module
             and should define the forward method
         2. trainData, testData and validateData -> these should be eegDatasets
             and data iterators will be forked out of them
             Each sample of these datasets should be a dictionary with two
             fields: 'data' and 'label'
         3. optimizer -> the optimizer of type torch.optim.
         4. outFolder -> the folder where the results will be stored.
         5. preferedDevice -> 'gpu'/'cpu' -> will run on gpu only if it's
             available
    TODO: Include a learning rate scheduler in _trainOE.
    TODO: Add a good hyper-parameter optimizer in the train.
    @author: Ravikiran Mane
"""

# To do deep learning
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import torch.utils.data.sampler as builtInSampler
import sys
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os
import pickle
import copy
#!pip install stopCriteria
#masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
import stopCriteria
import samplers

class baseModel():
    def __init__(
        self,
        net,
        resultsSavePath=None,
        seed=3141592,
        setRng=True,
        preferedDevice='gpu',
        nGPU=0,
        batchSize=1):
        self.net = net
        self.seed = seed
        self.preferedDevice = preferedDevice
        self.batchSize = batchSize
        self.setRng = setRng
        self.resultsSavePath = resultsSavePath
        self.device = None

        # Set RNG
        if self.setRng:
            self.setRandom(self.seed)

        # Set device
        self.setDevice(nGPU)
        self.net = net.to(self.device)

        # Check for the results save path.
        if self.resultsSavePath is not None:
            if not os.path.exists(self.resultsSavePath):
                os.makedirs(self.resultsSavePath)
            print('Results will be saved in folder : ' + self.resultsSavePath)

    def train(
        self,
        trainData,
        valData,
        testData=None,
        classes=None,
        lossFn='NLLLoss',
        loss_icp=None,
        loss_isp=None,
        optimFns='Adam',
        optim_icp=None,
        optim_isp=None,
        optimFnArgs={},
        sampler=None,
        lr=0.001,
        stopCondi={'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1000, 'varName' : 'epoch'}},
                                  'c2': {'NoDecrease': {'numEpochs' : 200, 'varName': 'valLoss'}} } }},
        loadBestModel=True,
        bestVarToCheck='valLoss',
        continueAfterEarlystop=False):
        """
        Apex function to train and test any network.
        Calls _trainOE for base training and adds the reporting capabilities.

        Parameters
        ----------
        trainData : eegDataset
            dataset used for training.
        valData : eegDataset
            dataset used for validation.
        testData : eegDataset, optional
            dataset to calculate the results on. The default is None.
        classes : list, optional
            List of classes to consider in evaluation matrices.
            None -> all classes.
            The default is None.
        lossFn : string from torch.nn, The default is 'NLLLoss'
            Name of the loss function from torch.nn which will be used for
            training.
        optimFns : string from torch.optim. The default is 'Adam'.
            Name of the optimization function from torch.nn which will be used
            for training.
        optimFnArgs : dict, optional
            Additional arguments to be passed to the optimizer.
            The default is {}.
        sampler : a string specifying sampler to be used in dataloader
            optional
            The sampler to use while training and validation.
            Function with this name will be searched at two places,
                1. torch.utils.data.sampler, 2. samplers
                if not found then error will be thrown.
            The default is None.
        lr : float, optional
            Learning rate. The default is 0.001.
        stopCondi : dict, optional
            Determines when to stop.
            It will be a dictionary which can be accepted by stopCriteria class
            The default is : no decrease in validation Inaccuracy in last
            200 epochs OR epoch > 1500
            This default is represented as:
            {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500, 'varName' : 'epoch'}},
                                  'c2': {'NoDecrease': {'numEpochs' : 200, 'varName': 'valLoss'}} } }}
        loadBestModel : bool, optional
            Whether to load the network with best validation loss/ accuracy
            at the end of training. The default is True.
        bestVarToCheck : 'valLoss'/'valLoss', optional
            the best value to check while determining the best model.
            The default is 'valLoss'.
        continueAfterEarlystop : bool, optional
            Whether to continue training after early stopping has reached.
            The default is False.

        Returns
        -------
        None.
        """
        # define the classes
        if classes is None:
            labels = [l[2] for l in trainData.labels]
            classes = list(set(labels))

        # Define the sampler
        if sampler is not None:
            sampler = self._findSampler(sampler)

        # Create the loss function
        lossFn = self._findLossFn(lossFn)(reduction='sum')
        if loss_icp is None:
            loss_icp = NormIncreaseLoss()
        if loss_isp is None:
            loss_isp = PrototypeLoss()
        # store the experiment details.
        self.expDetails = []

        # Lets run the experiment
        expNo = 0
        original_net_dict = copy.deepcopy(self.net.state_dict())

        # set the details
        expDetail = {'expNo': expNo, 'expParam': {'optimFn': optimFns,
                                                  'lossFn': lossFn, 'lr': lr,
                                                  'stopCondi': stopCondi}}

        # Reset the network to its initial form.
        self.net.load_state_dict(original_net_dict)

        # Run the training and get the losses.
        trainResults = self._trainOE(
        trainData, valData, lossFn, optimFns, lr, stopCondi,
        optimFnArgs, classes=classes, sampler=sampler,
        loadBestModel=loadBestModel, bestVarToCheck=bestVarToCheck,
        continueAfterEarlystop=continueAfterEarlystop,
        loss_icp=loss_icp, loss_isp=loss_isp,
        optim_icp=optim_icp, optim_isp=optim_isp
    )
        # store the results and netParm
        expDetail['results'] = {'train': trainResults}
        expDetail['netParam'] = copy.deepcopy(self.net.to('cpu').state_dict())

        self.net.to(self.device)
        # If you are restoring the best model at the end of training then get the final results again.
        pred, act, l = self.predict(trainData, sampler=sampler, lossFn=lossFn, loss_icp=loss_icp, loss_isp=loss_isp)
        trainResultsBest = self.calculateResults(pred, act, classes=classes)
        trainResultsBest['loss'] = l
        pred, act, l = self.predict(valData, sampler=sampler, lossFn=lossFn)
        valResultsBest = self.calculateResults(pred, act, classes=classes)
        valResultsBest['loss'] = l
        expDetail['results']['trainBest'] = trainResultsBest
        expDetail['results']['valBest'] = valResultsBest

        # if test data is present then get the results for the test data.
        if testData is not None:
            pred, act, l = self.predict(testData, sampler=sampler, lossFn=lossFn, loss_icp=loss_icp, loss_isp=loss_isp)
            testResults = self.calculateResults(pred, act, classes=classes)
            testResults['loss'] = l
            expDetail['results']['test'] = testResults

        # Print the final output to the console:
        print("Exp No. : " + str(expNo + 1))
        print('________________________________________________')
        print("\n Train Results: ")
        print(expDetail['results']['trainBest'])
        print('\n Validation Results: ')
        print(expDetail['results']['valBest'])
        if testData is not None:
            print('\n Test Results: ')
            print(expDetail['results']['test'])

        # save the results
        if self.resultsSavePath is not None:

            # Store the graphs
            self.plotLoss(trainResults['trainLoss'], trainResults['valLoss'],
                          savePath=os.path.join(self.resultsSavePath,
                                                'exp-'+str(expNo)+'-loss.png'))
            self.plotAcc(trainResults['trainResults']['acc'],
                         trainResults['valResults']['acc'],
                         savePath=os.path.join(self.resultsSavePath,
                                               'exp-'+str(expNo)+'-acc.png'))

            # Store the data in experimental details.
            with open(os.path.join(self.resultsSavePath, 'expResults' +
                                   str(expNo)+'.dat'), 'wb') as fp:
                pickle.dump(expDetail, fp)

        # Increment the expNo
        self.expDetails.append(expDetail)
        expNo += 1

    def _trainOE(
        self,
        trainData,
        valData,
        lossFn = 'NLLLoss',
        loss_icp=None,
        loss_isp=None,
        optim_icp=None,
        optim_isp=None,
        optimFn = 'Adam',
        lr = 0.001,
        stopCondi = {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1000, 'varName' : 'epoch'}},
                                               'c2': {'NoDecrease': {'numEpochs' : 300, 'varName': 'valLoss'}} } }},
        optimFnArgs = {},
        loadBestModel = True,
        bestVarToCheck = 'valLoss',
        continueAfterEarlystop = False,
        classes = None,
        sampler = None):
        '''
        Internal function to perform the training.
        Do not directly call this function. Use train instead

        Parameters
        ----------
        trainData : eegDataset
            dataset used for training.
        valData : eegDataset
            dataset used for validation.
        lossFn : function handle from torch.nn, The default is NLLLoss
            Loss function from torch.nn which will be used for training.
        optimFn : string from torch.optim. The default is 'Adam'.
            Name of the optimization function from torch.nn which will be used for training.
        lr : float, optional
            Learning rate. The default is 0.001.
        stopCondi : dict, optional
            Determines when to stop.
            It will be a dictionary which can be accepted by stopCriteria class.
            The default is : no decrease in validation Inaccuracy in last 200 epochs OR epoch > 1500
            This default is represented as:
            {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500, 'varName' : 'epoch'}},
                                  'c2': {'NoDecrease': {'numEpochs' : 200, 'varName': 'valLoss'}} } }}
        optimFnArgs : dict, optional
            Additional arguments to be passed to the optimizer. The default is {}.
        loadBestModel : bool, optional
            Whether to load the network with best validation loss/ acc at the end of training. The default is True.
        bestVarToCheck : 'valLoss'/'valLoss', optional
            the best value to check while determining the best model . The default is 'valLoss'.
        continueAfterEarlystop : bool, optional
            Whether to continue training after early stopping has reached. The default is False.
        classes : list, optional
            List of classes to consider in evaluation matrices.
            None -> all classes.
            The default is None.
        sampler : function handle to a sampler to be used in dataloader, optional
            The sampler to use while training and validation.
            The default is None.

        Returns
        -------
        dict
            a dictionary with all the training results.
        '''

        # For reporting.
        trainResults = []
        valResults = []
        trainLoss = []
        valLoss = []
        loss = []
        bestNet = copy.deepcopy(self.net.state_dict())
        bestValue = float('inf')
        earlyStopReached = False
        if optim_icp is None:
            optim_icp = torch.optim.Adam(self.net.icp.parameters(), lr=lr, **optimFnArgs)
        if optim_isp is None:
            optim_isp = torch.optim.Adam(self.net.isp.parameters(), lr=lr, **optimFnArgs)

        bestOptimizerState_icp = copy.deepcopy(optim_icp.state_dict())
        bestOptimizerState_isp = copy.deepcopy(optim_isp.state_dict())

        # Create optimizer
        self.optimizer = self._findOptimizer(optimFn)(self.net.parameters(), lr = lr, **optimFnArgs)
        bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())

        # Initialize the stop criteria
        stopCondition = stopCriteria.composeStopCriteria(**stopCondi)


        # lets start the training.
        monitors = {'epoch': 0, 'valLoss': 10000, 'valLoss': 1}
        doStop = False

        while not doStop:
            # train the epoch.
            loss.append(self.trainOneEpoch(trainData, lossFn, optim_icp, optim_isp, sampler=sampler))

            # evaluate the training and validation accuracy.
            pred, act, l = self.predict(trainData, sampler=sampler, lossFn=lossFn, loss_icp=loss_icp, loss_isp=loss_isp)
            trainResults.append(self.calculateResults(pred, act, classes=classes))
            trainLoss.append(l)
            monitors['trainLoss'] = l
            monitors['trainInacc'] = 1 - trainResults[-1]['acc']
            pred, act, l = self.predict(valData, sampler=sampler, lossFn=lossFn, loss_icp=loss_icp, loss_isp=loss_isp)
            valResults.append(self.calculateResults(pred, act, classes=classes))
            valLoss.append(l)
            monitors['valLoss'] = l
            monitors['valInacc'] = 1 - valResults[-1]['acc']

            # print the epoch info
            print("\t \t Epoch "+ str(monitors['epoch']+1))
            print("Train loss = "+ "%.3f" % trainLoss[-1] + " Train Acc = "+
                  "%.3f" % trainResults[-1]['acc']+
                  ' Val Acc = '+ "%.3f" % valResults[-1]['acc'] +
                  " Val loss = "+ "%.3f" % valLoss[-1])

            if loadBestModel:
                if monitors[bestVarToCheck] < bestValue:
                    bestValue = monitors[bestVarToCheck]
                    bestNet = copy.deepcopy(self.net.state_dict())
                    bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())
                    bestOptimizerState_icp = copy.deepcopy(optim_icp.state_dict())
                    bestOptimizerState_isp = copy.deepcopy(optim_isp.state_dict())
            #Check if to stop
            doStop = stopCondition(monitors)

            #Check if we want to continue the training after the first stop:
            if doStop:
                # first load the best model
                if loadBestModel and not earlyStopReached:
                    self.net.load_state_dict(bestNet)
                    self.optimizer.load_state_dict(bestOptimizerState)
                    optim_icp.load_state_dict(bestOptimizerState_icp)
                    optim_isp.load_state_dict(bestOptimizerState_isp)
                # Now check if  we should continue training:
                if continueAfterEarlystop:
                    if not earlyStopReached:
                        doStop = False
                        earlyStopReached = True
                        print('Early stop reached now continuing with full set')
                        # Combine the train and validation dataset
                        trainData = TensorDataset(
                         torch.cat((trainData.tensors[0], valData.tensors[0]), dim=0),  # Combina i dati
                         torch.cat((trainData.tensors[1], valData.tensors[1]), dim=0)   # Combina le etichette
                        )

                        # define new stop criteria which is the training loss.
                        monitors['epoch'] = 0
                        modifiedStop = {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 200, 'varName' : 'epoch'}},
                                               'c2': {'LessThan': {'minValue' : monitors['valLoss'], 'varName': 'valLoss'}} } }}
                        stopCondition = stopCriteria.composeStopCriteria(**modifiedStop)
                    else:
                        bestNet = copy.deepcopy(self.net.state_dict())

            # update the epoch
            monitors['epoch'] += 1


        # Make individual list for components of trainResults and valResults
        t = {}
        v = {}

        for key in trainResults[0].keys():
            t[key] = [result[key] for result in trainResults]
            v[key] = [result[key] for result in valResults]


        return {'trainResults': t, 'valResults': v,
                'trainLoss': trainLoss, 'valLoss' : valLoss}

    def trainOneEpoch(self, trainData, lossFn_cls, optimizer_cls, optimizer_isp, optimizer_icp, 
                      loss_icp, loss_pl, sampler=None):
        """
        Train the model for one epoch with custom losses and optimizers.

        Parameters
        ----------
        trainData : eegDataset
            Dataset used for training.
        lossFn_cls : torch.nn.Module
            Classification loss function (e.g., CrossEntropyLoss).
        optimizer_cls : torch.optim.Optimizer
            Optimizer for the encoder.
        optimizer_isp : torch.optim.Optimizer
            Optimizer for the ISP component.
        optimizer_icp : torch.optim.Optimizer
            Optimizer for the ICP component.
        loss_icp : torch.nn.Module
            Custom loss function for ICP.
        loss_pl : torch.nn.Module
            Custom prototype loss function.
        Lambda1 : float
            Weight for the prototype loss.
        Lambda2 : float
            Weight for the ICP loss.
        sampler : torch.utils.data.sampler.Sampler, optional
            Sampler for the DataLoader. Default is None.

        Returns
        -------
        float
            Average training loss for the epoch.
        """
        # Set the network in training mode
        self.net.train()

        # Initialize running loss
        running_loss = 0

        # Set shuffle or sampler
        if sampler is None:
            shuffle = True
        else:
            shuffle = False
            sampler = sampler(trainData)

        # Create the DataLoader
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, shuffle=shuffle, sampler=sampler)

        # Iterate over data batches
        for data_batch, labels_batch in dataLoader:
            # Move data to device
            data_batch, labels_batch = data_batch.to(self.device), labels_batch.to(self.device)

            # Zero the parameter gradients
            optimizer_cls.zero_grad()
            optimizer_isp.zero_grad()
            optimizer_icp.zero_grad()

            # Forward pass
            output = self.net(data_batch)
            loss_cls = lossFn_cls(output, labels_batch)

            # Compute custom losses
            icp_loss = loss_icp(self.net.icp)  # ICP loss
            features = self.net.get_features()  # Extract features
            proxy = self.net.icp  # Proxies (ICP component)
            pl_loss = loss_pl(features, proxy, labels_batch)  # Prototype loss

            # Total loss
            total_loss = loss_cls + 0.00001 * pl_loss + 0.001 * icp_loss

            # Backward pass
            total_loss.backward()
            optimizer_cls.step()
            optimizer_isp.step()
            optimizer_icp.step()

            # Accumulate loss
            running_loss += total_loss.item()

        return running_loss / len(dataLoader)

    def predict(self, data, lossFn_cls=None, lossFn_isp=None, lossFn_icp=None, sampler=None):
        '''
        Predict the class of the input data and optionally calculate losses.

        Parameters
        ----------
        data : eegDataset
            Dataset of type eegDataset.
        lossFn_cls : torch.nn.Module, optional
            Classification loss function (e.g., CrossEntropyLoss).
        lossFn_isp : torch.nn.Module, optional
            Separability loss function.
        lossFn_icp : torch.nn.Module, optional
            Compactness loss function.
        alpha : float, optional
            Weight for the ISP loss. Default is 0.
        lambda_ : float, optional
            Weight for the ICP loss. Default is 0.
        sampler : function handle of type torch.utils.data.sampler, optional
            Sampler for the DataLoader. Default is None.

        Returns
        -------
        predicted : list
            List of predicted labels.
        actual : list
            List of actual labels.
        loss : float or None
            Average total loss (if any loss function is provided), otherwise None.
        '''

        predicted = []
        actual = []
        total_loss = 0
        total_samples = 0

        # Set the network to evaluation mode
        self.net.eval()

        # Initiate the DataLoader
        dataLoader = DataLoader(data, batch_size=self.batchSize, shuffle=False, sampler=sampler)

        # Disable gradient calculation
        with torch.no_grad():
            for data_batch, labels_batch in dataLoader:
                # Move data to the appropriate device
                data_batch, labels_batch = data_batch.to(self.device), labels_batch.to(self.device)

                # Forward pass: compute predictions
                preds = self.net(data_batch)
                total_samples += data_batch.shape[0]

                # Compute classification loss (Ls), if provided
                if lossFn_cls is not None:
                    cls_loss = lossFn_cls(preds, labels_batch)
                else:
                    cls_loss = 0

                # Compute ISP loss (separability), if provided
                if lossFn_isp is not None:
                    features = self.net.get_features()  # Extract features
                    isp_loss = lossFn_isp(features)
                else:
                    isp_loss = 0

                # Compute ICP loss (compactness), if provided
                if lossFn_icp is not None:
                    icp_loss = lossFn_icp(self.net.icp)
                else:
                    icp_loss = 0

                # Total loss
                total_loss += cls_loss + 0.00001 * isp_loss + 0.001 * icp_loss

                # Convert the output of softmax to class labels
                _, preds = torch.max(preds, 1)
                predicted.extend(preds.data.tolist())
                actual.extend(labels_batch.tolist())

        # Return the predicted labels, actual labels, and average loss
        avg_loss = total_loss / total_samples if (lossFn_cls or lossFn_isp or lossFn_icp) else None
        return predicted, actual, avg_loss
    def calculateResults(self, yPredicted, yActual, classes = None):
        '''
        Calculate the results matrices based on the actual and predicted class.

        Parameters
        ----------
        yPredicted : list
            List of predicted labels.
        yActual : list
            List of actual labels.
        classes : list, optional
            List of labels to index the CM.
            This may be used to reorder or select a subset of class labels.
            If None then, the class labels that appear at least once in
            yPredicted or yActual are used in sorted order.
            The default is None.

        Returns
        -------
        dict
            a dictionary with fields:
                acc : accuracy.
                cm  : confusion matrix..

        '''

        acc = accuracy_score(yActual, yPredicted)
        if classes is not None:
            cm = confusion_matrix(yActual, yPredicted, labels= classes)
        else:
            cm = confusion_matrix(yActual, yPredicted)

        return {'acc': acc, 'cm': cm}

    def plotLoss(self, trainLoss, valLoss, savePath = None):
        '''
        Plot the training loss.

        Parameters
        ----------
        trainLoss : list
            Training Loss.
        valLoss : list
            Validation Loss.
        savePath : str, optional
            path to store the figure. The default is None: figure will be plotted.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.title("Training Loss vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1,len(trainLoss)+1),trainLoss,label="Train loss")
        plt.plot(range(1,len(valLoss)+1),valLoss,label="Validation Loss")
        plt.legend(loc='upper right')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            plt.show()
        plt.close()

    def plotAcc(self, trainAcc, valAcc, savePath= None):
        '''
        Plot the train and validation accuracy.
        '''
        plt.figure()
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,len(trainAcc)+1),trainAcc,label="Train Acc")
        plt.plot(range(1,len(valAcc)+1),valAcc,label="Validation Acc")
        plt.ylim((0,1.))
        plt.legend(loc='upper right')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            plt.show()
        plt.close()

    def setRandom(self, seed):
        '''
        Set all the random initializations with a given seed

        Parameters
        ----------
        seed : int
            seed.

        Returns
        -------
        None.

        '''
        self.seed = seed

        # Set np
        np.random.seed(self.seed)

        # Set torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Set cudnn
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setDevice(self, nGPU = 0):
        '''
        Set the device for training and testing

        Parameters
        ----------
        nGPU : int, optional
            GPU number to train on. The default is 0.

        Returns
        -------
        None.

        '''
        if self.device is None:
            if self.preferedDevice == 'gpu':
                self.device = torch.device("cuda:"+str(nGPU) if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device('cpu')
            print("Code will be running on device ", self.device)

    def _findOptimizer(self, optimString):
        '''
        Look for the optimizer with the given string and then return the function handle of that optimizer.
        '''
        out  = None
        if optimString in optim.__dict__.keys():
            out = optim.__dict__[optimString]
        else:
            raise AssertionError('No optimizer with name :' + optimString + ' can be found in torch.optim. The list of available options in this module are as follows: ' + str(optim.__dict__.keys()))
        return out

    def _findSampler(self, givenString):
        '''
        Look for the sampler with the given string and then return the function handle of the same.
        '''
        out  = None
        if givenString in builtInSampler.__dict__.keys():
            out = builtInSampler.__dict__[givenString]
        elif givenString in samplers.__dict__.keys():
            out = samplers.__dict__[givenString]
        else:
            raise AssertionError('No sampler with name :' + givenString + ' can be found')
        return out

    def _findLossFn(self, lossString):
        '''
        Look for the loss function with the given string and then return the function handle of that function.
        '''
        out  = None
        if lossString in nn.__dict__.keys():
            out = nn.__dict__[lossString]
        else:
            raise AssertionError('No loss function with name :' + lossString + ' can be found in torch.nn. The list of available options in this module are as follows: ' + str(nn.__dict__.keys()))

        return out


import torch
import torch.nn as nn


class PrototypeLoss(nn.Module):

    def forward(self, features, proxy, labels):

        label_prototypes = torch.index_select(proxy, dim=0, index=labels)

        pl = huber_loss(features, label_prototypes, sigma=1)
        pl_loss = torch.mean(pl)

        return pl_loss


def huber_loss(input, target, sigma=1):
    beta = 1.0 / (sigma**2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)

    return torch.sum(loss, dim=1)

class NormIncreaseLoss(nn.Module):
    def __init__(self):
        super(NormIncreaseLoss, self).__init__()

    def forward(self, mat):

        norms = torch.norm(mat, p=2, dim=1)
        loss = -norms
        return loss.mean()
