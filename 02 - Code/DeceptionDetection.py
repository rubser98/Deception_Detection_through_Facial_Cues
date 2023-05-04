import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from DeceptionDataset import DeceptionDataset
import os
import pandas as pd
import numpy as np

# t tensore, n numero di frame di un video
def pre_padding(t: torch.Tensor, n: int) -> torch.Tensor:
    numero_padding = abs(t.shape[0] - n)
    tt = torch.zeros(numero_padding, 4466)
    t_pad = torch.tensor([torch.cat([tt,t],dim=0).numpy()])
    return t_pad

#carico le feature in un numpy array, struttura riconosciuta dal DeceptionDataset
def feature_to_numpy():
        path_dec = 'C:\\Users\\Administrator\\Desktop\\Tesi\\Dataset\\Trial_finale\\Deceptive\\'
        path_truth = 'C:\\Users\\Administrator\\Desktop\\Tesi\\Dataset\\Trial_finale\\Truthful\\'
        lista_file_deceptive = [f for f in os.listdir(path_dec)]
        lista_file_truthful = [f for f in os.listdir(path_truth)]
        X_data = torch.zeros(1,1,4466)

        Y_data = []
        #carico i dati di bugia
        j = 1
        for dec in lista_file_deceptive:

            df = pd.read_csv(path_dec+dec,sep=';',header=0)
            #inizializzo vettore del file con [0,1,..,4665] per evitare errore nell'append di array di ogni frame
            #successivamente riga 0 viene scartata
            x_file = []
            for rows in df.iterrows():
                x_frame = np.array(rows[1][1:], dtype=np.float32)
                x_file.append(x_frame)
            #print(x_file)
            x_file = torch.from_numpy(np.array([x_file]))
            #se lunghezza sequenza attuale è maggiore rispetto al video preso in considerazione
            #faccio il pre padding della sequenza del video in considerazione e aggiungo il tensore alla lista
            if X_data.shape[1] > x_file.shape[1]:
                X_data = torch.cat([X_data, pre_padding(x_file[0],X_data.shape[1])],dim=0)
            #se il video è una sequenza più lunga faccio il pre padding di tutti i video nella lista 
            elif X_data.shape[1] < x_file.shape[1]:
                list_tensori = []
                for i in range(X_data.shape[0]):
                    t = pre_padding(X_data[i],x_file.shape[1])
                    list_tensori.append(t)
                list_tensori.append(x_file)
                X_data = torch.cat(list_tensori,dim=0)
            #se sono uguali aggiungo il vettore alla lista e basta
            else:
                X_data = torch.cat([X_data, x_file],dim=0)
            Y_data.append(1)
            j+=1
        j=1
        #carico i dati di verità
        for tru in lista_file_truthful:
            df = pd.read_csv(path_truth+tru,sep=';',header=0)
            #inizializzo vettore del file con [0,1,..,4665] per evitare errore nell'append di array di ogni frame
            #successivamente riga 0 viene scartata
            x_file = []
            for rows in df.iterrows():
                x_frame = np.array(rows[1][1:], dtype=np.float32)
                x_file.append(x_frame)
            x_file = torch.from_numpy(np.array([x_file]))
            #se lunghezza sequenza attuale è maggiore rispetto al video preso in considerazione
            #faccio il pre padding della sequenza del video in considerazione e aggiungo il tensore alla lista 
            if X_data.shape[1] > x_file.shape[1]:
                X_data = torch.cat([X_data, pre_padding(x_file[0],X_data.shape[1])],dim=0)
            #se il video è una sequenza più lunga faccio il pre padding di tutti i video nella lista 
            elif X_data.shape[1] < x_file.shape[1]:
                list_tensori = []
                for i in range(X_data.shape[0]):
                    t = pre_padding(X_data[i],x_file.shape[1])
                    list_tensori.append(t)
                list_tensori.append(x_file)
                X_data = torch.cat(list_tensori,dim=0)
            #aggiungo video alla lista e basta
            else:
                X_data = torch.cat([X_data, x_file],dim=0)
            Y_data.append(0)
            j+=1
        
        X_data = X_data[1:]
        return X_data.numpy(), np.array(Y_data, dtype=np.float32)



#classe che rappresenta LSTM 
class LSTMDeception(nn.Module):
    def __init__(self, input_size, n_hidden=51):
        super(LSTMDeception,self).__init__()
        self.n_hidden = n_hidden
        #input shape: batch_size, seq_length, input_size
        self.lstm = nn.LSTM(input_size,n_hidden,1,batch_first=True)
        #output layer
        self.linear = nn.Linear(self.n_hidden, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        #inizializzo hidden state e cell state
        h_t = torch.zeros(1,x.size(0), self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(1,x.size(0), self.n_hidden, dtype=torch.float32)
        #out dimension: batch_size, sequence_length, n_hidden
        out, _ = self.lstm(x,(h_t,c_t))
        #prendo output dell'ultimo frame
        #out dimension: batch_size, n_hidden
        out = out[:,-1, :]
        #applico sigmoide ad output per avere valori tra 0 e 1
        out = self.sigmoid(self.linear(out))
        return out.reshape(-1)

if __name__ == '__main__':
    #preparazione dataset
    X_data, y = feature_to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.3, random_state=42, shuffle=True, stratify=y )
    X_test = torch.from_numpy(X_test)

    #definizione parametri modello
    symmetry = True
    n_feature = 4466 if symmetry else 4464

    batch_size = 10
    #istanzio dataset che estende torch.Dataset
    train_dataset = DeceptionDataset(X_train, y_train, symmetry)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    learning_rate = 0.01
    model = LSTMDeception(n_feature)
    #binary cross entropy loss per classificazione binaria
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    n_epochs = 50
    #ciclo di training
    for epoch in range(n_epochs):
        for i, (input, label) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(input)
            l = loss(out,label)
            l.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
                print (f'Epoch [{epoch+1}/{n_epochs}], Loss: {l.item():.4f}')

    #fase di test
    with torch.no_grad():
        if not symmetry:
            X_test = X_test[:,:,:-2]
        y_pred = model(X_test)
        # classe verità: 0 <= x < 0.5 
        # classe bugia: 0.5 <= x <= 1
        accuracy = accuracy_score(y_test, torch.round(y_pred).numpy())
        print(f'accuracy = {accuracy}')
        #f1 = f1_score(y_test, torch.round(y_pred).numpy())
        #print(f'f1 = {f1}')

    
    
    
    
    
